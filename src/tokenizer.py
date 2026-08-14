"""トークナイザ2種.

CharTokenizer は1文字=1トークン。辞書は「コーパスに出てきた文字の集合」だけで
済むので学習が要らず、仕組みを読むのがいちばん簡単になる。そのかわり
1トークンあたりの情報量が小さく、文脈256トークンが256文字にしかならない。

SubwordTokenizer は SentencePiece (unigram) で語彙を学習する。
1トークンが平均で複数文字を運ぶぶん、同じ文脈長でより長い文章が入る。

どちらも同じインタフェース (encode / decode / vocab_size / *_id / save / load) を
持たせてあるので、model.py と train.py は差し替えを意識しなくていい。

会話の役割を表すマーカー (<|user|> など) は、どちらの方式でも
1トークンとして扱う。分解してしまうとモデルが「開始記号」を学ぶために
無駄な容量を使い、生成時に途中まで壊れた記号を出す原因になる。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END = "<|end|>"
UNK = "<|unk|>"

# UNK は語彙に必ず入れるが、コーパス側には現れない (推論時の未知文字用)。
SPECIAL_TOKENS = (UNK, USER, ASSISTANT, END)
_MARKER_RE = re.compile("(" + "|".join(re.escape(t) for t in (USER, ASSISTANT, END)) + ")")

CONFIG_NAME = "tokenizer.json"
SPM_NAME = "tokenizer.model"


class CharTokenizer:
    def __init__(self, itos: list[str]):
        self.itos = list(itos)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.unk_id = self.stoi[UNK]
        self.user_id = self.stoi[USER]
        self.assistant_id = self.stoi[ASSISTANT]
        self.end_id = self.stoi[END]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @classmethod
    def train(cls, text: str, min_freq: int = 1) -> CharTokenizer:
        """コーパスから語彙を作る. マーカーは除いた上で文字を数える."""
        counts = Counter()
        for chunk in _MARKER_RE.split(text):
            if chunk in SPECIAL_TOKENS:
                continue
            counts.update(chunk)
        chars = sorted(c for c, n in counts.items() if n >= min_freq)
        return cls(list(SPECIAL_TOKENS) + chars)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in _MARKER_RE.split(text):
            if chunk in SPECIAL_TOKENS:
                ids.append(self.stoi[chunk])
            else:
                ids.extend(self.stoi.get(c, self.unk_id) for c in chunk)
        return ids

    def decode(self, ids, skip_special: bool = True) -> str:
        out = []
        for i in ids:
            tok = self.itos[int(i)]
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        return "".join(out)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / CONFIG_NAME).write_text(
            json.dumps({"type": "char", "itos": self.itos}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> CharTokenizer:
        data = json.loads((Path(directory) / CONFIG_NAME).read_text(encoding="utf-8"))
        return cls(data["itos"])


class SubwordTokenizer:
    """SentencePiece (unigram) によるサブワード分割.

    設定で効いてくるのは次の3つ。

    byte_fallback
        未知の文字をバイト列に落として表現する。これを入れておけば
        「語彙に無い文字」が原理的に消えるので、コーパス側で低頻度文字を
        捨てる必要がなくなる。CharTokenizer の UNK に相当する事故が起きない。

    user_defined_symbols
        <|user|> などを分割対象から外して1トークンに固定する。

    split_by_whitespace=False
        日本語は単語が空白で区切られないうえ、こちらのコーパスは
        整形時に改行を半角スペースへ潰してある。空白で区切ると
        「スペースを含む長いトークン」が量産されて語彙が無駄になる。

    既定のままだと困るものが3つあるので、いずれも切ってある。
    add_dummy_prefix は文頭に ▁ を足してしまい、必ず <|user|> で始まる
    こちらの入力では1トークンぶん無駄になる。remove_extra_whitespaces は
    空白を勝手に詰める。そして normalization_rule_name の既定 (nmt_nfkc) は
    NFKC 正規化をかけるので、「？」「！」が半角の「?」「!」に化ける。
    日本語の見た目が変わるうえ、文字レベル版と同じ文章で比較できなくなる。
    """

    def __init__(self, model_path: str | Path):
        import sentencepiece as spm

        self.model_path = Path(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=str(self.model_path))
        self.unk_id = self.sp.piece_to_id(UNK)
        self.user_id = self.sp.piece_to_id(USER)
        self.assistant_id = self.sp.piece_to_id(ASSISTANT)
        self.end_id = self.sp.piece_to_id(END)
        self._special_ids = {self.unk_id, self.user_id, self.assistant_id, self.end_id}

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    # SentencePiece に渡す共通の設定。text からでもファイルからでも同じにする。
    _TRAIN_KWARGS = dict(
        model_type="unigram",
        byte_fallback=True,
        split_by_whitespace=False,
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        normalization_rule_name="identity",
        unk_piece=UNK,
        user_defined_symbols=[USER, ASSISTANT, END],
        # 既定の <s> / </s> は使わないので語彙から外す。
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
        minloglevel=1,
    )

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        model_dir: str | Path,
        character_coverage: float = 0.9995,
    ) -> SubwordTokenizer:
        """メモリに載る規模のコーパス (数千万文字まで) から語彙を学習する.

        1億文字を超えるコーパスでは text を丸ごと持てないので
        train_from_files を使うこと。
        """
        import sentencepiece as spm

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        prefix = model_dir / "tokenizer"

        # SentencePiece は入力をファイルか iterable で受ける。
        # 1会話1行なので、行を渡せばそのまま文の単位になる。
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(text.splitlines()),
            model_prefix=str(prefix),
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            train_extremely_large_corpus=False,
            **cls._TRAIN_KWARGS,
        )
        return cls(prefix.with_suffix(".model"))

    @classmethod
    def train_from_files(
        cls,
        input_files: list[str | Path],
        vocab_size: int,
        model_dir: str | Path,
        character_coverage: float = 0.9995,
        input_sentence_size: int = 2_000_000,
        seed: int = 1234,
        max_sentence_length: int = 8_192,
    ) -> SubwordTokenizer:
        """10億文字級のコーパスから語彙を学習する.

        train() との違いは4つあって、どれも入れないと落ちるか、質が下がる。

        input_sentence_size / shuffle_input_sentence
            unigram の学習は全文をメモリに持つので、10億文字を渡すと
            数十GB要求して落ちる。ここで指定した行数だけを標本にする。
            shuffle と併用しないと「先頭の n 行」だけを見ることになり、
            コーパスの先頭に偏ったデータ (同じサイトの連続など) が
            そのまま語彙に反映される。必ず両方入れる。

        train_extremely_large_corpus
            unigram の内部カウンタを64bitにする。大きいコーパスで
            これを切ったままだと桁溢れして語彙が壊れる。

        max_sentence_length
            **既定は 4192 バイトで、これを超える行は黙って捨てられる**。
            警告も出ない。事前学習コーパスは1行1文書で平均 1,500文字
            (UTF-8 で 4,500バイト前後) なので、既定のままだと
            半分以上が語彙の学習に使われない。気づかないまま
            「なぜか語彙の質が悪い」状態になる。
            呼び出し側で文単位に割ってから渡すのが本筋だが、
            保険としてここでも引き上げておく。

        input はファイルのパスで渡す。Python 側で行を作って
        sentence_iterator に流すと、そのイテレータが全部メモリに載る。
        """
        import sentencepiece as spm

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        prefix = model_dir / "tokenizer"

        # shuffle_input_sentence で標本を引くので、種を固定しないと
        # 同じコーパスから毎回違う語彙が出てくる。TrainerSpec には
        # seed のフィールドが無く、モジュール側の関数で設定する。
        spm.set_random_generator_seed(seed)
        spm.SentencePieceTrainer.train(
            input=[str(p) for p in input_files],
            model_prefix=str(prefix),
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            input_sentence_size=input_sentence_size,
            shuffle_input_sentence=True,
            train_extremely_large_corpus=True,
            max_sentence_length=max_sentence_length,
            num_threads=8,
            **cls._TRAIN_KWARGS,
        )
        return cls(prefix.with_suffix(".model"))

    def encode_iter(self, texts: list[str]) -> list[list[int]]:
        """複数の文字列をまとめて encode する. C++ 側で回るので速い."""
        return self.sp.encode(texts, out_type=int)

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids, skip_special: bool = True) -> str:
        ids = [int(i) for i in ids]
        if skip_special:
            ids = [i for i in ids if i not in self._special_ids]
        return self.sp.decode(ids)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / SPM_NAME).write_bytes(self.model_path.read_bytes())
        (directory / CONFIG_NAME).write_text(
            json.dumps({"type": "subword", "vocab_size": self.vocab_size}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> SubwordTokenizer:
        return cls(Path(directory) / SPM_NAME)


Tokenizer = CharTokenizer | SubwordTokenizer


def load_tokenizer(directory: str | Path) -> Tokenizer:
    """保存先のディレクトリから、方式を見て適切なトークナイザを返す.

    type を持たない tokenizer.json は文字レベルだった頃のもの。
    古いチェックポイントをそのまま読めるようにしておく。
    """
    directory = Path(directory)
    data = json.loads((directory / CONFIG_NAME).read_text(encoding="utf-8"))
    if data.get("type", "char") == "subword":
        return SubwordTokenizer.load(directory)
    return CharTokenizer(data["itos"])
