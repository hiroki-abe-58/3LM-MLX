"""コーパスのテキストファイルを、学習が直接読めるトークン列 (.bin) に変換する.

その2 までは「コーパス全文を str で読み、tokenizer.encode(text) で
list[int] を受け取る」で足りていた。949万文字・469万トークンなら、
リストが数十MBで済む。

この規模ではそれが通らない。5億トークンを Python の list[int] で持つと、
リスト本体のポインタ配列だけで 8 byte × 5億 = 4GB、さらに小整数キャッシュ
(-5〜256) を外れた要素は 1個 28 byte の int オブジェクトになるので
合計で 18GB 前後になる。ユニファイドメモリの Mac ではこれが
そのままカーネルパニックの引き金になる。その3 で一度やった。

なのでここでは、

  1. コーパスを行 (=1文書) 単位でストリーム読みする
  2. 数千行ずつまとめて encode する (SentencePiece の C++ 側で回るので速い)
  3. その場で np.uint16 にして memmap へ書き、Python 側には残さない

Python が同時に持つのはチャンク1個ぶんだけになる。

uint16 を選ぶのは語彙 32,000 が 65,536 に収まるから。int32 にすると
ファイルが倍になり、読み出しの帯域がそのまま学習速度に効いてくる。

文書の区切りには <|end|> を入れる。事前学習の時点で
「文章はここで終わる」を教えておくと、SFT で応答を止める挙動が乗りやすい。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tokenizer import END, SubwordTokenizer, load_tokenizer  # noqa: E402

# uint16 に入る上限。語彙をこれ以上にするなら dtype を変えること。
UINT16_MAX = 65_536

# 一度に encode する行数。大きいほど C++ 側の呼び出し回数が減るが、
# 中間の list[list[int]] がメモリに乗るので、このあたりで止める。
CHUNK_LINES = 4_000

# 文の切れ目。区切り文字は前の断片に残す ((?<=...) を使う)。
SENTENCE_END = re.compile(r"(?<=[。！？\n])")


def split_sentences(text: str, min_bytes: int, max_bytes: int) -> Iterator[str]:
    """1文書を、語彙の学習に渡しやすい長さの断片に割る.

    SentencePiece の unigram 学習は「1行=1文」を想定している。
    1行に1文書 (平均1,500文字) を渡すと2つ困ることがある。

      1. max_sentence_length を超えた行が黙って捨てられる
      2. 標本を「行」で引くので、長い文書も短い文書も1票になる。
         長い文書の中身がほとんど見られないまま語彙が決まる

    句点で割って、短すぎる断片は後ろとくっつける。日本語には空白の
    区切りが無いので、句点・感嘆符・疑問符・改行を境目に使う。
    """
    pieces = SENTENCE_END.split(text)
    buffer = ""
    for piece in pieces:
        if not piece:
            continue
        buffer += piece
        if len(buffer.encode("utf-8")) >= min_bytes:
            yield from _hard_split(buffer, max_bytes)
            buffer = ""
    if buffer.strip():
        yield from _hard_split(buffer, max_bytes)


def _hard_split(text: str, max_bytes: int) -> Iterator[str]:
    """句点が無いまま延々と続く文書を、上限のバイト数で切る."""
    if len(text.encode("utf-8")) <= max_bytes:
        yield text
        return
    # 3バイト文字を仮定して文字数に落とす。多少ぶれても上限内に収まる。
    step = max(1, max_bytes // 3)
    for i in range(0, len(text), step):
        chunk = text[i : i + step]
        if chunk.strip():
            yield chunk


def write_spm_sample(
    corpus_files: list[Path],
    out_path: Path,
    target_chars: int,
    every: int,
    min_bytes: int = 200,
    max_bytes: int = 6_000,
) -> dict:
    """語彙の学習用に、文単位に割った標本ファイルを書き出す.

    every 文書ごとに1本を採る。先頭から順に target_chars ぶん取ると
    コーパスの前の方 (このコーパスでは青空文庫) に偏る。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    docs = kept = lines = chars = 0
    started = time.time()
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for batch in iter_line_chunks(corpus_files, chunk_lines=CHUNK_LINES):
            for line in batch:
                take = docs % every == 0
                docs += 1
                if not take:
                    continue
                kept += 1
                for sentence in split_sentences(line, min_bytes, max_bytes):
                    fh.write(sentence.replace("\n", " ").strip() + "\n")
                    lines += 1
                    chars += len(sentence)
            print(
                f"\r  [標本] {kept:,} 文書 / {lines:,} 行 / {chars:,} 文字 "
                f"({chars / max(target_chars, 1) * 100:.0f}%)",
                end="", flush=True,
            )
            if chars >= target_chars:
                break
    print()
    return {
        "path": str(out_path), "documents_scanned": docs, "documents_kept": kept,
        "lines": lines, "chars": chars, "every": every,
        "seconds": round(time.time() - started, 1),
    }


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def iter_line_chunks(
    paths: list[Path],
    chunk_lines: int = CHUNK_LINES,
    max_chars: int = 0,
    every: int = 1,
) -> Iterator[list[str]]:
    """複数のファイルを1本の行ストリームとして扱い、chunk_lines ずつ返す.

    max_chars
        その文字数に達したところで打ち切る。コーパスの**先頭** N 文字。

    every
        every 文書ごとに1本だけ採る。コーパス全体に散らして
        部分集合を作るときに使う。
    """
    buffer: list[str] = []
    seen = 0
    index = 0
    for path in paths:
        # newline="" にはしない。コーパスは LF で書いてある前提。
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                take = index % every == 0
                index += 1
                if not take:
                    continue
                buffer.append(line)
                seen += len(line)
                if len(buffer) >= chunk_lines:
                    yield buffer
                    buffer = []
                if max_chars and seen >= max_chars:
                    if buffer:
                        yield buffer
                    return
    if buffer:
        yield buffer


def bytes_per_char(paths: list[Path], probe: int = 8 << 20) -> float:
    """先頭を少し読んで、1文字あたりのバイト数を実測する.

    「日本語は3バイト」と決め打ちにすると、ASCII の多いコーパスで
    総文字数を大きく外す。memmap の確保量にも使うので、
    外すと途中で「見積もりを超えました」で落ちる。
    """
    with paths[0].open("rb") as fh:
        raw = fh.read(probe)
    # 途中で切れたマルチバイト文字は捨てる
    text = raw.decode("utf-8", errors="ignore")
    return len(raw) / max(len(text), 1)


def estimate_chars(paths: list[Path]) -> float:
    return sum(p.stat().st_size for p in paths) / bytes_per_char(paths)


def stride_for(corpus_files: list[Path], sample_chars: int) -> int:
    """sample_chars ぶんを全体に散らして採るための間隔を返す.

    データ量を変えた比較で「コーパスの先頭 N 文字」を使うと、
    このコーパスは青空文庫 → FineWeb2 の順に書いてあるので、
    **小さい条件だけ文語100%**になる。量の効果を測りたいのに
    ドメインが一緒に動いてしまう。等間隔に採ればどの条件も
    同じ混ざり方になる。
    """
    if sample_chars <= 0:
        return 1
    return max(1, round(estimate_chars(corpus_files) / sample_chars))


class BinWriter:
    """uint16 のトークン列を memmap へ順に書き足す.

    総トークン数は encode し終わるまで分からないので、上限で memmap を作り、
    最後に os.truncate で切り詰める。上限は「1トークン=1文字以上」から
    文字数で取れる (どんな分割でも文字数を超えるトークン数にはならない)。
    """

    def __init__(self, path: Path, capacity: int):
        self.path = path
        self.capacity = capacity
        path.parent.mkdir(parents=True, exist_ok=True)
        self.arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(capacity,))
        self.written = 0

    def write(self, tokens: np.ndarray) -> None:
        n = tokens.size
        if self.written + n > self.capacity:
            raise RuntimeError(
                f"{self.path.name}: 見積もり {self.capacity:,} トークンを超えました "
                f"({self.written + n:,})。--capacity-scale を上げてください。"
            )
        self.arr[self.written : self.written + n] = tokens
        self.written += n

    def close(self) -> int:
        self.arr.flush()
        del self.arr  # memmap を閉じないと truncate がファイルサイズに反映されない
        os.truncate(self.path, self.written * 2)
        return self.written


def _capacity_for(paths: list[Path], max_chars: int, scale: float) -> int:
    """必要な memmap の大きさを見積もる.

    トークン数は文字数を超えないので、文字数の見積もりを上限に使える。
    等間隔サンプリングでは狙った文字数を少し超えることがあるので 1.3 倍。
    """
    chars = max_chars if max_chars else estimate_chars(paths)
    return int(chars * 1.3 * scale) + 1_000_000


def _encode_to(
    writer: BinWriter,
    corpus_files: list[Path],
    tokenizer: SubwordTokenizer,
    max_chars: int,
    label: str,
    every: int = 1,
) -> tuple[int, int]:
    """コーパスを1本の bin に流し込み、(文書数, 文字数) を返す."""
    end_id = tokenizer.end_id
    docs = 0
    chars = 0
    started = time.time()
    for lines in iter_line_chunks(corpus_files, max_chars=max_chars, every=every):
        chars += sum(len(s) for s in lines)
        buf: list[int] = []
        for line, ids in zip(lines, tokenizer.encode_iter(lines), strict=True):
            buf.extend(ids)
            # 対話コーパスは行末が既に <|end|> なので二重に足さない。
            if not line.endswith(END):
                buf.append(end_id)
            docs += 1
        writer.write(np.asarray(buf, dtype=np.uint16))
        elapsed = time.time() - started
        print(
            f"\r  [{label}] {docs:,} 文書 / {chars:,} 文字 / "
            f"{writer.written:,} トークン / {chars / max(elapsed, 1e-6) / 1e6:.1f}M文字每秒",
            end="",
            flush=True,
        )
    print()
    return docs, chars


def encode_corpus(
    corpus_files: list[Path],
    tokenizer: SubwordTokenizer,
    out_dir: Path,
    val_corpus: list[Path] | None = None,
    val_every: int = 1_000,
    val_max_tokens: int = 2_000_000,
    max_chars: int = 0,
    sample_chars: int = 0,
    capacity_scale: float = 1.0,
    extra: dict | None = None,
) -> dict:
    """コーパスを train.bin / val.bin に変換し、メタ情報を返す.

    検証データの作り方が2通りある。

    val_corpus を渡した場合
        そのファイルを丸ごと val.bin にする。**データ量を変えた比較を
        するときは必ずこちら**。条件ごとに違う検証セットを使うと、
        val_loss の差がデータ量の効果なのか検証セットの違いなのか
        分からなくなる。

    渡さない場合
        「val_every 文書に1本」を train から抜く。トークン列の末尾を
        切る方式だと train と val が同じ文書の前半・後半になり、
        検証が甘くなるので、文書単位で抜く。
    """
    if tokenizer.vocab_size > UINT16_MAX:
        raise SystemExit(f"語彙 {tokenizer.vocab_size} は uint16 に入りません")

    started = time.time()
    every = stride_for(corpus_files, sample_chars)
    capacity = _capacity_for(corpus_files, max_chars or sample_chars, capacity_scale)
    train = BinWriter(out_dir / "train.bin", capacity)

    if val_corpus:
        docs, chars = _encode_to(train, corpus_files, tokenizer, max_chars, "train", every)
        val = BinWriter(out_dir / "val.bin", _capacity_for(val_corpus, 0, capacity_scale))
        val_docs, val_chars = _encode_to(val, val_corpus, tokenizer, 0, "val")
        chars += val_chars
    else:
        val = BinWriter(out_dir / "val.bin", min(capacity, val_max_tokens * 2))
        docs = chars = val_docs = 0
        end_id = tokenizer.end_id
        for lines in iter_line_chunks(corpus_files, max_chars=max_chars, every=every):
            chars += sum(len(s) for s in lines)
            train_buf: list[int] = []
            val_buf: list[int] = []
            for line, ids in zip(lines, tokenizer.encode_iter(lines), strict=True):
                to_val = docs % val_every == 0 and val.written + len(ids) < val_max_tokens
                target = val_buf if to_val else train_buf
                val_docs += to_val
                target.extend(ids)
                if not line.endswith(END):
                    target.append(end_id)
                docs += 1
            if train_buf:
                train.write(np.asarray(train_buf, dtype=np.uint16))
            if val_buf:
                val.write(np.asarray(val_buf, dtype=np.uint16))
            elapsed = time.time() - started
            print(
                f"\r  {docs:,} 文書 / {chars:,} 文字 / {train.written:,} トークン / "
                f"{chars / max(elapsed, 1e-6) / 1e6:.1f}M文字每秒",
                end="",
                flush=True,
            )
        print()

    n_train = train.close()
    n_val = val.close()
    meta = {
        "vocab_size": tokenizer.vocab_size,
        "dtype": "uint16",
        "train_tokens": n_train,
        "val_tokens": n_val,
        "documents": docs,
        "val_documents": val_docs,
        "chars": chars,
        "chars_per_token": round(chars / max(1, n_train + n_val), 4),
        "max_chars": max_chars or None,
        "sample_chars": sample_chars or None,
        "sample_every": every if every > 1 else None,
        "corpus_files": [
            {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in corpus_files
        ],
        "val_corpus_files": [p.name for p in val_corpus] if val_corpus else None,
        "encoded_seconds": round(time.time() - started, 1),
        **(extra or {}),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="コーパスを train.bin / val.bin にする")
    ap.add_argument("--corpus", nargs="+", required=True, help="テキストファイル (1行1文書)")
    ap.add_argument("--out-dir", default="data/3lm")
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="学習済みトークナイザのディレクトリ。省略時は --corpus から学習する",
    )
    ap.add_argument("--val-corpus", nargs="*", default=None,
                    help="検証専用のファイル。データ量を変えた比較ではこれを固定する")
    ap.add_argument("--max-chars", type=int, default=0,
                    help="コーパスの先頭この文字数だけ使う")
    ap.add_argument("--sample-chars", type=int, default=0,
                    help="この文字数ぶんをコーパス全体から等間隔に採る "
                         "(データ量の比較はこちら。先頭だけだとドメインが偏る)")
    ap.add_argument("--vocab-size", type=int, default=32_000)
    ap.add_argument("--input-sentence-size", type=int, default=2_000_000)
    ap.add_argument("--spm-sample-chars", type=int, default=300_000_000,
                    help="語彙の学習に使う文字数。コーパス全体から等間隔に採る")
    ap.add_argument("--keep-spm-sample", action="store_true",
                    help="語彙の学習に使った標本ファイルを残す")
    ap.add_argument("--tokenizer-only", action="store_true",
                    help="語彙を学習して保存するだけで、エンコードはしない")
    ap.add_argument("--val-every", type=int, default=1_000)
    ap.add_argument("--val-max-tokens", type=int, default=2_000_000)
    ap.add_argument("--capacity-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    corpus_files = [Path(p) for p in args.corpus]
    val_corpus = [Path(p) for p in args.val_corpus] if args.val_corpus else None
    missing = [p for p in [*corpus_files, *(val_corpus or [])] if not p.exists()]
    if missing:
        raise SystemExit(f"見つかりません: {missing}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spm_sample = None
    if args.tokenizer:
        tokenizer = load_tokenizer(Path(args.tokenizer))
        print(f"トークナイザを読み込み: {args.tokenizer} (語彙 {tokenizer.vocab_size:,})")
        # 出力先にも必ず置く。src/train.py は「データと語彙は同じ場所にある」
        # 前提で読むので、共通の語彙を使い回すときは複製しておかないと
        # 学習側で FileNotFoundError になる。
        tokenizer.save(out_dir / "tokenizer")
    else:
        total_chars = sum(p.stat().st_size for p in corpus_files) / 3  # UTF-8 日本語の目安
        every = max(1, round(total_chars / max(args.spm_sample_chars, 1)))
        print(f"語彙の学習用に文単位の標本を作ります "
              f"({args.spm_sample_chars / 1e6:.0f}M文字 / {every} 文書ごとに1本)")
        spm_sample = write_spm_sample(
            corpus_files, out_dir / "spm_sample.txt", args.spm_sample_chars, every
        )

        print(f"SentencePiece {args.vocab_size:,} を学習します "
              f"(標本 {args.input_sentence_size:,} 行)")
        started = time.time()
        tokenizer = SubwordTokenizer.train_from_files(
            [Path(spm_sample["path"])],
            vocab_size=args.vocab_size,
            model_dir=out_dir / "tokenizer",
            input_sentence_size=args.input_sentence_size,
            seed=args.seed,
        )
        tokenizer.save(out_dir / "tokenizer")
        print(f"  語彙 {tokenizer.vocab_size:,} / {time.time() - started:.0f}秒")
        if not args.keep_spm_sample:
            Path(spm_sample["path"]).unlink(missing_ok=True)

    if args.tokenizer_only:
        (out_dir / "tokenizer_meta.json").write_text(
            json.dumps({"vocab_size": tokenizer.vocab_size, "spm_sample": spm_sample},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"トークナイザだけ作って終わります → {out_dir / 'tokenizer'}")
        return

    print(f"エンコード中 ({END} を文書の区切りに入れます)")
    meta = encode_corpus(
        corpus_files,
        tokenizer,
        out_dir,
        val_corpus=val_corpus,
        val_every=args.val_every,
        val_max_tokens=args.val_max_tokens,
        max_chars=args.max_chars,
        sample_chars=args.sample_chars,
        capacity_scale=args.capacity_scale,
        extra={"spm_sample": spm_sample} if spm_sample else None,
    )

    print()
    print(f"  train.bin : {meta['train_tokens']:,} トークン "
          f"({meta['train_tokens'] * 2 / 2**20:.0f}MiB)")
    print(f"  val.bin   : {meta['val_tokens']:,} トークン")
    if meta["val_corpus_files"]:
        # 検証は別ファイルなので、訓練の文書数とは別に数えている
        print(f"  文書数    : 訓練 {meta['documents']:,} / 検証 {meta['val_documents']:,}")
    else:
        print(f"  文書数    : {meta['documents']:,} (うち検証 {meta['val_documents']:,})")
    print(f"  文字数    : {meta['chars']:,}")
    print(f"  1トークンあたり {meta['chars_per_token']} 文字")
    print(f"  → {out_dir}/meta.json")


if __name__ == "__main__":
    main()
