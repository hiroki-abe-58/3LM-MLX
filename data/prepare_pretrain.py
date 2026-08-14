"""事前学習用の日本語コーパスを作る.

## 何を混ぜるか、なぜそれか

FineWeb2 の日本語 (jpn_Jpan) — **ODC-By 1.0**
    Common Crawl から言語判定と品質フィルタを通した大規模コーパス。
    日本語だけで474シャードあり、この用途には多すぎるほどある。

青空文庫 (globis-university/aozorabunko-clean) — **CC BY 4.0**
    著作権の切れた文学作品。文章が長くて構造がしっかりしているので、
    Web文だけだと足りない「段落として通った日本語」を補う。
    ただし文語・旧仮名が多いので入れすぎると口調が古くなる。全体の1割まで。

**どちらも継承条件 (ShareAlike) を持たない**のが選定の理由。
data/prepare_sft.py が明文化している「重みを Apache-2.0 相当で配布するため
CC BY-SA のデータは入れない」という方針を、事前学習側でも守る。
日本語の大規模コーパスで真っ先に候補になる Wikipedia は CC BY-SA なので、
ここでは意図的に外してある。

## 再現性: revision を必ず固定する

Windows 版の引き継ぎ資料で「HF のデータセットが revision 固定されていないと
再現できない」と書いたので、こちらでも守る。データセットは更新されるので、
`load_dataset("...")` と書いただけでは半年後に別の中身が来る。

  - revision をコミットハッシュで固定する
  - 実際に使ったシャード名と、その SHA256 を manifest.json に残す

これで「同じコーパスを作り直せる」ことが検証可能になる。

## ディスクを溜めない

空きが 64GiB しかない。シャードを1つ落として使ったら消す、を繰り返す。
`load_dataset(streaming=True)` でも似たことはできるが、シャード名と
ハッシュを自分で押さえたいので hf_hub_download を直接呼ぶ。

使い方:
    python data/prepare_pretrain.py --target-chars 1_200_000_000
    python data/prepare_pretrain.py --target-chars 10_000_000 --out /tmp/small.txt  # 試し
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

# --- データセットの固定 -------------------------------------------------------

FINEWEB_REPO = "HuggingFaceFW/fineweb-2"
FINEWEB_REVISION = "af9c13333eb981300149d5ca60a8e9d659b276b9"
FINEWEB_LICENSE = "ODC-By 1.0"
FINEWEB_SHARDS = 474  # data/jpn_Jpan/train/000_00000.parquet 〜 000_00473.parquet

AOZORA_REPO = "globis-university/aozorabunko-clean"
AOZORA_REVISION = "42a9c9c0f1d67e6a5554d9bea4201973dc9b049c"
AOZORA_FILE = "aozorabunko-dedupe-clean.jsonl.gz"
AOZORA_LICENSE = "CC BY 4.0"

# --- 整形とフィルタ ----------------------------------------------------------

_WS_RE = re.compile(r"\s+")

# 日本語として数える文字。ひらがな・カタカナ・漢字・日本語の句読点。
_JA_RE = re.compile(
    r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3005\u3006\u30FC\u3001\u3002\u300C\u300D]"
)

# Web ページに必ず混ざる定型文。この行だけを落とす (文書を捨てるほどではない)。
_BOILERPLATE = re.compile(
    r"(Cookie|クッキー).{0,20}(使用|利用|同意)"
    r"|JavaScript.{0,30}(有効|オフ|無効)"
    r"|^(ホーム|トップ|メニュー|サイトマップ|お問い合わせ|プライバシーポリシー"
    r"|利用規約|会社概要|採用情報|関連記事|人気記事|新着記事|カテゴリー?"
    r"|前の記事|次の記事|コメントを残す|コメントする|返信をキャンセル"
    r"|ページの先頭へ|検索|ログイン|新規登録|カートに入れる|続きを読む)$"
    r"|無断(転載|複写|複製)を?禁(じ|ず)"
    r"|All Rights Reserved"
)

# 文書ごと捨てる強い合図。アダルト誘導・広告の羅列・エラーページ。
_REJECT_DOC = re.compile(
    r"(18歳未満|アダルトサイト|出会い系)"
    r"|(404|403)\s*(Not Found|Forbidden)"
    r"|お探しのページは(見つかりません|移動)"
)


def normalize(text: str) -> str:
    """1文書を1行に整形する.

    data/prepare_sft.py の normalize と同じ方針で、空白と改行を
    半角スペース1個に潰す。事前学習と SFT で書式が違うと、
    SFT のときにモデルが「知らない書式」として扱ってしまう。

    NFKC は掛けない。全角の「？」「！」が半角に化けて、
    その2 で揃えた文字数の数え方が崩れる (src/tokenizer.py の
    normalization_rule_name="identity" と対応させている)。
    ただし互換文字だけは潰したいので、制御文字の除去はする。
    """
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    return _WS_RE.sub(" ", text).strip()


def clean_document(raw: str) -> str:
    """定型文の行を落としてから1行に整形する."""
    kept = [ln for ln in raw.splitlines() if ln.strip() and not _BOILERPLATE.search(ln.strip())]
    return normalize(" ".join(kept))


def japanese_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(_JA_RE.findall(text)) / len(text)


def is_repetitive(text: str, gram: int = 20, limit: int = 4) -> bool:
    """同じ並びが何度も出てくる文書を弾く.

    Web には「商品名 価格 商品名 価格 …」のような表の残骸が多い。
    これを学習させると、モデルが同じ語を繰り返す癖を覚える。
    その2 で最後まで残った「繰り返し率」の原因になりうる。
    """
    if len(text) < gram * limit:
        return False
    step = max(1, len(text) // 2000)  # 長い文書を全部数えると遅いので間引く
    counts: dict[str, int] = {}
    for i in range(0, len(text) - gram, step):
        key = text[i : i + gram]
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= limit:
            return True
    return False


class Filters:
    """採否の判定と、落ちた理由の集計をまとめて持つ."""

    def __init__(self, min_chars: int, max_chars: int, min_ja_ratio: float):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_ja_ratio = min_ja_ratio
        self.counts: dict[str, int] = {}
        self.seen: set[int] = set()

    def _reject(self, reason: str) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + 1

    def accept(self, raw: str) -> str | None:
        """通れば整形後の1行を返し、落ちれば None を返す."""
        if _REJECT_DOC.search(raw):
            self._reject("不適切・エラーページ")
            return None
        text = clean_document(raw)
        if len(text) < self.min_chars:
            self._reject("短すぎる")
            return None
        if len(text) > self.max_chars:
            text = text[: self.max_chars]
        if japanese_ratio(text) < self.min_ja_ratio:
            self._reject("日本語の比率が低い")
            return None
        if is_repetitive(text):
            self._reject("同じ並びの繰り返し")
            return None
        # 先頭200文字のハッシュで近い重複を落とす。完全一致より広く効く。
        key = hash(text[:200])
        if key in self.seen:
            self._reject("重複")
            return None
        self.seen.add(key)
        return text


# --- データの取得 ------------------------------------------------------------


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def drop_from_cache(path: Path) -> int:
    """HF キャッシュから実データを消し、空いたバイト数を返す.

    hf_hub_download が返すのは snapshots/ の下の**シンボリックリンク**で、
    実データは blobs/<sha> にある。リンクだけ消しても容量は戻らない。
    リンクの指す先を先に解決してから、両方消す。
    """
    freed = 0
    target = path.resolve() if path.is_symlink() else path
    for victim in {target, path}:
        if victim.exists() and not victim.is_symlink():
            freed += victim.stat().st_size
            victim.unlink(missing_ok=True)
        elif victim.is_symlink():
            victim.unlink(missing_ok=True)
    return freed


def iter_fineweb(shards: int, records: list[dict]) -> Iterator[str]:
    """FineWeb2 のシャードを1つずつ落として読み、読み終えたら消す.

    1シャードは 4.6GiB (2,762,000文書 / 約45億文字) ある。実測してみたら
    **1シャードだけで目標の数倍あった**ので、474シャードのうち
    使うのは1〜2個で足りる。

    read_table は使わない。4.6GiB の parquet を展開すると Arrow の
    文字列で十数GB になり、その3 のカーネルパニックの再現になる。
    iter_batches で row group ごとに読み、読んだぶんは捨てる。
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for index in range(shards):
        name = f"data/jpn_Jpan/train/000_{index:05d}.parquet"
        print(f"\n  シャード {index + 1}/{shards} を取得中: {name}")
        path = Path(
            hf_hub_download(
                FINEWEB_REPO, name, repo_type="dataset", revision=FINEWEB_REVISION
            )
        )
        digest = sha256_file(path)
        records.append({
            "shard": name, "bytes": path.stat().st_size, "sha256": digest,
        })
        print(f"  {path.stat().st_size / 2**30:.2f} GiB / sha256 {digest[:16]}…")
        # try/finally にするのが要点。目標文字数に達すると呼び出し側が
        # ループを抜け、この生成器は yield の位置で GeneratorExit になる。
        # 後片付けを finally に置かないと 4.6GiB が残ったままになる。
        try:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=1000, columns=["text"]):
                for value in batch.column("text"):
                    text = value.as_py()
                    if text:
                        yield text
            del parquet
        finally:
            freed = drop_from_cache(path)
            print(f"\n  シャードを削除しました ({freed / 2**30:.2f} GiB を解放)")


def iter_aozora(records: list[dict]) -> Iterator[str]:
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            AOZORA_REPO, AOZORA_FILE, repo_type="dataset", revision=AOZORA_REVISION
        )
    )
    records.append({
        "shard": AOZORA_FILE, "bytes": path.stat().st_size, "sha256": sha256_file(path),
    })
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text") or ""
            if text:
                yield text


# --- 本体 --------------------------------------------------------------------


def collect(
    source: Iterator[str], filters: Filters, target_chars: int, label: str, out
) -> tuple[int, int]:
    """target_chars に達するまで書き出し、(文書数, 文字数) を返す."""
    docs = chars = seen = 0
    started = time.time()
    for raw in source:
        seen += 1
        text = filters.accept(raw)
        if text is None:
            continue
        out.write(text + "\n")
        docs += 1
        chars += len(text)
        if docs % 2000 == 0:
            rate = chars / max(time.time() - started, 1e-6)
            print(
                f"\r  [{label}] {chars / 1e6:,.1f}M / {target_chars / 1e6:,.0f}M文字 "
                f"({chars / target_chars * 100:5.1f}%) / {docs:,}文書 / "
                f"採用率 {docs / seen * 100:.0f}% / {rate / 1e6:.1f}M文字每秒",
                end="", flush=True,
            )
        if chars >= target_chars:
            break
    print()
    return docs, chars


def main() -> None:
    ap = argparse.ArgumentParser(description="事前学習用の日本語コーパスを作る")
    ap.add_argument("--out", default=str(ROOT / "data" / "corpus_pretrain.txt"))
    ap.add_argument("--target-chars", type=int, default=1_200_000_000)
    ap.add_argument("--aozora-ratio", type=float, default=0.10,
                    help="全体に占める青空文庫の割合の上限")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=20_000)
    ap.add_argument("--min-ja-ratio", type=float, default=0.70)
    # 1シャードで約45億文字あるので、既定は2つまで。474個は要らない。
    ap.add_argument("--shards", type=int, default=2,
                    help=f"使う FineWeb2 シャードの上限 (全部で {FINEWEB_SHARDS} 個)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path.with_suffix(".manifest.json")

    aozora_target = int(args.target_chars * args.aozora_ratio)
    fineweb_target = args.target_chars - aozora_target

    print("=" * 70)
    print("  事前学習コーパスを構築します")
    print(f"    目標          : {args.target_chars:,} 文字")
    print(f"    FineWeb2      : {fineweb_target:,} 文字 ({FINEWEB_LICENSE})")
    print(f"    青空文庫      : {aozora_target:,} 文字 ({AOZORA_LICENSE})")
    print(f"    revision      : {FINEWEB_REVISION[:12]} / {AOZORA_REVISION[:12]}")
    print(f"    採用条件      : {args.min_chars}〜{args.max_chars}文字 / "
          f"日本語率 {args.min_ja_ratio:.0%} 以上")
    print("=" * 70)

    filters = Filters(args.min_chars, args.max_chars, args.min_ja_ratio)
    records: list[dict] = []
    started = time.time()

    # newline="\n" を明示する。Windows で動かしたときに CRLF にならないように。
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        aozora_docs, aozora_chars = collect(
            iter_aozora(records), filters, aozora_target, "青空文庫", out
        )
        fineweb_docs, fineweb_chars = collect(
            iter_fineweb(args.shards, records), filters, fineweb_target, "FineWeb2", out
        )

    total_chars = aozora_chars + fineweb_chars
    manifest = {
        "target_chars": args.target_chars,
        "total_chars": total_chars,
        "total_documents": aozora_docs + fineweb_docs,
        "elapsed_seconds": round(time.time() - started, 1),
        "filters": {
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "min_ja_ratio": args.min_ja_ratio,
            "rejected": filters.counts,
        },
        "sources": [
            {
                "repo": AOZORA_REPO, "revision": AOZORA_REVISION, "license": AOZORA_LICENSE,
                "documents": aozora_docs, "chars": aozora_chars,
                "share": round(aozora_chars / max(total_chars, 1), 4),
            },
            {
                "repo": FINEWEB_REPO, "revision": FINEWEB_REVISION, "license": FINEWEB_LICENSE,
                "documents": fineweb_docs, "chars": fineweb_chars,
                "share": round(fineweb_chars / max(total_chars, 1), 4),
            },
        ],
        "files": records,
        "output": {"name": out_path.name, "sha256": sha256_file(out_path),
                   "bytes": out_path.stat().st_size},
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print("=" * 70)
    print(f"  合計          : {total_chars:,} 文字 / "
          f"{aozora_docs + fineweb_docs:,} 文書")
    print(f"  青空文庫      : {aozora_chars:,} 文字 "
          f"({aozora_chars / max(total_chars, 1):.1%})")
    print(f"  FineWeb2      : {fineweb_chars:,} 文字 "
          f"({fineweb_chars / max(total_chars, 1):.1%}) / "
          f"{len(records) - 1} シャード")
    print(f"  所要          : {(time.time() - started) / 60:.1f} 分")
    print("  落とした理由:")
    for reason, count in sorted(filters.counts.items(), key=lambda kv: -kv[1]):
        print(f"    {reason:<24} {count:,}")
    print(f"  → {out_path} ({out_path.stat().st_size / 2**30:.2f} GiB)")
    print(f"  → {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
