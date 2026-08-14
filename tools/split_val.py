"""コーパスから検証セットを切り出し、訓練側からは確実に取り除く.

    python3 tools/split_val.py --corpus data/corpus_pretrain.txt

## なぜ専用の検証セットを作るのか

データ量を変えた比較 (tools/run_data_scaling.py) では、**全条件で同じ
検証セットを使わないと結果を読めない**。条件ごとに違う文章で採点すると、
val loss の差が「データ量の効果」なのか「採点した文章の難しさ」なのか
区別できなくなる。その2 の対照実験はここが弱かった。

## 等間隔で抜く理由

末尾から取ると、このコーパスは青空文庫→FineWeb2 の順に書いてあるので
**FineWeb2 だけの検証セット**になってしまう。青空文庫の文語が
訓練にだけ入って検証に入らないと、そのぶん val loss が実態より低く出る。

そこで --every 文書ごとに1本ずつ抜き、両方のソースが比率どおりに入るようにする。

## 訓練側から必ず消す

抜いた文書を訓練側に残したままにすると、検証は「見たことのある文章」の
採点になる。乖離点 (val が train を追い越す地点) を測るのが目的なので、
ここが漏れると測定そのものが無意味になる。
入力を読み直して、抜いた行を除いたファイルを書き出す。
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="検証セットを切り出して訓練側から除く")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--val-out", default="", help="既定: <corpus の隣>/val_pretrain.txt")
    ap.add_argument("--train-out", default="", help="既定: 元のファイルを置き換える")
    ap.add_argument("--every", type=int, default=400, help="この文書数ごとに1本を検証へ")
    ap.add_argument("--max-chars", type=int, default=3_000_000,
                    help="検証セットの文字数の上限")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    if not corpus.exists():
        raise SystemExit(f"{corpus} がありません")
    val_path = Path(args.val_out) if args.val_out else corpus.parent / "val_pretrain.txt"
    train_path = Path(args.train_out) if args.train_out else corpus.with_suffix(".train.tmp")

    val_docs = val_chars = train_docs = train_chars = 0
    index = 0
    with (
        corpus.open("r", encoding="utf-8") as src,
        val_path.open("w", encoding="utf-8", newline="\n") as val,
        train_path.open("w", encoding="utf-8", newline="\n") as train,
    ):
        for line in src:
            stripped = line.strip()
            if not stripped:
                continue
            if index % args.every == 0 and val_chars < args.max_chars:
                val.write(stripped + "\n")
                val_docs += 1
                val_chars += len(stripped)
            else:
                train.write(stripped + "\n")
                train_docs += 1
                train_chars += len(stripped)
            index += 1

    if not args.train_out:
        # 元のファイルを置き換える。検証に使う文書が訓練側に残らないようにする。
        train_path.replace(corpus)
        train_path = corpus

    print("=" * 66)
    print(f"  検証セット : {val_path}")
    print(f"    {val_docs:,} 文書 / {val_chars:,} 文字")
    print(f"    sha256 {sha256_file(val_path)[:32]}…")
    print(f"  訓練データ : {train_path}")
    print(f"    {train_docs:,} 文書 / {train_chars:,} 文字")
    print(f"  検証の割合 : {val_chars / max(1, val_chars + train_chars):.3%}")
    print("=" * 66)
    print("  検証に使う文書は訓練側から取り除いてあります。")


if __name__ == "__main__":
    main()
