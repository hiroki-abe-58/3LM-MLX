"""任意のテキストに対して、任意のチェックポイントの bits/char を測る.

`eval/run.py` の採点は「公開データ由来の会話」という一つの土俵しか見ない。
そこは前作 (2LM) が事前学習ごと浸かっていた領域なので、
**前作に有利な土俵**である可能性がある。

そこで土俵を変えて測れるようにする。Web文の検証セットで測れば、
今作 (3LM) が事前学習した領域での勝ち負けが分かる。

bits/char は分母が文字数なので、語彙の大きさが違うモデル同士でも比べられる。

使い方:
    python tools/bpc.py --ckpt checkpoints/sft-final --text data/val_pretrain.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run import bits_per_char, tidy_path  # noqa: E402
from src.generate import load_bundle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument(
        "--max-chars",
        type=int,
        default=400_000,
        help="長すぎる検証ファイルの先頭だけを使う (0 で全部)",
    )
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    text = Path(args.text).read_text(encoding="utf-8")
    if args.max_chars and len(text) > args.max_chars:
        text = text[: args.max_chars]

    model, tokenizer = load_bundle(Path(args.ckpt))
    started = time.time()
    bpc = bits_per_char(model, tokenizer, text)

    label = args.label or tidy_path(args.ckpt)
    print(
        f"{label:<26} {Path(args.text).name:<22} "
        f"{len(text):>9,}文字  bits/char = {bpc:.3f}  ({time.time() - started:.1f}秒)"
    )


if __name__ == "__main__":
    main()
