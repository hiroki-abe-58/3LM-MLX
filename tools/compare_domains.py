"""同じモデル群を、性質の違う2つの検証テキストで採点して並べる.

`eval/run.py` の 4指標は「公開データ由来の会話」だけで測る。
だが前作 (2LM) は**その公開データで事前学習した**モデルなので、
そこは前作のホームグラウンドにあたる。
そこだけを見て「大きくしたのに負けた」と読むと、結論を間違える。

そこで土俵を2つ用意して、同じ物差し (bits/char) で並べる。

    土俵A  eval/holdout_clean.txt  公開データ由来の会話 (前作の領域)
    土俵B  data/val_pretrain.txt   Web文 + 青空文庫    (今作の領域)

bits/char は分母が文字数なので、語彙 8k と 32k のモデルを直接比べられる。

使い方:
    python tools/compare_domains.py --baseline ../2LM-MLX-GAL/checkpoints/final
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run import bits_per_char, tidy_path  # noqa: E402
from src.generate import load_bundle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

DOMAINS: list[tuple[str, Path]] = [
    ("A 公開データ由来の会話", ROOT / "eval" / "holdout_clean.txt"),
    ("B Web文+青空文庫", ROOT / "data" / "val_pretrain.txt"),
]

MAX_CHARS = 400_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs" / "3lm" / "domain_bpc.json"))
    ap.add_argument(
        "--baseline",
        default="",
        help="前作 (2LM) の重みの場所。このリポジトリの外にあるので明示的に渡す",
    )
    args = ap.parse_args()

    models: list[tuple[str, Path]] = []
    if args.baseline:
        models.append(("2LM 13.81M", Path(args.baseline).expanduser()))
    models += [
        ("3LM 35.66M 事前学習のみ", ROOT / "checkpoints" / "pretrain-final"),
        ("3LM 35.66M SFT済み", ROOT / "checkpoints" / "sft-final"),
        ("3LM 35.66M 口調あり", ROOT / "checkpoints" / "gal-final"),
    ]

    texts: dict[str, str] = {}
    for name, path in DOMAINS:
        raw = path.read_text(encoding="utf-8")
        texts[name] = raw[:MAX_CHARS]

    records: list[dict] = []
    for label, ckpt in models:
        if not (ckpt / "model.safetensors").exists():
            print(f"  飛ばす: {label} ({ckpt} が無い)")
            continue
        model, tokenizer = load_bundle(ckpt)
        for domain, _ in DOMAINS:
            bpc = bits_per_char(model, tokenizer, texts[domain])
            records.append(
                {
                    "model": label,
                    "ckpt": tidy_path(str(ckpt)),
                    "domain": domain,
                    "chars": len(texts[domain]),
                    "bits_per_char": round(bpc, 3),
                }
            )
            print(f"  {label:<26} {domain:<24} bits/char = {bpc:.3f}")

    width = max(len(m) for m, _ in models) + 2
    print()
    print("  " + " " * width + "".join(f"{d:>26}" for d, _ in DOMAINS))
    for label, _ in models:
        row = [r for r in records if r["model"] == label]
        if not row:
            continue
        cells = "".join(
            f"{next(r['bits_per_char'] for r in row if r['domain'] == d):>26.3f}"
            for d, _ in DOMAINS
        )
        print(f"  {label:<{width}}{cells}")
    print("\n  低いほうが良い。土俵ごとに勝者が入れ替わることを確認する。")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"max_chars": MAX_CHARS, "eval_block": 256, "records": records},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
