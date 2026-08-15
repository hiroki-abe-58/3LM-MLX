"""口調を乗せるために何を失ったかを、前作と並べる.

同じ 2,610会話・同じ検証セット・同じサンプリング条件で、
13.81M と 35.66M に口調を移植した前後の差を出す。

見せたいのは絶対値ではなく **差** (口調を得るために払った代償)。
大きいモデルのほうが払う代償が小さければ、
「大きくすると必要なキャラクターデータも増える」という予想は外れたことになる。

使い方:
    python tools/plot_gal_cost.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.plotting import SERIES_COLORS, dark_axes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

# (見出し, 指標のキー, 良い向き)
METRICS = [
    ("bits/char の悪化", "bits_per_char", "低いほど良い"),
    ("主題保持率の低下", "topic_rate", "高いほど良い"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "images" / "3lm-gal-cost.png"))
    args = ap.parse_args()

    pairs = [
        ("2LM 13.81M", "eval_2lm_holdout_clean.json", "eval_2lm_gal_clean.json"),
        ("3LM 35.66M", "eval_3lm_sft.json", "eval_3lm_gal.json"),
    ]

    fig, ax = dark_axes(figsize=(8.2, 4.4))
    x = np.arange(len(METRICS))
    width = 0.34

    for i, (label, before_file, after_file) in enumerate(pairs):
        before = json.loads((RUNS / before_file).read_text(encoding="utf-8"))
        after = json.loads((RUNS / after_file).read_text(encoding="utf-8"))
        # どちらも「悪くなった量」を正の値にそろえて並べる。
        costs = [
            after[key] - before[key] if better.startswith("低い")
            else before[key] - after[key]
            for _, key, better in METRICS
        ]
        offset = (i - (len(pairs) - 1) / 2) * width
        bars = ax.bar(
            x + offset, costs, width * 0.92, label=label,
            color=SERIES_COLORS[i % len(SERIES_COLORS)],
        )
        for bar, value in zip(bars, costs, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:+.3f}",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _, _ in METRICS])
    ax.set_ylabel("失った量（大きいほど代償が重い）")
    ax.set_title("同じ 2,610会話で口調を乗せたとき、何を失ったか")
    ax.legend(fontsize=9, framealpha=0.2)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
