"""土俵を変えると勝敗が入れ替わることを、1枚の図にする.

`tools/compare_domains.py` が出した json を読んで棒グラフにする。
**同じモデル群を2つの検証セットで測ると、順位が逆になる**という一点だけを見せる。

使い方:
    python tools/plot_domains.py
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "runs" / "3lm" / "domain_bpc.json"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "images" / "3lm-domain-bpc.png"))
    args = ap.parse_args()

    records = json.loads(Path(args.json).read_text(encoding="utf-8"))["records"]
    models = list(dict.fromkeys(r["model"] for r in records))
    domains = list(dict.fromkeys(r["domain"] for r in records))

    fig, ax = dark_axes(figsize=(9, 4.6))
    x = np.arange(len(domains))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        values = [
            next(
                (r["bits_per_char"] for r in records
                 if r["model"] == model and r["domain"] == d),
                np.nan,
            )
            for d in domains
        ]
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(
            x + offset, values, width * 0.92,
            label=model, color=SERIES_COLORS[i % len(SERIES_COLORS)],
        )
        for bar, value in zip(bars, values, strict=True):
            if np.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2, value + 0.08, f"{value:.2f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylabel("bits / char（低いほど良い）")
    ax.set_title("同じモデルでも、測る土俵で勝敗が変わる")
    ax.set_ylim(0, max(r["bits_per_char"] for r in records) * 1.18)
    ax.legend(fontsize=8, framealpha=0.2)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
