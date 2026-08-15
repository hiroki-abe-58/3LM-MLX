"""kill して再開したときの検査結果を1枚にする.

「再開できました」と言うだけなら誰でも書ける。この図は
**壊れている場合と比べて初めて意味を持つ**ことを見せるためのもの。

    雑音の床  : 同じ設定を2回走らせただけの差 (GPU の加算順序で毎回揺れる)
    kill 再開 : わざと kill -9 して再開した場合の差
    対照      : optimizer の m, v をわざと捨てた「壊れた再開」

kill 再開が雑音の床の近くにあり、対照だけが飛び抜けていれば、
**検査に感度がある**と言える。

使い方:
    python tools/plot_killtest.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.plotting import SERIES_COLORS, dark_figure  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "runs" / "3lm" / "killtest.json"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "images" / "3lm-killtest.png"))
    args = ap.parse_args()

    d = json.loads(Path(args.json).read_text(encoding="utf-8"))
    fig, (ax1, ax2) = dark_figure(2, figsize=(8.4, 6.0))
    fig.subplots_adjust(hspace=0.45)

    labels = ["雑音の床\n(同設定を2回)", "kill -9 して再開", "対照\n(m, v を捨てた)"]
    weights = [d["noise"], d["killed_diff"], d["noopt_diff"]]
    colors = [SERIES_COLORS[0], SERIES_COLORS[2], SERIES_COLORS[1]]

    bars = ax1.bar(labels, weights, color=colors, width=0.55)
    ax1.set_yscale("log")
    ax1.set_ylabel("重みの最大差（対数）")
    ax1.set_title("再開したあとの重みが、どれだけずれたか")
    for bar, value in zip(bars, weights, strict=True):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, value * 1.25, f"{value:.2e}",
            ha="center", va="bottom", fontsize=9,
        )
    ax1.set_ylim(min(weights) * 0.35, max(weights) * 4)

    gaps = [0.0, d["loss_gap"], d["noopt_loss_gap"]]
    bars = ax2.bar(labels, gaps, color=colors, width=0.55)
    ax2.set_ylabel("学習曲線の段差")
    ax2.set_title("再開の前後で、損失が飛んでいないか")
    for bar, value in zip(bars, gaps, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, value + max(gaps) * 0.03, f"{value:.4f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax2.set_ylim(0, max(gaps) * 1.25)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
