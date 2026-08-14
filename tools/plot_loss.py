"""runs/loss.csv から学習曲線の画像を作る (記事用).

    pip install matplotlib
    python tools/plot_loss.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.loss_log import read_curve  # noqa: E402
from tools.plotting import SERIES_COLORS, dark_axes  # noqa: E402


def elapsed_by_step(path: str) -> dict[int, float]:
    """step -> 経過秒。同じ step が複数あれば後の行で上書きする."""
    out: dict[int, float] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(ln for ln in fh if not ln.startswith("#")):
            if row.get("elapsed_sec"):
                out[int(row["step"])] = float(row["elapsed_sec"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="runs/3lm/pretrain_loss.csv")
    ap.add_argument("--out", default="docs/images/3lm-pretrain-loss.png")
    ap.add_argument("--title", default="3LM 事前学習 (35.66M / 3.68億トークン)")
    ap.add_argument("--hours", action="store_true",
                    help="横軸を step ではなく経過時間にする (一晩の様子を見せる用)")
    args = ap.parse_args()

    curve = read_curve(args.csv)
    train_y = [v for _, v in curve.train]
    val_y = [v for _, v in curve.val]

    if args.hours:
        # 一晩の様子は step より時間のほうが伝わる。
        # 再開をはさむと同じ step が2回出るので、後の行 (=生き残った run) を採る。
        elapsed = elapsed_by_step(args.csv)
        train_x = [elapsed.get(s, 0) / 3600 for s, _ in curve.train]
        val_x = [elapsed.get(s, 0) / 3600 for s, _ in curve.val]
    else:
        train_x = [s for s, _ in curve.train]
        val_x = [s for s, _ in curve.val]

    fig, ax = dark_axes()
    ax.plot(train_x, train_y, color=SERIES_COLORS[0], lw=1.6, label="train loss")
    ax.plot(val_x, val_y, color=SERIES_COLORS[1], lw=1.8, marker="o", ms=3, label="val loss")
    ax.set_xlabel("経過時間 (時間)" if args.hours else "step")
    # サブワードなので1トークンあたり。文字あたりに直すには
    # 1トークン=2.101文字で割る必要があり、別物になる。
    ax.set_ylabel("交差エントロピー (nats / トークン)")
    ax.set_title(args.title, loc="left", fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"保存: {out}  (train {len(train_x)}点 / val {len(val_x)}点)")


if __name__ == "__main__":
    main()
