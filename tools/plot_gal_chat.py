"""口調を乗せた版どうしの会話を、チャット画面の形で並べる.

`tools/compare_replies.py --gal` が保存した json から作る。
**選ばない。** 走らせた質問を全部、出てきた順にそのまま並べる。
良い返答だけを拾うと「良くなった」ではなく「良いものを選んだ」を見せることになる。

その3の記事では端末の画面を貼ったが、口調の変化は非エンジニアにも
伝わってほしいので、見慣れたチャットの形にする。

使い方:
    python tools/plot_gal_chat.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402

from tools.plotting import apply_style, clip, wrap_ja  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

OLD, NEW = "2LM 13.81M", "3LM 35.66M"

PAGE_BG = "#0b1024"
PANEL_BG = "#151925"
PANEL_EDGE = "#2c3446"
HEADER_BG = "#222839"
USER_BG = "#3d5afe"
GAL_BG = "#2b3142"
TEXT = "#f2f5ff"
MUTED = "#8fa0c2"
AVATAR = "#ff7ba8"

FIG_W = 12.0
PANEL_W = 5.55
PANEL_X = (0.35, 6.10)
MARGIN = 0.34          # 吹き出しと枠の間
BUBBLE_MAX = 3.55      # 吹き出しの最大幅
CHARS_PER_LINE = 22
LINE_H = 0.235
PAD_Y = 0.115
GAP = 0.14             # 同じ人の発言のあいだ
TURN_GAP = 0.30        # 相手が変わるところ
HEADER_H = 0.78
TITLE_H = 1.28
BAND_H = 1.30
FOOTER_H = 0.62
PANEL_PAD = 0.26


def bubble_lines(text: str) -> list[str]:
    return wrap_ja(clip(text, 120), CHARS_PER_LINE)


def turn_height(user: str, reply: str) -> float:
    rows = len(bubble_lines(user)) + len(bubble_lines(reply))
    return rows * LINE_H + PAD_Y * 4 + GAP + TURN_GAP


class Panel:
    """チャット画面1枚ぶん."""

    def __init__(self, ax, x: float, top: float, height: float, label: str, note: str):
        self.ax = ax
        self.x = x
        ax.add_patch(
            FancyBboxPatch(
                (x, top - height), PANEL_W, height,
                boxstyle="round,pad=0.02,rounding_size=0.20",
                facecolor=PANEL_BG, edgecolor=PANEL_EDGE, linewidth=1.3,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (x, top - HEADER_H), PANEL_W, HEADER_H,
                boxstyle="round,pad=0.02,rounding_size=0.20",
                facecolor=HEADER_BG, edgecolor="none",
            )
        )
        cy = top - HEADER_H / 2
        ax.add_patch(Circle((x + 0.42, cy), 0.20, facecolor=AVATAR, edgecolor="none"))
        ax.text(x + 0.42, cy - 0.012, "ギ", fontsize=11, color="#1a1020",
                ha="center", va="center", fontweight="bold")
        ax.text(x + 0.76, cy + 0.11, label, fontsize=11.5, color=TEXT,
                ha="left", va="center", fontweight="bold")
        ax.text(x + 0.76, cy - 0.17, note, fontsize=8.6, color=MUTED,
                ha="left", va="center")
        self.y = top - HEADER_H - 0.24

    def _bubble(self, text: str, outgoing: bool) -> None:
        lines = bubble_lines(text)
        width = min(
            BUBBLE_MAX,
            max(len(line) for line in lines) * 0.128 + 0.30,
        )
        height = len(lines) * LINE_H + PAD_Y * 2
        left = (
            self.x + PANEL_W - MARGIN - width if outgoing else self.x + MARGIN
        )
        self.ax.add_patch(
            FancyBboxPatch(
                (left, self.y - height), width, height,
                boxstyle="round,pad=0.03,rounding_size=0.13",
                facecolor=USER_BG if outgoing else GAL_BG, edgecolor="none",
            )
        )
        for i, line in enumerate(lines):
            self.ax.text(
                left + 0.15, self.y - PAD_Y - LINE_H * (i + 0.58), line,
                fontsize=9.8, color=TEXT, ha="left", va="center",
            )
        self.y -= height

    def turn(self, user: str, reply: str) -> None:
        self._bubble(user, outgoing=True)
        self.y -= GAP
        self._bubble(reply, outgoing=False)
        self.y -= TURN_GAP


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "runs" / "3lm" / "gal_reply_compare.json"))
    ap.add_argument("--old-eval", default=str(ROOT / "runs" / "eval_2lm_gal_clean.json"))
    ap.add_argument("--new-eval", default=str(ROOT / "runs" / "eval_3lm_gal.json"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "images" / "3lm-gal-chat.png"))
    args = ap.parse_args()

    before = json.loads(Path(args.old_eval).read_text(encoding="utf-8"))
    after = json.loads(Path(args.new_eval).read_text(encoding="utf-8"))
    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    rec, s = data["records"], data["sampling"]
    prompts = list(dict.fromkeys(r["prompt"] for r in rec))

    def reply_of(model: str, prompt: str) -> str:
        for r in rec:
            if r["model"] == model and r["prompt"] == prompt:
                return r["reply"].strip()
        raise SystemExit(f"見つかりません: {model} / {prompt}")

    body = max(
        sum(turn_height(p, reply_of(m, p)) for p in prompts) for m in (OLD, NEW)
    )
    panel_h = HEADER_H + 0.24 + body + PANEL_PAD
    total_h = TITLE_H + panel_h + BAND_H + FOOTER_H

    apply_style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(FIG_W, total_h), dpi=150)
    fig.patch.set_facecolor(PAGE_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    ax.text(FIG_W / 2, total_h - 0.40, "同じ 2,610会話で口調を乗せた、前作と今作",
            fontsize=17, color="#ffffff", ha="center", va="center", fontweight="bold")
    ax.text(FIG_W / 2, total_h - 0.78,
            "同じ質問・同じ乱数・同じサンプリング条件。走らせた6問を、選ばずに全部並べています",
            fontsize=10, color=MUTED, ha="center", va="center")

    top = total_h - TITLE_H
    panels = [
        Panel(ax, PANEL_X[0], top, panel_h, "2LM-MLX-GAL",
              "13.81M / 事前学習 949万文字 / その3"),
        Panel(ax, PANEL_X[1], top, panel_h, "3LM-MLX-GAL",
              "35.66M / 事前学習 12億文字 / その4"),
    ]
    for model, panel in zip((OLD, NEW), panels, strict=True):
        for prompt in prompts:
            panel.turn(prompt, reply_of(model, prompt))

    band_top = FOOTER_H + BAND_H
    ax.add_patch(
        FancyBboxPatch(
            (PANEL_X[0], FOOTER_H + 0.10), PANEL_W * 2 + (PANEL_X[1] - PANEL_X[0] - PANEL_W),
            BAND_H - 0.20,
            boxstyle="round,pad=0.02,rounding_size=0.16",
            facecolor="#141a2b", edgecolor=PANEL_EDGE, linewidth=1.0,
        )
    )
    left_x, right_x = FIG_W * 0.30, FIG_W * 0.72
    ax.text(left_x, band_top - 0.34, "変わったこと", fontsize=10.5, color="#8ce99a",
            ha="center", va="center", fontweight="bold")
    ax.text(
        left_x, band_top - 0.68,
        f"主題保持率 {before['topic_rate']:.3f} → {after['topic_rate']:.3f}",
        fontsize=11, color=TEXT, ha="center", va="center",
    )
    ax.text(
        left_x, band_top - 0.97,
        f"bits/char {before['bits_per_char']:.3f} → {after['bits_per_char']:.3f}（低いほど良い）",
        fontsize=11, color=TEXT, ha="center", va="center",
    )
    ax.text(right_x, band_top - 0.34, "変わらなかったこと", fontsize=10.5, color="#ffd43b",
            ha="center", va="center", fontweight="bold")
    ax.text(right_x, band_top - 0.68, "どちらも、何を聞いてもすぐお腹が空く",
            fontsize=11, color=TEXT, ha="center", va="center")
    ax.text(right_x, band_top - 0.97, "この癖は会話データ側にあり、大きくしても消えない",
            fontsize=11, color=TEXT, ha="center", va="center")

    ax.text(FIG_W / 2, FOOTER_H / 2 + 0.08,
            "口調の追加学習は 30ステップ・48秒。会話データは両方まったく同じものです",
            fontsize=9.4, color=MUTED, ha="center", va="center")
    ax.text(FIG_W / 2, FOOTER_H / 2 - 0.18,
            f"temperature {s['temperature']} / top_k {s['top_k']} / "
            f"repetition_penalty {s['repetition_penalty']}　"
            "全文は runs/3lm/gal_reply_compare.json",
            fontsize=8.4, color="#6d7d9c", ha="center", va="center")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
