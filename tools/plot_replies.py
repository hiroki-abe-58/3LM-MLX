"""前作と今作の返答を、吹き出しで並べた画像にする.

`tools/compare_replies.py` が保存した json から作る。
**画像に書く文字を手で打たない**ためで、こうしておけば
モデルを作り直したときに図だけ古い返答のまま残ることがない。

2枚出す。

    chat : 同じ質問への返答。誰が見ても分かる違いを見せる
    raw  : 素の日本語の続き。事前学習の厚みが出る土俵

長い返答は途中で切る。**どこを切ったかが分かるように「…」を付ける**。
全文は json に残っているので、切り方で印象を作っていないか確認できる。

座標は inch で持つ。文字数に応じて吹き出しの高さが変わるので、
先に全部の高さを積んでから図の大きさを決めないと、下がはみ出す。

使い方:
    python tools/plot_replies.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matplotlib.patches import FancyBboxPatch  # noqa: E402

from tools.plotting import BACKGROUND, apply_style, clip  # noqa: E402
from tools.plotting import wrap_ja as _wrap_ja  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

OLD = "2LM 13.81M"
NEW = "3LM 35.66M"

# 前作を青、今作を緑にする。良し悪しの色分けには使わない。
COLOR_OLD, EDGE_OLD = "#16273f", "#6ea8ff"
COLOR_NEW, EDGE_NEW = "#14301f", "#8ce99a"

FIG_W = 11.5
COL_W = 5.15
COL_X = (0.45, 5.9)

CHARS_PER_LINE = 33
LINE_H = 0.255
PAD_Y = 0.13
HEADER_H = 1.30
PROMPT_H = 0.50
GAP_H = 0.38
FOOTER_H = 0.46


def wrap_ja(text: str, width: int = CHARS_PER_LINE) -> list[str]:
    return _wrap_ja(text, width)


class Sheet:
    """inch 単位で上から順に積んでいく紙."""

    def __init__(self, height: float) -> None:
        apply_style()
        import matplotlib.pyplot as plt

        self.fig = plt.figure(figsize=(FIG_W, height), dpi=150)
        self.fig.patch.set_facecolor(BACKGROUND)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, FIG_W)
        self.ax.set_ylim(0, height)
        self.ax.axis("off")
        self.ax.set_facecolor(BACKGROUND)
        self.height = height
        self.y = height

    def header(self, title: str, subtitle: str) -> None:
        ax = self.ax
        ax.text(FIG_W / 2, self.y - 0.34, title, fontsize=17, color="#ffffff",
                ha="center", va="center", fontweight="bold")
        ax.text(FIG_W / 2, self.y - 0.68, subtitle, fontsize=10.2, color="#9fb3d9",
                ha="center", va="center")
        for x, label, color in (
            (COL_X[0] + COL_W / 2, f"{OLD}　前作 / 949万文字 / 27分", EDGE_OLD),
            (COL_X[1] + COL_W / 2, f"{NEW}　今回 / 12億文字 / 8時間", EDGE_NEW),
        ):
            ax.text(x, self.y - 1.06, label, fontsize=11, color=color,
                    ha="center", va="center", fontweight="bold")
        self.y -= HEADER_H

    def prompt(self, text: str) -> None:
        self.ax.text(FIG_W / 2, self.y - PROMPT_H / 2, text, fontsize=12,
                     color="#ffd43b", ha="center", va="center", fontweight="bold")
        self.y -= PROMPT_H

    def bubbles(self, old: str, new: str) -> None:
        pair = (
            (COL_X[0], wrap_ja(old), COLOR_OLD, EDGE_OLD),
            (COL_X[1], wrap_ja(new), COLOR_NEW, EDGE_NEW),
        )
        height = max(len(lines) for _, lines, _, _ in pair) * LINE_H + PAD_Y * 2
        for x, lines, face, edge in pair:
            self.ax.add_patch(
                FancyBboxPatch(
                    (x, self.y - height), COL_W, height,
                    boxstyle="round,pad=0.04,rounding_size=0.12",
                    facecolor=face, edgecolor=edge, linewidth=1.2,
                )
            )
            for i, line in enumerate(lines):
                self.ax.text(x + 0.18, self.y - PAD_Y - LINE_H * (i + 0.6), line,
                             fontsize=10, color="#e8eefc", va="center", ha="left")
        self.y -= height + GAP_H

    def footer(self, text: str) -> None:
        self.ax.text(FIG_W / 2, FOOTER_H / 2, text, fontsize=8.6, color="#7d8fb3",
                     ha="center", va="center")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, facecolor=self.fig.get_facecolor())
        print(f"保存: {path}")


def pair_height(old: str, new: str) -> float:
    lines = max(len(wrap_ja(old)), len(wrap_ja(new)))
    return PROMPT_H + lines * LINE_H + PAD_Y * 2 + GAP_H


class Bpc:
    """bits/char を json から引く.

    図の中の数字を手で書くと、モデルを取り直したときに図だけ古くなる。
    どの重みの文章に、どの重みの点を並べているかもここで固定する。
    """

    # 吹き出しの文章を出した重み → domain_bpc.json 側の呼び名
    ALIAS = {OLD: "2LM 13.81M", NEW: "3LM 35.66M SFT済み"}
    CHAT = "A 公開データ由来の会話"
    PLAIN = "B Web文+青空文庫"

    def __init__(self, path: Path) -> None:
        self.records = json.loads(path.read_text(encoding="utf-8"))["records"]

    def get(self, model: str, domain: str) -> float:
        name = self.ALIAS.get(model, model)
        for r in self.records:
            if r["model"] == name and r["domain"] == domain:
                return r["bits_per_char"]
        raise SystemExit(f"bits/char が見つかりません: {name} / {domain}")


def pick(records: list[dict], mode: str, model: str, prompt: str, limit: int) -> str:
    for r in records:
        if r["mode"] == mode and r["model"] == model and r["prompt"] == prompt:
            return clip(r["reply"].strip(), limit)
    raise SystemExit(f"見つかりません: {mode} / {model} / {prompt}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "runs" / "3lm" / "reply_compare.json"))
    ap.add_argument("--bpc", default=str(ROOT / "runs" / "3lm" / "domain_bpc.json"))
    ap.add_argument("--outdir", default=str(ROOT / "docs" / "images"))
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    rec, s = data["records"], data["sampling"]
    bpc = Bpc(Path(args.bpc))
    outdir = Path(args.outdir)

    # --- 1枚目: 会話 -------------------------------------------------------
    pairs = [
        (p, pick(rec, "chat", OLD, p, 190), pick(rec, "chat", NEW, p, 190))
        for p in ("おすすめの本を3つ挙げてください", "日本の首都はどこですか")
    ]
    total = HEADER_H + FOOTER_H + sum(pair_height(o, n) for _, o, n in pairs)
    sheet = Sheet(total)
    sheet.header(
        "同じ質問を、同じ設定で聞き比べる",
        "違うのはモデルだけ。質問も乱数もサンプリング条件もそろえてあります",
    )
    for prompt, old, new in pairs:
        sheet.prompt(prompt)
        sheet.bubbles(old, new)
    sheet.footer(
        f"temperature {s['temperature']} / top_k {s['top_k']} / "
        f"repetition_penalty {s['repetition_penalty']}　"
        "長い返答は途中で切っています（…）。全文は runs/3lm/reply_compare.json"
    )
    sheet.save(outdir / "3lm-vs-2lm-chat.png")

    # --- 2枚目: 素の文章 ---------------------------------------------------
    raw = "梅雨が明けると、庭の紫陽花は"
    old = pick(rec, "raw", OLD, raw, 200)
    new = pick(rec, "raw", NEW, raw, 200)
    score_h = 1.62
    sheet = Sheet(HEADER_H + FOOTER_H + pair_height(old, new) + score_h)
    sheet.header(
        "対話ではなく、素の日本語の続きを書かせる",
        "事前学習の量がそのまま出る土俵。会話データには出てこない語彙と文体を与えます",
    )
    sheet.prompt(f"「{raw}」に続けて書かせる")
    sheet.bubbles(old, new)

    ax = sheet.ax
    ax.text(FIG_W / 2, sheet.y - 0.22, "この土俵での bits/char（低いほど良い）",
            fontsize=10.4, color="#9fb3d9", ha="center", va="center")
    for x, model, color in (
        (COL_X[0] + COL_W / 2, OLD, EDGE_OLD),
        (COL_X[1] + COL_W / 2, NEW, EDGE_NEW),
    ):
        ax.text(x, sheet.y - 0.80, f"{bpc.get(model, Bpc.PLAIN):.3f}", fontsize=30,
                color=color, ha="center", va="center", fontweight="bold")
    ax.text(FIG_W / 2, sheet.y - 1.36,
            f"SFT で会話に寄せる前は {bpc.get('3LM 35.66M 事前学習のみ', Bpc.PLAIN):.3f}。"
            "口調や作法を覚えるかわりに、素の日本語は少し戻ります。",
            fontsize=9.6, color="#9fb3d9", ha="center", va="center")
    sheet.footer(
        "一方、会話の採点では前作のほうが良い点を取ります"
        f"（{bpc.get(OLD, Bpc.CHAT):.3f} 対 {bpc.get(NEW, Bpc.CHAT):.3f}）。"
        "その検証セットが前作の学習データと同じ出どころだからです。"
    )
    sheet.save(outdir / "3lm-vs-2lm-raw.png")


if __name__ == "__main__":
    main()
