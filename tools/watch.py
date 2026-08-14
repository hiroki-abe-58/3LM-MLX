"""別の端末から学習の進捗を覗く.

    python3 tools/watch.py

一晩の学習は tmux か nohup で端末から切り離して走らせるので、
標準出力を直接見られない。src/train.py が定期的に書いている
runs/3lm/heartbeat.json を読んで表示する。

見るのは進捗だけではない。**tok/s と CPU の速度制限**を並べて出す。
朝になって「思ったより進んでいない」となったとき、
熱で絞られていたのか、そもそも見積もりが甘かったのかを切り分けられる。
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def bar(fraction: float, width: int = 32) -> str:
    filled = int(max(0.0, min(1.0, fraction)) * width)
    return "█" * filled + "░" * (width - filled)


def render(hb: dict, age: float) -> str:
    step, total = hb.get("step", 0), hb.get("total_steps", 1)
    tokens, budget = hb.get("tokens_seen", 0), hb.get("token_budget", 1)
    eta = hb.get("eta_hours", 0.0)
    done = datetime.now() + timedelta(hours=eta)
    limit = hb.get("cpu_speed_limit")

    # 最初の検証が走るまで best_val は inf。そのまま出すと壊れて見える。
    best = hb.get("best_val")
    best_text = "未測定" if best is None or best == float("inf") else f"{best}"

    lines = [
        f"  {bar(step / max(total, 1))}  {step / max(total, 1) * 100:5.1f}%",
        f"  step        {step:,} / {total:,}",
        f"  トークン    {tokens:,} / {budget:,}",
        f"  train_loss  {hb.get('train_loss')}      最良 val_loss {best_text}",
        f"  学習率      {hb.get('lr', 0):.3e}",
        f"  速度        {hb.get('tok_per_sec', 0) / 1e3:.1f}k tok/s",
        f"  経過        {hb.get('elapsed_hours', 0):.2f} 時間  "
        f"(再開 {hb.get('resumes', 0)} 回)",
        f"  残り        {eta:.2f} 時間  → 終了は {done:%m/%d %H:%M} ごろ",
        f"  ピークメモリ {hb.get('peak_gb', 0):.1f}GB",
    ]
    if limit is not None and limit < 100:
        lines.append(f"  熱          CPU が {limit}% に絞られています")
    else:
        lines.append("  熱          絞られていません")

    # 終了しているのに古い step が残っていると、落ちたのか終わったのか
    # 区別できない。train.py は終了時に finished を立てて上書きする。
    if hb.get("finished"):
        lines.append(f"  状態        終了しました ({hb.get('stop_reason', '理由不明')})")
    elif age > 180:
        lines.append(f"  状態        {age / 60:.0f}分 更新がありません。落ちている可能性があります")
    else:
        lines.append(f"  更新        {age:.0f}秒前")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(ROOT / "runs" / "3lm" / "heartbeat.json"))
    ap.add_argument("--once", action="store_true", help="1回だけ表示して終わる")
    ap.add_argument("--interval", type=float, default=10.0)
    args = ap.parse_args()

    path = Path(args.path)
    while True:
        if not path.exists():
            print(f"{path} がまだありません。学習の開始を待っています…")
        else:
            try:
                hb = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                hb = None
            if hb:
                if not args.once:
                    print("\033[2J\033[H", end="")  # 画面を消して左上へ
                print(f"3LM 事前学習  {datetime.now():%H:%M:%S}")
                print("=" * 56)
                print(render(hb, time.time() - hb.get("updated_at", 0)))
                print("=" * 56)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
