#!/usr/bin/env bash
# 一晩の学習を、落ちても勝手に立ち上がる形で走らせる。
#
# 使い方:
#   ./scripts/train_overnight.sh --data data/3lm --tokens 514000000 --max-hours 8.5
#
# 端末を閉じても続けたい場合は tmux か nohup を併用する:
#   tmux new -s 3lm './scripts/train_overnight.sh --tokens 514000000 --max-hours 8.5'
#   nohup ./scripts/train_overnight.sh --tokens 514000000 --max-hours 8.5 &
#
# ここでやっていることは4つ。
#
# 1. caffeinate でスリープを止める
#    寝ている間にMacがスリープすると GPU の処理が止まる。復帰後に続く場合も
#    あるが、止まったまま朝まで放置になることもある。
#    -d ディスプレイ / -i アイドルスリープ / -m ディスクスリープ / -s システムスリープ
#
# 2. 落ちたら --resume auto で再投入する
#    src/train.py は落ちる直前のチェックポイントから続けられる。
#
# 3. step が進んでいなければ再投入をやめる
#    ここが肝。設定ミスやデータの不備で「起動して即落ちる」状態になると、
#    再投入ラッパーは朝まで無限に起動を繰り返し、何も進まないまま
#    「動いていた」ように見えてしまう。前回の step を覚えておいて、
#    進んでいなければ止める。
#
# 4. 各回の標準出力を別ファイルに残す
#    何回落ちたか、どこで落ちたかを後から追えるようにする。

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

MAX_RESTARTS="${MAX_RESTARTS:-40}"
OUT_DIR=""
LOG_DIR="runs/3lm"

# --out を拾う (step の確認に使う)。それ以外の引数はそのまま train.py へ渡す。
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  if [[ "${ARGS[i]}" == "--out" ]]; then OUT_DIR="${ARGS[i + 1]}"; fi
done
OUT_DIR="${OUT_DIR:-checkpoints/pretrain}"

mkdir -p "$LOG_DIR"

current_step() {
  # CURRENT が指すチェックポイントの step を読む。無ければ -1。
  python3 - "$OUT_DIR" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
marker = root / "CURRENT"
if not marker.exists():
    print(-1); raise SystemExit
slot = marker.read_text(encoding="utf-8").strip()
state = root / slot / "train_state.json"
print(json.loads(state.read_text(encoding="utf-8"))["step"] if state.exists() else -1)
PY
}

echo "======================================================================"
echo " 一晩の学習を開始します"
echo "   引数        : ${ARGS[*]}"
echo "   出力        : $OUT_DIR"
echo "   再投入の上限: $MAX_RESTARTS 回"
echo "   進捗の確認  : python3 tools/watch.py"
echo "======================================================================"

attempt=0
last_step=-1

while ((attempt < MAX_RESTARTS)); do
  attempt=$((attempt + 1))
  run_log="$LOG_DIR/run_$(printf '%02d' "$attempt").log"

  echo ""
  echo "--- $attempt 回目 $(date '+%Y-%m-%d %H:%M:%S') → $run_log ---"

  # caffeinate に子プロセスを渡すと、その子が終わるまで assertion が続く。
  # 学習が終われば自動的にスリープ抑止も解除される。
  caffeinate -dimsu python3 src/train.py "${ARGS[@]}" --resume auto 2>&1 | tee "$run_log"
  rc="${PIPESTATUS[0]}"

  step="$(current_step)"
  echo "--- 終了コード $rc / step $step ---"

  if ((rc == 0)); then
    echo "正常終了しました。"
    exit 0
  fi

  # 進んでいないなら、再投入しても同じところで落ちる。
  if [[ "$step" == "$last_step" ]]; then
    echo "======================================================================"
    echo " step が $step から進みませんでした。再投入を中止します。"
    echo " $run_log の末尾を見て原因を確かめてください。"
    echo "======================================================================"
    tail -20 "$run_log"
    exit 1
  fi
  last_step="$step"

  echo "10秒後に再開します (step $step から)"
  sleep 10
done

echo "再投入の上限 $MAX_RESTARTS 回に達しました。中止します。"
exit 1
