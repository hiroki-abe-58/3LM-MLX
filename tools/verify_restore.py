"""チェックポイントを別プロセスで戻し、保存時の状態と完全に一致するか確かめる.

tools/killtest.py の検査 [A]。ここだけは**ビット単位の一致を要求する**。
戻した瞬間にずれていたら、その後の学習はもう別物なので、
許容差でごまかしてはいけない。

見るのは4つ。

  重み                  model.safetensors と、読み込んだモデルの値
  オプティマイザの m,v  Adam の1次・2次モーメント。捨てると再開直後に更新量が跳ねる
  step                  学習率スケジュールの現在地がここから決まる
  learning_rate         step から再計算された値が、保存時と同じか

step が戻れば learning_rate も自動的に戻る、というのが MLX の設計。
それを口で言うのではなく、ここで確かめる。
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import checkpoint  # noqa: E402
from src.model import GPTConfig, MiniGPT  # noqa: E402
from src.train import build_schedule  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("使い方: python tools/verify_restore.py <チェックポイントのディレクトリ>")
    ckpt = Path(sys.argv[1])

    saved_weights = {k: np.asarray(v) for k, v in mx.load(str(ckpt / "model.safetensors")).items()}
    saved_opt = {k: np.asarray(v) for k, v in mx.load(str(ckpt / "optimizer.safetensors")).items()}

    cfg = GPTConfig.load(ckpt / "config.json")
    model = MiniGPT(cfg)
    mx.eval(model.parameters())
    args = checkpoint.TrainState.from_path(ckpt / "train_state.json").args
    total_steps = args.get("steps") or 1
    optimizer = optim.AdamW(
        learning_rate=build_schedule(
            args.get("lr", 6e-4),
            total_steps,
            max(100, int(total_steps * args.get("warmup_frac", 0.02))),
            args.get("min_lr_ratio", 0.1),
        ),
        weight_decay=args.get("weight_decay", 0.1),
    )
    state = checkpoint.restore(ckpt, model, optimizer)

    problems: list[str] = []

    got_weights = dict(tree_flatten(model.parameters()))
    for key, want in saved_weights.items():
        if key not in got_weights:
            problems.append(f"重み {key} が読み込まれていない")
            continue
        if not np.array_equal(np.asarray(got_weights[key]), want):
            problems.append(f"重み {key} が一致しない")

    got_opt = dict(tree_flatten(optimizer.state))
    for key, want in saved_opt.items():
        if key not in got_opt:
            problems.append(f"オプティマイザ状態 {key} が復元されていない")
            continue
        if not np.array_equal(np.asarray(got_opt[key]), want):
            problems.append(f"オプティマイザ状態 {key} が一致しない")

    n_moments = sum(1 for k in saved_opt if k.endswith((".m", ".v")))
    opt_step = int(np.asarray(optimizer.state["step"]))
    if opt_step != state.step:
        problems.append(
            f"optimizer.state['step'] {opt_step} と train_state.json の step {state.step} が違う"
        )

    print(f"  重み {len(saved_weights)} 個 / m,v {n_moments} 個 / step {opt_step} / "
          f"lr {float(np.asarray(optimizer.state['learning_rate'])):.6e}")
    if problems:
        print("  一致しません:")
        for p in problems[:10]:
            print(f"    - {p}")
        raise SystemExit(1)
    print("  すべてビット単位で一致しました")


if __name__ == "__main__":
    main()
