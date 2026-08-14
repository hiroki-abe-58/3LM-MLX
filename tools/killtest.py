"""再開機構が本当に効くかを、わざと kill -9 して確かめる.

一晩の学習に入る前の必須ゲート。「再開できるはず」で8時間を投じてはいけない。

## 途中で分かったこと: 重みの一致では検証できない

最初はハッシュで比べようとして落ちた。差は 1.19e-07 で float32 の 1 ULP。
「では許容差で見よう」と 1e-5 にしたら、今度は kill 後の差が 5.3e-03 で
また落ちた。97ステップぶん Adam を回すあいだに、1 ULP の差が育っていた。

そこで **同じ設定を中断なしで2回走らせて比べた**。結果は 7.19e-03。
つまり **MLX / Metal は同じ入力でも走らせるたびに最後の桁が変わる**。
GPU の総和は足す順番が実行のたびに変わりうるので、これは避けられない。

kill を挟んだときの差 (5.3e-03) は、**何もしなくても出る差 (7.2e-03) より
小さかった**。つまり再開機構は正しく、比較の基準が間違っていた。

## だからこの検証はこう組む

  [B] バッチが (seed, step) だけで決まるか        ← 厳密に一致すること
  [A] 戻した瞬間の状態が保存時と同じか            ← 厳密に一致すること
  [雑音] 同じ設定を2回走らせたときの差             ← これが「一致」の下限
  [C] kill -9 を挟んだときの差                    ← 雑音と同じ桁に収まること
  [対照] m,v を捨てて再開したときの差              ← 雑音より明らかに大きいこと
  [D] 学習曲線に段差が出ていないか                ← 実用上いちばん大事

[A] と [B] は決定的な処理なので厳密に見る。[C] は GPU の非決定性が混ざるので
雑音と比べる。[対照] を置くのは、許容差をゆるめた結果
**何も検出しない検証**になっていないことを示すため。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint import read_current  # noqa: E402

# kill 後の差は「雑音の何倍まで許すか」で見る。
NOISE_MULTIPLE_OK = 3.0
# 対照実験は「雑音の何倍以上あれば、この検証が機能していると言えるか」。
NOISE_MULTIPLE_CONTROL = 5.0
# 学習曲線の段差。m,v を忘れると 0.1 以上跳ねるので、その手前に置く。
LOSS_GAP_OK = 0.05

# src/train.py に渡す --log-interval と同じ値。
# 曲線を比較するとき「平均を取る窓の長さ」として使う。
LOG_INTERVAL = 10


def load_weights(root: Path) -> dict[str, np.ndarray]:
    path = current_dir(root) / "model.safetensors"
    return {k: np.asarray(v, dtype=np.float32) for k, v in mx.load(str(path)).items()}


def max_abs_diff(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> tuple[float, str]:
    if set(a) != set(b):
        raise SystemExit(f"テンソルの構成が違います: {set(a) ^ set(b)}")
    worst, where = 0.0, ""
    for key in a:
        diff = float(np.max(np.abs(a[key] - b[key])))
        if diff > worst:
            worst, where = diff, key
    return worst, where


def current_dir(root: Path) -> Path:
    slot = read_current(root)
    if slot is None:
        raise SystemExit(f"{root} に CURRENT がありません")
    return root / slot


def train(
    out: Path, steps: int, data: Path, log: Path, resume: str = "auto",
    kill_after: float | None = None, quiet: bool = False,
) -> int:
    """学習を別プロセスで起動する. kill_after 秒後に SIGKILL する.

    SIGKILL (kill -9) を使うのが要点。SIGTERM だと train.py が
    「保存して終了」の道を通ってしまい、本当に落ちたときの経路を試せない。
    """
    cmd = [
        sys.executable, str(ROOT / "src" / "train.py"),
        "--data", str(data), "--out", str(out),
        "--arch", "2lm", "--block-size", "256",
        "--n-layer", "6", "--n-head", "6", "--n-embd", "384",
        "--steps", str(steps), "--log-interval", str(LOG_INTERVAL),
        "--eval-interval", str(steps * 10),  # 検証は走らせない (時間の無駄)
        "--save-interval-min", "0.05",       # 3秒ごとに保存し、殺す位置を選ばない
        "--seed", "4242", "--resume", resume,
        "--log", str(log), "--heartbeat", str(log.with_suffix(".hb.json")),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if kill_after is None:
        text, _ = proc.communicate()
        if not quiet:
            tail = [ln for ln in text.splitlines() if ln.startswith(("step", "  [", "終了"))]
            print("\n".join(f"    {ln}" for ln in tail[-3:]))
        return proc.returncode

    deadline = time.time() + kill_after
    while time.time() < deadline and proc.poll() is None:
        proc.stdout.readline()
    if proc.poll() is None:
        print(f"    >>> kill -9 (pid {proc.pid})")
        os.kill(proc.pid, signal.SIGKILL)
    proc.wait()
    return proc.returncode


def read_losses(path: Path) -> dict[int, float]:
    """ログから step -> train_loss を読む. コメント行と検証行は飛ばす."""
    losses: dict[int, float] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(ln for ln in fh if not ln.startswith("#")):
            if row.get("train_loss"):
                losses[int(row["step"])] = float(row["train_loss"])
    return losses


def usable_steps(shared: list[int], marks: list[int], interval: int) -> list[int]:
    """曲線の比較に使える step だけを返す.

    csv の train_loss は「直近 interval ステップの平均」で、再開すると
    その平均を取る窓が途中から始まる。step 20 の行が、中断なしなら
    11〜20 の平均なのに、step 14 で再開した側は 15〜20 の平均になる。
    **後半6ステップのほうが学習が進んでいるので、必ず低く出る。**

    これは再開が壊れているのではなく、平均する範囲が違うだけ。
    ここを比較に入れると「再開直後に損失が跳ねた」と誤読する
    (実測では 0.46 の差が出て、壊れた対照実験の 0.12 より大きく見えた)。
    """
    return [s for s in shared if not any(s - interval < m < s for m in marks)]


def resumed_steps(path: Path) -> list[int]:
    steps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# resumed at step"):
            steps.append(int(line.split()[4]))
    return steps


def check_batch_determinism(data: Path) -> bool:
    """バッチが (seed, step) だけで決まっていることを、別プロセス2本で確かめる.

    同じプロセス内で2回呼んで一致しても、それは「乱数生成器を作り直している」
    ことの証明にしかならない。プロセスをまたいで一致することを見る。
    """
    script = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
        "from src.data import TokenBin, get_batch\n"
        "import numpy as np\n"
        f"b = TokenBin({str(data / 'train.bin')!r})\n"
        "for s in (1, 7, 4242, 100000):\n"
        "    x, y = get_batch(b, 4, 64, 4242, s)\n"
        "    print(s, int(np.asarray(x).sum()), int(np.asarray(y).sum()))\n"
    )
    runs = [
        subprocess.run([sys.executable, "-c", script], capture_output=True, text=True).stdout
        for _ in range(2)
    ]
    return runs[0] == runs[1] and runs[0].strip() != ""


def main() -> None:
    ap = argparse.ArgumentParser(description="kill -9 して再開が効くかを検証する")
    ap.add_argument("--data", default=str(ROOT / "data" / "sft8k"))
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--mid", type=int, default=40, help="対照実験で使う中間のステップ")
    ap.add_argument("--kill-after", type=float, default=4.0)
    ap.add_argument("--kills", type=int, default=2)
    ap.add_argument("--work", default="/tmp/3lm_killtest")
    args = ap.parse_args()

    data = Path(args.data)
    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    results: dict[str, object] = {"steps": args.steps}

    def header(text: str) -> None:
        print()
        print("=" * 70)
        print(text)
        print("=" * 70)

    header("[B] バッチが (seed, step) だけで決まるかを、別プロセス2本で確認")
    deterministic = check_batch_determinism(data)
    results["batch_deterministic"] = deterministic
    print(f"  {'厳密に一致した' if deterministic else '一致しない'}")

    header(f"[雑音] 同じ設定を中断なしで2回走らせ、GPU の非決定性を測る ({args.steps}ステップ)")
    runs = []
    for tag in ("base1", "base2"):
        print(f"  --- {tag} ---")
        if train(work / tag, args.steps, data, work / f"{tag}.csv", resume="never") != 0:
            raise SystemExit(f"{tag} の学習が失敗しました")
        runs.append(load_weights(work / tag))
    noise, noise_where = max_abs_diff(runs[0], runs[1])
    results["noise"] = noise
    print(f"  重みの最大差: {noise:.3e}  ({noise_where})")
    print("  ※ これが「一致」の下限。kill の検証はこれと比べる")

    header(f"[A] step {args.mid} で保存し、別プロセスで戻して厳密な一致を見る")
    mid = work / "mid"
    if train(mid, args.mid, data, work / "mid.csv", resume="never") != 0:
        raise SystemExit("中間の学習が失敗しました")
    check = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "verify_restore.py"), str(current_dir(mid))],
        capture_output=True, text=True,
    )
    print(check.stdout.rstrip() or check.stderr.rstrip())
    results["restore_exact"] = check.returncode == 0

    header(f"[C] kill -9 を {args.kills} 回はさんで {args.steps} ステップ")
    killed = work / "killed"
    log = work / "killed.csv"
    for i in range(args.kills):
        print(f"  --- {i + 1}回目 ---")
        train(killed, args.steps, data, log, resume="auto", kill_after=args.kill_after)
        left = json.loads((current_dir(killed) / "train_state.json").read_text(encoding="utf-8"))
        print(f"    残った step: {left['step']}")
        if left["step"] >= args.steps:
            print("    もう終わっているので kill を打ち切ります")
            break
    print("  --- 最後まで回す ---")
    if train(killed, args.steps, data, log, resume="auto") != 0:
        raise SystemExit("再開後の学習が失敗しました")
    killed_state = json.loads((current_dir(killed) / "train_state.json").read_text(encoding="utf-8"))
    killed_diff, killed_where = max_abs_diff(runs[0], load_weights(killed))
    results["resumes"] = killed_state["resumes"]
    results["killed_diff"] = killed_diff
    print(f"  再開 {killed_state['resumes']} 回 / 重みの最大差 {killed_diff:.3e}  ({killed_where})")

    header("[D] 学習曲線に段差が出ていないか")
    base_losses = read_losses(work / "base1.csv")
    killed_losses = read_losses(log)
    marks = resumed_steps(log)
    shared = sorted(set(base_losses) & set(killed_losses))
    comparable = usable_steps(shared, marks, LOG_INTERVAL)
    truncated = [s for s in shared if s not in set(comparable)]

    gap = max((abs(base_losses[s] - killed_losses[s]) for s in comparable), default=0.0)
    at = max(comparable, key=lambda s: abs(base_losses[s] - killed_losses[s])) if comparable else 0
    results["loss_gap"] = gap
    results["resumed_at"] = marks
    results["loss_compared_steps"] = comparable
    results["loss_excluded_steps"] = truncated
    print(f"  再開した step: {marks}")
    print(f"  中断なしの曲線との最大差: {gap:.4f} (step {at}, {len(comparable)}点で比較)")
    if truncated:
        # ここを比較に入れると再開が壊れているように見える。理由を出しておく。
        print(f"  比較から外した step: {truncated}")
        print(f"    train_loss は直近 {LOG_INTERVAL} ステップの平均で、"
              "再開直後の行はその窓が短くなる")
        for s in truncated:
            print(f"    step {s:4d}: 中断なし {base_losses[s]:.4f}"
                  f" / kill後 {killed_losses[s]:.4f}  (窓が短いため比較できない)")

    header("[対照] optimizer.safetensors を捨てて再開したらどうなるか")
    noopt = work / "noopt"
    shutil.copytree(mid, noopt)
    (current_dir(noopt) / "optimizer.safetensors").unlink()
    print("  Adam の m, v を消しました。ここから最後まで回します")
    if train(noopt, args.steps, data, work / "noopt.csv", resume="auto") != 0:
        raise SystemExit("対照実験が失敗しました")
    noopt_diff, noopt_where = max_abs_diff(runs[0], load_weights(noopt))
    noopt_losses = read_losses(work / "noopt.csv")
    # 対照も同じ物差しで測る。窓が切られた行を外さないと、
    # 「壊れているから差が出た」のか「窓が短いから差が出た」のか区別できない。
    noopt_shared = usable_steps(
        sorted(set(base_losses) & set(noopt_losses)),
        resumed_steps(work / "noopt.csv"),
        LOG_INTERVAL,
    )
    noopt_gap = max((abs(base_losses[s] - noopt_losses[s]) for s in noopt_shared), default=0.0)
    results["noopt_diff"] = noopt_diff
    results["noopt_loss_gap"] = noopt_gap
    print(f"  重みの最大差 {noopt_diff:.3e}  ({noopt_where})")
    print(f"  学習曲線の最大差 {noopt_gap:.4f}")

    header("結果")
    print(f"  [B] バッチの決定性              : {'OK' if deterministic else 'NG'}")
    print(f"  [A] 戻した瞬間の厳密な一致      : {'OK' if results['restore_exact'] else 'NG'}")
    print()
    print(f"  [雑音] 同設定を2回走らせた差    : {noise:.3e}   ← 下限")
    print(f"  [C]  kill -9 {killed_state['resumes']}回のあとの差     : {killed_diff:.3e}"
          f"   (雑音の {killed_diff / noise:.2f} 倍)")
    print(f"  [対照] m,v を捨てた場合の差     : {noopt_diff:.3e}"
          f"   (雑音の {noopt_diff / noise:.1f} 倍)")
    print()
    print(f"  [D]  kill 後の学習曲線の段差    : {gap:.4f}   (許容 {LOSS_GAP_OK})")
    print(f"  [対照] m,v を捨てた場合の段差   : {noopt_gap:.4f}")

    checks = {
        "バッチが step の関数になっている": deterministic,
        "戻した瞬間の状態が保存時と厳密に一致する": bool(results["restore_exact"]),
        "kill 後の差が雑音の範囲に収まっている": killed_diff <= noise * NOISE_MULTIPLE_OK,
        "kill 後の学習曲線に段差が無い": gap <= LOSS_GAP_OK,
        "対照実験では差が雑音より明らかに大きい": noopt_diff >= noise * NOISE_MULTIPLE_CONTROL,
        "実際に再開が起きた": killed_state["resumes"] > 0,
    }
    print()
    for label, ok in checks.items():
        print(f"  {'OK ' if ok else 'NG '} {label}")

    # 記事で数字を引くので、作業用の一時ディレクトリではなく runs/ に残す。
    results["checks"] = {k: bool(v) for k, v in checks.items()}
    results["passed"] = all(checks.values())
    for path in (work / "result.json", ROOT / "runs" / "3lm" / "killtest.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print()
    if all(checks.values()):
        print("合格。重み・オプティマイザの m,v・学習率の位置・データ順がすべて戻っています。")
        print("kill 後の差は「同じ設定を2回走らせたときの差」の範囲に収まっており、")
        print("m,v を捨てた場合には桁違いの差が出ることも確認できました。")
        print("一晩の学習に進んでよい状態です。")
    else:
        print("不合格。上の NG を見て原因を切り分けてください。")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
