"""一晩走らせるための安全装置.

その3 で一度カーネルパニックでMacを落とした。落ちたときのログはこう:

    panic(cpu 9): IOGPUGroupMemory.cpp:220 Assertion failed:
    result != kIOReturnSuccess
    Panicked task ...: pid 3625: python3.11

Apple Silicon はユニファイドメモリなので GPU の割り当ても本体RAMから取る。
MLX は既定で上限を持たず、要求が通らなくなると GPU ドライバ (IOGPUFamily) の側が
確保失敗をアサーションで扱ってカーネルごと落とす。つまり **アプリの例外にならず
マシンが落ちる**。だからアプリ側で先に止める。

そのときはローカルLLMの推論で踏んだが、8時間の学習でも条件は同じ。
むしろ学習の方が危ない。推論は数分で終わるので「今のメモリ状況で大丈夫か」を
一度確かめれば済むが、学習は8時間のあいだに他のアプリが起動したり
ブラウザのタブが増えたりして、途中で条件が変わる。

やっていることは4つ。

  1. MLX に明示的な上限を与える。超えたら Python の例外になり、カーネルまで行かない
  2. 開始前にスワップと空きメモリを見て、余裕がなければ走らせない
  3. 定期的にピーク使用量を測り、上限に近づいたら**保存してから**止める
  4. 熱による速度低下を記録する (原因の分からない「遅くなった」を潰すため)
"""

from __future__ import annotations

import subprocess

import mlx.core as mx

GB = 2**30


def device_summary() -> dict:
    info = mx.device_info()
    return {
        "name": info["device_name"],
        "memory_gb": info["memory_size"] / GB,
        # GPU が快適に使える上限。物理メモリより小さい値が返る
        "working_set_gb": info["max_recommended_working_set_size"] / GB,
        "max_buffer_gb": info["max_buffer_length"] / GB,
    }


def _sysctl(name: str) -> str:
    try:
        return subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def swap_used_gb() -> float:
    """使用中のスワップ量をGBで返す. 取れなければ 0 を返す."""
    # 例: total = 1024.00M  used = 12.25M  free = 1011.75M  (encrypted)
    parts = _sysctl("vm.swapusage").replace("=", " ").split()
    for i, token in enumerate(parts):
        if token == "used" and i + 1 < len(parts):
            value = parts[i + 1]
            scale = {"M": 1 / 1024, "G": 1.0, "K": 1 / 1024**2}.get(value[-1], 0)
            try:
                return float(value[:-1]) * scale
            except ValueError:
                return 0.0
    return 0.0


def cpu_speed_limit() -> int | None:
    """熱で CPU のクロックが制限されているかを % で返す. 100 なら無制限.

    pmset -g therm は、制限が一度も起きていないと
    「No CPU power status has been recorded」だけを返す。
    つまり **値が取れないこと自体が「まだ絞られていない」の意味**になる。

    夜通し回すと筐体内に熱が溜まり、ここが 100 を割ってくる。
    tok/s が落ちたときに「熱なのか、実装を変えたせいなのか」を
    切り分けられるようにしておく。
    """
    try:
        out = subprocess.run(
            ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if "CPU_Speed_Limit" in line:
            try:
                return int(line.split("=")[-1].strip())
            except ValueError:
                return None
    return None


def configure(limit_gb: float | None = None, cache_gb: float = 4.0) -> float:
    """MLX に使用量の上限を与え、実際に設定した上限を返す.

    上限は「GPUが快適に使える上限」よりさらに下に置く。ここを物理メモリいっぱいに
    しても意味はなく、超えた瞬間に落ちる相手はアプリではなくカーネルなので、
    余裕を持たせる側に倒す。
    """
    device = device_summary()
    ceiling = device["working_set_gb"] * 0.75
    if limit_gb is None or limit_gb > ceiling:
        limit_gb = ceiling
    mx.set_memory_limit(int(limit_gb * GB))
    mx.set_cache_limit(int(cache_gb * GB))
    print(
        f"  {device['name']} / 実装 {device['memory_gb']:.0f}GB / "
        f"GPU推奨上限 {device['working_set_gb']:.0f}GB"
    )
    print(f"  MLXの上限 {limit_gb:.1f}GB / キャッシュ {cache_gb:.1f}GB")
    return limit_gb


def preflight(required_gb: float = 4.0, max_swap_gb: float = 8.0) -> None:
    """走らせて安全かを開始前に確かめる. 危なければ例外で止める."""
    device = device_summary()
    if device["memory_gb"] < required_gb:
        raise SystemExit(
            f"実装メモリ {device['memory_gb']:.0f}GB では足りません "
            f"(この設定には {required_gb:.0f}GB 必要)。"
            "--batch-size か --n-embd を下げてください。"
        )
    swap = swap_used_gb()
    print(f"  スワップ使用量: {swap:.1f}GB")
    if swap > max_swap_gb:
        raise SystemExit(
            f"スワップを {swap:.1f}GB 使っています。この状態でGPUメモリを大量に要求すると\n"
            "カーネルパニックでmacOSごと落ちることがあります。\n"
            "他のアプリを終了するか、再起動してから実行してください。"
        )


class MemoryGuard:
    """定期的にピーク使用量を見て、上限に近づいたら知らせる.

    学習では「例外を投げて終わり」にはできない。8時間目に投げられたら
    そこまでの学習が消える。なので危険域では例外を投げず、
    **呼び出し側に「保存して止めろ」を返す**。実際の保存は train.py が行う。
    """

    def __init__(self, limit_gb: float, headroom: float = 0.85) -> None:
        self.threshold_gb = limit_gb * headroom
        self.peak_gb = 0.0

    def over_threshold(self) -> bool:
        peak = mx.get_peak_memory() / GB
        self.peak_gb = max(self.peak_gb, peak)
        return peak > self.threshold_gb

    def release(self) -> None:
        """区切りでキャッシュを返す. 断片化と積み上がりを防ぐ."""
        mx.clear_cache()
        mx.reset_peak_memory()
