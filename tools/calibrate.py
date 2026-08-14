"""実測の tok/s から、一晩に入るトークン数とモデルの大きさを決める.

    python3 tools/calibrate.py --data data/3lm --hours 8

## なぜ測ってから決めるのか

その2 の実測は非埋め込み 10.6M・文脈256で 47k tok/s だった。
そこから「文脈512・非埋め込み 25.7M なら何 tok/s か」を計算で出そうとすると、
理屈の上では パラメータが2.4倍なので 20k tok/s 前後、になる。

だがこの見積もりはよく外れる。文脈を伸ばすと行列積が大きくなって
GPU の使用効率が上がる (小さい行列だと起動のたびのオーバーヘッドが効く)。
逆にアテンションは文脈長の2乗で増える。どちらが勝つかは実際に測るまで
分からない。**8時間を投じる前に、ここだけは実測する。**

## 何を出すか

候補の構成それぞれについて短く回して tok/s を測り、

  - 指定した時間で何トークン学習できるか
  - そのとき D/N (学習トークン ÷ 非埋め込みパラメータ) がいくつになるか
  - Chinchilla の目安 20 に対して、どの構成がいちばん近いか

を並べる。D/N が 20 から離れているほど「モデルが大きすぎて学習不足」
または「モデルが小さすぎて頭打ち」になる。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import runtime  # noqa: E402
from src.data import TokenBin, get_batch  # noqa: E402
from src.model import GPTConfig, MiniGPT  # noqa: E402
from src.tokenizer import load_tokenizer  # noqa: E402

CHINCHILLA = 20.0


@dataclass
class Candidate:
    label: str
    n_layer: int
    n_embd: int
    n_head: int
    block_size: int
    batch_size: int
    arch: str = "3lm"


# 8層512次元を中心に、前後を見る。バッチはメモリに収まる範囲で。
DEFAULT_CANDIDATES = (
    Candidate("6層 384次元 (その2と同じ形で ctx 512)", 6, 384, 6, 512, 24, "2lm"),
    Candidate("6層 512次元", 6, 512, 8, 512, 24),
    Candidate("8層 512次元 (計画の本命)", 8, 512, 8, 512, 24),
    Candidate("8層 512次元 / バッチ 32", 8, 512, 8, 512, 32),
    Candidate("10層 640次元", 10, 640, 10, 512, 16),
)


def measure(
    cand: Candidate, train_bin: TokenBin, vocab_size: int, steps: int, warmup: int
) -> dict:
    """1つの構成を steps ステップ回して tok/s を測る.

    最初の warmup ステップは mx.compile のコンパイルとメモリ確保が入るので
    計測から外す。ここを入れると短い測定ほど遅く見える。
    """
    mx.random.seed(0)
    cfg = GPTConfig(
        vocab_size=vocab_size, block_size=cand.block_size, n_layer=cand.n_layer,
        n_head=cand.n_head, n_embd=cand.n_embd, dropout=0.0, arch=cand.arch,
    )
    model = MiniGPT(cfg)
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=3e-4, weight_decay=0.1)
    loss_and_grad = nn.value_and_grad(model, lambda m, x, y: m.loss(x, y))
    state = [model.state, optimizer.state, mx.random.state]

    def _step(x, y):
        loss, grads = loss_and_grad(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        return loss

    step_fn = partial(mx.compile, inputs=state, outputs=state)(_step)
    model.train()

    tokens_per_step = cand.batch_size * cand.block_size
    started = None
    for i in range(1, steps + warmup + 1):
        x, y = get_batch(train_bin, cand.batch_size, cand.block_size, 0, i)
        step_fn(x, y)
        mx.eval(state)
        if i == warmup:
            started = time.time()
    elapsed = time.time() - started
    tps = steps * tokens_per_step / elapsed

    result = {
        "label": cand.label,
        "params": model.n_params,
        "non_embed": model.n_params_non_embedding,
        "tokens_per_step": tokens_per_step,
        "tok_per_sec": tps,
        "peak_gb": mx.get_peak_memory() / 2**30,
    }
    # 次の候補を測る前にメモリを返す。ここを忘れると、後の候補が
    # 前の候補の残したキャッシュのせいで遅く見える。
    mx.clear_cache()
    mx.reset_peak_memory()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="実測 tok/s からトークン予算を決める")
    ap.add_argument("--data", default=str(ROOT / "data" / "sft8k"))
    ap.add_argument("--hours", type=float, default=8.0, help="一晩に使える時間")
    ap.add_argument("--steps", type=int, default=40, help="計測に使うステップ数")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--vocab-size", type=int, default=0,
                    help="0 なら --data のトークナイザから読む")
    ap.add_argument("--only", default="", help="ラベルの一部で候補を絞る")
    ap.add_argument("--spec", action="append", default=[],
                    help="候補を直接指定する: 層,次元,ヘッド,文脈,バッチ[,arch] (複数指定可)")
    args = ap.parse_args()

    runtime.preflight()
    runtime.configure()

    data_dir = Path(args.data)
    train_bin = TokenBin(data_dir / "train.bin")
    vocab_size = args.vocab_size or load_tokenizer(data_dir / "tokenizer").vocab_size

    if args.spec:
        candidates = []
        for spec in args.spec:
            parts = spec.split(",")
            layer, embd, head, block, batch = (int(p) for p in parts[:5])
            arch = parts[5] if len(parts) > 5 else "3lm"
            candidates.append(Candidate(
                f"{layer}層 {embd}次元 batch{batch}", layer, embd, head, block, batch, arch
            ))
    else:
        candidates = [c for c in DEFAULT_CANDIDATES if args.only in c.label]
    # 実測に使う効率は 90% で見る。検証・保存・生成例に時間を取られるぶん。
    usable_seconds = args.hours * 3600 * 0.90

    print()
    print(f"語彙 {vocab_size:,} / 予算 {args.hours} 時間 "
          f"(実効 {usable_seconds / 3600:.2f} 時間として計算)")
    print(f"各構成を {args.warmup} ステップ暖気 + {args.steps} ステップ計測")
    print()

    rows = []
    for cand in candidates:
        print(f"  測定中: {cand.label} …", end="", flush=True)
        r = measure(cand, train_bin, vocab_size, args.steps, args.warmup)
        r["tokens_in_budget"] = int(r["tok_per_sec"] * usable_seconds)
        r["d_over_n"] = r["tokens_in_budget"] / r["non_embed"]
        r["steps_in_budget"] = r["tokens_in_budget"] // r["tokens_per_step"]
        rows.append(r)
        print(f" {r['tok_per_sec'] / 1e3:.1f}k tok/s")

    print()
    print("=" * 108)
    print(f"{'構成':<34} {'総params':>9} {'非埋め込み':>10} {'tok/s':>8} "
          f"{'ピーク':>7} {'入るトークン':>14} {'ステップ':>9} {'D/N':>6}")
    print("-" * 108)
    for r in rows:
        print(f"{r['label']:<34} {r['params'] / 1e6:>8.2f}M {r['non_embed'] / 1e6:>9.2f}M "
              f"{r['tok_per_sec'] / 1e3:>7.1f}k {r['peak_gb']:>6.1f}G "
              f"{r['tokens_in_budget']:>14,} {r['steps_in_budget']:>9,} "
              f"{r['d_over_n']:>6.1f}")
    print("=" * 108)

    out_path = ROOT / "runs" / "3lm" / "calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "hours": args.hours, "usable_seconds": usable_seconds,
            "vocab_size": vocab_size, "measured_steps": args.steps,
            "device": runtime.device_summary()["name"], "results": rows,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best = min(rows, key=lambda r: abs(r["d_over_n"] - CHINCHILLA))
    print()
    print(f"Chinchilla の目安 D/N = {CHINCHILLA:.0f} に最も近いのは:")
    print(f"  {best['label']}")
    print(f"  非埋め込み {best['non_embed'] / 1e6:.2f}M / "
          f"{best['tokens_in_budget']:,} トークン / D/N {best['d_over_n']:.1f}")
    print()
    print("この設定で走らせるには:")
    print(f"  ./scripts/train_overnight.sh --data {args.data} \\")
    print(f"      --tokens {best['tokens_in_budget']} --max-hours {args.hours + 0.5:.1f}")
    print()
    print("必要なコーパスの文字数 (データを1周で使い切る場合):")
    for r in rows:
        print(f"  {r['label']:<34} {r['tokens_in_budget']:>13,} トークン "
              f"→ 1トークン2.8文字なら約 {r['tokens_in_budget'] * 2.8 / 1e8:.1f} 億文字")


if __name__ == "__main__":
    main()
