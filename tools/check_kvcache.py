"""KVキャッシュが「毎回作り直した場合」と同じ結果になるか確かめる.

    python3 tools/check_kvcache.py

## なぜ確認するのか

KVキャッシュは速くするための仕組みなので、**間違っていても動く**。
出てくる文章が少し変になるだけで、例外も出ない。しかもモデルが小さいうちは
「まだ賢くないだけ」と区別がつかない。何時間も学習したあとに
「生成がおかしい」と気づいて、原因がキャッシュだと分かるのが最悪の順序。

間違え方は決まっている。

  RoPE の位置がずれる
      キャッシュを使うと、渡すトークンは常に1個になる。位置を 0 として
      回転させてしまうと、2トークン目以降が全部同じ位置だと思われる。
      offset を渡し忘れるとこうなる。
  マスクが合わない
      キャッシュありでは query が1個・key が n 個の長方形になる。
      正方形の因果マスクを当てると形が合わないか、未来を隠しすぎる。
  学習した位置埋め込み (2lm) のずれ
      RoPE ではなく学習した位置埋め込みを使う構成では、
      キャッシュ利用時に pos_emb の添字を offset ぶんずらす必要がある。

どれも「1トークンずつ入れた結果」と「全系列を一度に入れた結果」を
突き合わせれば分かる。因果的なモデルなら最後の位置の logits は一致するはず。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import GPTConfig, MiniGPT  # noqa: E402

# GPU の加算順序で下位ビットは揺れる。桁落ちしない範囲の許容値。
TOLERANCE = 2e-4


def compare(arch: str, n_steps: int, seed: int) -> tuple[float, float]:
    """(キャッシュあり vs 一括, 最終位置の最大差) を返す."""
    mx.random.seed(seed)
    cfg = GPTConfig(
        vocab_size=512, block_size=64, n_layer=3, n_head=4, n_embd=64,
        dropout=0.0, arch=arch,
    )
    model = MiniGPT(cfg)
    model.eval()
    mx.eval(model.parameters())

    prompt = mx.random.randint(0, cfg.vocab_size, (1, 5))

    # (A) キャッシュを使って1トークンずつ進める
    caches = model.make_caches()
    logits = model(prompt, caches=caches)
    ids = prompt
    cached_last = [logits[:, -1, :]]
    for _ in range(n_steps):
        nxt = mx.argmax(cached_last[-1], axis=-1, keepdims=True)
        ids = mx.concatenate([ids, nxt], axis=1)
        logits = model(nxt, caches=caches)
        cached_last.append(logits[:, -1, :])
    mx.eval(ids, *cached_last)

    # (B) 同じ系列を、毎回まるごと入れ直す
    full_last = []
    for length in range(prompt.shape[1], ids.shape[1] + 1):
        out = model(ids[:, :length])
        full_last.append(out[:, -1, :])
    mx.eval(*full_last)

    diffs = [
        float(mx.max(mx.abs(a - b)).item())
        for a, b in zip(cached_last, full_last, strict=True)
    ]
    return max(diffs), sum(diffs) / len(diffs)


def check_overflow(arch: str) -> float:
    """文脈長を超えて生成したときに落ちないか / 値が壊れないか.

    src/generate.py は文脈が溢れたら古いトークンを捨ててキャッシュを
    作り直す。その経路を通す。
    """
    mx.random.seed(7)
    cfg = GPTConfig(
        vocab_size=512, block_size=16, n_layer=2, n_head=4, n_embd=64,
        dropout=0.0, arch=arch,
    )
    model = MiniGPT(cfg)
    model.eval()
    mx.eval(model.parameters())

    from src.generate import generate_stream

    prompt = mx.random.randint(0, cfg.vocab_size, (1, 12)).reshape(-1).tolist()
    produced = list(generate_stream(model, prompt, max_new_tokens=40, temperature=0.0))
    return len(produced)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    print("=" * 70)
    print("  KVキャッシュの一致検査")
    print(f"    許容値 {TOLERANCE:.0e} (GPU の加算順序で下位ビットは揺れる)")
    print("=" * 70)

    failed = False
    for arch in ("3lm", "2lm"):
        worst, mean = compare(arch, args.steps, args.seed)
        status = "一致" if worst <= TOLERANCE else "不一致"
        note = "RoPE" if arch == "3lm" else "学習した位置埋め込み"
        print(f"  [{status}] {arch} ({note})  最大差 {worst:.2e} / 平均 {mean:.2e}")
        if worst > TOLERANCE:
            failed = True

    print()
    for arch in ("3lm", "2lm"):
        made = check_overflow(arch)
        ok = made == 40
        print(f"  [{'合格' if ok else '失敗'}] {arch} 文脈長を超えて生成: "
              f"{made} トークン (期待 40)")
        if not ok:
            failed = True

    print("=" * 70)
    if failed:
        raise SystemExit("KVキャッシュが一括計算と一致しません。生成が壊れます。")
    print("  1トークンずつ進めても、まるごと入れ直したのと同じ logits が出ている。")


if __name__ == "__main__":
    main()
