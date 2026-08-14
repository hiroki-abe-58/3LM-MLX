"""学習データの読み出しと、バッチの決定的な作り方.

## memmap で読む理由

train.bin は 5億トークン × 2 byte = 約1GB ある。np.load で読めば1GBの
配列がメモリに載るが、この先チェックポイントとモデルとオプティマイザ状態が
同じユニファイドメモリを取り合うので、削れるものは削っておく。
np.memmap ならページキャッシュ経由になり、実際に触った範囲しか常駐しない。

## バッチをステップの関数にする理由

一晩走らせる学習は必ずどこかで落ちる。落ちた後に「同じ続き」から
再開できないと、再開のたびに同じデータを二度見たり、まだ見ていない
データを飛ばしたりする。学習曲線に段差が出て、何が原因か分からなくなる。

素直な実装は np.random.Generator を1個持って回すことだが、その状態を
保存するのは面倒で、しかも巨大になる (MT19937 なら 624 word)。

そこで乱数生成器の状態を持ち歩くのをやめて、
**バッチの内容を (seed, step) だけから決まる純粋関数にする**。

    rng = np.random.default_rng([seed, step])

default_rng は整数のリストを SeedSequence のエントロピーとして受け、
そこから決定的に初期化する。step ごとに独立な生成器を作り捨てにすれば、
保存すべき状態は step 1個の整数だけになる。再開したら
step から作り直すので、バッチは1トークンも違わない。

作り捨てのコストは、実測で1ステップ 0.05ms 程度。1ステップの学習が
数十msなので無視できる。
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np


class TokenBin:
    """uint16 のトークン列を memmap で読む."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise SystemExit(
                f"{self.path} がありません。data/encode.py でコーパスを変換してください。"
            )
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return len(self.data)

    @property
    def tokens(self) -> int:
        return len(self.data)


def load_meta(bin_dir: str | Path) -> dict:
    path = Path(bin_dir) / "meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_batch(
    bin_data: TokenBin,
    batch_size: int,
    block_size: int,
    seed: int,
    step: int,
) -> tuple[mx.array, mx.array]:
    """(seed, step) から決まるバッチを1つ作る.

    y は x を1トークンずらしたもの。「次のトークンを当てる」課題なので、
    位置 i の入力に対する正解は位置 i+1 のトークンになる。

    切り出しは復元ありのランダム位置。文書の途中から始まるバッチも混ざるが、
    事前学習ではそれで構わない (むしろ文脈の途中から続きを書く練習になる)。
    """
    rng = np.random.default_rng([seed, step])
    high = len(bin_data) - block_size - 1
    if high <= 0:
        raise SystemExit(
            f"{bin_data.path.name} が短すぎます "
            f"({len(bin_data):,} トークン < block_size {block_size})"
        )
    ix = rng.integers(0, high, size=batch_size)
    # memmap から必要な窓だけを取り出す。int32 にするのは MLX の埋め込みが
    # 整数インデックスを取るため。uint16 のまま渡すと型が合わない。
    xs = np.stack([bin_data.data[i : i + block_size] for i in ix]).astype(np.int32)
    ys = np.stack([bin_data.data[i + 1 : i + 1 + block_size] for i in ix]).astype(np.int32)
    return mx.array(xs), mx.array(ys)


def iter_eval_batches(
    bin_data: TokenBin,
    batch_size: int,
    block_size: int,
    n_batches: int,
    seed: int = 0,
):
    """検証用のバッチ列. step の代わりに 0..n を使うので、いつ呼んでも同じ.

    毎回同じ検証バッチにしないと、val_loss の上下がデータのばらつきか
    学習の進みかを区別できなくなる。
    """
    for i in range(n_batches):
        yield get_batch(bin_data, batch_size, block_size, seed, i)
