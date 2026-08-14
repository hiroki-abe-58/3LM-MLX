"""ミニGPT本体 (MLX実装).

やっていることは1つだけ: これまでのトークン列から「次の1トークン」の
確率分布を出す。会話が成立するのは、この予測を繰り返しているだけである。

## 構成を2つ持たせてある

    arch="2lm"  学習された位置埋め込み / LayerNorm / GELU の MLP (4倍)
    arch="3lm"  RoPE / RMSNorm / SwiGLU (1408) / bias なし

その2 の 13.81M モデルは "2lm" で、そのまま読み直せる。これを残すのは
懐古趣味ではなく、**「データ量だけを変えた比較」を成立させるため**。
アーキテクチャも一緒に変えてしまうと、乖離点が動いた理由が
データなのか構成なのか分からなくなる。

## 3lm 側で変えた4点と、その理由

RoPE (回転位置埋め込み)
    位置情報を埋め込みの足し算ではなく、q と k の回転で与える。
    学習された位置埋め込みは block_size ぶんの表を持つので、
    文脈を 256 → 512 にすると表も倍になり、しかも
    「学習中にあまり出てこなかった後ろの位置」の行が育たない。
    RoPE は表を持たないので位置ごとの学習量の偏りが出ない。

RMSNorm
    LayerNorm から平均の引き算と bias を落としたもの。
    パラメータが半分で、実測でわずかに速い。精度はほぼ変わらない。

SwiGLU
    GELU の MLP は fc → 活性化 → proj の2行列。SwiGLU は
    gate と up の2本を掛け合わせてから down に通す3行列で、
    同じパラメータ数なら性能が出やすいことが知られている。
    3行列にするぶん、中間次元は 4倍 (2048) ではなく
    **8/3倍あたり (1408)** に落として総数を揃える。
    1408 は 512 × 8/3 = 1365 を 128 の倍数に丸めた値。
    行列の幅を 128 の倍数に揃えると GPU のタイル割りが素直になる。

bias なし
    Transformer の Linear から bias を落としても性能はほぼ変わらない、
    というのは近年のモデルでほぼ共通の選択。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    arch: str = "3lm"
    # SwiGLU の中間次元。None なら n_embd × 8/3 を128の倍数に丸めた値。
    mlp_hidden: int | None = None
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.arch not in ("2lm", "3lm"):
            raise ValueError(f"知らない arch: {self.arch}")
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd {self.n_embd} が n_head {self.n_head} で割れません")
        if self.mlp_hidden is None and self.arch == "3lm":
            self.mlp_hidden = round(self.n_embd * 8 / 3 / 128) * 128

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> GPTConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # その2 の config.json には arch が無い。無ければ 2lm とみなす。
        data.setdefault("arch", "2lm")
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class KVCache:
    """生成時に、過去のキーと値を取り置きしておく箱.

    キャッシュが無いと、1トークン出すたびに文脈全体を全層通し直すことになる。
    文脈512で200トークン生成すると、のべ 512×200 = 10万トークンぶんの
    前向き計算を回すので、体感で使えない速さになる。

    キャッシュがあれば、毎回計算するのは新しい1トークンぶんだけになる。
    増えていくのは「過去の k と v を持っておくメモリ」で、
    これは 2 × 層数 × ヘッド数 × head_dim × 長さ × 4byte。
    このモデルなら 512トークンで約17MB。安い買い物である。
    """

    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None

    @property
    def offset(self) -> int:
        return 0 if self.keys is None else self.keys.shape[2]

    def update(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            # 毎回の連結は確保し直しになるが、200トークン規模なら
            # 前向き計算を省ける分の方がはるかに大きい。
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        return self.keys, self.values


def _attention_mask(n_query: int, n_key: int) -> mx.array | str | None:
    """クエリ n_query 本が、キー n_key 本のどこまで見てよいかを表すマスク.

    キャッシュを使うと「クエリ1本・キー300本」のような形になる。
    このときクエリは全部のキーを見てよいのでマスクは不要。

    クエリが複数あって、なおかつキャッシュがある場合 (途中まで生成した続きを
    まとめて流すとき) は、クエリがキー列の末尾 n_query 本に対応する、
    という前提で下三角を作る。mask="causal" は正方行列を前提にしているので
    ここでは使えない。
    """
    if n_query == 1:
        return None
    if n_query == n_key:
        return "causal"
    offset = n_key - n_query
    q_pos = mx.arange(offset, n_key)[:, None]
    k_pos = mx.arange(n_key)[None, :]
    return q_pos >= k_pos


class CausalSelfAttention(nn.Module):
    """因果マスク付きの自己注意.

    「未来のトークンを見てはいけない」という制約 (causal mask) が
    言語モデルの心臓部。ここを外すとカンニングになり、
    学習損失は下がるのに生成は破綻する。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim**-0.5
        self.use_rope = cfg.arch == "3lm"
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def __call__(self, x: mx.array, cache: KVCache | None = None) -> mx.array:
        B, T, C = x.shape
        offset = cache.offset if cache is not None else 0
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        # (B, T, C) -> (B, n_head, T, head_dim)
        shape = (B, T, self.n_head, self.head_dim)
        q = q.reshape(shape).transpose(0, 2, 1, 3)
        k = k.reshape(shape).transpose(0, 2, 1, 3)
        v = v.reshape(shape).transpose(0, 2, 1, 3)

        if self.use_rope:
            # offset を渡すのが要点。キャッシュを使った生成では、
            # 新しいトークンの絶対位置は 0 ではなく offset から始まる。
            # ここを 0 のままにすると、生成中ずっと「文頭のつもり」で回る。
            q = mx.fast.rope(
                q, self.head_dim, traditional=False, base=self.cfg.rope_theta,
                scale=1.0, offset=offset,
            )
            k = mx.fast.rope(
                k, self.head_dim, traditional=False, base=self.cfg.rope_theta,
                scale=1.0, offset=offset,
            )

        if cache is not None:
            k, v = cache.update(k, v)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=_attention_mask(T, k.shape[2])
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.drop(self.proj(out))


class MLP(nn.Module):
    """arch="2lm" 用. fc → GELU → proj の素直な2行列."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def __call__(self, x: mx.array) -> mx.array:
        return self.drop(self.proj(nn.gelu(self.fc(x))))


class SwiGLU(nn.Module):
    """arch="3lm" 用. gate を SiLU に通して up と掛け、down に通す."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = cfg.mlp_hidden
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def __call__(self, x: mx.array) -> mx.array:
        return self.drop(self.down(nn.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    """Pre-LN + 残差接続. この形が深くしても学習が壊れにくい."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.arch == "3lm":
            self.ln1 = nn.RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
            self.ln2 = nn.RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
            self.mlp = SwiGLU(cfg)
        else:
            self.ln1 = nn.LayerNorm(cfg.n_embd)
            self.ln2 = nn.LayerNorm(cfg.n_embd)
            self.mlp = MLP(cfg)
        self.attn = CausalSelfAttention(cfg)

    def __call__(self, x: mx.array, cache: KVCache | None = None) -> mx.array:
        x = x + self.attn(self.ln1(x), cache)
        return x + self.mlp(self.ln2(x))


class MiniGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # 3lm は RoPE を使うので位置埋め込みの表を持たない。
        if cfg.arch == "2lm":
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layer)]
        self.ln_f = (
            nn.RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
            if cfg.arch == "3lm"
            else nn.LayerNorm(cfg.n_embd)
        )
        # 出力層は埋め込み行列を転用する (weight tying)。
        # 語彙32,000×次元512ぶんのパラメータを節約でき、小さいモデルでは効きが良い。

    def __call__(self, idx: mx.array, caches: list[KVCache] | None = None) -> mx.array:
        x = self.tok_emb(idx)
        if self.cfg.arch == "2lm":
            offset = caches[0].offset if caches else 0
            x = x + self.pos_emb(mx.arange(offset, offset + idx.shape[1]))
        x = self.drop(x)
        for i, block in enumerate(self.blocks):
            x = block(x, caches[i] if caches else None)
        return self.tok_emb.as_linear(self.ln_f(x))

    def make_caches(self) -> list[KVCache]:
        return [KVCache() for _ in range(self.cfg.n_layer)]

    def loss(
        self, idx: mx.array, targets: mx.array, weights: mx.array | None = None
    ) -> mx.array:
        """次トークン予測の交差エントロピー.

        weights は「その位置の損失を数えるか」を 0/1 で指定するもの。
        SFT で「アシスタントの返答だけを学習させる」ときに使う
        (instruction masking)。None なら全位置を等しく数える。
        """
        logits = self(idx)
        flat = nn.losses.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1), reduction="none"
        )
        if weights is None:
            return flat.mean()
        w = weights.reshape(-1)
        # 分母を「数えた位置の数」にする。ここを全位置数にすると、
        # マスクの割合が変わるだけで損失の絶対値が動き、比較できなくなる。
        return (flat * w).sum() / mx.maximum(w.sum(), 1.0)

    @property
    def n_params(self) -> int:
        from mlx.utils import tree_flatten

        return sum(p.size for _, p in tree_flatten(self.parameters()))

    @property
    def n_params_non_embedding(self) -> int:
        """埋め込みを除いたパラメータ数.

        Chinchilla の「学習トークン数 ≈ 20 × パラメータ数」で使うのはこちら。
        埋め込みは語彙の大きさで決まるだけで、計算量にほとんど寄らない。
        """
        from mlx.utils import tree_flatten

        return sum(
            p.size
            for name, p in tree_flatten(self.parameters())
            if not name.startswith(("tok_emb", "pos_emb"))
        )
