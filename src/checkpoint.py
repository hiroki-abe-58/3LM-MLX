"""落ちても失われないチェックポイント.

## その2 までの保存は、運が悪いと全部消える

    def save_checkpoint(path, model, tokenizer):
        tmp.mkdir()
        ...書き込み...
        if path.exists():
            shutil.rmtree(path)   # ← 古い世代を消す
        tmp.rename(path)          # ← ここまでの間に落ちると何も残らない

rmtree と rename の間には数十msの隙間がある。27分の学習なら
「まあ引き直せばいい」で済むが、8時間走らせて7時間目にここを踏むと
一晩が消える。実際に踏む確率は低いが、踏んだときの損失が大きい。

## 世代を2つ持って交互に書く

    checkpoints/pretrain/
    ├── CURRENT     # "ckpt-A" か "ckpt-B" の1行。os.replace で原子的に差し替える
    ├── ckpt-A/
    │   └── COMPLETE  # 全部書き終わってから最後に置く
    └── ckpt-B/

書くのは常に「今 CURRENT でない方」。書いている最中に落ちても、
CURRENT はまだ前の世代を指しているので、前の世代から再開できる。
最悪でも「保存間隔ぶんの学習」しか失わない。

COMPLETE マーカーは、書き込みが途中で終わったディレクトリを見分けるため。
safetensors が中途半端な長さで残っていると、読み込みは例外ではなく
「壊れたテンソル」で通ってしまうことがある。ファイルの存在チェックでは
足りないので、最後に置くマーカーで判定する。

## 何を保存すれば再開できるか

重みだけでは足りない。AdamW は各パラメータについて1次・2次モーメント
(m, v) を持っていて、これを捨てると再開直後に実効学習率が跳ね、
loss が目に見えて悪化する。

さらに step も要る。step から学習率スケジュールの現在地が決まり、
データの切り出し位置も決まる (src/data.py 参照)。

MLX の optimizer.state は step / learning_rate / 各パラメータの m,v を
含む入れ子の dict なので、mx.utils.tree_flatten で平らにすれば
safetensors にそのまま入る。復元は tree_unflatten で戻す。
step を戻せば **学習率スケジュールも自動的に元の位置に戻る**
(MLX のスケジュールは optimizer.state["step"] を引数に取るため)。
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

CURRENT_NAME = "CURRENT"
COMPLETE_NAME = "COMPLETE"
SLOTS = ("ckpt-A", "ckpt-B")
STATE_NAME = "train_state.json"


@dataclass
class TrainState:
    """再開に必要な、モデル・オプティマイザ以外の情報."""

    step: int = 0
    tokens_seen: int = 0
    best_val: float = float("inf")
    seed: int = 1234
    elapsed_sec: float = 0.0
    resumes: int = 0
    data_meta: dict[str, Any] = field(default_factory=dict)
    args: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_path(cls, path: Path) -> TrainState:
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def _flatten_opt_state(state: dict) -> dict[str, mx.array]:
    """optimizer.state を safetensors に入れられる平らな dict にする.

    step や learning_rate はスカラーの mx.array なので、そのまま入る。
    mx.array でない値 (あれば) は落とす。
    """
    flat = {}
    for key, value in tree_flatten(state):
        if isinstance(value, mx.array):
            flat[key] = value
    return flat


def save(
    root: Path,
    model,
    optimizer,
    tokenizer,
    state: TrainState,
) -> Path:
    """CURRENT でない側のスロットへ書き、最後に CURRENT を差し替える."""
    root.mkdir(parents=True, exist_ok=True)
    current = read_current(root)
    slot = SLOTS[1] if current == SLOTS[0] else SLOTS[0]
    target = root / slot

    # 書き始める前に、前回この枠に書いたものを消す。COMPLETE も一緒に消える。
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    model.save_weights(str(target / "model.safetensors"))
    mx.save_safetensors(str(target / "optimizer.safetensors"), _flatten_opt_state(optimizer.state))
    model.cfg.save(target / "config.json")
    tokenizer.save(target)
    (target / STATE_NAME).write_text(state.to_json(), encoding="utf-8")

    # ここまで無事に書けたことを示す。以降のクラッシュでは前世代へ落ちる。
    (target / COMPLETE_NAME).write_text(
        f"{time.time():.0f}\nstep={state.step}\n", encoding="utf-8"
    )
    write_current(root, slot)
    return target


def read_current(root: Path) -> str | None:
    path = root / CURRENT_NAME
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value if value in SLOTS else None


def write_current(root: Path, slot: str) -> None:
    """CURRENT を原子的に差し替える.

    同じファイルシステム上の rename は POSIX で原子的なので、
    「古い内容」か「新しい内容」のどちらかしか観測されない。
    直接 write_text すると、切り詰めた直後に落ちて空ファイルになりうる。
    """
    tmp = root / (CURRENT_NAME + ".tmp")
    tmp.write_text(slot + "\n", encoding="utf-8")
    os.replace(tmp, root / CURRENT_NAME)


def _is_complete(path: Path) -> bool:
    return (path / COMPLETE_NAME).exists() and (path / "model.safetensors").exists()


def find_resumable(root: Path) -> Path | None:
    """再開できるチェックポイントを探す.

    CURRENT の指す先を最優先。壊れていれば、もう一方に落ちる。
    どちらも駄目なら None (最初から学習する)。
    """
    if not root.exists():
        return None
    current = read_current(root)
    order = [current, *[s for s in SLOTS if s != current]] if current else list(SLOTS)
    for slot in order:
        if slot and _is_complete(root / slot):
            return root / slot
    return None


def restore(path: Path, model, optimizer) -> TrainState:
    """チェックポイントからモデル・オプティマイザ・進捗を戻す.

    optimizer.state を戻す前に optimizer.init(model.trainable_parameters()) で
    形を作っておく必要がある。空の state に tree_unflatten した dict を
    代入しても、MLX 側が持つ入れ子の形と合わないことがある。
    """
    model.load_weights(str(path / "model.safetensors"))
    state = TrainState.from_path(path / STATE_NAME)

    opt_path = path / "optimizer.safetensors"
    if opt_path.exists():
        optimizer.init(model.trainable_parameters())
        loaded = mx.load(str(opt_path))
        restored = tree_unflatten(list(loaded.items()))
        optimizer.state = _merge(optimizer.state, restored)
    mx.eval(model.parameters(), optimizer.state)
    return state


def _merge(base: Any, incoming: Any) -> Any:
    """optimizer.state に、保存してあった値を上書きしていく.

    丸ごと差し替えないのは、保存できなかったキー (mx.array でない値) を
    init が作った既定値のまま残したいから。
    """
    if isinstance(base, dict) and isinstance(incoming, dict):
        merged = dict(base)
        for key, value in incoming.items():
            merged[key] = _merge(base.get(key), value) if key in base else value
        return merged
    if isinstance(base, list) and isinstance(incoming, list) and len(base) == len(incoming):
        return [_merge(b, i) for b, i in zip(base, incoming, strict=True)]
    return incoming


def export(target: Path, source: Path, extra: dict[str, Any] | None = None) -> None:
    """配布用に、推論で必要なファイルだけを別ディレクトリへ出す.

    optimizer.safetensors は重みと同じくらい大きいので入れない。
    """
    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for name in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer.model"):
        src = source / name
        if src.exists():
            shutil.copy2(src, tmp / name)
    if extra:
        (tmp / "metrics.json").write_text(
            json.dumps(extra, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    # export は「消えても学習には影響しない」ので、素直に差し替えてよい。
    if target.exists():
        shutil.rmtree(target)
    tmp.rename(target)
