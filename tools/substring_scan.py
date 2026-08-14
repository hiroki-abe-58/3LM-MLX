"""巨大なテキストから「複数の長い文字列」を1回のスキャンで探す.

なぜ自前で書くか。macOS 標準の grep は BSD 版で、`-F -f パターンファイル`
に数百本を渡すと極端に遅くなる。3.4GB のコーパスに 668 本を投げたら
17分で終わらなかった。GNU grep や ripgrep を入れれば速いが、
検査スクリプトが「先に別のツールを入れてください」になるのは避けたい。

## やっていること (Rabin-Karp の変種)

長さ W バイトの窓それぞれについて、2つの数を出す。

    A[i] = Σ c[i+k]                (窓の中のバイトの和)
    B[i] = Σ c[i+k] × (k+1)        (位置で重み付けした和)

A だけだと「同じバイトを並べ替えただけ」を区別できない。B を足すと
順番の情報が入る。この2つはどちらも **累積和の差** で書けるので、
numpy で窓の数だけ一気に計算できる。1バイトずつ Python で回すのと違って、
3.4GB でも数分で終わる。

    A[i] = S[i+W] - S[i]                    S = cumsum(c)
    B[i] = (T[i+W] - T[i]) - (i-1) × A[i]   T = cumsum(c × 位置)

B の式は Σ c[j]×(j-i+1) = Σ c[j]×j - (i-1)×Σ c[j] から出る。

## 誤検出の扱い

(A, B) が一致しても、同じ文字列とは限らない。だから一致した位置だけを
取り出して、実際のバイト列を突き合わせる。候補は全体のごく一部なので、
ここが Python でも問題にならない。**取りこぼしは無い**
(同じ文字列なら A も B も必ず一致するので、偽陰性は原理的に起きない)。

## 窓をまたぐ一致

チャンクに切って読むので、境目に跨がる一致を落とさないよう
W-1 バイトを重ねて読む。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

# 署名を取る窓の長さ (バイト)。日本語 UTF-8 で1文字3バイトなので、
# 96バイトは約32文字にあたる。長いほど誤検出が減る。
WINDOW_BYTES = 96

# 一度に読むバイト数。累積和を uint64 で2本持つので、
# ここを大きくすると 16 倍のメモリを使う (16MB → 256MB)。
CHUNK_BYTES = 16 << 20


def _signature(buf: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """buf の全ての window バイト窓について (A, B) を返す."""
    c = buf.astype(np.uint64)
    idx = np.arange(1, c.size + 1, dtype=np.uint64)

    s = np.zeros(c.size + 1, dtype=np.uint64)
    np.cumsum(c, out=s[1:])
    t = np.zeros(c.size + 1, dtype=np.uint64)
    np.cumsum(c * idx, out=t[1:])

    n = c.size - window + 1
    starts = np.arange(n, dtype=np.uint64)
    a = s[window:] - s[:n]
    b = (t[window:] - t[:n]) - starts * a
    return a, b


def _probe_signature(probe: bytes, window: int) -> tuple[int, int]:
    head = np.frombuffer(probe[:window], dtype=np.uint8)
    a, b = _signature(head, window)
    return int(a[0]), int(b[0])


def _iter_chunks(path: Path, overlap: int, chunk: int) -> Iterator[tuple[int, bytes]]:
    """(ファイル先頭からのオフセット, バイト列) を重ねながら返す."""
    with path.open("rb") as fh:
        offset = 0
        tail = b""
        while True:
            block = fh.read(chunk)
            if not block:
                break
            buf = tail + block
            yield offset - len(tail), buf
            tail = buf[-overlap:] if overlap else b""
            offset += len(block)


def scan(
    path: Path,
    probes: list[str],
    window: int = WINDOW_BYTES,
    chunk: int = CHUNK_BYTES,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """path の中に現れた probe とその出現回数を返す.

    probe は window バイト以上の長さが必要 (短いと署名が取れない)。
    """
    encoded = [p.encode("utf-8") for p in probes]
    usable = [(p, e) for p, e in zip(probes, encoded, strict=True) if len(e) >= window]
    if not usable:
        raise ValueError(f"{window} バイト以上の断片がありません")

    # 署名 -> その署名を持つ (元の文字列, バイト列) の一覧。
    table: dict[tuple[int, int], list[tuple[str, bytes]]] = {}
    for text, raw in usable:
        table.setdefault(_probe_signature(raw, window), []).append((text, raw))

    keys = np.array(sorted(table), dtype=np.uint64)  # shape (n, 2)
    key_a, key_b = keys[:, 0], keys[:, 1]

    hits: dict[str, int] = {}
    total = path.stat().st_size
    seen_positions: set[int] = set()

    for base, buf in _iter_chunks(path, window - 1, chunk):
        arr = np.frombuffer(buf, dtype=np.uint8)
        if arr.size < window:
            continue
        a, b = _signature(arr, window)

        # A が候補に無い窓を先に落とす。ここで大半が消える。
        maybe = np.isin(a, key_a)
        if not maybe.any():
            if progress:
                progress(base + len(buf), total)
            continue
        cand = np.flatnonzero(maybe)
        # 残りを B でも絞る
        cand = cand[np.isin(b[cand], key_b)]

        for pos in cand.tolist():
            bucket = table.get((int(a[pos]), int(b[pos])))
            if not bucket:
                continue  # A と B が別々の断片と一致しただけ
            for text, raw in bucket:
                if buf[pos : pos + len(raw)] == raw:
                    absolute = base + pos
                    if absolute in seen_positions:
                        continue  # 重ねて読んだ部分の二重計上を防ぐ
                    seen_positions.add(absolute)
                    hits[text] = hits.get(text, 0) + 1
        if progress:
            progress(base + len(buf), total)

    return hits
