"""固定検証セットが学習コーパスに混ざっていないか調べる.

    python3 tools/check_leak.py --corpus data/corpus_pretrain.txt

なぜ必要か。eval/holdout.txt で測る bits/char は「モデルが見たことのない
文章をどれだけ予測できるか」の数字です。もし holdout の文章が学習コーパスに
入っていたら、それは記憶を測っているだけで、数字が良く出ても意味がありません。

対話コーパス側は data/prepare_sft.py が --exclude で行ごとに除いています。
問題は事前学習コーパスです。こちらは FineWeb2 (Common Crawl) なので、
「ウェブのどこかに同じ文章が転載されていた」という混入があり得ます。
青空文庫も同様で、対話データの中に引用があれば重なります。

やり方は素朴です。holdout の各行から特徴的な断片を抜き、それが
コーパス本文に現れるかを探します。断片は長め (既定48文字) に取ります。
日本語で48文字が一致するのは、偶然ではなく同じ文章です。

探索は tools/substring_scan.py に置いた Rabin-Karp の変種で行います。
macOS 標準の grep では 3.4GB × 数百パターンが終わらなかったためです
(詳しい理由はそちらのファイルの冒頭に書いてあります)。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from substring_scan import WINDOW_BYTES, scan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# 対話コーパスの話者マーカー。断片を取る前に外す。
MARKERS = re.compile(r"<\|(?:user|assistant|end|endoftext)\|>")


def probes_from_line(line: str, length: int, per_line: int) -> list[str]:
    """1行から特徴的な断片を per_line 本取り出す.

    行の先頭・中央・末尾から取る。先頭だけだと「よくある挨拶」に当たって
    偽陽性が出やすく、末尾だけだと定型の締め文に当たりやすい。
    """
    text = MARKERS.sub("", line).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) < length:
        return []
    spans = []
    usable = len(text) - length
    for i in range(per_line):
        start = 0 if per_line == 1 else round(usable * i / (per_line - 1))
        end = start + length
        # 走査器は先頭 WINDOW_BYTES バイトで署名を取るので、そこに届かない
        # 断片は使えない。日本語なら48文字=144バイトで足りるが、
        # ASCII 主体の行だと48文字=48バイトしかないので伸ばす。
        while end < len(text) and len(text[start:end].encode("utf-8")) < WINDOW_BYTES:
            end += 1
        piece = text[start:end]
        if len(piece.encode("utf-8")) >= WINDOW_BYTES:
            spans.append(piece)
    return spans


def build_probes(holdout: Path, length: int, per_line: int) -> dict[str, int]:
    """断片 -> 由来した holdout の行番号."""
    probes: dict[str, int] = {}
    for lineno, line in enumerate(holdout.read_text(encoding="utf-8").splitlines(), 1):
        for piece in probes_from_line(line, length, per_line):
            probes.setdefault(piece, lineno)
    return probes


def scan_probes(corpus: Path, probes: dict[str, int]) -> list[tuple[str, int]]:
    """コーパスに現れた断片を、多い順に返す."""
    started = time.time()

    def show(done: int, total: int) -> None:
        gb = done / 2**30
        rate = gb / max(time.time() - started, 1e-6)
        print(f"\r    {done / max(total, 1) * 100:5.1f}%  {gb:.2f} GiB  "
              f"{rate * 1024:.0f} MiB/秒", end="", flush=True)

    hits = scan(corpus, list(probes), progress=show)
    print()
    return sorted(hits.items(), key=lambda kv: -kv[1])


def self_test() -> None:
    """既知の答えで走査器を確かめる.

    「混入なし」という結果は、走査器が壊れていても同じように出る。
    出方が同じなので、検査そのものが働いていることを別に示す必要がある。
    わざと混入させた文章を見つけられるか、混入していない文章を
    誤って報告しないかを、この場で確認する。
    """
    import tempfile

    needle_a = "この文章は検査器を確かめるために置いた目印です。" * 4
    needle_b = "こちらはコーパスに入れていない文章なので見つかってはいけません。" * 4
    filler = "".join(f"{i}番目の埋め草の文章です。日本語の文字を並べています。" for i in range(4000))

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "corpus.txt"
        # 目印を先頭・中間・末尾に置く。チャンクの境目でも拾えることを見る。
        body = filler + needle_a + filler + needle_a + filler
        target.write_text(body + "\n", encoding="utf-8")

        # チャンクをわざと小さくして、境目をまたぐ経路を通す
        hits = scan(target, [needle_a, needle_b], chunk=8192)

        problems = []
        if hits.get(needle_a) != 2:
            problems.append(f"混入した文章の検出数が {hits.get(needle_a)} で、期待した 2 と違う")
        if needle_b in hits:
            problems.append("混入していない文章を誤って報告した")
        if problems:
            for p in problems:
                print(f"  [自己検査 失敗] {p}")
            raise SystemExit("走査器が期待どおりに動いていません。検査を中止します。")
        print(f"  [自己検査] 合格 (混入 2 件を検出 / 誤検出 0 件 / 断片 {len(needle_a)} 文字)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", required=True,
                    help="調べるコーパス. 複数指定可")
    ap.add_argument("--holdout", default=str(ROOT / "eval" / "holdout.txt"))
    ap.add_argument("--length", type=int, default=48, help="断片の長さ (文字)")
    ap.add_argument("--skip-self-test", action="store_true",
                    help="走査器の自己検査を省く (普段は省かない)")
    ap.add_argument("--per-line", type=int, default=3, help="1行から取る断片の本数")
    ap.add_argument("--out", default=str(ROOT / "runs" / "3lm" / "leak_check.json"))
    args = ap.parse_args()

    holdout = Path(args.holdout)
    if not holdout.exists():
        raise SystemExit(f"{holdout} がありません。eval/run.py --make-holdout で作ってください。")

    probes = build_probes(holdout, args.length, args.per_line)
    n_lines = len(holdout.read_text(encoding="utf-8").splitlines())

    print("=" * 66)
    print("  固定検証セットの混入検査")
    print(f"    holdout   : {holdout} ({n_lines} 行)")
    print(f"    断片      : {len(probes):,} 本 x {args.length} 文字")
    print("=" * 66)
    if not args.skip_self_test:
        self_test()

    report = {"holdout": str(holdout.relative_to(ROOT)), "holdout_lines": n_lines,
              "probe_chars": args.length, "probes": len(probes), "corpora": []}
    leaked = False

    for name in args.corpus:
        corpus = Path(name)
        if not corpus.exists():
            print(f"  [飛ばす] {corpus} がありません")
            continue
        size_gb = corpus.stat().st_size / 2**30
        print(f"\n  走査中: {corpus} ({size_gb:.2f} GiB)")
        hits = scan_probes(corpus, probes)
        affected = sorted({probes[p] for p, _ in hits})
        entry = {
            "corpus": corpus.name, "bytes": corpus.stat().st_size,
            "hit_probes": len(hits), "affected_holdout_lines": affected,
            "examples": [{"probe": p, "count": c} for p, c in hits[:5]],
        }
        report["corpora"].append(entry)

        if not hits:
            print(f"    混入なし ({len(probes):,} 本すべて不一致)")
        else:
            leaked = True
            print(f"    混入の疑い: 断片 {len(hits)} 本 / holdout {len(affected)} 行")
            for probe, count in hits[:5]:
                print(f"      行 {probes[probe]:3d} x{count}: {probe[:40]}…")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report["leaked"] = leaked
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    if leaked:
        print("  混入が見つかりました。該当行を holdout から外すか、")
        print("  コーパス側から除いてから評価してください。")
    else:
        print("  すべてのコーパスで混入なし。bits/char は素の予測性能として読めます。")
    print(f"  → {out}")
    print("=" * 66)
    sys.exit(1 if leaked else 0)


if __name__ == "__main__":
    main()
