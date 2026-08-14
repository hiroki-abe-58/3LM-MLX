"""混入していた行を除いた検証セットを作る.

    python3 tools/make_clean_holdout.py

## なぜ必要か

その2 の 2.584 bits/char は、**混入込みの数字**である。
`eval/holdout.txt` 249行のうち 29行が、近似重複として学習データに残っていた
(検出の経緯は tools/check_leak.py と docs/notes-part4.md を参照)。
その2 のモデルはこの29行を学習時に見ている。

その4 では混入を直したので、その4 のモデルは249行すべてを見ていない。
このまま両者を同じ holdout で比べると、**その2 に下駄を履かせたまま
「その4 が勝った/負けた」を言うことになる**。

そこで「どちらのモデルも見ていない 220行」だけの検証セットを作る。
この上で両方を測り直せば、条件が揃った比較になる。

3つの数字を並べて記事に書けるようにしておく。

    その2 / 249行 (混入込み)  ← 既に公開してしまった数字
    その2 / 220行 (混入なし)  ← 測り直し
    その4 / 220行 (混入なし)  ← 本来の比較対象
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=str(ROOT / "eval" / "holdout.txt"))
    ap.add_argument("--report", default=str(ROOT / "runs" / "3lm" / "leak_check.json"),
                    help="tools/check_leak.py が書いた検査結果")
    ap.add_argument("--corpus", default="corpus_sft.txt",
                    help="どのコーパスへの混入を除くか (検査結果の中の名前)")
    ap.add_argument("--out", default=str(ROOT / "eval" / "holdout_clean.txt"))
    args = ap.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    entry = next((c for c in report["corpora"] if c["corpus"] == args.corpus), None)
    if entry is None:
        names = [c["corpus"] for c in report["corpora"]]
        raise SystemExit(f"{args.corpus} が検査結果にありません。あるのは {names}")

    leaked = set(entry["affected_holdout_lines"])
    lines = Path(args.holdout).read_text(encoding="utf-8").splitlines()
    kept = [ln for i, ln in enumerate(lines, 1) if i not in leaked]

    out = Path(args.out)
    out.write_text("\n".join(kept) + "\n", encoding="utf-8")

    print("=" * 66)
    print("  混入行を除いた検証セット")
    print(f"    元         : {Path(args.holdout).name} ({len(lines)} 行)")
    print(f"    除いた行   : {len(leaked)} 行 (→ {args.corpus} に混入していた)")
    print(f"    残した行   : {len(kept)} 行")
    print(f"    → {out}")
    print("=" * 66)
    print("  両方のモデルをこの検証セットで測り直すと、条件の揃った比較になる:")
    print(f"    python3 eval/run.py --ckpt <その2> --holdout {out.relative_to(ROOT)} --tag 2lm_clean")
    print(f"    python3 eval/run.py --ckpt <その4> --holdout {out.relative_to(ROOT)} --tag 3lm_clean")


if __name__ == "__main__":
    main()
