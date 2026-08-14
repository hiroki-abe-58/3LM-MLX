"""その2 の「データが必要量の 1.7% しかない」という診断の答え合わせ.

    python3 tools/run_data_scaling.py

## 何を確かめるのか

その2 で 13.81M パラメータのモデルを学習したとき、val loss が train loss を
追い越す地点 (乖離点) がこう動いた。

    405万文字 → step   750
    949万文字 → step 1,750

コーパスを2.15倍にしたら、乖離点が2.3倍うしろにずれた。ここから
「暗記に入るのが早いのはデータが足りないからだ」と診断した。

この診断が正しいなら、**データをさらに10倍・100倍にすれば乖離点は
さらにうしろへ動き、いずれ予算内では追い越されなくなる**はず。
そうならなければ診断が誤りで、原因は別にある。それを測る。

## 前提を3つ揃えてある

その2 の3本は、コーパスが違うと検証セットも違っていた。それでは
val loss の水準を比べられない (難しさの違う文章を採点している)。
この実験では、**訓練データの量以外を全部固定する**。

1. **検証セットを共通にする** (`--val-corpus`)
   条件ごとに違う文章で採点すると、val loss の差が「データ量の効果」なのか
   「採点した文章の難しさ」なのか区別できない。

2. **語彙を共通にする** (語彙 8,000 を事前学習コーパスで1回だけ学習)
   条件ごとに語彙を学習し直すと、データが多い条件のほうが良い語彙になり、
   トークンあたりの情報量が変わってしまう。loss の比較ができなくなる。
   語彙を 8,000 に留めるのは、モデルを**その2 と同じ 13.81M に保つ**ため
   (32,000 にすると埋め込みだけで 12.3M 増えて別のモデルになる)。

3. **部分集合はコーパス全体から等間隔に採る** (`--sample-chars`)
   「先頭 N 文字」にすると、このコーパスは青空文庫 → FineWeb2 の順なので
   **小さい条件だけ文語100%**になる。量を変えたつもりでドメインも
   変わってしまい、何を測ったのか分からなくなる。

そのぶん「その2 の step 1,750 をそのまま再現する」ことは狙わない
(コーパスの中身が違うので同じ数字にはならない)。見るのは**乖離点が動く向き**である。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.loss_log import read_curve  # noqa: E402

# その2 と同じ形・同じステップ数にする。変えるのは訓練データの量だけ。
STEPS = 3600
VOCAB_SIZE = 8_000
BATCH_SIZE = 32
BLOCK_SIZE = 256
ARCH_ARGS = [
    "--arch", "2lm", "--block-size", str(BLOCK_SIZE),
    "--n-layer", "6", "--n-head", "6", "--n-embd", "384",
    "--batch-size", str(BATCH_SIZE),
    # その2 は dropout 0.1 で学習していた。乖離点の定義が dropout に
    # 依存する (tools/loss_log.py の divergence_step) ので、ここも揃える。
    "--dropout", "0.1",
]

CONDITIONS = (
    ("949万文字", 9_480_000),
    ("1億文字", 100_000_000),
    ("全量", 0),  # 0 は「コーパスを全部使う」
)


def run(cmd: list[str], log: Path) -> None:
    print(f"  $ {' '.join(str(c) for c in cmd[:6])} …")
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"  失敗しました。{log} の末尾:")
        print("\n".join(log.read_text(encoding="utf-8").splitlines()[-15:]))
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="データ量だけを変えて乖離点の動きを測る")
    ap.add_argument("--corpus", default=str(ROOT / "data" / "corpus_pretrain.txt"))
    ap.add_argument("--tokenizer", default=str(ROOT / "data" / "scaling_tok" / "tokenizer"),
                    help=f"全条件で共通に使うトークナイザ (語彙 {VOCAB_SIZE:,})。無ければ作る")
    ap.add_argument("--val-corpus", default=str(ROOT / "data" / "val_pretrain.txt"),
                    help="全条件で共通に使う検証セット")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--work", default=str(ROOT / "runs" / "3lm" / "scaling"))
    ap.add_argument("--skip-encode", action="store_true")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    corpus = Path(args.corpus)
    if not corpus.exists():
        raise SystemExit(f"{corpus} がありません。data/prepare_pretrain.py を先に実行してください。")

    # 語彙は全条件で共通。無ければコーパス全体から1回だけ学習する。
    tokenizer_dir = Path(args.tokenizer)
    if not (tokenizer_dir / "tokenizer.model").exists():
        print("=" * 70)
        print(f"  共通の語彙 {VOCAB_SIZE:,} を学習します (コーパス全体から1回だけ)")
        print("=" * 70)
        run([
            sys.executable, str(ROOT / "data" / "encode.py"),
            "--corpus", str(corpus),
            "--out-dir", str(tokenizer_dir.parent),
            "--vocab-size", str(VOCAB_SIZE),
            "--tokenizer-only",
        ], work / "encode_tokenizer.log")
        print(f"  → {tokenizer_dir}")
        print()

    summary = []
    for label, sample_chars in CONDITIONS:
        tag = label.replace("万", "man").replace("億", "oku").replace("文字", "")
        data_dir = ROOT / "data" / f"scaling_{tag}"
        ckpt = work / f"ckpt_{tag}"
        log_csv = work / f"loss_{tag}.csv"

        print("=" * 70)
        print(f"  条件: {label}")
        print("=" * 70)

        if not args.skip_encode or not (data_dir / "train.bin").exists():
            encode = [
                sys.executable, str(ROOT / "data" / "encode.py"),
                "--corpus", str(corpus),
                "--val-corpus", args.val_corpus,
                "--out-dir", str(data_dir),
                "--tokenizer", args.tokenizer,
            ]
            if sample_chars:
                # 先頭 N 文字ではなく、全体から等間隔に採る
                encode += ["--sample-chars", str(sample_chars)]
            run(encode, work / f"encode_{tag}.log")
        meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
        print(f"  訓練 {meta['train_tokens']:,} トークン / "
              f"検証 {meta['val_tokens']:,} / {meta['chars']:,} 文字")

        # ログは条件ごとに作り直す (追記だと前回の曲線が混ざる)
        log_csv.unlink(missing_ok=True)
        started = time.time()
        run([
            sys.executable, str(ROOT / "src" / "train.py"),
            "--data", str(data_dir), "--out", str(ckpt),
            "--steps", str(args.steps), *ARCH_ARGS,
            "--eval-interval", "100", "--log-interval", "50",
            "--save-interval-min", "30",
            "--log", str(log_csv),
            "--heartbeat", str(work / f"hb_{tag}.json"),
            "--resume", "never", "--seed", "1234",
        ], work / f"train_{tag}.log")

        curve = read_curve(log_csv, label)
        best_step, best_val = curve.best_val
        summary.append({
            "label": label,
            "chars": meta["chars"],
            "train_tokens": meta["train_tokens"],
            "epochs": round(
                args.steps * BATCH_SIZE * BLOCK_SIZE / meta["train_tokens"], 3
            ),
            "divergence_step": curve.divergence_step,
            "best_val": round(best_val, 4),
            "best_step": best_step,
            "final_train": round(curve.final_train, 4),
            "minutes": round((time.time() - started) / 60, 1),
            "csv": str(log_csv.relative_to(ROOT)),
        })
        print(f"  乖離点 {curve.divergence_step or 'なし'} / "
              f"最良 val {best_val:.4f} (step {best_step}) / "
              f"{summary[-1]['minutes']:.1f}分")
        print()

    out = work / "summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 78)
    print(f"{'条件':<12} {'文字数':>14} {'トークン':>13} {'周回':>6} "
          f"{'乖離点':>8} {'最良val':>9}")
    print("-" * 78)
    for row in summary:
        onset = str(row["divergence_step"]) if row["divergence_step"] else "なし"
        print(f"{row['label']:<12} {row['chars']:>14,} {row['train_tokens']:>13,} "
              f"{row['epochs']:>6.2f} {onset:>8} {row['best_val']:>9.4f}")
    print("=" * 78)

    onsets = [r["divergence_step"] for r in summary]
    print()
    if all(o == 0 for o in onsets[1:]):
        print("データを増やした条件では、3,600ステップの予算内で val が train を")
        print("追い越しませんでした。診断は正しかったことになります。")
    elif onsets[0] and all(
        (b == 0 or b > a) for a, b in zip(onsets, onsets[1:], strict=False)
    ):
        print("乖離点がデータ量に応じてうしろへ動きました。診断は正しかったことになります。")
    else:
        print("乖離点がデータ量どおりに動きませんでした。診断が誤りだった可能性があります。")
        print("この場合は訂正として書くこと。原因の候補は、モデルの容量が先に")
        print("飽和している / 検証セットが訓練データと近すぎる / 学習率が高すぎる。")

    print()
    print("図を作る:")
    specs = " ".join(f"{r['csv']}:{r['label']}" for r in summary)
    print(f"  python3 tools/compare_runs.py {specs} \\")
    print("      --out docs/images/3lm-data-scaling.png")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
