"""事前学習したモデルを、対話の書式に合わせる (SFT).

事前学習が終わった時点のモデルは「日本語の続きを書く」ことしかできない。
「こんにちは」と入れても、挨拶の続きに見える文章を書くだけで、
返事はしてくれない。対話の形を教えるのがこの段階。

## その2 との違い: instruction masking

その2 は会話まるごとに損失をかけていた。つまり

    <|user|>おすすめの本はありますか？<|assistant|>SFがお好きなら…<|end|>
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ここも学習していた

質問文の予測まで学習させると、容量の一部が「ユーザーの言いそうなこと」の
モデル化に使われる。返答の質に効かないうえ、質問を勝手に続けて書く癖
(<|user|> を自分で出してしまう) の原因にもなる。

ここでは **<|assistant|> より後ろだけに損失をかける**。

    <|user|>おすすめの本はありますか？<|assistant|>SFがお好きなら…<|end|>
                                              ~~~~~~~~~~~~~~~~~~~~~ ここだけ

<|end|> は損失に含める。「ここで止まる」を学ばせたいので、
これを外すと返答が終わらなくなる。

## 詰め込み (packing)

1会話は平均で190トークンしかないので、文脈512に1会話だけ入れて
残りを詰め物で埋めると、計算の6割が無駄になる。会話を連結して
512トークンごとに切る。マスクがあるので、境界で会話が混ざっても
損失は各会話の返答部分にしか掛からない。

使い方:
    python src/sft.py --init-from checkpoints/pretrain-final --corpus data/corpus_sft.txt
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import checkpoint, runtime  # noqa: E402
from src.generate import chat_stream  # noqa: E402
from src.model import GPTConfig, MiniGPT  # noqa: E402
from src.tokenizer import Tokenizer, load_tokenizer  # noqa: E402
from src.train import Heartbeat, append_log, build_schedule, open_log  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PROMPTS = ("こんにちは", "おすすめの本を教えてください", "猫について教えて")

_stop_requested = False


def _request_stop(signum, _frame) -> None:
    global _stop_requested
    if _stop_requested:
        raise KeyboardInterrupt
    _stop_requested = True
    print(f"\n[シグナル {signum}] 次の区切りで保存して終了します。", flush=True)


def build_sft_arrays(
    lines: list[str], tokenizer: Tokenizer
) -> tuple[np.ndarray, np.ndarray, dict]:
    """会話の行を、トークン列と「損失を数えるか」の 0/1 列に変換する.

    <|user|> <|assistant|> <|end|> はどれも語彙に1トークンとして
    入っているので (user_defined_symbols)、ID を見るだけで区切りが分かる。
    文字列を探して位置を数える必要はない。
    """
    assistant_id, end_id, user_id = tokenizer.assistant_id, tokenizer.end_id, tokenizer.user_id
    tokens: list[int] = []
    weights: list[int] = []
    for line in lines:
        ids = tokenizer.encode(line)
        in_answer = False
        for token_id in ids:
            if token_id == assistant_id:
                # <|assistant|> 自体は入力側なので数えない。
                tokens.append(token_id)
                weights.append(0)
                in_answer = True
                continue
            if token_id == user_id:
                in_answer = False
            tokens.append(token_id)
            weights.append(1 if in_answer else 0)
            if token_id == end_id:
                # <|end|> は数える。止まり方を学ばせたいので。
                in_answer = False
    return (
        np.asarray(tokens, dtype=np.uint16),
        np.asarray(weights, dtype=np.uint8),
        {"tokens": len(tokens), "target_tokens": int(np.sum(weights)), "conversations": len(lines)},
    )


def get_batch(
    tokens: np.ndarray, weights: np.ndarray, batch_size: int, block_size: int,
    seed: int, step: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """src/data.py と同じ (seed, step) 決定論で、マスク付きのバッチを作る."""
    rng = np.random.default_rng([seed, step])
    high = len(tokens) - block_size - 1
    ix = rng.integers(0, max(high, 1), size=batch_size)
    xs = np.stack([tokens[i : i + block_size] for i in ix]).astype(np.int32)
    ys = np.stack([tokens[i + 1 : i + 1 + block_size] for i in ix]).astype(np.int32)
    ws = np.stack([weights[i + 1 : i + 1 + block_size] for i in ix]).astype(np.float32)
    return mx.array(xs), mx.array(ys), mx.array(ws)


def main() -> None:
    ap = argparse.ArgumentParser(description="事前学習したモデルを対話に合わせる")
    ap.add_argument("--init-from", required=True, help="事前学習のチェックポイント")
    ap.add_argument("--corpus", default=str(ROOT / "data" / "corpus_sft.txt"))
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "sft"))
    ap.add_argument("--export", default="", help="既定: <out>-final")
    ap.add_argument("--log", default="", help="既定: runs/3lm/sft_loss.csv")
    ap.add_argument("--heartbeat", default="")
    ap.add_argument("--resume", default="auto", choices=("auto", "never"))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05,
                    help="SFT はデータを何周もするので、少しだけ入れる")
    ap.add_argument("--lr", type=float, default=6e-5,
                    help="事前学習の 1/10 程度。大きいと事前学習した知識を壊す")
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--eval-interval", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--save-interval-min", type=float, default=10.0)
    ap.add_argument("--sample-interval", type=int, default=0)
    ap.add_argument("--no-mask", action="store_true",
                    help="instruction masking を切る (その2 と同じ条件にして比べるため)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    base = Path(args.init_from)
    out_dir = Path(args.out)
    export_dir = Path(args.export) if args.export else out_dir.parent / f"{out_dir.name}-final"
    log_path = Path(args.log) if args.log else ROOT / "runs" / "3lm" / "sft_loss.csv"
    hb_path = Path(args.heartbeat) if args.heartbeat else ROOT / "runs" / "3lm" / "sft_heartbeat.json"

    print("=" * 66)
    runtime.preflight()
    limit_gb = runtime.configure()
    guard = runtime.MemoryGuard(limit_gb)
    mx.random.seed(args.seed)

    # 語彙は事前学習したものをそのまま使う。作り直すとIDの対応が変わり、
    # 事前学習した重みが全部無意味になる。
    tokenizer = load_tokenizer(base)
    lines = [
        ln.strip()
        for ln in Path(args.corpus).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(lines)
    n_val = max(1, int(len(lines) * args.val_frac))
    val_lines, train_lines = lines[:n_val], lines[n_val:]

    train_tokens, train_weights, train_meta = build_sft_arrays(train_lines, tokenizer)
    val_tokens, val_weights, val_meta = build_sft_arrays(val_lines, tokenizer)
    if args.no_mask:
        train_weights = np.ones_like(train_weights)
        val_weights = np.ones_like(val_weights)

    cfg = GPTConfig.load(base / "config.json")
    cfg.dropout = args.dropout
    model = MiniGPT(cfg)
    model.load_weights(str(base / "model.safetensors"))
    mx.eval(model.parameters())

    tokens_per_step = args.batch_size * cfg.block_size
    total_steps = max(1, int(train_meta["tokens"] * args.epochs / tokens_per_step))
    mask_ratio = train_meta["target_tokens"] / max(1, train_meta["tokens"])

    schedule = build_schedule(
        args.lr, total_steps, max(20, int(total_steps * args.warmup_frac)), args.min_lr_ratio
    )
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=args.weight_decay)

    state = checkpoint.TrainState(
        seed=args.seed, data_meta={"train": train_meta, "val": val_meta},
        args=dict(vars(args)),
    )
    resume_from = checkpoint.find_resumable(out_dir) if args.resume == "auto" else None
    if resume_from is not None:
        restored = checkpoint.restore(resume_from, model, optimizer)
        state.step = restored.step
        state.tokens_seen = restored.tokens_seen
        state.best_val = restored.best_val
        state.elapsed_sec = restored.elapsed_sec
        state.resumes = restored.resumes + 1
        state.seed = restored.seed
        print(f"  再開          : {resume_from.name} / step {state.step:,}")

    print(f"  元のモデル    : {base}")
    print(f"  構成          : {cfg.arch} / {cfg.n_layer}層 / n_embd {cfg.n_embd} / "
          f"ctx {cfg.block_size}")
    print(f"  パラメータ数  : {model.n_params / 1e6:.2f}M")
    print(f"  会話数        : 学習 {train_meta['conversations']:,} / "
          f"検証 {val_meta['conversations']:,}")
    print(f"  トークン      : {train_meta['tokens']:,}")
    print(f"  損失を数える  : {train_meta['target_tokens']:,} "
          f"({mask_ratio:.1%})" + ("  ※マスク無効" if args.no_mask else ""))
    print(f"  予算          : {args.epochs} エポック = {total_steps:,} ステップ")
    print(f"  学習率        : {args.lr:.1e}")
    print("=" * 66)

    def loss_fn(m: MiniGPT, x: mx.array, y: mx.array, w: mx.array) -> mx.array:
        return m.loss(x, y, w)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    compile_state = [model.state, optimizer.state, mx.random.state]

    def _step(x: mx.array, y: mx.array, w: mx.array) -> mx.array:
        loss, grads = loss_and_grad(model, x, y, w)
        if args.grad_clip > 0:
            grads, _ = optim.clip_grad_norm(grads, args.grad_clip)
        optimizer.update(model, grads)
        return loss

    step_fn = (
        _step if args.no_compile
        else partial(mx.compile, inputs=compile_state, outputs=compile_state)(_step)
    )

    def evaluate() -> float:
        model.eval()
        total = 0.0
        for i in range(args.eval_batches):
            x, y, w = get_batch(
                val_tokens, val_weights, args.batch_size, cfg.block_size, 0, i
            )
            total += float(model.loss(x, y, w).item())
        model.train()
        return total / args.eval_batches

    open_log(log_path)
    if state.step:
        append_log(log_path, f"# resumed at step {state.step} ({state.resumes}回目)")
    heartbeat = Heartbeat(hb_path)

    session_start = time.time()
    base_elapsed = state.elapsed_sec

    def elapsed_total() -> float:
        return base_elapsed + (time.time() - session_start)

    def save(reason: str) -> None:
        state.elapsed_sec = elapsed_total()
        checkpoint.save(out_dir, model, optimizer, tokenizer, state)
        print(f"  [保存] step {state.step:,} ({reason})", flush=True)

    model.train()
    window: list[float] = []
    last_save = time.time()
    mark_time, mark_step = time.time(), state.step
    step = state.step
    stop_reason = "エポック数に到達"

    while step < total_steps:
        step += 1
        x, y, w = get_batch(
            train_tokens, train_weights, args.batch_size, cfg.block_size, state.seed, step
        )
        loss = step_fn(x, y, w)
        mx.eval(compile_state)
        window.append(float(loss.item()))
        state.step = step
        state.tokens_seen = step * tokens_per_step

        if step % args.log_interval == 0:
            train_loss = sum(window) / len(window)
            window.clear()
            now = time.time()
            tps = (step - mark_step) * tokens_per_step / max(now - mark_time, 1e-6)
            mark_time, mark_step = now, step
            lr_now = float(schedule(mx.array(step)).item())
            epoch = state.tokens_seen / max(1, train_meta["tokens"])
            print(f"step {step:6d}/{total_steps} | loss {train_loss:.4f} | "
                  f"lr {lr_now:.2e} | {tps / 1e3:.0f}k tok/s | {epoch:.2f} エポック",
                  flush=True)
            append_log(
                log_path,
                f"{step},{elapsed_total():.1f},{state.tokens_seen},{lr_now:.6e},"
                f"{train_loss:.4f},,{tps:.0f}",
            )
            heartbeat.write({
                "stage": "sft", "step": step, "total_steps": total_steps,
                "tokens_seen": state.tokens_seen, "epoch": round(epoch, 3),
                "train_loss": round(train_loss, 4),
                "best_val": Heartbeat.finite(state.best_val),
                "lr": lr_now, "tok_per_sec": round(tps),
                "elapsed_hours": round(elapsed_total() / 3600, 3),
                "peak_gb": round(guard.peak_gb, 2), "resumes": state.resumes,
            })

        if step % args.eval_interval == 0 or step == total_steps:
            val_loss = evaluate()
            marker = ""
            if val_loss < state.best_val:
                state.best_val = val_loss
                marker = "  <- 最良"
            print(f"  [検証] step {step:,} val_loss {val_loss:.4f} "
                  f"(最良 {state.best_val:.4f}){marker}", flush=True)
            append_log(
                log_path, f"{step},{elapsed_total():.1f},{state.tokens_seen},,,{val_loss:.4f},"
            )
            if marker:
                save("検証が最良")
                checkpoint.export(
                    export_dir, out_dir / checkpoint.read_current(out_dir),
                    extra={"step": step, "val_loss": round(val_loss, 4),
                           "stage": "sft", "instruction_masking": not args.no_mask,
                           "mask_ratio": round(mask_ratio, 4)},
                )
                last_save = time.time()

        if args.sample_interval and step % args.sample_interval == 0:
            model.eval()
            for prompt in SAMPLE_PROMPTS:
                reply = "".join(
                    chat_stream(model, tokenizer, [], prompt, max_new_tokens=80, temperature=0.8)
                )
                print(f"  [試し] {prompt} -> {reply}", flush=True)
            model.train()

        if time.time() - last_save > args.save_interval_min * 60:
            save("定期")
            last_save = time.time()
        if guard.over_threshold():
            stop_reason = f"メモリのピークが危険域 ({guard.peak_gb:.1f}GB)"
            break
        if step % 200 == 0:
            guard.release()
        if _stop_requested:
            stop_reason = "シグナルを受けた"
            break

    save("終了時")
    print("=" * 66)
    print(f"終了: {stop_reason}")
    print(f"  step {state.step:,} / {total_steps:,} / 最良 val_loss {state.best_val:.4f}")
    print(f"  経過 {elapsed_total() / 60:.1f} 分")
    print(f"  配布用: {export_dir}")
    print("=" * 66)

    model.eval()
    print("生成例:")
    for prompt in SAMPLE_PROMPTS:
        reply = "".join(
            chat_stream(model, tokenizer, [], prompt, max_new_tokens=100, temperature=0.8)
        )
        print(f"  {prompt} -> {reply}")

    (log_path.parent / "sft_summary.json").write_text(
        json.dumps({
            "step": state.step, "total_steps": total_steps,
            "best_val": state.best_val, "mask_ratio": mask_ratio,
            "instruction_masking": not args.no_mask,
            "epochs": args.epochs, "lr": args.lr,
            "train": train_meta, "val": val_meta,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
