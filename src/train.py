"""事前学習ループ.

やることは3行で書ける:
  1. コーパスから連続したトークン列をランダムに切り出す
  2. 「1トークンずらした列」を正解として次トークン予測の誤差を計算する
  3. 誤差が小さくなる方向にパラメータを動かす

これを何万回繰り返すだけで、モデルは日本語の並び方を覚える。

## その2 から変えたところ

**予算の単位をトークンにした。**
その2 は `--steps 4300` と `--minutes 35` の二重管理で、実際には時間側が
先に効いて **学習率が下がりきらないまま step 3,600 で止まった**。
cosine 減衰は「予定した終点」に向けて下がるので、途中で切ると
最後まで学習率が高いままになり、重みが落ち着かないところで終わる。

そこで `--tokens` を第一の予算にした。スケジュールもここから組む。
`--max-hours` は保険で、到達したら**保存して正常終了**する (黙って切らない)。

**途中から再開できるようにした。**
8時間走らせれば、どこかで落ちる。詳しくは src/checkpoint.py と src/data.py。

使い方:
    # 較正 (5分だけ回して tok/s を測る)
    python src/train.py --data data/3lm --max-hours 0.1 --tokens 100_000_000

    # 本番
    python src/train.py --data data/3lm --tokens 514_000_000 --max-hours 8.5

    # 落ちた後
    python src/train.py --data data/3lm --tokens 514_000_000 --resume auto
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import checkpoint, runtime  # noqa: E402
from src.data import TokenBin, get_batch, iter_eval_batches, load_meta  # noqa: E402
from src.generate import chat_stream  # noqa: E402
from src.model import GPTConfig, MiniGPT  # noqa: E402
from src.tokenizer import load_tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PROMPTS = ("こんにちは", "おすすめの本を教えてください")

# SIGTERM / SIGINT を受けたら、その場で死ぬのではなく次の区切りで保存して抜ける。
_stop_requested = False


def _request_stop(signum, _frame) -> None:
    global _stop_requested
    if _stop_requested:
        # 2回目は即座に落とす (保存が固まったときの逃げ道)
        raise KeyboardInterrupt
    _stop_requested = True
    print(f"\n[シグナル {signum}] 次の区切りで保存して終了します。もう一度送ると即停止します。",
          flush=True)


class Heartbeat:
    """進捗を1つの JSON に書き出す.

    ターミナルを閉じても学習は続くので (nohup / tmux)、
    別の端末から `tools/watch.py` で覗けるようにしておく。
    """

    def __init__(self, path: Path, interval_sec: float = 30.0):
        self.path = path
        self.interval = interval_sec
        self.last = 0.0
        path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def finite(value: float) -> float | None:
        """inf を None にする.

        best_val は最初の検証まで inf。json.dump はこれを `Infinity` と
        書くが、これは JSON の規格に無い綴りで、Python 以外のパーサでは
        読めなくなる。heartbeat は他のツールから読む前提のファイルなので
        規格内に収める。
        """
        return None if value in (float("inf"), float("-inf")) else round(value, 4)

    def write(self, payload: dict, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last < self.interval:
            return
        self.last = now
        payload = {"updated_at": now, **payload}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)  # 読み手が半端な JSON を見ないように差し替える


def open_log(path: Path) -> None:
    """損失ログを追記で開く. 既にあればヘッダを書かない.

    その2 は write_text でログを毎回切り捨てていた。再開すると
    それまでの学習曲線が消えるので、一晩の記録が取れない。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("step,elapsed_sec,tokens,lr,train_loss,val_loss,tok_per_sec\n",
                        encoding="utf-8")


def append_log(path: Path, row: str) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(row + "\n")


def build_schedule(lr: float, total_steps: int, warmup: int, min_ratio: float):
    """線形ウォームアップ → cosine 減衰.

    total_steps は**トークン予算から決めた最終ステップ**。ここを実際の
    終点と一致させることが肝で、一致していないと学習率が下がりきらない。
    """
    warmup = max(1, min(warmup, total_steps - 1))
    return optim.join_schedules(
        [
            optim.linear_schedule(lr * 0.02, lr, warmup),
            optim.cosine_decay(lr, max(total_steps - warmup, 1), lr * min_ratio),
        ],
        [warmup],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="3LM の事前学習")
    ap.add_argument("--data", default=str(ROOT / "data" / "3lm"),
                    help="train.bin / val.bin / tokenizer/ があるディレクトリ")
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "pretrain"),
                    help="ckpt-A / ckpt-B / CURRENT を置く場所")
    ap.add_argument("--export", default="",
                    help="配布用に重みだけ出す先 (既定: <out>/../pretrain-final)")
    ap.add_argument("--log", default="", help="既定: runs/3lm/pretrain_loss.csv")
    ap.add_argument("--heartbeat", default="", help="既定: runs/3lm/heartbeat.json")
    ap.add_argument("--resume", default="auto", choices=("auto", "never"),
                    help="auto なら <out> に再開できるものがあれば続ける")
    ap.add_argument("--init-from", default="",
                    help="別のチェックポイントの重みから始める (再開とは別)")

    # 予算。--tokens が主で、--steps は「答え合わせ」用に固定したいとき使う。
    ap.add_argument("--tokens", type=float, default=0,
                    help="学習トークン数の予算。スケジュールもここから組む")
    ap.add_argument("--steps", type=int, default=0,
                    help="ステップ数で直接指定する (--tokens と排他)")
    ap.add_argument("--max-hours", type=float, default=0,
                    help="保険。到達したら保存して正常終了する (0 で無効)")

    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--n-embd", type=int, default=512)
    ap.add_argument("--arch", default="3lm", choices=("3lm", "2lm"))
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="事前学習はデータを1周もしないので既定で切ってある")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup-frac", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--eval-interval", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--save-interval-min", type=float, default=10.0,
                    help="この分数ごとに再開用チェックポイントを書く")
    ap.add_argument("--sample-interval", type=int, default=0,
                    help="この間隔で生成例を出す (0 なら出さない)")
    ap.add_argument("--memory-limit-gb", type=float, default=0,
                    help="MLX に与える上限 (0 なら GPU 推奨上限の 75%%)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args(argv)

    if bool(args.tokens) == bool(args.steps):
        ap.error("--tokens か --steps のどちらか一方を指定してください")
    return args


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    log_path = Path(args.log) if args.log else ROOT / "runs" / "3lm" / "pretrain_loss.csv"
    hb_path = Path(args.heartbeat) if args.heartbeat else ROOT / "runs" / "3lm" / "heartbeat.json"
    export_dir = Path(args.export) if args.export else out_dir.parent / f"{out_dir.name}-final"

    print("=" * 66)
    runtime.preflight()
    limit_gb = runtime.configure(args.memory_limit_gb or None)
    guard = runtime.MemoryGuard(limit_gb)

    mx.random.seed(args.seed)
    train_bin = TokenBin(data_dir / "train.bin")
    val_bin = TokenBin(data_dir / "val.bin")
    tokenizer = load_tokenizer(data_dir / "tokenizer")
    data_meta = load_meta(data_dir)

    tokens_per_step = args.batch_size * args.block_size
    if args.steps:
        total_steps = args.steps
        token_budget = total_steps * tokens_per_step
    else:
        token_budget = int(args.tokens)
        total_steps = max(1, token_budget // tokens_per_step)

    if args.init_from:
        cfg = GPTConfig.load(Path(args.init_from) / "config.json")
        cfg.dropout = args.dropout
    else:
        cfg = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            arch=args.arch,
        )
    model = MiniGPT(cfg)
    if args.init_from:
        model.load_weights(str(Path(args.init_from) / "model.safetensors"))
    mx.eval(model.parameters())

    schedule = build_schedule(
        args.lr, total_steps, max(100, int(total_steps * args.warmup_frac)), args.min_lr_ratio
    )
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=args.weight_decay)

    # 再開はコンパイル前に済ませる。mx.compile は state のオブジェクトを
    # 捕まえるので、コンパイル後に optimizer.state を差し替えると噛み合わない。
    state = checkpoint.TrainState(
        seed=args.seed, data_meta=data_meta, args={k: v for k, v in vars(args).items()}
    )
    resume_from = checkpoint.find_resumable(out_dir) if args.resume == "auto" else None
    if resume_from is not None:
        restored = checkpoint.restore(resume_from, model, optimizer)
        state.step = restored.step
        state.tokens_seen = restored.tokens_seen
        state.best_val = restored.best_val
        state.elapsed_sec = restored.elapsed_sec
        state.resumes = restored.resumes + 1
        state.seed = restored.seed  # データ順は保存時の seed に従う
        print(f"  再開          : {resume_from.name} / step {state.step:,} "
              f"({state.resumes}回目)")

    print(f"  構成          : {cfg.arch} / {cfg.n_layer}層 / n_embd {cfg.n_embd} / "
          f"{cfg.n_head}ヘッド / ctx {cfg.block_size}")
    print(f"  語彙数        : {cfg.vocab_size:,}")
    print(f"  パラメータ数  : {model.n_params / 1e6:.2f}M "
          f"(非埋め込み {model.n_params_non_embedding / 1e6:.2f}M)")
    print(f"  学習データ    : {train_bin.tokens:,} トークン (検証 {val_bin.tokens:,})")
    if data_meta.get("chars_per_token"):
        print(f"  1トークン     : {data_meta['chars_per_token']} 文字")
    print(f"  1ステップ     : {args.batch_size} x {args.block_size} = "
          f"{tokens_per_step:,} トークン")
    print(f"  予算          : {token_budget:,} トークン / {total_steps:,} ステップ"
          + (f" / 保険 {args.max_hours}時間" if args.max_hours else ""))
    epochs = token_budget / max(1, train_bin.tokens)
    print(f"  コーパス周回  : {epochs:.2f} 周")
    d_over_n = token_budget / max(1, model.n_params_non_embedding)
    print(f"  D/N           : {d_over_n:.1f} (Chinchilla の目安 20)")
    print("=" * 66)

    def loss_fn(m: MiniGPT, x: mx.array, y: mx.array) -> mx.array:
        return m.loss(x, y)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    # mx.random.state を inputs/outputs に含めないと Dropout の乱数が固定され、
    # コンパイル済み関数が毎回同じマスクを使ってしまう。
    compile_state = [model.state, optimizer.state, mx.random.state]

    def _step(x: mx.array, y: mx.array) -> mx.array:
        loss, grads = loss_and_grad(model, x, y)
        if args.grad_clip > 0:
            grads, _ = optim.clip_grad_norm(grads, args.grad_clip)
        optimizer.update(model, grads)
        return loss

    step_fn = (
        _step if args.no_compile else partial(mx.compile, inputs=compile_state, outputs=compile_state)(_step)
    )

    def evaluate() -> float:
        model.eval()
        total = 0.0
        for x, y in iter_eval_batches(val_bin, args.batch_size, args.block_size, args.eval_batches):
            total += float(model.loss(x, y).item())
        model.train()
        return total / args.eval_batches

    open_log(log_path)
    if state.step:
        append_log(log_path, f"# resumed at step {state.step} ({state.resumes}回目)")
    heartbeat = Heartbeat(hb_path)

    def save(reason: str) -> None:
        state.elapsed_sec = elapsed_total()
        checkpoint.save(out_dir, model, optimizer, tokenizer, state)
        print(f"  [保存] step {state.step:,} ({reason})", flush=True)

    session_start = time.time()
    base_elapsed = state.elapsed_sec

    def elapsed_total() -> float:
        return base_elapsed + (time.time() - session_start)

    model.train()
    window: list[float] = []
    last_save = time.time()
    # 速度は「直近の log_interval ステップ」で測る。開始からの平均にすると
    # 最初のコンパイル時間が薄まりながら効き続け、熱による低下と混ざる。
    tps = 0.0
    tps_history: list[float] = []
    last_train_loss: float | None = None
    mark_time = time.time()
    mark_step = state.step
    stop_reason = "トークン予算に到達"
    step = state.step

    while step < total_steps:
        step += 1
        # バッチは (seed, step) だけで決まる。再開しても1トークンも変わらない。
        x, y = get_batch(train_bin, args.batch_size, args.block_size, state.seed, step)
        loss = step_fn(x, y)
        mx.eval(compile_state)  # ここで初めて実際に計算される (MLXは遅延評価)
        window.append(float(loss.item()))
        state.step = step
        state.tokens_seen = step * tokens_per_step

        if step % args.log_interval == 0:
            train_loss = sum(window) / len(window)
            last_train_loss = train_loss
            window.clear()
            now = time.time()
            tps = (step - mark_step) * tokens_per_step / max(now - mark_time, 1e-6)
            mark_time, mark_step = now, step
            tps_history.append(tps)
            lr_now = float(schedule(mx.array(step)).item())
            remaining = (total_steps - step) * tokens_per_step / max(tps, 1)
            print(
                f"step {step:6d}/{total_steps} | loss {train_loss:.4f} | "
                f"lr {lr_now:.2e} | {tps / 1e3:.0f}k tok/s | "
                f"{elapsed_total() / 3600:.2f}h | 残り {remaining / 3600:.1f}h",
                flush=True,
            )
            append_log(
                log_path,
                f"{step},{elapsed_total():.1f},{state.tokens_seen},{lr_now:.6e},"
                f"{train_loss:.4f},,{tps:.0f}",
            )
            heartbeat.write({
                "step": step, "total_steps": total_steps,
                "tokens_seen": state.tokens_seen, "token_budget": token_budget,
                "train_loss": round(train_loss, 4),
                "best_val": Heartbeat.finite(state.best_val),
                "lr": lr_now, "tok_per_sec": round(tps),
                "elapsed_hours": round(elapsed_total() / 3600, 3),
                "eta_hours": round(remaining / 3600, 2),
                "peak_gb": round(guard.peak_gb, 2),
                "cpu_speed_limit": runtime.cpu_speed_limit(),
                "resumes": state.resumes,
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
                log_path,
                f"{step},{elapsed_total():.1f},{state.tokens_seen},,,{val_loss:.4f},",
            )
            if marker:
                save("検証が最良")
                checkpoint.export(
                    export_dir,
                    out_dir / checkpoint.read_current(out_dir),
                    extra={"step": step, "val_loss": round(val_loss, 4),
                           "tokens_seen": state.tokens_seen},
                )
                last_save = time.time()

        if args.sample_interval and step % args.sample_interval == 0:
            model.eval()
            for prompt in SAMPLE_PROMPTS:
                reply = "".join(
                    chat_stream(model, tokenizer, [], prompt, max_new_tokens=60, temperature=0.8)
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
        if args.max_hours and elapsed_total() > args.max_hours * 3600:
            stop_reason = f"時間の保険 {args.max_hours} 時間に到達"
            break

    save("終了時")

    # heartbeat は30秒ごとにしか書かないので、最後は必ず上書きする。
    # これを忘れると、朝 watch.py を見たときに終わっているのに
    # 途中の step が残っていて、落ちたのか終わったのか分からない。
    heartbeat.write({
        "step": state.step, "total_steps": total_steps,
        "tokens_seen": state.tokens_seen, "token_budget": token_budget,
        "train_loss": round(sum(window) / len(window), 4) if window
                      else (round(last_train_loss, 4) if last_train_loss else None),
        "best_val": Heartbeat.finite(state.best_val),
        "lr": float(schedule(mx.array(state.step)).item()),
        "tok_per_sec": round(tps_history[-1]) if tps_history else 0,
        "elapsed_hours": round(elapsed_total() / 3600, 3),
        "eta_hours": 0.0,
        "peak_gb": round(guard.peak_gb, 2),
        "cpu_speed_limit": runtime.cpu_speed_limit(),
        "resumes": state.resumes,
        "finished": True,
        "stop_reason": stop_reason,
    }, force=True)

    print("=" * 66)
    print(f"終了: {stop_reason}")
    print(f"  step {state.step:,} / {total_steps:,} "
          f"({state.tokens_seen:,} トークン = 予算の {state.tokens_seen / token_budget * 100:.1f}%)")
    print(f"  経過 {elapsed_total() / 3600:.2f} 時間 (再開 {state.resumes} 回)")
    print(f"  最良 val_loss {state.best_val:.4f}")
    if len(tps_history) >= 2:
        # 1区間目はモデルのコンパイルを含むので、2区間目を基準にする。
        first, last = tps_history[1], tps_history[-1]
        print(f"  速度 {first / 1e3:.0f}k → {last / 1e3:.0f}k tok/s "
              f"({(last / first - 1) * 100:+.1f}%)")
    limit = runtime.cpu_speed_limit()
    if limit is not None and limit < 100:
        print(f"  CPU が熱で {limit}% に絞られています")
    print(f"  ピークメモリ {guard.peak_gb:.1f}GB")
    print(f"  再開用: {out_dir} / 配布用: {export_dir}")
    print("=" * 66)

    if state.step < total_steps:
        print("続きから再開するには:")
        print(f"  python src/train.py --data {args.data} "
              + (f"--tokens {token_budget}" if args.tokens else f"--steps {total_steps}")
              + " --resume auto")


if __name__ == "__main__":
    main()
