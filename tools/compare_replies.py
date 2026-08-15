"""同じ入力を前作と今作に与えて、返答を並べて記録する.

記事に載せる比較例を**手で選ばない**ための道具。
同じ prompt・同じ seed・同じサンプリング条件で両方を回し、
出てきたものをそのまま json に落とす。

2種類を測る。

    chat    : 対話形式。SFT で合わせた土俵
    raw     : 素の文章の続き。事前学習の厚みがそのまま出る土俵

**raw のほうが差が大きい**というのが今回の結論なので、
対話だけを見せると実態を取り違える。

使い方:
    python tools/compare_replies.py --baseline ../2LM-MLX-GAL/checkpoints/final
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402

from src.generate import (  # noqa: E402
    chat_stream,
    decode_incrementally,
    generate_stream,
    load_bundle,
)

ROOT = Path(__file__).resolve().parents[1]

CHAT_PROMPTS = [
    "おすすめの本を3つ挙げてください",
    "短く自己紹介してください",
    "日本の首都はどこですか",
    "疲れました",
]

# 事前学習コーパス (ウェブ + 青空文庫) にありそうな書き出し。
# 対話データには出てこない語彙と文体を選ぶ。
RAW_PROMPTS = [
    "梅雨が明けると、庭の紫陽花は",
    "この製品の保証期間は購入日から",
    "駅前の再開発について、市は",
]

# 口調を乗せた版どうしの比較。その3の記事に載せた画面と**同じ質問**にする。
# 別の質問で比べると「質問が易しくなっただけ」の可能性を潰せない。
GAL_PROMPTS = [
    "おはよう",
    "今日バイト行きたくない",
    "お金貯めたい",
    "AIって将来どうなると思う？",
    "日本の首都はどこ？",
    "猫について教えて",
]


def collect(models, groups, sampling, seed) -> list[dict]:
    """モデル × 質問群を総当たりして記録を返す.

    種は「何番目の質問か」だけで決まるようにする。モデルごとに違う種を
    使うと、差がモデルのせいなのか運のせいなのか分けられなくなる。
    """
    records: list[dict] = []
    for label, ckpt in models:
        model, tokenizer = load_bundle(ckpt)
        for offset, (mode, prompts) in enumerate(groups):
            for i, prompt in enumerate(prompts):
                mx.random.seed(seed + offset * 100 + i)
                if mode == "raw":
                    ids = generate_stream(model, tokenizer.encode(prompt), **sampling)
                    reply = "".join(decode_incrementally(tokenizer, ids))
                else:
                    reply = "".join(chat_stream(model, tokenizer, [], prompt, **sampling))
                records.append(
                    {"mode": mode, "model": label, "prompt": prompt, "reply": reply}
                )
                print(f"  [{mode:<4}] {label}  {prompt}\n         {reply}")
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="前作 (2LM) の重み")
    ap.add_argument("--new", default="")
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--gal",
        action="store_true",
        help="口調を乗せた版どうしを比べる (既定は対話調整版どうし)",
    )
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--max-new-tokens", type=int, default=90)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    if args.gal:
        default_new = ROOT / "checkpoints" / "gal-final"
        default_out = ROOT / "runs" / "3lm" / "gal_reply_compare.json"
        groups = [("gal", GAL_PROMPTS)]
    else:
        default_new = ROOT / "checkpoints" / "sft-final"
        default_out = ROOT / "runs" / "3lm" / "reply_compare.json"
        groups = [("chat", CHAT_PROMPTS), ("raw", RAW_PROMPTS)]

    models = [
        ("2LM 13.81M", Path(args.baseline).expanduser()),
        ("3LM 35.66M", Path(args.new).expanduser() if args.new else default_new),
    ]
    sampling = dict(
        temperature=args.temperature, top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )

    records = collect(models, groups, sampling, args.seed)

    out = Path(args.out) if args.out else default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"sampling": sampling, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
