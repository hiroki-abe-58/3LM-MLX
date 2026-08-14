"""instruction masking が狙った位置だけを数えているか確かめる.

    python3 tools/check_mask.py --tokenizer data/3lm/tokenizer

## なぜ確認するのか

マスクは間違っていても学習が動く。損失が下がるので、
「学習できている」ようにしか見えない。ずれ方が2つあって、どちらも痛い。

  1 トークンぶん後ろにずれている
      返答の1文字目が数えられず、質問の最後の1トークンが数えられる。
      「質問の続きを書く」癖が残る。

  <|end|> が数えられていない
      止まり方を学ばないので、返答が終わらずに次の質問を自分で書き始める。

どちらも生成してみるまで気づかない。ここで位置を突き合わせておく。

損失の位置は「入力 x に対する正解 y」で決まる。y は1つずらしたトークン列
なので、重みも y に合わせてずらす必要がある。この対応がこの検査の本体。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sft import build_sft_arrays  # noqa: E402
from src.tokenizer import ASSISTANT, END, USER, load_tokenizer  # noqa: E402

CASES = [
    f"{USER}こんにちは{ASSISTANT}こんにちは。今日はどうしましたか。{END}",
    f"{USER}1たす1は{ASSISTANT}2です。{END}",
    f"{USER}長い質問をします。これは学習の対象外であるべき部分です。"
    f"{ASSISTANT}短い返答。{END}",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=str(ROOT / "data" / "3lm" / "tokenizer"))
    args = ap.parse_args()

    tokenizer = load_tokenizer(Path(args.tokenizer))
    tokens, weights, meta = build_sft_arrays(CASES, tokenizer)

    print("=" * 70)
    print("  instruction masking の位置あわせ")
    print(f"    語彙 {tokenizer.vocab_size:,} / "
          f"user={tokenizer.user_id} assistant={tokenizer.assistant_id} "
          f"end={tokenizer.end_id}")
    print("=" * 70)

    problems: list[str] = []

    # 1. マーカーそのものの扱い
    for name, token_id, want in (
        ("<|user|>", tokenizer.user_id, 0),
        ("<|assistant|>", tokenizer.assistant_id, 0),
        ("<|end|>", tokenizer.end_id, 1),
    ):
        got = weights[tokens == token_id]
        if got.size == 0:
            problems.append(f"{name} がトークン列に見つからない")
        elif not np.all(got == want):
            problems.append(f"{name} の重みが {set(got.tolist())} で、期待 {want} と違う")
        else:
            label = "数える" if want else "数えない"
            print(f"  {name:<14} 重み {want} ({label})  x{got.size}")

    # 2. 返答の本文が全部数えられているか / 質問が数えられていないか
    print()
    for case in CASES:
        one_tokens, one_weights, _ = build_sft_arrays([case], tokenizer)
        a_pos = int(np.flatnonzero(one_tokens == tokenizer.assistant_id)[0])
        e_pos = int(np.flatnonzero(one_tokens == tokenizer.end_id)[0])

        # 質問側 (先頭から <|assistant|> まで) は全部 0 であるべき
        if one_weights[: a_pos + 1].any():
            problems.append("質問側に重み 1 が混ざっている")
        # 返答側 (<|assistant|> の次から <|end|> まで) は全部 1 であるべき
        if not one_weights[a_pos + 1 : e_pos + 1].all():
            problems.append("返答側に重み 0 が混ざっている")

        counted = tokenizer.decode(
            one_tokens[one_weights == 1].tolist(), skip_special=False
        )
        ignored = tokenizer.decode(
            one_tokens[one_weights == 0].tolist(), skip_special=False
        )
        print(f"  数える : {counted}")
        print(f"  無視   : {ignored}")
        print()

    # 3. y と重みのずれを確認する
    #    x[t] を入れて y[t] を当てる。y[t] = tokens[t+1] なので、
    #    重みも weights[t+1] を使わなければ1つぶんずれる。
    block = 8
    i = 0
    y = tokens[i + 1 : i + 1 + block]
    w = weights[i + 1 : i + 1 + block]
    print("  y と重みの対応 (先頭8トークン):")
    for token_id, weight in zip(y.tolist(), w.tolist(), strict=True):
        piece = tokenizer.decode([token_id], skip_special=False)
        print(f"    {'数える' if weight else '無視  '}  {piece!r}")

    # 4. 割合が妥当か
    ratio = meta["target_tokens"] / meta["tokens"]
    print()
    print(f"  損失を数える割合: {ratio:.1%} "
          f"({meta['target_tokens']} / {meta['tokens']} トークン)")
    if not 0.1 < ratio < 0.95:
        problems.append(f"数える割合 {ratio:.1%} が極端。マスクがほぼ全部か、ほぼ空になっている")

    print("=" * 70)
    if problems:
        for p in problems:
            print(f"  [失敗] {p}")
        raise SystemExit(1)
    print("  マスクは <|assistant|> より後ろと <|end|> だけを数えている。")


if __name__ == "__main__":
    main()
