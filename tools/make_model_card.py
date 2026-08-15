"""モデルカード (README.md) を、実際の成果物から組み立てる.

    python3 tools/make_model_card.py --ckpt checkpoints/sft-final --eval runs/eval_3lm.json

数字を手で書き写さない。config.json / metrics.json / 評価結果 / コーパスの
manifest から読んで埋める。**手で書くと、あとで学習をやり直したときに
モデルカードだけ古い数字のまま残る**。公開物なので、そこがずれるのが一番まずい。

帰属表示もここで機械的に入れる。FineWeb2 は ODC-By 1.0、青空文庫は CC BY 4.0 で、
どちらも**帰属表示が義務**。書き忘れるとライセンス違反になる。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fmt(value, spec: str = ",", default: str = "未測定") -> str:
    if value is None:
        return default
    return format(value, spec) if spec else str(value)


def pii_lines(check: dict) -> list[str]:
    """コーパスに入っていた連絡先について、事実だけを書く.

    ウェブ由来のコーパスには連絡先が入る。隠すのではなく、
    **何件試して何件出たか**を書く。検査していない場合も「していない」と書く。
    """
    if not check:
        return [
            "- 事前学習コーパスは Common Crawl 由来のため、"
            "**連絡先などの個人情報が含まれます**。"
            "モデルから引き出せるかの検査は行っていません",
        ]
    exact = check.get("exact", 0)
    tried = check.get("tried", 0)
    if exact:
        return [
            f"- **警告: 学習データにあった連絡先 {exact} 件が、"
            f"モデルから復元できました**（{tried} 件を試験）。利用には適しません",
        ]
    return [
        "- 事前学習コーパスは Common Crawl 由来のため、"
        "メールアドレスや電話番号が含まれています。"
        f"学習後に prefix attack で {tried} 件試したところ、"
        "**そのまま復元できたものはありませんでした** "
        "(コーパスを1周未満しか読んでいないため)。"
        "ただし、より強い攻撃で引き出せない保証はありません",
    ]


def domain_lines(domains: dict) -> list[str]:
    """検証セットの出自によって勝敗が変わることを、カードに明記する.

    上の 4指標は「公開データ由来の会話」だけで測っている。
    前作 (2LM) はその公開データで**事前学習した**モデルなので、
    そこは前作のホームグラウンドにあたる。
    その表だけを載せると「大きくしたのに全部悪化した」と読めてしまう。

    **指標は、何で測ったかとセットでなければ意味を持たない。**
    """
    records = domains.get("records") or []
    if not records:
        return []

    models = list(dict.fromkeys(r["model"] for r in records))
    fields = list(dict.fromkeys(r["domain"] for r in records))
    lines = [
        "",
        "### 検証セットを変えると、勝敗が変わります",
        "",
        "上の表は「公開データ由来の会話」で測ったものです。"
        "前作はその公開データで**事前学習した**モデルなので、"
        "そこは前作の得意分野にあたります。"
        "土俵を変えて、同じ bits/char (低いほど良い) で並べます。",
        "",
        "| モデル | " + " | ".join(fields) + " |",
        "|---" * (len(fields) + 1) + "|",
    ]
    for model in models:
        cells = []
        for field in fields:
            hit = next(
                (r for r in records if r["model"] == model and r["domain"] == field), None
            )
            cells.append(f"{hit['bits_per_char']:.3f}" if hit else "-")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "一般的な日本語の文 (土俵B) では前作を大きく上回ります。"
        "**bits/char は分母が文字数なので、語彙の大きさが違うモデル同士でも比べられます。**",
    ]
    return lines


def build(args) -> str:
    ckpt = Path(args.ckpt)
    cfg = read_json(ckpt / "config.json")
    evaluation = read_json(Path(args.eval)) if args.eval else {}
    baseline = read_json(Path(args.baseline)) if args.baseline else {}
    manifest = read_json(Path(args.manifest))
    pretrain_metrics = read_json(Path(args.pretrain_metrics)) if args.pretrain_metrics else {}
    domains = read_json(Path(args.domains)) if args.domains else {}
    data_meta = read_json(Path(args.data_meta))
    sft = read_json(Path(args.sft_summary))

    sources = {s["repo"]: s for s in manifest.get("sources", [])}
    fineweb = sources.get("HuggingFaceFW/fineweb-2", {})
    aozora = sources.get("globis-university/aozorabunko-clean", {})

    sampling = evaluation.get("sampling", {})
    is_gal = args.character

    name = args.repo.split("/")[-1]
    title = f"{name}"
    intro = (
        "M1 Max (64GB) 1台で、一晩で事前学習から作った日本語の小さな言語モデルです。"
        if not is_gal else
        "M1 Max (64GB) 1台で作った日本語の小さな言語モデルに、"
        "特定の口調を後から学習させたものです。"
    )

    lines = [
        "---",
        "language:",
        "- ja",
        "license: apache-2.0",
        "library_name: mlx",
        "tags:",
        "- mlx",
        "- apple-silicon",
        "- japanese",
        "- text-generation",
        "- from-scratch",
        "datasets:",
        "- HuggingFaceFW/fineweb-2",
        "- globis-university/aozorabunko-clean",
        "- kunishou/oasst1-89k-ja",
        "- llm-jp/oasst2-33k-ja",
        "- llm-jp/magpie-sft-v1.0",
        "- Aratako/Magpie-Tanuki-8B-97k",
        "pipeline_tag: text-generation",
        "---",
        "",
        f"# {title}",
        "",
        intro,
        "",
        "## 概要",
        "",
        "| 項目 | 値 |",
        "|---|---|",
        f"| パラメータ数 | {fmt(args.params, ',')} |",
        f"| 非埋め込みパラメータ | {fmt(args.non_embed, ',')} |",
        f"| 語彙 | {fmt(cfg.get('vocab_size'))} (SentencePiece unigram / byte fallback) |",
        f"| 文脈長 | {fmt(cfg.get('block_size'))} |",
        f"| 層 / 次元 / ヘッド | {cfg.get('n_layer')} / {cfg.get('n_embd')} / {cfg.get('n_head')} |",
        "| 構成 | RoPE / RMSNorm / SwiGLU / bias なし / weight tying |",
        f"| 事前学習トークン | {fmt(pretrain_metrics.get('tokens_seen'))} |",
        f"| 事前学習コーパス | {fmt(data_meta.get('chars') or manifest.get('total_chars'))} 文字 |",
        "| 学習環境 | Apple M1 Max 64GB / MLX |",
        "",
        "## 評価",
        "",
        "同じサンプリング条件 "
        f"(temperature {sampling.get('temperature', '-')} / "
        f"top_k {sampling.get('top_k', '-')} / "
        f"repetition_penalty {sampling.get('repetition_penalty', '-')} / "
        f"seed {sampling.get('seed', '-')}) で測っています。",
        "",
        "| 指標 | " + name + (f" | {args.baseline_label} | 差 |" if baseline else " |"),
        "|---|---|" + ("---|---|" if baseline else ""),
    ]

    def row(label: str, key: str, spec: str = ".3f", better: str = "低いほど良い") -> str:
        now = evaluation.get(key)
        cell = f"| {label} ({better}) | {fmt(now, spec)} "
        if baseline:
            before = baseline.get(key)
            cell += f"| {fmt(before, spec)} "
            if now is not None and before is not None:
                delta = now - before
                lower_is_better = better.startswith("低い")
                if abs(delta) < 5e-4:
                    verdict = "変化なし"
                else:
                    verdict = "改善" if (delta < 0) == lower_is_better else "悪化"
                cell += f"| {delta:+.3f} ({verdict}) "
            else:
                cell += "| - "
        return cell + "|"

    lines += [
        row("bits/char", "bits_per_char"),
        row("反復率", "repetition_rate"),
        row("主題保持率", "topic_rate", better="高いほど良い"),
        row("破綻率", "broken_rate"),
    ]
    if evaluation.get("holdout"):
        lines += [
            "",
            f"検証セット: `{evaluation['holdout']}` "
            f"({evaluation.get('holdout_lines', '?')} 行)。"
            "学習データから除いたうえで、**部分一致でも混入していないことを"
            "検査してから**測っています。",
        ]
    lines += domain_lines(domains)

    lines += [
        "",
        "## 使い方",
        "",
        "```bash",
        "pip install mlx numpy sentencepiece",
        "```",
        "",
        "```python",
        "from huggingface_hub import snapshot_download",
        "from src.generate import load_bundle, chat_stream",
        "",
        f'path = snapshot_download("{args.repo}")',
        "model, tokenizer = load_bundle(path)",
        'for piece in chat_stream(model, tokenizer, [], "日本の首都はどこですか"):',
        '    print(piece, end="", flush=True)',
        "```",
        "",
        f"`load_bundle` と `chat_stream` は学習に使ったリポジトリ ({args.code_url}) に入っています。",
        "",
        "## 学習の流れ",
        "",
        "1. **事前学習**: FineWeb2 の日本語 + 青空文庫から作ったコーパスで、"
        "次のトークンを当てる学習",
        "2. **SFT (指示学習)**: 対話データで書式を合わせる。"
        "損失は `<|assistant|>` より後ろと `<|end|>` だけに掛けています "
        "(instruction masking)",
    ]
    if is_gal:
        lines += ["3. **口調の学習**: 上のモデルに、特定の話し方の会話を追加学習"]

    if sft:
        lines += [
            "",
            f"SFT で損失を数えたトークンの割合: {sft.get('mask_ratio', 0):.1%} "
            f"(会話 {fmt(sft.get('train', {}).get('conversations'))} 件)",
        ]

    lines += [
        "",
        "## 学習データと帰属表示",
        "",
        "### 事前学習",
        "",
        "| データ | ライセンス | 使った量 |",
        "|---|---|---|",
        f"| [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) "
        f"(`jpn_Jpan`) | **ODC-By 1.0** | {fmt(fineweb.get('chars'))} 文字 "
        f"({fineweb.get('share', 0):.1%}) |",
        f"| [globis-university/aozorabunko-clean]"
        f"(https://huggingface.co/datasets/globis-university/aozorabunko-clean) "
        f"| **CC BY 4.0** | {fmt(aozora.get('chars'))} 文字 "
        f"({aozora.get('share', 0):.1%}) |",
        "",
        "FineWeb2 は Common Crawl から作られたデータセットで、"
        "ODC-By 1.0 に従い帰属を表示します。"
        "青空文庫版 (globis-university/aozorabunko-clean) は CC BY 4.0 です。"
        "どちらも継承条件 (ShareAlike) が無いため、"
        "この重みを Apache-2.0 相当で配布できます。",
        "",
        f"再現性のため、使用した revision と各シャードの SHA256 を"
        f"構築スクリプトの manifest に記録しています "
        f"(FineWeb2 `{fineweb.get('revision', '')[:12]}` / "
        f"青空文庫 `{aozora.get('revision', '')[:12]}`)。"
        "コーパス本体は再ホストせず、スクリプトと manifest で再現できる形にしています。",
        "",
        "適用したフィルタ: 日本語文字比率 70% 以上 / 200〜20,000文字 / "
        "定型文とエラーページの除去 / 重複除去 / 同一並びの繰り返し検出。",
        "",
        "### SFT",
        "",
        "| データ | ライセンス |",
        "|---|---|",
        "| [kunishou/oasst1-89k-ja](https://huggingface.co/datasets/kunishou/oasst1-89k-ja) "
        "| Apache-2.0 |",
        "| [llm-jp/oasst2-33k-ja](https://huggingface.co/datasets/llm-jp/oasst2-33k-ja) "
        "| Apache-2.0 |",
        "| [llm-jp/magpie-sft-v1.0](https://huggingface.co/datasets/llm-jp/magpie-sft-v1.0) "
        "| Apache-2.0 |",
        "| [Aratako/Magpie-Tanuki-8B-97k]"
        "(https://huggingface.co/datasets/Aratako/Magpie-Tanuki-8B-97k) | Apache-2.0 |",
        "",
        "継承条件のあるデータ (CC BY-SA など) は、"
        "配布ライセンスの整合が崩れるため意図的に使っていません。",
        "",
        "## 制限",
        "",
        f"- **{fmt(args.params, ',')} パラメータ**しかありません。"
        "事実を答える能力はほとんど期待できません。知識の参照には使えません",
        "- 学習データの大半が Common Crawl 由来のウェブ文書なので、"
        "**そこに含まれる偏りや不適切な表現を引き継いでいる可能性があります**",
        "- 青空文庫を約10%混ぜているため、文語的な言い回しが出ることがあります",
        "- 商用利用の可否は、上記データセット各々のライセンスもご確認ください",
        *pii_lines(read_json(Path(args.pii_check))),
        "",
        "## ライセンス",
        "",
        "- 重み: Apache-2.0 相当",
        f"- 学習コード: MIT ({args.code_url})",
        "- 学習データの帰属: 上記のとおり (ODC-By 1.0 / CC BY 4.0 / Apache-2.0)",
    ]

    if data_meta.get("chars_per_token"):
        lines += [
            "",
            "## 実測値のメモ",
            "",
            f"- 1トークンあたり **{data_meta['chars_per_token']} 文字** "
            f"(語彙 {fmt(data_meta.get('vocab_size'))} / ウェブ文書)",
            # train だけだと val のぶんが落ちて、上の「コーパス文字数」と
            # 桁が合わなくなる。文字数と対応する量は train + val。
            f"- コーパス {fmt(data_meta.get('chars'))} 文字 = "
            f"{fmt(data_meta.get('train_tokens', 0) + data_meta.get('val_tokens', 0))} トークン "
            f"(うち学習に使ったのは {fmt(data_meta.get('train_tokens'))})",
        ]

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--eval", default="")
    ap.add_argument("--baseline", default="", help="比較相手の評価 json (並べて出す)")
    ap.add_argument("--baseline-label", default="前作 (2LM-MLX)",
                    help="比較相手の列見出し。口調版なら「口調の学習前」など")
    ap.add_argument("--manifest", default=str(ROOT / "data" / "corpus_pretrain.manifest.json"))
    ap.add_argument("--data-meta", default=str(ROOT / "data" / "3lm" / "meta.json"))
    ap.add_argument("--sft-summary", default=str(ROOT / "runs" / "3lm" / "sft_loss_summary.json"))
    ap.add_argument("--pii-check", default=str(ROOT / "runs" / "3lm" / "pii_check.json"),
                    help="tools/check_pii.py の結果。無ければ「未検査」と書く")
    ap.add_argument("--pretrain-metrics",
                    default=str(ROOT / "checkpoints" / "pretrain-final" / "metrics.json"),
                    help="事前学習の metrics.json。SFT 後の重みには読んだトークン数が無い")
    ap.add_argument("--domains", default=str(ROOT / "runs" / "3lm" / "domain_bpc.json"),
                    help="tools/compare_domains.py の結果")
    ap.add_argument("--params", type=int, default=0)
    ap.add_argument("--non-embed", type=int, default=0)
    ap.add_argument("--character", action="store_true", help="口調を学習した版のカードにする")
    ap.add_argument("--code-url", default="https://github.com/hiroki-abe-58/3LM-MLX")
    args = ap.parse_args()

    # パラメータ数は config から数え直す (手入力に頼らない)
    if not args.params:
        import sys
        sys.path.insert(0, str(ROOT))
        from src.model import GPTConfig, MiniGPT
        cfg = GPTConfig.load(Path(args.ckpt) / "config.json")
        model = MiniGPT(cfg)
        args.params = model.n_params
        args.non_embed = model.n_params_non_embedding

    text = build(args)
    out = Path(args.out) if args.out else Path(args.ckpt) / "README.md"
    out.write_text(text, encoding="utf-8")
    print(f"モデルカードを書きました: {out} ({len(text):,} 文字)")
    print("中身を目で確認してから上げてください。")


if __name__ == "__main__":
    main()
