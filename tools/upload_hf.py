"""学習した重みを Hugging Face に上げる.

    python3 tools/upload_hf.py --ckpt checkpoints/sft-final --repo GeneLab/3LM-MLX --dry-run
    python3 tools/upload_hf.py --ckpt checkpoints/sft-final --repo GeneLab/3LM-MLX

## 許可リスト方式にする理由

upload_folder には ignore_patterns (除外リスト) もあるが、こちらは
**allow_patterns (許可リスト) だけ**を使う。

除外リストは「書き忘れたものが上がる」設計になっている。あとで
`optimizer.safetensors` や `runs/` を置いたときに、除外リストを
更新し忘れれば黙って公開される。許可リストなら「書いたものだけ上がる」ので、
書き忘れの結果は「上がらない」になる。事故の向きが逆になる。

## トークンの扱い

認証トークンは `huggingface-cli login` で
`~/.cache/huggingface/token` に入る。**リポジトリの外**にあるので、
コードにもファイルにも書かない。ここでは引数でも受け取らない。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 上げるものだけを列挙する。ここに無いものは上がらない。
ALLOW_PATTERNS = [
    "model.safetensors",
    "config.json",
    "tokenizer.model",
    "tokenizer.json",
    "metrics.json",
    "README.md",
    "NOTICE",
    "LICENSE",
]


def check_secrets(path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "secret_scan.py"), "--dir", str(path)],
        capture_output=True, text=True,
    )
    print(result.stdout.rstrip())
    if result.returncode != 0:
        raise SystemExit(
            "機密の検査に落ちました。中身を直してから上げ直してください。"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="重みを Hugging Face に上げる")
    ap.add_argument("--ckpt", required=True, help="上げるディレクトリ")
    ap.add_argument("--repo", required=True, help="例: GeneLab/3LM-MLX")
    ap.add_argument("--card", default="", help="モデルカード (README.md) の元ファイル")
    ap.add_argument("--message", default="Add 3LM weights")
    ap.add_argument("--private", action="store_true", default=True,
                    help="private で作る (既定)。public 化は検査後に手で行う")
    ap.add_argument("--dry-run", action="store_true", help="上げずに何が上がるかだけ見る")
    ap.add_argument("--delete", action="append", default=[],
                    help="リポジトリ側から消すファイル (複数指定可)")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.is_dir():
        raise SystemExit(f"{ckpt} がありません")

    if args.card:
        card = Path(args.card)
        if not card.exists():
            raise SystemExit(f"{card} がありません")
        (ckpt / "README.md").write_text(card.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"モデルカードを置きました: {ckpt / 'README.md'}")

    print("=" * 70)
    print(f"  上げ先: {args.repo} ({'private' if args.private else 'public'})")
    print(f"  元    : {ckpt}")
    print("=" * 70)

    present = []
    for pattern in ALLOW_PATTERNS:
        path = ckpt / pattern
        if path.exists():
            present.append((pattern, path.stat().st_size))
    skipped = sorted(
        p.name for p in ckpt.iterdir()
        if p.is_file() and p.name not in {n for n, _ in present}
    )

    print("  上げるもの (許可リストに載っているもの):")
    for name, size in present:
        print(f"    {name:<24} {size / 2**20:>8.2f} MiB")
    if skipped:
        print("  上げないもの (許可リストに無い):")
        for name in skipped:
            print(f"    {name}")
    total = sum(size for _, size in present) / 2**20
    print(f"  合計 {total:.1f} MiB")

    if not any(name == "model.safetensors" for name, _ in present):
        raise SystemExit("model.safetensors がありません")
    if not any(name == "README.md" for name, _ in present):
        print("\n  ※ README.md (モデルカード) がありません。--card で指定してください。")

    print()
    check_secrets(ckpt)

    if args.dry_run:
        print("\n--dry-run なので、ここで終わります。")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()
    print(f"\n  ログイン中: {who.get('name')}")

    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

    for name in args.delete:
        try:
            api.delete_file(path_in_repo=name, repo_id=args.repo, repo_type="model")
            print(f"  削除しました: {name}")
        except Exception as exc:  # noqa: BLE001 - 無ければ無いで良い
            print(f"  削除できませんでした ({name}): {type(exc).__name__}")

    api.upload_folder(
        folder_path=str(ckpt),
        repo_id=args.repo,
        repo_type="model",
        allow_patterns=ALLOW_PATTERNS,
        commit_message=args.message,
    )
    print(f"\n  完了: https://huggingface.co/{args.repo}")
    print("  中身を目で確認してから、Settings で public に切り替えてください。")

    listed = api.list_repo_files(args.repo, repo_type="model")
    print("\n  リポジトリにあるファイル:")
    for name in sorted(listed):
        mark = "  " if name in {n for n, _ in present} or name == ".gitattributes" else " ←想定外"
        print(f"    {name}{mark}")
    (ROOT / "runs" / "3lm" / "hf_upload.json").write_text(
        json.dumps({"repo": args.repo, "files": sorted(listed), "private": args.private},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
