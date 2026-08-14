"""公開する前に、機密が混ざっていないかを機械的に調べる.

    python3 tools/secret_scan.py                    # 作業ツリーを見る
    python3 tools/secret_scan.py --dir checkpoints/sft-final   # 上げる直前のものだけ
    python3 tools/secret_scan.py --git              # 過去のコミットも見る

リポジトリと Hugging Face を public にする前に必ず通す。
一度 public にした内容は、あとで消しても取り消せない
(クローンされている / キャッシュに載っている)。

見るのは5つ。

  1. トークンや鍵に見える文字列 (hf_ / sk- / ghp_ / Bearer など)
  2. 置いてはいけないファイル (.env / *.pem / id_rsa / credentials)
  3. ローカルの絶対パス (/Users/<名前>/... が入っていると本名が漏れる)
  4. メールアドレス
  5. 過去のコミット (--git。今のツリーが綺麗でも履歴に残っていることがある)

3 を入れているのは、runs/ のログや config に絶対パスが残りやすいから。
学習ログには --data や --out がそのまま書かれる。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 実際に漏れる形のものだけを拾う。"token" という単語自体は
# コードやコメントに山ほど出てくるので、そこは見ない。
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Hugging Face トークン", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("OpenAI 風の鍵", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")),
    ("GitHub トークン", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("AWS アクセスキー", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Google API キー", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("Slack トークン", re.compile(r"\bxox[baprs]-[0-9A-Za-z\-]{10,}")),
    ("秘密鍵ファイルの中身", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("Authorization ヘッダ", re.compile(r"Authorization\s*[:=]\s*['\"]?(Bearer|Basic)\s+\S+")),
    ("代入された秘密", re.compile(
        r"(?i)\b(api_?key|secret|password|passwd|access_token)\b\s*[:=]\s*"
        r"['\"][^'\"\s{}$]{12,}['\"]"
    )),
    ("ローカルの絶対パス", re.compile(r"/(Users|home)/[A-Za-z0-9._\-]+/")),
    ("メールアドレス", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
)

FORBIDDEN_NAMES = (
    ".env", ".env.local", ".netrc", "credentials", "credentials.json",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", ".htpasswd",
)
FORBIDDEN_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore")

SKIP_DIRS = {".git", "__pycache__", ".ruff_cache", ".venv", "node_modules", ".pytest_cache"}
# 中身を文字列として見ても意味が無い / 巨大なもの。
SKIP_SUFFIXES = {
    ".safetensors", ".bin", ".npy", ".npz", ".model", ".png", ".jpg", ".jpeg",
    ".gif", ".webp", ".zip", ".gz", ".parquet", ".mp4", ".pdf", ".woff", ".woff2",
}
MAX_BYTES = 8 << 20


def iter_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if SKIP_DIRS & set(path.relative_to(root).parts):
            continue
        files.append(path)
    return sorted(files)


def scan_text(path: Path, root: Path) -> list[str]:
    findings = []
    rel = path.relative_to(root)
    if path.suffix.lower() in SKIP_SUFFIXES or path.stat().st_size > MAX_BYTES:
        return findings
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return findings
    for label, pattern in PATTERNS:
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            snippet = match.group(0)
            if len(snippet) > 70:
                snippet = snippet[:67] + "…"
            findings.append(f"{rel}:{line}  [{label}]  {snippet}")
    return findings


def scan_names(root: Path) -> list[str]:
    findings = []
    for path in iter_files(root):
        rel = path.relative_to(root)
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append(f"{rel}  [置いてはいけないファイル]")
    return findings


def scan_git_history() -> list[str]:
    """過去のコミットに秘密が入っていないか見る.

    今のファイルを消しても、コミット履歴には残る。GitHub に上げた時点で
    誰でも git log -p で読める。
    """
    findings = []
    try:
        diff = subprocess.run(
            ["git", "log", "-p", "--all", "--no-color"],
            capture_output=True, text=True, cwd=ROOT, timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [f"git の履歴を読めませんでした: {exc}"]
    if diff.returncode != 0:
        return ["git リポジトリではないか、コミットがありません"]
    # 履歴の中では「絶対パス」と「メール」は必ず出る (コミッタ情報)。
    # 秘密そのものだけを見る。
    for label, pattern in PATTERNS:
        if label in ("ローカルの絶対パス", "メールアドレス"):
            continue
        for match in pattern.finditer(diff.stdout):
            snippet = match.group(0)[:70]
            findings.append(f"git履歴  [{label}]  {snippet}")
    return findings


def main() -> None:
    ap = argparse.ArgumentParser(description="公開前に機密の混入を調べる")
    ap.add_argument("--dir", default=str(ROOT), help="調べる場所 (既定: リポジトリ全体)")
    ap.add_argument("--git", action="store_true", help="過去のコミットも調べる")
    args = ap.parse_args()

    root = Path(args.dir).resolve()
    print("=" * 70)
    print(f"  機密の検査: {root}")
    print("=" * 70)

    files = iter_files(root)
    findings: list[str] = []
    for path in files:
        findings += scan_text(path, root)
    findings += scan_names(root)
    print(f"  調べたファイル: {len(files):,}")

    if args.git:
        print("  git の履歴も調べます (時間がかかります)")
        findings += scan_git_history()

    # 同じものが何度も出るので畳む。
    unique = sorted(set(findings))
    print()
    if unique:
        print(f"  {len(unique)} 件見つかりました:")
        for line in unique[:80]:
            print(f"    {line}")
        if len(unique) > 80:
            print(f"    … 他 {len(unique) - 80} 件")
        print()
        print("  対応:")
        print("    - 鍵やトークンなら、まず**その鍵を無効化**してから消す")
        print("      (消すだけでは、既に見られていた場合に間に合わない)")
        print("    - 絶対パスやメールは、ログや config を書き換えるか公開対象から外す")
        print("    - git 履歴に入っていた場合、消すには履歴の書き換えが必要")
        sys.exit(1)
    print("  見つかりませんでした。公開して問題ない状態です。")


if __name__ == "__main__":
    main()
