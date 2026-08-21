"""本地仓库 ⇄ 主办方 JupyterHub 的单向同步管道。

用途：把本地改动推到云端 `~/Quant_trade`，把云端跑出来的结果拉回本地 `outputs/cloud/`。
云端是「第二台开发机」，代码的唯一真值源始终是本地（见 CLAUDE.md 的可信度顺序）。

## 为什么只用标准库

仓库当前的第三方依赖面只有 numpy/pandas/pyarrow/sklearn/scipy/lightgbm/pytest。
一个运维脚本不该把它扩大，所以这里只用 `urllib.request` 打 Jupyter Contents API。

## 为什么按文件增量传，而不是打 tar 包

云端存一份 `.sync_manifest.sha256`（上次推送时的逐文件哈希）。push 时先取回这份清单、
和本地 `git ls-files` 的哈希求差集，**只 PUT 变动的文件**。改三行代码就只传三个文件，
既快，也不需要远端有解包权限。

## Token

从 `$JHUB_TOKEN` 读，回落到 `~/.config/quant_jhub/token`（建议 chmod 600）。
token 不写进本文件，也不该进版本控制。

## 用法

    python scripts/cloud_sync.py status              # 只比对，列出差异
    python scripts/cloud_sync.py push [--dry-run]    # 增量上传
    python scripts/cloud_sync.py pull                # 云端 outputs/experiments → 本地 outputs/cloud
    python scripts/cloud_sync.py log <名字> [--tail N]  # 看云端 outputs/logs/<名字> 的尾部
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE = "https://jupyter.xhth.cn/user/ustc064"
REMOTE_ROOT = "Quant_trade"
# 不带点开头：Jupyter Contents API 默认 allow_hidden=False，PUT 任何点开头的
# 文件或目录都会 400（实测 body: "Cannot create file or directory"）。
MANIFEST_NAME = "sync_manifest.sha256"
TOKEN_FILE = Path.home() / ".config" / "quant_jhub" / "token"


# ---------------------------------------------------------------- Contents API


def read_token() -> str:
    token = os.environ.get("JHUB_TOKEN")
    if token:
        return token.strip()
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text(encoding="utf-8").strip()
    raise SystemExit(
        f"找不到 token：设置 $JHUB_TOKEN，或写入 {TOKEN_FILE}（建议 chmod 600）")


class Contents:
    """Jupyter Contents API 的最小客户端。路径都相对于云端 home。"""

    def __init__(self, base: str, token: str) -> None:
        self.base = base.rstrip("/")
        self.headers = {"Authorization": f"token {token}"}

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        url = f"{self.base}/api/contents/{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        headers = dict(self.headers)
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            # 裸的 "HTTP Error 400: Bad Request" 什么都说明不了；服务端的 message
            # 才是有用的那一半（例如 allow_hidden 拒绝点开头的路径）。
            error.detail = error.read().decode(errors="replace")[:500]
            raise
        return json.loads(body) if body else {}

    def get_bytes(self, path: str) -> bytes | None:
        """读取一个文件；不存在返回 None（而不是抛异常）。"""
        try:
            meta = self._request("GET", f"{path}?format=base64&type=file")
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise
        return base64.b64decode(meta["content"])

    def put_bytes(self, path: str, raw: bytes) -> None:
        self._request("PUT", path, {"type": "file", "format": "base64",
                                    "content": base64.b64encode(raw).decode()})

    def listdir(self, path: str) -> list[dict]:
        try:
            return self._request("GET", f"{path}?content=1").get("content", [])
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return []
            raise

    def mkdirs(self, path: str) -> None:
        """Contents API 没有 mkdir -p，逐级建；已存在会 409，忽略。"""
        parts = [p for p in path.split("/") if p]
        for index in range(1, len(parts) + 1):
            prefix = "/".join(parts[:index])
            try:
                self._request("PUT", prefix, {"type": "directory"})
            except urllib.error.HTTPError as error:
                if error.code not in (400, 409):
                    raise


# ---------------------------------------------------------------- 清单


def local_manifest() -> dict[str, str]:
    """工作树里该同步的文件 → sha256。

    用 `--cached --others --exclude-standard` 而不是光 `git ls-files`：新写的文件
    还没 commit 也要能推上去（这是开发同步工具，不是发布工具），同时 `--exclude-standard`
    保证 .gitignore 口径一致 —— data/、.venv/、outputs/cache/ 不会被卷进来。
    """
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=_REPO_ROOT, capture_output=True, check=True).stdout
    manifest = {}
    for name in listing.split(b"\0"):
        if not name:
            continue
        relative = name.decode()
        path = _REPO_ROOT / relative
        if not path.is_file():          # 删掉但还没 git rm 的，跳过
            continue
        if is_hidden(relative):         # Contents API 传不了，见 skipped_hidden()
            continue
        manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def is_hidden(relative: str) -> bool:
    return any(part.startswith(".") for part in Path(relative).parts)


def skipped_hidden() -> list[str]:
    """同步范围内、但因 allow_hidden=False 传不上去的路径（目前只有 .gitignore）。

    这些文件首轮已经随 tar 包落到云端了；之后要更新只能走 JupyterLab 的终端。
    列出来是为了让"零差异"这句话诚实——它指的是可同步范围内的零差异。
    """
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=_REPO_ROOT, capture_output=True, check=True).stdout
    return sorted(n.decode() for n in listing.split(b"\0")
                  if n and is_hidden(n.decode()))


def parse_manifest(raw: bytes | None) -> dict[str, str]:
    if not raw:
        return {}
    manifest = {}
    for line in raw.decode().splitlines():
        if line.strip():
            digest, _, relative = line.partition("  ")
            manifest[relative] = digest
    return manifest


def render_manifest(manifest: dict[str, str]) -> bytes:
    lines = [f"{manifest[k]}  {k}" for k in sorted(manifest)]
    return ("\n".join(lines) + "\n").encode()


def diff(local: dict[str, str], remote: dict[str, str]) -> tuple[list[str], list[str]]:
    changed = sorted(k for k, v in local.items() if remote.get(k) != v)
    # 本地已删除的：只报告，不替用户删云端文件（CLAUDE.md 第 1 条红线）。
    vanished = sorted(set(remote) - set(local))
    return changed, vanished


# ---------------------------------------------------------------- 子命令


def cmd_status(api: Contents, args: argparse.Namespace) -> int:
    local = local_manifest()
    remote = parse_manifest(api.get_bytes(f"{REMOTE_ROOT}/{MANIFEST_NAME}"))
    if not remote:
        print(f"云端还没有 {MANIFEST_NAME}（首次 push 会建立）；本地 {len(local)} 个受控文件")
        return 0
    changed, vanished = diff(local, remote)
    for relative in skipped_hidden():
        print(f"  跳过（Contents API 传不了点开头的路径）  {relative}")
    if not changed and not vanished:
        print(f"零差异：{len(local)} 个可同步文件本地与云端一致")
        return 0
    for relative in changed:
        print(f"  变动  {relative}")
    for relative in vanished:
        print(f"  本地已删（云端仍在，不自动删）  {relative}")
    print(f"合计：{len(changed)} 个待上传，{len(vanished)} 个本地已删")
    return 0


def cmd_push(api: Contents, args: argparse.Namespace) -> int:
    local = local_manifest()
    remote = parse_manifest(api.get_bytes(f"{REMOTE_ROOT}/{MANIFEST_NAME}"))
    changed, vanished = diff(local, remote)

    for relative in skipped_hidden():
        print(f"  跳过（Contents API 传不了点开头的路径）  {relative}")
    for relative in vanished:
        print(f"  ! 本地已删，云端保留（需要清理请自行处理）：{relative}")
    if not changed:
        print(f"零差异，无需上传（{len(local)} 个文件）")
        return 0
    if args.dry_run:
        for relative in changed:
            print(f"  会上传  {relative}")
        print(f"--dry-run：{len(changed)} 个文件未实际上传")
        return 0

    made = set()
    for index, relative in enumerate(changed, 1):
        parent = str(Path(relative).parent)
        if parent not in (".", "") and parent not in made:
            api.mkdirs(f"{REMOTE_ROOT}/{parent}")
            made.add(parent)
        api.put_bytes(f"{REMOTE_ROOT}/{relative}", (_REPO_ROOT / relative).read_bytes())
        print(f"  [{index}/{len(changed)}] {relative}")

    api.put_bytes(f"{REMOTE_ROOT}/{MANIFEST_NAME}", render_manifest(local))
    print(f"已上传 {len(changed)} 个文件，清单已更新")
    return 0


def cmd_pull(api: Contents, args: argparse.Namespace) -> int:
    """云端 outputs/experiments → 本地 outputs/cloud。

    刻意不落到 outputs/experiments：云端是 py3.11/numpy1.24 口径，和本地
    py3.13/numpy2.5 的结果混在同一个目录里排序会踩 CLAUDE.md §5.5。
    """
    target = _REPO_ROOT / "outputs" / "cloud"
    source = _REPO_ROOT / "outputs" / "experiments"
    target.mkdir(parents=True, exist_ok=True)
    entries = api.listdir(f"{REMOTE_ROOT}/outputs/experiments")
    fetched = mirrored = 0
    for entry in entries:
        if entry["type"] != "file":
            continue
        name = entry["name"]
        raw = api.get_bytes(f"{REMOTE_ROOT}/outputs/experiments/{name}")
        if raw is None:
            continue
        # 云端那个目录里既有我们推上去的报告，也有云端自己跑出来的。前者内容与本地
        # 逐字节相同，拉回来只是把同一份文件复制成两份 —— 只取真正的云端产物。
        local_copy = source / name
        if local_copy.exists() and local_copy.read_bytes() == raw:
            mirrored += 1
            continue
        destination = target / name
        if destination.exists() and destination.read_bytes() == raw:
            continue
        destination.write_bytes(raw)
        print(f"  拉取  outputs/cloud/{name}  ({len(raw):,} bytes)")
        fetched += 1
    print(f"完成：{fetched} 个云端产物落在 {target}"
          f"（另有 {mirrored} 个与本地相同，是我们推上去的，已跳过）")
    return 0


def cmd_log(api: Contents, args: argparse.Namespace) -> int:
    raw = api.get_bytes(f"{REMOTE_ROOT}/outputs/logs/{args.name}")
    if raw is None:
        listing = api.listdir(f"{REMOTE_ROOT}/outputs/logs")
        names = ", ".join(e["name"] for e in listing) or "(空)"
        raise SystemExit(f"云端没有 outputs/logs/{args.name}；现有：{names}")
    lines = raw.decode(errors="replace").splitlines()
    for line in lines[-args.tail:]:
        print(line)
    print(f"--- 共 {len(lines)} 行，显示末 {min(args.tail, len(lines))} 行 ---")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=os.environ.get("JHUB_BASE", DEFAULT_BASE))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="比对本地与云端，不传输")

    push = sub.add_parser("push", help="增量上传变动文件")
    push.add_argument("--dry-run", action="store_true", help="只列出会传什么")

    sub.add_parser("pull", help="云端 outputs/experiments → 本地 outputs/cloud")

    log = sub.add_parser("log", help="查看云端 outputs/logs/<名字>")
    log.add_argument("name")
    log.add_argument("--tail", type=int, default=40)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api = Contents(args.base, read_token())
    handlers = {"status": cmd_status, "push": cmd_push, "pull": cmd_pull, "log": cmd_log}
    try:
        return handlers[args.command](api, args)
    except urllib.error.HTTPError as error:
        raise SystemExit(f"HTTP {error.code} {error.reason}\n"
                         f"服务端说：{getattr(error, 'detail', '(无正文)')}")


if __name__ == "__main__":
    sys.exit(main())
