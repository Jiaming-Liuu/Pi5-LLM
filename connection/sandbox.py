"""Sandbox backends.

The agent's filesystem skills go through a `Sandbox` instance instead of
manipulating paths directly. This lets us swap the backend at runtime:

- `LocalSandbox` — files live in a folder on the Pi (the original behavior).
- `BrowserSandbox` — files live on the user's PC; ops are proxied to the
  browser via the RPC `Bridge` and executed against a
  `FileSystemDirectoryHandle` obtained from `window.showDirectoryPicker()`.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Protocol

MAX_PC_BYTES = 20 * 1024 * 1024  # 20 MB cap on read_bytes through the bridge


class Sandbox(Protocol):
    label: str
    def list_dir(self, path: str) -> dict[str, Any]: ...
    def read_text(self, path: str) -> str: ...
    def read_bytes(self, path: str) -> bytes: ...
    def write_text(self, path: str, content: str) -> dict[str, Any]: ...
    def exists(self, path: str) -> bool: ...
    def is_dir(self, path: str) -> bool: ...


class LocalSandbox:
    """Files under a fixed root directory on the Pi."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.label = f"Pi: {self.root}"

    def _resolve(self, path: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        candidate = (self.root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if self.root not in candidate.parents and candidate != self.root:
            raise PermissionError(f"path escapes sandbox {self.root}")
        return candidate

    def list_dir(self, path: str) -> dict[str, Any]:
        target = self._resolve(path) if path else self.root
        if not target.exists():
            return {"error": f"directory not found: {target}"}
        if not target.is_dir():
            return {"error": f"not a directory: {target}"}
        entries = []
        for entry in sorted(target.iterdir()):
            entries.append({
                "name": entry.name,
                "type": "dir" if entry.is_dir() else "file",
                "bytes": entry.stat().st_size if entry.is_file() else None,
            })
        return {"dir": str(target), "entries": entries}

    def read_text(self, path: str) -> str:
        return self._resolve(path).read_text(encoding="utf-8", errors="replace")

    def read_bytes(self, path: str) -> bytes:
        return self._resolve(path).read_bytes()

    def write_text(self, path: str, content: str) -> dict[str, Any]:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"file": str(target), "bytes": len(content.encode("utf-8")), "status": "written"}

    def exists(self, path: str) -> bool:
        try:
            return self._resolve(path).exists()
        except PermissionError:
            return False

    def is_dir(self, path: str) -> bool:
        try:
            return self._resolve(path).is_dir()
        except PermissionError:
            return False

    def delete(self, path: str) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.exists():
            return {"error": f"not found: {path}"}
        if target.is_dir():
            return {"error": "cannot delete directories"}
        target.unlink()
        return {"ok": True, "deleted": str(target)}


def _safe_path(path: str) -> str:
    """Reject path traversal before sending to the browser."""
    parts = [p for p in path.replace("\\", "/").split("/") if p and p != "."]
    if any(p == ".." for p in parts):
        raise PermissionError(f"path contains '..': {path}")
    return "/".join(parts)


class BrowserSandbox:
    """Proxies all filesystem ops to the connected browser via the Bridge."""

    def __init__(self, bridge, display_name: str):
        self.bridge = bridge
        self.display_name = display_name
        self.label = f"PC: {display_name}"

    def _request(self, op: str, args: dict[str, Any]) -> Any:
        res = self.bridge.request(op, args)
        if not res.get("ok"):
            raise RuntimeError(res.get("error", "browser sandbox error"))
        return res.get("result")

    def list_dir(self, path: str) -> dict[str, Any]:
        return self._request("list", {"path": _safe_path(path)})

    def read_text(self, path: str) -> str:
        return self._request("read_text", {"path": _safe_path(path)})

    def read_bytes(self, path: str) -> bytes:
        result = self._request("read_bytes", {"path": _safe_path(path), "max_bytes": MAX_PC_BYTES})
        b64 = result.get("b64", "") if isinstance(result, dict) else ""
        data = base64.b64decode(b64)
        if len(data) > MAX_PC_BYTES:
            raise RuntimeError(f"file exceeds {MAX_PC_BYTES // (1024 * 1024)} MB cap")
        return data

    def write_text(self, path: str, content: str) -> dict[str, Any]:
        return self._request("write_text", {"path": _safe_path(path), "content": content})

    def exists(self, path: str) -> bool:
        return bool(self._request("exists", {"path": _safe_path(path)}))

    def is_dir(self, path: str) -> bool:
        return bool(self._request("is_dir", {"path": _safe_path(path)}))
