"""Skill registry for the Pi5 agent.

Two skills:
  - read_status(): BME280 + UPS HAT + CPU temp summary, with graceful fallback
    to psutil-only output when I2C hardware is unavailable.
  - read_PDF(path): extract text from a PDF, sandboxed to SANDBOX_ROOT.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

SANDBOX_ROOT = Path(os.environ.get("PI5_SANDBOX", Path.home() / "pi5_sandbox")).resolve()
PDF_TOKEN_BUDGET = 700  # ~chars*0.25; we cap by characters as a proxy
PDF_CHAR_BUDGET = PDF_TOKEN_BUDGET * 4


# ---------- read_status ----------

def _bme280_reading() -> dict[str, float] | None:
    try:
        import smbus2
        from bme280 import BME280  # type: ignore
        bus = smbus2.SMBus(1)
        sensor = BME280(i2c_dev=bus)
        return {
            "temp_c": round(sensor.get_temperature(), 1),
            "humidity_pct": round(sensor.get_humidity(), 1),
            "pressure_hpa": round(sensor.get_pressure(), 1),
        }
    except Exception:
        return None


def _ups_reading() -> dict[str, float] | None:
    """INA219-style UPS HAT on I2C bus 1. Address varies (0x40, 0x41, 0x42, 0x43)."""
    try:
        import smbus2
        bus = smbus2.SMBus(1)
        for addr in (0x41, 0x42, 0x43, 0x40):
            try:
                # Bus voltage register 0x02, shift >>3, *4mV
                raw = bus.read_word_data(addr, 0x02)
                raw = ((raw & 0xFF) << 8) | (raw >> 8)
                voltage = ((raw >> 3) * 4) / 1000.0
                if 2.5 < voltage < 12.6:
                    pct = max(0, min(100, round((voltage - 3.0) / (4.2 - 3.0) * 100)))
                    return {"battery_v": round(voltage, 2), "battery_pct": pct}
            except OSError:
                continue
        return None
    except Exception:
        return None


def _cpu_reading() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        import psutil
        out["cpu_pct"] = psutil.cpu_percent(interval=0.2)
        vm = psutil.virtual_memory()
        out["ram_used_mb"] = round((vm.total - vm.available) / 1024 / 1024)
        out["ram_total_mb"] = round(vm.total / 1024 / 1024)
    except Exception:
        pass
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            out["cpu_temp_c"] = round(int(f.read().strip()) / 1000, 1)
    except Exception:
        pass
    return out


def read_status() -> dict[str, Any]:
    status: dict[str, Any] = {"cpu": _cpu_reading()}
    bme = _bme280_reading()
    if bme:
        status["environment"] = bme
    ups = _ups_reading()
    if ups:
        status["battery"] = ups
    if "environment" not in status and "battery" not in status:
        status["note"] = "I2C sensors unavailable; CPU stats only."
    return status


# ---------- read_PDF ----------

def _resolve_sandboxed(path: str) -> Path:
    SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
    candidate = (SANDBOX_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if SANDBOX_ROOT not in candidate.parents and candidate != SANDBOX_ROOT:
        raise PermissionError(f"path escapes sandbox {SANDBOX_ROOT}")
    return candidate


def list_dir(path: str = "") -> dict[str, Any]:
    target = _resolve_sandboxed(path) if path else SANDBOX_ROOT
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


def read_file(path: str) -> dict[str, Any]:
    file_path = _resolve_sandboxed(path)
    if not file_path.exists():
        return {"error": f"file not found: {file_path}"}
    if file_path.suffix.lower() == ".pdf":
        return {"error": "use read_PDF for PDF files"}
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    truncated = False
    if len(text) > PDF_CHAR_BUDGET:
        text = text[:PDF_CHAR_BUDGET] + "\n[truncated]"
        truncated = True
    return {
        "file": file_path.name,
        "chars": len(text),
        "truncated": truncated,
        "text": text,
    }


def write_file(path: str, content: str) -> dict[str, Any]:
    file_path = _resolve_sandboxed(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return {
        "file": str(file_path),
        "bytes": len(content.encode("utf-8")),
        "status": "written",
    }


def read_PDF(path: str) -> dict[str, Any]:
    from pypdf import PdfReader
    file_path = _resolve_sandboxed(path)
    if not file_path.exists():
        return {"error": f"file not found: {file_path}"}
    if file_path.suffix.lower() != ".pdf":
        return {"error": f"not a PDF: {file_path.name}"}

    reader = PdfReader(str(file_path))
    pages = [p.extract_text() or "" for p in reader.pages]
    full_text = "\n".join(pages).strip()
    truncated = False
    if len(full_text) > PDF_CHAR_BUDGET:
        full_text = full_text[:PDF_CHAR_BUDGET] + "\n[truncated]"
        truncated = True
    return {
        "file": file_path.name,
        "pages": len(reader.pages),
        "chars": len(full_text),
        "truncated": truncated,
        "text": full_text,
    }


# ---------- registry ----------

@dataclass
class Skill:
    name: str
    description: str
    fn: Callable[..., Any]


REGISTRY: dict[str, Skill] = {
    "read_status": Skill(
        name="read_status",
        description="Return current Pi 5 status: CPU temp/load/RAM, BME280 environment (temp/humidity/pressure), and UPS HAT battery. No arguments.",
        fn=lambda **_: read_status(),
    ),
    "read_PDF": Skill(
        name="read_PDF",
        description="Extract text from a PDF file (.pdf) inside the sandbox. Argument: path (string).",
        fn=lambda **kw: read_PDF(kw["path"]),
    ),
    "read_file": Skill(
        name="read_file",
        description="Read a plain text file (.txt, .md, .csv, .json, .py, etc.) inside the sandbox. Argument: path (string).",
        fn=lambda **kw: read_file(kw["path"]),
    ),
    "list_dir": Skill(
        name="list_dir",
        description="List files and folders inside the sandbox. Use this FIRST when you don't know what files exist. Optional argument: path (string); empty {} lists the sandbox root.",
        fn=lambda **kw: list_dir(kw.get("path", "")),
    ),
    "write_file": Skill(
        name="write_file",
        description="Write a UTF-8 text file inside the sandbox. Arguments: path (string), content (string). Overwrites if exists.",
        fn=lambda **kw: write_file(kw["path"], kw["content"]),
    ),
}


def skill_descriptions() -> str:
    return "\n".join(f"- {s.name}: {s.description}" for s in REGISTRY.values())


def execute(skill_name: str, args: dict[str, Any]) -> dict[str, Any]:
    if skill_name not in REGISTRY:
        return {"error": f"unknown skill: {skill_name}"}
    try:
        return REGISTRY[skill_name].fn(**(args or {}))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
