"""Skill registry for the Pi5 agent.

Two skills:
  - read_status(): BME280 + UPS HAT + CPU temp summary, with graceful fallback
    to psutil-only output when I2C hardware is unavailable.
  - read_PDF(path): extract text from a PDF, sandboxed to SANDBOX_ROOT.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from connection.sandbox import LocalSandbox, Sandbox

SANDBOX_ROOT = Path(os.environ.get("PI5_SANDBOX", Path.home() / "pi5_sandbox")).resolve()
PDF_TOKEN_BUDGET = 3000  # ~chars*0.25; we cap by characters as a proxy
PDF_CHAR_BUDGET = PDF_TOKEN_BUDGET * 4

_active_sandbox: Sandbox = LocalSandbox(SANDBOX_ROOT)


def set_sandbox(sb: Sandbox) -> None:
    global _active_sandbox
    _active_sandbox = sb


def get_sandbox() -> Sandbox:
    return _active_sandbox


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


# ---------- sandbox-backed file skills ----------

def list_dir(path: str = "") -> dict[str, Any]:
    return _active_sandbox.list_dir(path)


def read_file(path: str) -> dict[str, Any]:
    if path.lower().endswith(".pdf"):
        return {"error": "use read_PDF for PDF files"}
    sb = _active_sandbox
    if not sb.exists(path):
        return {"error": f"file not found: {path}"}
    try:
        text = sb.read_text(path)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    truncated = False
    if len(text) > PDF_CHAR_BUDGET:
        text = text[:PDF_CHAR_BUDGET] + "\n[truncated]"
        truncated = True
    return {
        "file": path,
        "chars": len(text),
        "truncated": truncated,
        "text": text,
    }


def write_file(path: str, content: str) -> dict[str, Any]:
    return _active_sandbox.write_text(path, content)


def read_PDF(path: str) -> dict[str, Any]:
    from pypdf import PdfReader
    if not path.lower().endswith(".pdf"):
        return {"error": f"not a PDF: {path}"}
    sb = _active_sandbox
    if not sb.exists(path):
        return {"error": f"file not found: {path}"}
    try:
        data = sb.read_bytes(path)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    reader = PdfReader(io.BytesIO(data))
    pages = [p.extract_text() or "" for p in reader.pages]
    full_text = "\n".join(pages).strip()
    truncated = False
    if len(full_text) > PDF_CHAR_BUDGET:
        full_text = full_text[:PDF_CHAR_BUDGET] + "\n[truncated]"
        truncated = True
    return {
        "file": path,
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
