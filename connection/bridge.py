"""RPC bridge between Pi-side Python skills and the browser's File System Access API.

The Pi-side skill calls `bridge.request(op, args)` which:
  1. Generates a uuid op_id
  2. Pushes (op_id, op, args) onto an outbound queue
  3. Blocks on a per-id `threading.Event` until the browser POSTs back

The browser-side consumer holds a long-lived SSE stream from `/fs/poll`
(implemented as `bridge.next_op()` in a Flask route) and POSTs results to
`/fs/result/<op_id>` (which calls `bridge.complete(op_id, payload)`).

Only one SSE consumer is allowed at a time (`try_connect()` returns False
otherwise). When the consumer disconnects, every pending op is woken with
an error so skills don't hang for the full 30s timeout.
"""

from __future__ import annotations

import queue
import threading
import uuid


class BridgeDisconnected(RuntimeError):
    """Raised when no browser is connected as the bridge consumer."""


class Bridge:
    def __init__(self):
        self._pending: dict[str, threading.Event] = {}
        self._results: dict[str, dict] = {}
        self._outbound: queue.Queue = queue.Queue()
        self._connected = threading.Event()
        self._connect_lock = threading.Lock()
        self._registry_lock = threading.Lock()

    def try_connect(self) -> bool:
        """Acquire single-consumer slot. Returns False if a tab is already connected."""
        with self._connect_lock:
            if self._connected.is_set():
                return False
            self._connected.set()
            return True

    def disconnect(self) -> None:
        """Release the consumer slot and fail any in-flight ops fast."""
        with self._connect_lock:
            self._connected.clear()
        # Drain anything queued but never delivered.
        try:
            while True:
                self._outbound.get_nowait()
        except queue.Empty:
            pass
        # Wake every pending request with a synthetic disconnect error.
        with self._registry_lock:
            ids = list(self._pending.keys())
        for op_id in ids:
            self.complete(op_id, {"ok": False, "error": "disconnected"})

    def request(self, op: str, args: dict, timeout: float = 30.0) -> dict:
        """Block until the browser returns a result. Raises BridgeDisconnected
        if no consumer is connected within ~2s of the call."""
        if not self._connected.wait(timeout=2.0):
            raise BridgeDisconnected("PC sandbox bridge disconnected")

        op_id = uuid.uuid4().hex
        event = threading.Event()
        with self._registry_lock:
            self._pending[op_id] = event
        self._outbound.put((op_id, op, args))
        try:
            if not event.wait(timeout=timeout):
                raise TimeoutError(f"bridge op {op} timed out after {timeout}s")
        finally:
            with self._registry_lock:
                self._pending.pop(op_id, None)
                result = self._results.pop(op_id, None)
        if result is None:
            return {"ok": False, "error": "no result"}
        return result

    def next_op(self, block_timeout: float = 15.0):
        """Used by the SSE poll route. Returns (op_id, op, args) or None on heartbeat tick."""
        try:
            return self._outbound.get(timeout=block_timeout)
        except queue.Empty:
            return None

    def complete(self, op_id: str, payload: dict) -> None:
        """Called by the result POST handler with `{ok, result|error}`."""
        with self._registry_lock:
            event = self._pending.get(op_id)
            if event is None:
                return  # late arrival or duplicate; drop
            self._results[op_id] = payload
        event.set()
