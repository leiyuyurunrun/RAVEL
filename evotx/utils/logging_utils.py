from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Optional, TextIO
import atexit


class TeeStream:
    """Mirror writes to both the original stream and a UTF-8 log file."""

    def __init__(self, original: TextIO, logfile: TextIO):
        self.original = original
        self.logfile = logfile
        self.encoding = getattr(original, "encoding", "utf-8")

    def write(self, data: str) -> int:
        self.original.write(data)
        self.logfile.write(data)
        return len(data)

    def flush(self) -> None:
        self.original.flush()
        self.logfile.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.original, "isatty", lambda: False)())


def configure_session_logging(
    stage_name: str,
    logs_dir: str | Path = "data/log",
    suffix: Optional[str] = None,
) -> Path:
    """
    Tee stdout/stderr to a stage-specific UTF-8 log file.

    Returns the created log path. Safe to call once per process entrypoint.
    """
    log_root = Path(logs_dir)
    log_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stage = _sanitize_component(stage_name)
    safe_suffix = _sanitize_component(suffix or "")
    name_parts = [safe_stage]
    if safe_suffix:
        name_parts.append(safe_suffix)
    name_parts.append(timestamp)
    log_path = log_root / ("__".join(name_parts) + ".log")

    logfile = open(log_path, "a", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    if not isinstance(sys.stdout, TeeStream):
        sys.stdout = TeeStream(sys.stdout, logfile)
    if not isinstance(sys.stderr, TeeStream):
        sys.stderr = TeeStream(sys.stderr, logfile)

    def _restore_and_close() -> None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        try:
            logfile.flush()
        finally:
            logfile.close()

    atexit.register(_restore_and_close)

    print(f"[Log] Session log: {log_path}")
    return log_path


def _sanitize_component(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    allowed = []
    for ch in text:
        if ch.isalnum():
            allowed.append(ch.lower())
        else:
            allowed.append("_")
    collapsed = "_".join(part for part in "".join(allowed).split("_") if part)
    return collapsed[:80]
