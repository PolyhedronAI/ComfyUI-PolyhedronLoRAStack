"""Staging-log mute (v558) - one source of truth for the whole pack.

ComfyUI's model manager logs an INFO line every time it stages a model:
VAE encode, sample, decode, and - the loud one - EVERY preview decode
(TAEHV/TAESD). A tiled dual-stage run or a long sampling run repeats the
same three or four lines dozens of times.

v553 muted these with the process-wide logging gate - a sledgehammer: it
silenced EVERY record at or below INFO for the duration, ours and other
packs' alike.
v558 replaces it with a scalpel: a logging.Filter that drops ONLY the known
staging lines and lets everything else through untouched - a real warning or
another node's INFO message still reaches the console.

Two details that matter:

* The filter is installed on the ROOT logger's HANDLERS, not on the root
  logger itself. A record emitted by a CHILD logger (comfy.model_management
  and friends) never runs the root logger's own filters - it only runs the
  handlers'. Filtering at the logger would silently do nothing.
* Our own telemetry is print(), never logging, so it always passes.

The capsule counts what it swallowed and says so once per run (measure beats
believe): nothing disappears silently.
"""
import logging

# Substrings of the staging chatter. Matched case-sensitively against the
# formatted message, and only at level <= INFO, so a WARNING that happens to
# contain one of these still gets through.
STAGING_PATTERNS = (
    "prepared for dynamic VRAM loading",
    "Force pre-loaded",
    "models unloaded",
    "Requested to load",
    "Loading 1 new model",
    "loaded completely",
    "loaded partially",
)


class _StagingFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.muted = 0

    def filter(self, record):   # True = keep, False = drop
        if record.levelno > logging.INFO:
            return True
        try:
            msg = record.getMessage()
        except Exception:       # a broken record is not ours to judge
            return True
        for pat in STAGING_PATTERNS:
            if pat in msg:
                self.muted += 1
                return False
        return True


class MuteStagingLogs:
    """Context manager: silence the staging chatter for exactly this scope and
    restore the logging setup byte-exactly afterwards (try/finally). Failure
    to install is never fatal - the run matters, the logs do not."""

    def __init__(self, enabled=True, label="run"):
        self.enabled = bool(enabled)
        self.label = str(label)
        self._filter = None
        self._handlers = []

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            self._filter = _StagingFilter()
            self._handlers = list(logging.root.handlers)
            for h in self._handlers:
                h.addFilter(self._filter)
        except Exception:       # pragma: no cover - never cost a run
            self._filter = None
            self._handlers = []
        return self

    def __exit__(self, *exc):
        if self._filter is None:
            return False
        for h in self._handlers:
            try:
                h.removeFilter(self._filter)
            except Exception:
                pass
        if self._filter.muted:
            print(f"[PLS] {self.label}: {self._filter.muted} staging log "
                  f"line(s) muted")
        self._filter = None
        self._handlers = []
        return False
