"""Guard v558 -- the staging-log mute: sledgehammer -> scalpel, and the
Sampler finally gets it too.

BEHAVIOURAL: the capsule is imported and EXERCISED against a CHILD logger
(comfy.model_management is a child - the real case). Pinned:

  * only the known staging lines are dropped; a normal INFO from another node
    and any WARNING still get through (the whole point of the scalpel),
  * the filter sits on the root HANDLERS, not on the root logger - a record
    from a child logger never runs the root logger's own filters, so
    filtering there would silently do nothing,
  * the logging setup is restored byte-exactly afterwards,
  * the capsule counts what it swallowed and says so (nothing vanishes
    silently),
  * our own telemetry is print(), never logging, so it always passes the mute.

Structure pins: ONE source of truth (nodes/ph_logmute.py - no second copy),
the Sampler wraps its whole run (including every preview decode) WITHOUT
touching its 17-name index-based widget layout, and `logging.disable` (the
v553 sledgehammer) may never come back.

Script-style: exit 0 = pass.
"""
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v558_logmute] FAIL: " + msg)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def main():
    sys.path.insert(0, os.path.join(ROOT, "nodes"))
    try:
        from ph_logmute import MuteStagingLogs, STAGING_PATTERNS
    except Exception as exc:
        _fail(f"the capsule must stay import-light (stdlib only): {exc}")

    cap = _Capture()
    prev_handlers = list(logging.root.handlers)
    prev_level = logging.root.level
    logging.root.handlers = [cap]
    logging.root.setLevel(logging.INFO)
    child = logging.getLogger("comfy.model_management")   # the REAL case
    try:
        with MuteStagingLogs(True, label="Guard") as m:
            child.info("Model TAEHV prepared for dynamic VRAM loading. 21MB Staged.")
            child.info("0 models unloaded.")
            child.info("something useful from another pack")
            child.warning("a real warning")
            # a WARNING that happens to contain a pattern must still pass
            child.warning("Requested to load - but this is a WARNING")
        inside = list(cap.lines)
        cap.lines = []
        child.info("Model TAEHV prepared for dynamic VRAM loading.")
        after = list(cap.lines)
    finally:
        logging.root.handlers = prev_handlers
        logging.root.setLevel(prev_level)

    if any("prepared for dynamic VRAM" in ln for ln in inside):
        _fail("the staging chatter was NOT muted")
    if any("models unloaded" in ln for ln in inside):
        _fail("the 'models unloaded' line was NOT muted")
    if not any("something useful" in ln for ln in inside):
        _fail("a normal INFO from another pack must survive - that is the "
              "whole point of the scalpel (v553 killed these too)")
    if not any("a real warning" in ln for ln in inside):
        _fail("a WARNING must always survive")
    if not any("but this is a WARNING" in ln for ln in inside):
        _fail("the filter must only look at records <= INFO")
    if not after:
        _fail("the logging setup was not restored after the scope")
    if getattr(m, "_filter", None) is not None:
        _fail("the filter must be dropped on exit")

    # disabled capsule = a true no-op (the level must be INFO for this to
    # mean anything - the first block restored it, so set it again)
    logging.root.handlers = [cap]
    logging.root.setLevel(logging.INFO)
    cap.lines = []
    try:
        with MuteStagingLogs(False):
            child.info("Model TAEHV prepared for dynamic VRAM loading.")
    finally:
        logging.root.handlers = prev_handlers
        logging.root.setLevel(prev_level)
    if not cap.lines:
        _fail("MuteStagingLogs(False) must be a no-op (VRAM debugging)")

    if not STAGING_PATTERNS or "prepared for dynamic VRAM loading" not in \
            STAGING_PATTERNS:
        _fail("the pattern list lost the main offender")

    # ---- structure ------------------------------------------------------------
    lm = _read("nodes", "ph_logmute.py")
    pu = _read("nodes", "ph_power_upscale.py")
    smp = _read("nodes", "uls_sampler.py")
    fu = _read("nodes", "ph_fast_upscale.py")

    if "addFilter" not in lm or "logging.root.handlers" not in lm:
        _fail("the filter must be installed on the root HANDLERS (a child "
              "logger's record never runs the root LOGGER's filters)")
    if "logging.disable" in lm or "logging.disable" in pu:
        _fail("the v553 sledgehammer (logging.disable) must not come back - "
              "it silenced other packs' INFO messages too")
    if "class _MuteInfoLogs" in pu:
        _fail("the capsule must live ONCE, in ph_logmute.py")
    if "from .ph_logmute import MuteStagingLogs" not in pu or \
       "from ph_logmute import MuteStagingLogs" not in pu:
        _fail("ph_power_upscale must IMPORT the capsule in both branches")
    if 'label="Power Upscale"' not in pu or 'label="Fast Upscale"' not in fu:
        _fail("each node must label its own mute (so the count says who)")

    # ---- the Sampler wrap -------------------------------------------------------
    if "def sample(self, **kwargs):" not in smp:
        _fail("the Sampler's thin mute wrapper is gone")
    if "def _sample_impl(self, model, positive, negative" not in smp:
        _fail("the original sampler body must live on as _sample_impl "
              "(the MoE loop is no-touch)")
    if '_MuteStagingLogs(True, label="Sampler")' not in smp:
        _fail("the Sampler run is no longer inside the mute")
    if 'FUNCTION = "sample"' not in smp:
        _fail("ComfyUI must still call sample()")
    if "mute_staging_logs" in smp:
        _fail("no new widget in the Sampler: uls_sampler.js is index-based "
              "over 17 serialised names with length heuristics - shifting that "
              "layout for a logging tweak is exactly what a stability-first "
              "house does not do")

    print("PASS: v558 -- scalpel exercised against a CHILD logger (staging "
          "dropped, INFO/WARNING survive, restore + count), one source, "
          "Sampler wrapped without touching its widget layout")
    sys.exit(0)


if __name__ == "__main__":
    main()
