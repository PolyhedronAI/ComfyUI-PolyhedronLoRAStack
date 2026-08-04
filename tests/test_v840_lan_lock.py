#!/usr/bin/env python3
"""Guard v840 -- the LAN browse lock (audit B1) and the event-loop house rule.

WHY THIS EXISTS
The media routes anchor on a CLIENT-supplied folder/path. Local bind: that is
the feature. Bound to the LAN (--listen): every client could read arbitrary
directories, upload into them, and pop native dialogs on the host. Frank's
call (04.08.): keep local behaviour untouched, LOCK the path-anchored routes
under a non-local bind, ULS_ALLOW_LAN_BROWSE=1 opens them deliberately.

WHAT THIS PINS (driven where there is machinery, read where there is law):
  P1 policy      -- _lan_browse_locked, LIFTED and exec-driven through the
                    full matrix: localhost/::1 open, any non-local bind
                    locked, comma lists locked on ONE foreign entry, the env
                    override opens, and an UNKNOWN bind reads LOCAL
                    (fail-open: the lock must never break a localhost
                    setup).
  P1b the door   -- _lan_deny driven: locked -> 403 naming the override;
                    open -> None.
  P2 wiring      -- every path-anchored handler carries the gate; the
                    deliberately OPEN handlers (seq_list, seq_delete,
                    proc_count) carry NONE -- openness is part of the
                    design, not an omission.
  P4 tree law    -- asyncio.get_event_loop( appears NOWHERE under nodes/
                    (vendor excluded): the house uses get_running_loop
                    (v577 rule, deprecated 3.12+).

MUTATIONS LANDED DURING THE BUILD (each turned this guard red):
  M1 gate removed from handle_media_file        -> P2
  M2 env override dropped from the policy       -> P1
  M3 fail-open flipped to locked-on-unknown     -> P1
  M4 get_event_loop reinstated                  -> P4
"""
import glob
import os
import re
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = "v840"

_PUB = os.path.join(ROOT, "nodes", "ph_media_routes.py")
_INT = os.path.join(ROOT, "nodes", "uls_routes.py")
FILE = _PUB if os.path.isfile(_PUB) else _INT
SRC = open(FILE, encoding="utf-8").read()
BASE = os.path.basename(FILE)


def _fail(msg):
    print("[test_%s_lan_lock] FAIL -- %s" % (NAME, msg))
    sys.exit(1)


def _need(cond, msg):
    if not cond:
        _fail(msg)


# ---- P1: the policy, lifted and driven -------------------------------------
i0 = SRC.find("_LAN_BROWSE_ENV = ")
_need(i0 != -1, "P1: the lock is missing from %s" % BASE)
i1 = SRC.index("def _lan_deny(", i0)
i1 = SRC.index("\n\n", SRC.index("status=403)", i1))
policy = SRC[i0:i1]


class _FakeWeb:
    def __init__(self):
        self.calls = []

    def json_response(self, payload, status=200):
        self.calls.append((payload, status))
        return ("RESP", status, payload)


def _drive(bind, env=None, missing=False):
    """Run the lifted policy with a controlled comfy.cli_args + environ."""
    import types
    saved = sys.modules.get("comfy.cli_args")
    saved_c = sys.modules.get("comfy")
    try:
        if missing:
            sys.modules.pop("comfy.cli_args", None)
            sys.modules.pop("comfy", None)
        else:
            m = types.ModuleType("comfy.cli_args")
            m.args = types.SimpleNamespace(listen=bind)
            c = sys.modules.get("comfy") or types.ModuleType("comfy")
            c.cli_args = m
            sys.modules["comfy"] = c
            sys.modules["comfy.cli_args"] = m
        old_env = os.environ.pop("ULS_ALLOW_LAN_BROWSE", None)
        if env is not None:
            os.environ["ULS_ALLOW_LAN_BROWSE"] = env
        web = _FakeWeb()
        ns = {"os": os, "web": web}
        exec(compile(textwrap.dedent(policy), "<policy>", "exec"), ns)
        locked = ns["_lan_browse_locked"]()
        deny = ns["_lan_deny"](object())
        return locked, deny
    finally:
        os.environ.pop("ULS_ALLOW_LAN_BROWSE", None)
        if old_env is not None:
            os.environ["ULS_ALLOW_LAN_BROWSE"] = old_env
        if saved is not None:
            sys.modules["comfy.cli_args"] = saved
        elif not missing:
            sys.modules.pop("comfy.cli_args", None)
        if saved_c is not None:
            sys.modules["comfy"] = saved_c
        elif not missing:
            sys.modules.pop("comfy", None)


for bind, env, want in (("127.0.0.1", None, False), ("localhost", None, False),
                        ("::1", None, False), ("", None, False),
                        ("0.0.0.0", None, True), ("192.168.1.5", None, True),
                        ("127.0.0.1,0.0.0.0", None, True),
                        ("0.0.0.0", "1", False), ("0.0.0.0", "true", False),
                        ("0.0.0.0", "0", True)):
    locked, _ = _drive(bind, env)
    _need(locked is want,
          "P1: bind=%r env=%r -> locked=%r, want %r" % (bind, env, locked, want))
locked, _ = _drive(None, None, missing=True)
_need(locked is False,
      "P1: an UNKNOWN bind must read LOCAL (fail-open) -- the lock may never "
      "break a localhost setup")

# ---- P1b: the door ---------------------------------------------------------
_, deny = _drive("0.0.0.0", None)
_need(deny is not None and deny[1] == 403,
      "P1b: locked must answer 403 -- got %r" % (deny,))
_need("ULS_ALLOW_LAN_BROWSE" in str(deny[2]),
      "P1b: the refusal must NAME the override, or nobody finds the key")
_, deny = _drive("127.0.0.1", None)
_need(deny is None, "P1b: open must answer None (the handler proceeds)")

# ---- P2: wiring ------------------------------------------------------------
GATED = ["handle_media_folders", "handle_media_list", "handle_media_dims",
         "handle_media_thumb", "handle_media_file", "handle_media_native_pick",
         "handle_media_upload", "handle_media_seq_build", "handle_media_resolve",
         "handle_media_locate", "handle_media_open_folder"]
OPEN = ["handle_media_seq_list", "handle_media_seq_delete",
        "handle_media_proc_count"]


def _span(name):
    m = re.search(r"^async def %s\(request" % name, SRC, re.M)
    if not m:
        return None
    nxt = re.search(r"^async def ", SRC[m.end():], re.M)
    return SRC[m.start():m.end() + (nxt.start() if nxt else len(SRC))]


present = 0
for h in GATED:
    s = _span(h)
    if s is None:
        continue
    present += 1
    _need("_lan_deny(request)" in s,
          "P2: %s takes a client path but carries NO gate" % h)
for h in OPEN:
    s = _span(h)
    if s is None:
        continue
    _need("_lan_deny(request)" not in s,
          "P2: %s is gated -- it operates inside the managed project dir and "
          "must stay OPEN (deliberate design, see the helper comment)" % h)
_need(present >= 5,
      "P2: only %d gated handlers found in %s -- the core browse family "
      "(folders/list/dims/thumb/file) must exist" % (present, BASE))

# ---- P4: tree law ----------------------------------------------------------
hits = []
for p in glob.glob(os.path.join(ROOT, "nodes", "**", "*.py"), recursive=True):
    if os.sep + "vendor" + os.sep in p:
        continue
    if "asyncio.get_event_loop(" in open(p, encoding="utf-8").read():
        hits.append(os.path.relpath(p, ROOT))
_need(not hits,
      "P4: asyncio.get_event_loop( is back in %r -- the house uses "
      "get_running_loop (v577 rule, deprecated since Python 3.12)" % hits)

print("[test_%s_lan_lock] OK -- %s: policy matrix true (fail-open on unknown), "
      "the door answers 403 naming the override, %d handlers gated / %d open "
      "by design, and the tree is get_event_loop-free"
      % (NAME, BASE, present, sum(1 for h in OPEN if _span(h))))
