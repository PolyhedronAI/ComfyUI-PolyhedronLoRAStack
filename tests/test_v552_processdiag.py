"""Guard v552 -- process view: routing fix + a gap-free evidence chain.

The v550 listener routed by `comfyClass`, which may be UNSET on the graph
node object - every event was then dropped SILENTLY (Frank's "keine Spur").
v552 pins the LIVE-PROVEN uls_live_preview pattern instead: `node.type`,
robust `(e && e.detail) || {}` extraction, `async setup()`. And because two
backend exits were also silent, the whole chain must now state itself once
per run (measure > believe):

  server log:  "process view armed"        -> the probe was built
               "first frame sent"          -> the send side works
               "unavailable (no node id)"  -> the former silent exit speaks
               "idle (sampler callback"    -> the x0 silent exit speaks
  F12 console: "process listener armed"    -> setup() actually ran
               "first event"               -> the receive side works
               "result viewer: first payload" -> the v549 chain works

Script-style: exit 0 = pass.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v552_processdiag] FAIL: " + msg)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def main():
    py = _read("nodes", "ph_power_upscale.py")
    js = _read("web", "js", "ph_power_upscale.js")

    # ---- backend: every exit states itself -----------------------------------
    if "process view armed" not in py:
        _fail("the armed marker is gone - a built probe must say so")
    if "first frame sent" not in py:
        _fail("the first-send proof line is gone")
    if "process view unavailable (no node id)" not in py:
        _fail("the node-id exit went silent again")
    if "sampler \\\ncallback carries no x0" in py:
        pass  # formatting-agnostic; real pin below
    if "carries no x0" not in py:
        _fail("the x0 exit went silent again")
    if '"sent": False' not in py or '"x0w": False' not in py:
        _fail("the once-per-run state flags are gone")

    # ---- frontend: the live-proven routing + armed/first markers -------------
    # The exact banner version is pinned by the file's NEWEST guard (v553);
    # existence + format is pinned by test_v548. No banner pin here.
    if "async setup()" not in js:
        _fail("setup must be async (the uls_live_preview pattern)")
    if "PU process listener armed" not in js:
        _fail("the listener-armed console marker is gone")
    if "node.type !== NODE_TYPE" not in js:
        _fail("routing must use node.type (LIVE-PROVEN; comfyClass may be "
              "unset and silently drops every event)")
    if "comfyClass === NODE_TYPE" in js:
        _fail("a comfyClass routing check crept back in")
    if "(e && e.detail) || {}" not in js:
        _fail("the robust detail extraction (uls_live_preview pattern) is gone")
    if "PU process view: first event" not in js:
        _fail("the first-receive console marker is gone")
    if "PU result viewer: first payload" not in js:
        _fail("the v549 result-chain proof marker is gone")
    if "_pvFirstLogged" not in js or "_procFirstLogged" not in js:
        _fail("the once-per-session marker flags are gone")

    print("PASS: v552 -- node.type routing (live-proven pattern) + gap-free "
          "armed/first-sent/first-received evidence chain")
    sys.exit(0)


if __name__ == "__main__":
    main()
