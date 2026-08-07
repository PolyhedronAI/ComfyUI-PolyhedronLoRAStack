"""Guard v554 -- Polyhedron Fast Upscale (the fast path next to Power
Upscale).

BEHAVIOURAL where it can be: `_target` is pure math, so it is extracted
verbatim from the module and EXECUTED against the decision matrix (factor /
exact / zero-side aspect / divisible_by floor-snap / the nvidia_rtx_vsr /8
snap / both-zero raise). Text pins hold what needs comfy to run: the house
imports (one source of truth in ph_power_upscale - the PU file itself must
stay byte-untouched by this cut), the fail-loud gates (lanczos+gpu,
vsr+cpu, missing nvvfx), the per-frame Maxine lifecycle with a finally
close, the sub-batching, the v553-style telemetry, and the ISOLATED
registration (_FUP_OK - a bug in the new file must never fell the Power
Upscale). Script-style: exit 0 = pass.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v554_fastupscale] FAIL: " + msg)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def main():
    py = _read("nodes", "ph_fast_upscale.py")
    pu = _read("nodes", "ph_power_upscale.py")   # v555: the machine lives here
    init = _read("__init__.py")

    # ---- _target: extracted and EXECUTED (pure math) --------------------------
    m = re.search(r"def _target\(.*?\n(?=\ndef |\nclass )", py, re.S)
    if not m:
        _fail("_target not extractable")
    ns = {}
    exec(m.group(0), ns)  # noqa: S102 - our own source, measured not believed
    T = ns["_target"]
    # (mode, by, w, h, src_w, src_h, div, method) -> expected (tw, th)
    matrix = [
        (("factor", 2.0, 0, 0, 832, 1088, 16, "bicubic"), (1664, 2176)),
        (("factor", 1.05, 0, 0, 1024, 1024, 16, "bicubic"), (1072, 1072)),
        (("exact", 1.0, 1024, 1024, 640, 480, 16, "bicubic"), (1024, 1024)),
        (("exact", 1.0, 1024, 0, 640, 480, 16, "bicubic"), (1024, 768)),
        (("exact", 1.0, 0, 768, 1024, 768, 16, "bicubic"), (1024, 768)),
        (("factor", 2.0, 0, 0, 833, 1089, 16, "bicubic"), (1664, 2176)),
        (("exact", 1.0, 1020, 1020, 640, 640, 16, "nvidia_rtx_vsr"),
         (1016, 1016)),   # 1020 -> /16 floor 1008? no: 1020//16*16=1008 ...
    ]
    # correct the vsr expectation by the pinned rule itself: /16 floor, then /8 nearest
    matrix[-1] = ((("exact", 1.0, 1020, 1020, 640, 640, 16, "nvidia_rtx_vsr")),
                  (1008, 1008))
    for args, want in matrix:
        got = T(*args)
        if got != want:
            _fail(f"_target{args} -> {got}, expected {want}")
    try:
        T("exact", 1.0, 0, 0, 640, 480, 16, "bicubic")
        _fail("exact mode with both sides zero must raise")
    except ValueError:
        pass
    try:
        T("factor", 0.0, 0, 0, 640, 480, 16, "bicubic")
        _fail("factor <= 0 must raise")
    except ValueError:
        pass

    # ---- house imports: ONE source of truth ----------------------------------
    # RE-GROUNDED v836 (audit C): _lanczos_to left this file entirely --
    # it was imported and never used. The PROMISE ("helpers are imported,
    # never duplicated") survives in its direct form below: the file must
    # never grow its OWN copy of it either.
    for name in ("_resolve_input", "_build_video", "_MODEL_UPSCALER",
                 "_MuteInfoLogs", "_METHODS", "_chunks",
                 "_resize_chunked", "_vsr_resize"):
        if py.count(name) < 2:
            _fail(f"{name} must be IMPORTED from ph_power_upscale in BOTH "
                  "branches (house pattern)")
    if ("def _resolve_input" in py or "def _build_video" in py
            or "def _lanczos_to" in py):
        _fail("helpers must be imported, never duplicated")

    # ---- fail-loud gates -------------------------------------------------------
    if "cannot run on the GPU" not in pu:
        _fail("the lanczos+gpu gate must fail LOUD (the KJ lesson; v555: in PU)")
    if "runs on the GPU" not in py:
        _fail("the vsr+cpu gate is gone")
    if "import nvvfx" not in pu or "import failed" not in pu:
        _fail("the nvvfx import capsule with a plain message is gone (in PU)")

    # ---- Maxine lifecycle (v555: the machine lives in ph_power_upscale) ---------
    if "finally:" not in pu or "ctx.__exit__(None, None, None)" not in pu:
        _fail("the VSR context must ALWAYS close (try/finally)")
    if "per-frame: the Maxine API" not in pu:
        _fail("the per-frame SDK constraint must stay documented at the loop")
    if "from_dlpack" not in pu:
        _fail("the DLPack hand-over is gone")
    if "def _resize_chunked" in py or "def _vsr_resize" in py:
        _fail("the resize machine must live ONCE, in ph_power_upscale (v555)")

    # ---- sub-batching + telemetry ----------------------------------------------
    if "def _chunks" not in pu or "_esrgan_chunked" not in py:
        _fail("the VRAM sub-batching is gone")
    if "Fast Upscale: begin ->" not in py:
        _fail("the v553-style begin line is gone")
    if "ms/frame" not in py or "time.monotonic() - t0" not in py:
        _fail("the done line must carry the MEASURED duration and ms/frame")

    # ---- required order pinned at birth (index stability from day one) ---------
    req = py[py.index('"required"'):py.index('"optional"')]
    names = re.findall(r'"([a-z_0-9]+)":\s*\(', req)
    BIRTH = ["size_mode", "upscale_by", "width", "height", "resize_method",
             "device", "divisible_by", "per_batch", "mute_staging_logs"]
    # v559 hardening: prefix pin instead of equality - later fields APPEND
    # behind the birth order (index stability from day one).
    if names[:len(BIRTH)] != BIRTH:
        _fail(f"the birth order must keep indices 0-{len(BIRTH) - 1}: "
              f"{names[:len(BIRTH)]} != {BIRTH}")

    # ---- isolated registration ---------------------------------------------------
    if "_FUP_OK" not in init:
        _fail("Fast Upscale must register through its own _FUP_OK flag")
    if 'NODE_CLASS_MAPPINGS["ULSFastUpscale"]' not in init:
        _fail("central registration is gone")
    if "\u2b21 Polyhedron Fast Upscale" not in init:
        _fail("the display name must carry the \u2b21 prefix (house style)")
    pup_try = init[init.index("from .nodes.ph_power_upscale import"):
                   init.index("_FUP_OK")]
    if "ph_fast_upscale" in pup_try:
        _fail("the Fast Upscale import crept back into the Power Upscale "
              "try block - a bug here must never fell the PU")

    print("PASS: v554 Fast Upscale -- _target matrix executed, house imports, "
          "fail-loud gates, Maxine lifecycle, telemetry, isolated registration")
    sys.exit(0)


if __name__ == "__main__":
    main()
