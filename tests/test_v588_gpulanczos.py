"""v588 guard: TRUE lanczos on the GPU + the canon marker.

Two laws in one cut, both pinned by RETURN VALUES where possible (lesson 4):

1. The GPU lanczos is the SAME windowed sinc PIL uses - _fit_method calls
   PIL's lanczos the reference, so the new path must match the MATH, not an
   approximation of it. The kernel and the 1-D weight builder are pure
   python by design; this guard exec()s them without torch and pins values
   computed INDEPENDENTLY here (the sinc formula written out a second time).

2. The canon marker: a save must PROVE its widget order. v584 measured the
   cost of a guessed order (the seed landed in cfg_low). Only the branch
   that provably wrote canon order may set the marker; configure must read
   it and SAY which path it took; the display re-sort stays illegal (v585
   law, test_v546) until the marker is proven in the field.
"""
import math, pathlib, re, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PU = ROOT / "nodes" / "ph_power_upscale.py"
JS = ROOT / "web" / "js" / "ph_power_upscale.js"


def _fail(msg):
    print(f"[test_v588_gpulanczos] FAIL: {msg}")
    sys.exit(1)


def main():
    pu = PU.read_text(encoding="utf-8")
    js = JS.read_text(encoding="utf-8")

    # ---- 1: the kernel math, exec'd in isolation, pinned against an
    #         INDEPENDENT computation of the same formula --------------------
    src = pu[pu.index("_LANCZOS_A = 3"):pu.index("_LANCZOS_MATS = {}")]
    ns = {"math": math}
    exec(compile(src, "<lanczos>", "exec"), ns)
    k, w1d = ns["_lanczos_kernel"], ns["_lanczos_weights_1d"]
    if k(0.0) != 1.0 or k(3.0) != 0.0 or k(-3.0) != 0.0:
        _fail("the window must be 1 at 0 and 0 at |a| - it is the definition")
    ref = 3.0 * math.sin(1.5 * math.pi) * math.sin(0.5 * math.pi) / (math.pi * 1.5) ** 2
    if abs(k(1.5) - ref) > 1e-12 or abs(k(1.5) - k(-1.5)) > 1e-15:
        _fail(f"L(1.5) must equal the sinc formula ({ref:.12f}) and be "
              f"symmetric - got {k(1.5):.12f}")
    # 2x shrink: the support stretches (that IS the antialias) -> >= 7 taps.
    rows = w1d(8, 4)
    if len(rows) != 4 or any(len(r) < 7 for r in rows):
        _fail("a 2x shrink must stretch the kernel support (>=7 taps/row) - "
              "an unstretched kernel is an aliasing interpolator")
    for j, r in enumerate(rows):
        if abs(sum(w for _, w in r) - 1.0) > 1e-9:
            _fail(f"row {j} weights must sum to 1 (got "
                  f"{sum(w for _, w in r):.12f}) - unnormalised kernels shift "
                  f"brightness")
        if any(i < 0 or i > 7 for i, _ in r):
            _fail("tap indices must clamp to the source (replicate edges)")
    rows = w1d(4, 8)   # upscale: classic support, still normalised
    if len(rows) != 8 or any(abs(sum(w for _, w in r) - 1.0) > 1e-9 for r in rows):
        _fail("upscale rows must exist and sum to 1")

    # ---- 2: the wiring ------------------------------------------------------
    if '"lanczos (gpu)": None' not in pu:
        _fail("the _METHODS entry is gone - one source of truth feeds every "
              "resize_method list")
    if 'if method == "lanczos (gpu)":' not in pu or "_lanczos_gpu_to(image, width, height)" not in pu:
        _fail("_fit_to must route 'lanczos (gpu)' to its own path")
    if "'lanczos (gpu)' runs its matmul" not in pu:
        _fail("_resize_chunked must fail LOUD on a cpu device choice - the "
              "mirror of the lanczos (cpu)/gpu rule (the KJ lesson)")
    if '"default": "lanczos (cpu)"' not in pu:
        _fail("the widget default must stay 'lanczos (cpu)' - a new option is "
              "an offer, never a silent swap (v555 promise)")
    if pu.count("torch.einsum") < 2:
        _fail("the separable two-pass (H then W) is the whole speed argument")

    # ---- 3: the canon marker ------------------------------------------------
    if 'const CANON_MARKER = "pls_widgets_canon";' not in js:
        _fail("the marker constant is gone")
    if js.count("o.properties[CANON_MARKER] = 588;") != 1:
        _fail("exactly ONE branch may mark - the one that provably wrote "
              "canon order (inside the _plsDisplayReordered serialize path)")
    mark_at = js.index("o.properties[CANON_MARKER] = 588;")
    dtc_at = js.index("o.widgets_values = _displayToCanon(o.widgets_values);")
    if mark_at < dtc_at:
        _fail("the marker must be set AFTER the canon mapping it certifies")
    if "canon-marked save" not in js or "legacy save (no canon marker)" not in js:
        _fail("configure must SAY which path it took - the field proof lives "
              "in these two lines")

    # ---- 4: the mapping roundtrip, executed ---------------------------------
    m = re.search(r"const ORDER_CANON = \[(.*?)\];", js, re.S)
    n = len(re.findall(r'"([a-z_ +()]+)"', m.group(1)))
    harness = "\n".join([
        re.search(r"const ORDER_CANON = \[[\s\S]*?\];", js).group(0),
        re.search(r"const DISPLAY_ORDER = \[[\s\S]*?\];", js).group(0),
        re.search(r"const CANON_IDX_AT_DISPLAY[\s\S]*?;", js).group(0),
        re.search(r"const DISPLAY_POS_OF_CANON[\s\S]*?;", js).group(0),
        re.search(r"function _canonToDisplay[\s\S]*?\n\}", js).group(0),
        re.search(r"function _displayToCanon[\s\S]*?\n\}", js).group(0),
        f"const seq = Array.from({{length: {n}}}, (_, i) => 'v' + i);",
        "const rt = _displayToCanon(_canonToDisplay(seq));",
        "if (JSON.stringify(rt) !== JSON.stringify(seq)) {",
        "    console.error('FAIL roundtrip'); process.exit(1);",
        "}",
        "const rt2 = _canonToDisplay(_displayToCanon(seq));",
        "if (JSON.stringify(rt2) !== JSON.stringify(seq)) {",
        "    console.error('FAIL roundtrip 2'); process.exit(1);",
        "}",
        "console.log('OK');",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(harness)
        tmp = f.name
    r = subprocess.run(["node", tmp], capture_output=True, text=True, timeout=30)
    if r.returncode != 0 or "OK" not in r.stdout:
        _fail(f"mapping roundtrip failed: {r.stdout} {r.stderr}")

    print("PASS: v588 -- lanczos kernel matches the independent sinc formula, "
          "shrink stretches support, rows sum to 1, edges clamp; wiring + "
          "loud cpu-mismatch + untouched default; marker set only by the "
          "proven branch, spoken on load; canon<->display roundtrip identity")


if __name__ == "__main__":
    main()
