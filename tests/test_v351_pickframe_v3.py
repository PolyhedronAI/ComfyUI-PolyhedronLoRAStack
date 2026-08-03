"""
test_v351_pickframe_v3.py  (script-style, GATE-3b)

Guards Stage 1 of the V3 migration: ULSImagePickFrame -> V3 schema.

After the v352 registry refactor, the single comfy_entrypoint lives in
nodes/uls_v3_extension.py (guarded by test_v352), not in the node file. This test
therefore guards what is enduring about Stage 1:
  [1] nodes/uls_pick_frame_v3.py holds the V3 node class with the right schema,
      node_id identical to the legacy key, and a stateless execute() — and it no
      longer carries an entrypoint of its own (that moved to the registry).
  [2] The live version triple.
  [3] Behaviour: the legacy pick() logic, which execute() mirrors verbatim.

Run as a plain script:  python3 tests/test_v351_pickframe_v3.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
if PACK not in sys.path:
    sys.path.insert(0, PACK)


def _read(*parts):
    with open(os.path.join(PACK, *parts), encoding="utf-8") as fh:
        return fh.read()


def main():
    fails = []

    def check(num, label, cond):
        print(f"[{num}] {label}: {'PASS' if cond else 'FAIL'}")
        if not cond:
            fails.append(label)

    # ---- [1] V3 node source structure -------------------------------------
    v3 = _read("nodes", "uls_pick_frame_v3.py")
    check(1, "class ULSImagePickFrameV3(io.ComfyNode)",
          "class ULSImagePickFrameV3(io.ComfyNode)" in v3)
    check(1, "define_schema present", "def define_schema" in v3)
    check(1, "node_id identical to legacy key",
          'node_id="ULSImagePickFrame"' in v3)
    check(1, "category preserved", 'category="Polyhedron/Wan"' in v3)
    check(1, "io.Image.Input('images')", 'io.Image.Input("images")' in v3)
    check(1, "io.Int.Input frame_index",
          ('io.Int.Input(' in v3) and ('"frame_index"' in v3))
    check(1, "io.Image.Output present", "io.Image.Output(" in v3)
    check(1, "stateless execute -> NodeOutput",
          ("def execute" in v3) and ("io.NodeOutput" in v3))
    check(1, "entrypoint moved out to the registry",
          "def comfy_entrypoint" not in v3)

    # ---- [2] live triple --------------------------------------------------
    check(2, 'pyproject version = "3.65.0"',
          'version = "3.65.0"' in _read("pyproject.toml"))
    check(2, "banner 'Polyhedron Suite  v365' (two spaces)",
          "Polyhedron Suite  v365" in _read("__init__.py"))
    check(2, 'uls_compat PLUGIN_VERSION = "v365"',
          'const PLUGIN_VERSION = "v365";' in _read("web", "js", "uls_compat.js"))

    # ---- [3] behaviour of the verbatim-ported pick() logic -----------------
    try:
        import numpy as np
    except Exception:
        print("[3] behaviour: SKIPPED (numpy not available)")
    else:
        from nodes.wan_frame_inflate import ULSImagePickFrame
        node = ULSImagePickFrame()
        arr = np.zeros((5, 2, 2, 3), dtype=np.float32)
        for i in range(5):
            arr[i] = float(i)
        mid = node.pick(arr, -1)[0]
        check(3, "middle frame (-1 -> n//2 == 2)",
              mid.shape[0] == 1 and float(mid[0, 0, 0, 0]) == 2.0)
        exp = node.pick(arr, 0)[0]
        check(3, "explicit index 0", float(exp[0, 0, 0, 0]) == 0.0)
        hi = node.pick(arr, 99)[0]
        check(3, "clamp high (99 -> 4)", float(hi[0, 0, 0, 0]) == 4.0)
        empty = np.zeros((0, 2, 2, 3), dtype=np.float32)
        out = node.pick(empty, -1)[0]
        check(3, "empty batch passes through", out.shape[0] == 0)

    print("=" * 60)
    if fails:
        print("RESULT: *** FAIL *** ->", ", ".join(fails))
        return 1
    print("RESULT: ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
