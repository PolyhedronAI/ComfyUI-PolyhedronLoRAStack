"""
test_v352_inflate_v3.py  (script-style, GATE-3b)

Guards Stage 2 of the V3 migration:

  [1] nodes/uls_v3_extension.py is the single central registry: it exports
      V3_NODE_CLASSES (node_id -> V3 class) for BOTH migrated nodes; no
      comfy_entrypoint (the loader ignores it when NODE_CLASS_MAPPINGS exists).
  [2] nodes/uls_pick_frame_v3.py no longer defines an entrypoint/extension (moved
      to the registry); the node class remains; import narrowed to io only.
  [3] nodes/wan_frame_inflate_v3.py has the V3 class with node_id identical to the
      legacy key and the io.Custom("WANVIDIMAGE_EMBEDS") type as BOTH in and out.
  [4] __init__.py uses the central _V3_OK flag, imports V3_NODE_CLASSES, and
      registers BOTH Inflate and Pick into NODE_CLASS_MAPPINGS (V3 when on,
      legacy when off).
  [5] The v352 version triple.
  [6] Behaviour: the legacy inflate() logic, which execute() mirrors verbatim,
      handles inflate / no-op / skip / missing / malformed correctly.

Run as a plain script:  python3 tests/test_v352_inflate_v3.py
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

    # ---- [1] central registry (node_id -> V3 class dict) ------------------
    reg = _read("nodes", "uls_v3_extension.py")
    check(1, "registry imports ULSImagePickFrameV3",
          "from .uls_pick_frame_v3 import ULSImagePickFrameV3" in reg)
    check(1, "registry imports ULSWanFrameInflateV3",
          "from .wan_frame_inflate_v3 import ULSWanFrameInflateV3" in reg)
    check(1, "registry exports V3_NODE_CLASSES dict", "V3_NODE_CLASSES = {" in reg)
    dict_body = reg[reg.find("V3_NODE_CLASSES = {"):]
    check(1, "V3_NODE_CLASSES maps both node_ids",
          ('"ULSImagePickFrame":' in dict_body and "ULSImagePickFrameV3" in dict_body)
          and ('"ULSWanFrameInflate":' in dict_body and "ULSWanFrameInflateV3" in dict_body))
    # the loader ignores comfy_entrypoint when NODE_CLASS_MAPPINGS exists, so v354
    # registers via NODE_CLASS_MAPPINGS and the entrypoint/extension are gone (code)
    reg_code = reg[reg.rfind('"""') + 3:]
    check(1, "no comfy_entrypoint in registry code", "def comfy_entrypoint" not in reg_code)
    check(1, "no PolyhedronV3Extension in registry code", "class PolyhedronV3Extension" not in reg_code)

    # ---- [2] pick_frame stripped of entrypoint ----------------------------
    pf = _read("nodes", "uls_pick_frame_v3.py")
    check(2, "pick_frame keeps its node class",
          "class ULSImagePickFrameV3(io.ComfyNode)" in pf)
    check(2, "pick_frame no longer defines comfy_entrypoint",
          "def comfy_entrypoint" not in pf)
    check(2, "pick_frame no longer defines an extension",
          "PolyhedronV3Extension" not in pf)
    # v364: the invariant is the NARROWING -- io comes in, ComfyExtension does
    # not. HOW io arrives is not the invariant: since v577 it may come through
    # the one door (nodes/ph_comfyapi.py) instead of comfy_api.latest directly.
    # Pinning the spelling made this check go red on a change that kept the
    # invariant perfectly intact.
    check(2, "pick_frame import narrowed (io only, no ComfyExtension)",
          ("import io" in pf) and ("ComfyExtension" not in pf))

    # ---- [3] inflate_v3 structure + custom type ---------------------------
    inf = _read("nodes", "wan_frame_inflate_v3.py")
    check(3, "class ULSWanFrameInflateV3(io.ComfyNode)",
          "class ULSWanFrameInflateV3(io.ComfyNode)" in inf)
    check(3, "node_id identical to legacy key",
          'node_id="ULSWanFrameInflate"' in inf)
    check(3, "custom type io.Custom('WANVIDIMAGE_EMBEDS')",
          'io.Custom("WANVIDIMAGE_EMBEDS")' in inf)
    check(3, "custom type used as input",
          '.Input("image_embeds")' in inf)
    check(3, "custom type used as output",
          '.Output(display_name="image_embeds")' in inf)
    check(3, "io.Int.Input target_latent_frames",
          ('io.Int.Input(' in inf) and ('"target_latent_frames"' in inf))
    check(3, "io.Boolean.Input only_if_single_frame",
          ('io.Boolean.Input(' in inf) and ('"only_if_single_frame"' in inf))
    check(3, "stateless execute -> NodeOutput",
          ("def execute" in inf) and ("io.NodeOutput" in inf))

    # ---- [4] __init__ central flag + V3-or-legacy registration ------------
    init = _read("__init__.py")
    check(4, "central _V3_OK flag", "_V3_OK = False" in init)
    check(4, "imports V3_NODE_CLASSES (not entrypoint)",
          ("from .nodes.uls_v3_extension import V3_NODE_CLASSES" in init)
          and ("import comfy_entrypoint" not in init))
    check(4, "old per-node flag gone", "_PICKFRAME_V3_OK" not in init)
    check(4, "no `if not _V3_OK` blocks remain", "if not _V3_OK" not in init)
    check(4, "Inflate registered V3-or-legacy into NODE_CLASS_MAPPINGS",
          ('NODE_CLASS_MAPPINGS["ULSWanFrameInflate"] = '
           'V3_NODE_CLASSES["ULSWanFrameInflate"] if _V3_OK else ULSWanFrameInflate') in init)
    check(4, "Pick registered V3-or-legacy into NODE_CLASS_MAPPINGS",
          ('NODE_CLASS_MAPPINGS["ULSImagePickFrame"] = '
           'V3_NODE_CLASSES["ULSImagePickFrame"] if _V3_OK else ULSImagePickFrame') in init)

    # ---- [5] v352 triple --------------------------------------------------
    check(5, 'pyproject version = "3.68.0"',
          'version = "3.68.0"' in _read("pyproject.toml"))
    check(5, "banner 'Polyhedron Suite  v368' (two spaces)",
          "Polyhedron Suite  v368" in init)
    check(5, 'uls_compat PLUGIN_VERSION = "v368"',
          'const PLUGIN_VERSION = "v368";' in _read("web", "js", "uls_compat.js"))

    # ---- [6] behaviour of the verbatim-ported inflate() logic --------------
    from nodes.wan_frame_inflate import ULSWanFrameInflate
    node = ULSWanFrameInflate()

    # inflate 1 -> 5 latent frames; num_frames = 4*(5-1)+1 = 17
    e = node.inflate({"target_shape": (16, 1, 60, 104), "num_frames": 1}, 5, True)[0]
    check(6, "inflate 1->5 rewrites target_shape[1] and num_frames",
          e["target_shape"][1] == 5 and e["num_frames"] == 17
          and e.get("_uls_inflated_from") == 1)

    # no-op when already at target (only_if_single_frame=False)
    e = node.inflate({"target_shape": (16, 5, 60, 104), "num_frames": 17}, 5, False)[0]
    check(6, "no-op when already at target",
          e["target_shape"][1] == 5 and "_uls_inflated_from" not in e)

    # skip when only_if_single_frame=True and not single
    e = node.inflate({"target_shape": (16, 3, 60, 104), "num_frames": 9}, 5, True)[0]
    check(6, "skip when only_if_single_frame and frames!=1",
          e["target_shape"][1] == 3 and "_uls_inflated_from" not in e)

    # missing target_shape -> pass through
    e = node.inflate({"num_frames": 1}, 5, True)[0]
    check(6, "missing target_shape passes through",
          "target_shape" not in e)

    # malformed target_shape (len 1) -> pass through unchanged
    e = node.inflate({"target_shape": (16,), "num_frames": 1}, 5, True)[0]
    check(6, "malformed target_shape passes through",
          e["target_shape"] == (16,) and "_uls_inflated_from" not in e)

    print("=" * 60)
    if fails:
        print("RESULT: *** FAIL *** ->", ", ".join(fails))
        return 1
    print("RESULT: ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
