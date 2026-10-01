# v385 -- 3.85.0: the V3 node path loads again

Backend only. Restart ComfyUI. No node, input, output or widget changed
(baselines stay v384); old workflows load unchanged.

## What was wrong

`nodes/uls_v3_extension.py` collects the V3 (comfy_api) versions of seven
nodes. It also imported three V3 modules of nodes that are not part of this
build (Mesh Render, Mesh to File 3D, Camera). That import failed inside
`__init__.py`'s guard, so the V3 path never loaded and all seven nodes ran as
their legacy V1 classes -- silently, since the fallback exists exactly so
nothing disappears.

## The fix

The registry now imports only the V3 modules this build ships. The seven nodes
run as their V3 classes, the same path the development tree runs:

Inspector, Resolve Inspector, Wan Frame Inflate, Pick Frame, Dual Sigma Curve,
Sigma Curve, Sigma List.

On a ComfyUI without `comfy_api.latest` the legacy classes still take over,
as before.

## Checked

- Server interface of the seven nodes, V1 vs V3: same inputs in the same
  order, same defaults, ranges and choices, same outputs, same display names.
- Same results: Sigma List (own list and the HyperFlow preset), Sigma Curve and
  Dual Sigma Curve produce identical sigmas through both classes.
- All bundled templates and a real H3 workflow: load -> save gives the same
  widget values with v384 (V1) and v385 (V3), in both renderers.
- `test_v365_public_build` now pins it: every module the V3 registry imports
  must exist in the build, every key it maps must be a registered node.
