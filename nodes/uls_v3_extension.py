"""
Polyhedron V3 node registry.

ComfyUI's custom-node loader registers a pack as EITHER V1 (NODE_CLASS_MAPPINGS)
OR V3 (comfy_entrypoint): nodes.py processes NODE_CLASS_MAPPINGS and `return`s
True BEFORE it reaches the `elif hasattr(module, "comfy_entrypoint")` branch.
Because this pack always exports a non-empty NODE_CLASS_MAPPINGS (the legacy,
non-migrated nodes), a comfy_entrypoint would be silently ignored. So the pack
does NOT use comfy_entrypoint.

Instead the V3 nodes are registered through the SAME NODE_CLASS_MAPPINGS as the
legacy nodes. A V3 io.ComfyNode exposes the full V1 interface (INPUT_TYPES,
RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY, OUTPUT_NODE) via @classproperty,
and ComfyUI's executor unwraps the NodeOutput it returns (execution.py checks
isinstance(r, _NodeOutputInternal)). So a V3 class is a drop-in for the V1 path.

This module is the single place that collects the V3 node classes: one import +
one dict entry per node. __init__.py imports V3_NODE_CLASSES inside a try/except
(the central _V3_OK flag); if comfy_api.latest is unavailable (older ComfyUI),
this import fails, _V3_OK stays False, and __init__.py registers the proven legacy
node for every migrated node instead, so nothing disappears.

Each migrated node lives in its own nodes/*_v3.py module (class only). The dict
key is the node_id (identical to the legacy key) so V3 and legacy are drop-in
interchangeable in NODE_CLASS_MAPPINGS.

Public build (v385): only the V3 classes of nodes this build REGISTERS are
imported here. Until v384 the list still named three modules of nodes that are
not public (Mesh Render, Mesh to File 3D, Camera); the import failed, _V3_OK
stayed False, and every node below ran as its legacy V1 class. The list now
matches the build, so the V3 path loads -- the same path the internal tree runs.

Migrated so far:
  • ULSImagePickFrame        (v351)  — nodes/uls_pick_frame_v3.py
  • ULSWanFrameInflate       (v352)  — nodes/wan_frame_inflate_v3.py
  • ULSInspector             (v353)  — nodes/uls_inspector_v3.py
  • ULSResolveInspector      (v353)  — nodes/uls_resolve_inspector_v3.py
  • ULSWanSplitNoiseSchedule (v353)  — nodes/wan_split_sigma_v3.py
  • ULSUniversalSigmaCurve   (v353)  — nodes/wan_universal_sigma_v3.py
  • ULSSigmaList             (v384)  — nodes/wan_sigma_list_v3.py
"""

from .uls_pick_frame_v3 import ULSImagePickFrameV3
from .wan_frame_inflate_v3 import ULSWanFrameInflateV3
from .uls_inspector_v3 import ULSInspectorV3
from .uls_resolve_inspector_v3 import ULSResolveInspectorV3
from .wan_split_sigma_v3 import ULSWanSplitNoiseScheduleV3
from .wan_universal_sigma_v3 import ULSUniversalSigmaCurveV3
from .wan_sigma_list_v3 import ULSSigmaListV3


# node_id -> V3 class. Keys match the legacy NODE_CLASS_MAPPINGS keys exactly, so
# __init__.py registers the V3 class when comfy_api is available and the proven
# legacy class otherwise, both through NODE_CLASS_MAPPINGS.
V3_NODE_CLASSES = {
    "ULSImagePickFrame":        ULSImagePickFrameV3,
    "ULSWanFrameInflate":       ULSWanFrameInflateV3,
    "ULSInspector":             ULSInspectorV3,
    "ULSResolveInspector":      ULSResolveInspectorV3,
    "ULSWanSplitNoiseSchedule": ULSWanSplitNoiseScheduleV3,
    "ULSUniversalSigmaCurve":   ULSUniversalSigmaCurveV3,
    "ULSSigmaList":             ULSSigmaListV3,
}
