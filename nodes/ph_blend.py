# -*- coding: ascii -*-
"""Polyhedron Blend (v1218) -- up to three layers over a base, Photoshop modes, pictures or whole clips.

Frank 23.10.: "eine Blend-Node ... damit koennen wir Open-Pose und Depth und Canny mergen und gemeinsam als ein Video
oder Bild in Controlnet einspeisen und muessen nicht mehrere Sachen hintereinander fahren" -- "Diese Blend-Node braucht
diverse Einstellungen wie Multiply und so, die klassischen Photoshop-Sachen".

The pack had the maths already: Relight Blend (ph_relight_blend.blend_images) carries the full Photoshop set,
spec-exact, but blends ONE layer onto ONE picture. This node stacks: a base and up to three layers, each layer with
its own mode and opacity, applied bottom to top like a layer panel -- over single pictures or over clips frame by
frame. It is a general image node (category Polyhedron/Image), not a control special case.

One mode is added for maps on black (pose skeletons, canny / scribble / soft edges): "over (black = clear)" -- the
layer's black is transparent, everything else lies on top at the layer's opacity (alpha = the layer's brightest
channel, so anti-aliased lines keep soft edges). That is the usual way to put a skeleton onto a depth map.

Clips meet like this: every input is resized to the base (bilinear); a single frame lies over every frame; a
shorter clip holds its last frame or loops (`at_the_end`); the result has the base's length, or the longest input's
(`length`). The frames are shown in the node (an animated WebP, as Control Map does) when `preview` is on.

About control maps (said, not hidden): a Fun ControlNet Union is trained on one control kind per clip. A blended
map (a skeleton on depth) is a common practice, not a documented input of those models -- compare it against the
chained Apply Controls (one per map, strengths adding up) before relying on it.

numpy core per frame (blend_images), torch at the edges. ASCII only.
"""
import numpy as np

try:
    from .ph_relight_blend import BLEND_MODES as _PS_MODES, blend_images
except ImportError:                      # flat import in the guards
    from ph_relight_blend import BLEND_MODES as _PS_MODES, blend_images

OVER = "over (black = clear)"
MODES = [OVER] + [m for m in _PS_MODES if m != "relight_gain"]
LENGTHS = ["base", "longest"]
AT_END = ["hold last frame", "loop"]
N_LAYERS = 3


def over_black(base, layer, opacity):
    """Layer on base where the layer is not black. base/layer float [..., 3] 0..1."""
    a = np.clip(layer.max(axis=-1, keepdims=True), 0.0, 1.0) * float(opacity)
    return base * (1.0 - a) + layer * a


def frame_at(n_src, i, at_end):
    """Which source frame lies under output frame i (a single frame lies everywhere)."""
    if n_src <= 1:
        return 0
    if i < n_src:
        return i
    return (i % n_src) if at_end == "loop" else n_src - 1


def blend_stack(base, layers, length="base", at_end="hold last frame"):
    """base: float [N,H,W,3]; layers: [(array [M,h,w,3] already at H x W, mode, opacity)] -> float [L,H,W,3]."""
    n = int(base.shape[0])
    if length == "longest":
        n = max([n] + [int(l.shape[0]) for l, _, _ in layers])
    out = np.empty((n,) + tuple(base.shape[1:3]) + (3,), dtype=np.float32)
    for i in range(n):
        f = np.clip(base[frame_at(int(base.shape[0]), i, at_end), ..., :3], 0.0, 1.0).astype(np.float32)
        for arr, mode, op in layers:
            s = np.clip(arr[frame_at(int(arr.shape[0]), i, at_end), ..., :3], 0.0, 1.0).astype(np.float32)
            if float(op) <= 0.0:
                continue
            f = over_black(f, s, op) if mode == OVER else blend_images(f, s, mode, opacity=float(op))
        out[i] = f
    return out


class ULSBlend:
    @classmethod
    def INPUT_TYPES(cls):
        # written out (no loop): the widget order is pinned by the WIDGET_ORDER baseline (#577 law)
        mode_tip = ("How this layer meets what lies below it. 'over (black = clear)': the layer's black is "
                    "transparent (skeletons, canny lines on a depth map). The rest is the Photoshop set: normal, "
                    "multiply, screen, overlay, soft light, difference, hue / color / luminosity ...")
        op_tip = "Opacity of this layer (0 = off)."
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The base (bottom layer): a picture or a clip. Every layer is "
                                               "resized to it."}),
                "mode_1": (MODES, {"default": OVER, "tooltip": mode_tip}),
                "opacity_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": op_tip}),
                "mode_2": (MODES, {"default": OVER, "tooltip": mode_tip}),
                "opacity_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": op_tip}),
                "mode_3": (MODES, {"default": OVER, "tooltip": mode_tip}),
                "opacity_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": op_tip}),
                "length": (LENGTHS, {"default": LENGTHS[0],
                    "tooltip": "Length of the result: the base's, or the longest input's."}),
                "at_the_end": (AT_END, {"default": AT_END[0],
                    "tooltip": "A clip shorter than the result: hold its last frame, or loop it. A single "
                               "picture lies over every frame."}),
                "preview": ("BOOLEAN", {"default": True,
                    "tooltip": "Show the result in the node (a clip as a small animated WebP). To keep it, wire "
                               "`image` into a Save node."}),
            },
            "optional": {
                "layer_1": ("IMAGE", {"tooltip": "Layer 1, right above the base: a picture or a clip."}),
                "layer_2": ("IMAGE", {"tooltip": "Layer 2, above layer 1: a picture or a clip."}),
                "layer_3": ("IMAGE", {"tooltip": "Layer 3, on top: a picture or a clip."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    OUTPUT_TOOLTIPS = ("The blended picture or clip -- e.g. into Apply Control's `control`, or into a Save node.",
                       "What lay where.")
    FUNCTION = "blend"
    CATEGORY = "Polyhedron/Image"
    DESCRIPTION = ("Stacks up to three layers on a base, each with its own blend mode (the Photoshop set plus 'over "
                   "(black = clear)' for lines and skeletons on black) and opacity -- over single pictures or whole "
                   "clips, frame by frame. E.g. pose + canny on depth as ONE control video.")

    def blend(self, image, mode_1, opacity_1, mode_2, opacity_2, mode_3, opacity_3, length, at_the_end, preview,
              layer_1=None, layer_2=None, layer_3=None):
        import torch
        import torch.nn.functional as F
        H, W = int(image.shape[1]), int(image.shape[2])

        def fit(x):
            x = x[..., :3].detach().float().cpu()
            if (int(x.shape[1]), int(x.shape[2])) != (H, W):
                x = F.interpolate(x.movedim(-1, 1), size=(H, W), mode="bilinear", align_corners=False).movedim(1, -1)
            return x.clamp(0, 1).numpy()

        layers, lines = [], []
        for k, (lay, mode, op) in enumerate(((layer_1, mode_1, opacity_1), (layer_2, mode_2, opacity_2),
                                             (layer_3, mode_3, opacity_3)), start=1):
            if lay is None:
                continue
            layers.append((fit(lay), mode, float(op)))
            lines.append("layer %d: %d frame(s) %dx%d -> %s %.2f" % (k, int(lay.shape[0]), int(lay.shape[2]),
                                                                       int(lay.shape[1]), mode, float(op)))
        base = fit(image)
        out = torch.from_numpy(blend_stack(base, layers, length, at_the_end))
        info = "Blend: base %d frame(s) %dx%d, %d layer(s) -> %d frame(s)" % (int(image.shape[0]), W, H, len(layers),
                                                                              int(out.shape[0]))
        if lines:
            info += "\n  " + "\n  ".join(lines)
        else:
            info += " -- no layer wired: the base passes through"
        print("[PLS] " + info.replace("\n", "\n[PLS]"))
        result = (out, info)
        if not preview:
            return result
        try:
            from .ph_control import preview_ui
        except ImportError:
            from ph_control import preview_ui
        ui = preview_ui(out, "ph_blend")
        return {"ui": ui, "result": result} if ui else result


NODE_CLASS_MAPPINGS = {"ULSBlend": ULSBlend}
NODE_DISPLAY_NAME_MAPPINGS = {"ULSBlend": "\u2b21 Polyhedron Blend"}
