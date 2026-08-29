# ph_media_info.py — ⬡ Polyhedron Media Info
#
# A tiny, general-purpose probe: hand it any IMAGE batch and it reads the things
# you usually want to branch on — width, height, frame count, plus duration and
# aspect ratio — and exposes them as typed outputs while showing a one-line
# readout in the node.
#
# Why this is source-agnostic: width/height/frame-count live in the tensor
# itself ([N, H, W, C]), so this works for ANY image stream — not just the
# Media Loader, but also mid-graph after a resize or a sampler. The only thing
# a bare IMAGE can't know is timing, so fps is an optional input (wire the
# Media Loader's fps in); 0 = unknown, and duration is reported as 0.
#
# No heavy imports on purpose: the node only touches `image.shape` and does
# arithmetic, so it loads even when torch/Pillow/OpenCV are unavailable.

OUTPUT_NODE_TAG = "pls_mediainfo"   # UI channel key, read by web/js/ph_media_info.js


def _dims(image):
    """Best-effort (N, H, W) from a ComfyUI IMAGE tensor [N, H, W, C].

    Defensive about odd shapes so the node never throws on a weird input — it
    just reports what it can read.
    """
    shp = list(getattr(image, "shape", []) or [])
    n = int(shp[0]) if len(shp) > 0 else 0
    h = int(shp[1]) if len(shp) > 1 else 0
    w = int(shp[2]) if len(shp) > 2 else 0
    return n, h, w


def _readout(w, h, n, fps, duration, aspect):
    ar_txt = f"{aspect:.2f}:1" if aspect > 0 else "—"
    dur_txt = f"{duration:.2f} s" if duration > 0 else "— s"
    fps_txt = f"{fps:.2f} fps" if fps > 0 else "— fps"
    plural = "s" if n != 1 else ""
    return f"{w} x {h} · {n} frame{plural} · {fps_txt} · {dur_txt} · {ar_txt}"


class ULSMediaInfo:
    """Read width/height/frame-count/fps/duration/aspect from an IMAGE batch
    (plus an optional fps) and expose them as typed outputs with a readout."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Any IMAGE batch [N,H,W,C]. Width/height/frame-count are read "
                               "from the tensor itself, so this works for any source — not "
                               "just the Media Loader, but also mid-graph.",
                }),
            },
            "optional": {
                "fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Optional source fps (e.g. wired from the Media Loader). "
                               "0 = unknown → duration is reported as 0.",
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("width", "height", "frame_count", "fps", "duration_sec", "aspect_ratio")
    FUNCTION = "info"
    CATEGORY = "Polyhedron/IO"
    DESCRIPTION = ("Reads width, height, frame count, fps, duration and aspect out of a "
                   "wired IMAGE batch and exposes them as typed outputs. Feed those into "
                   "latent sizes, save nodes or frame maths instead of typing numbers by "
                   "hand, which a changed input silently invalidates.")
    OUTPUT_NODE = True

    def info(self, image, fps=0.0):
        n, h, w = _dims(image)
        fps = float(fps or 0.0)
        duration = (n / fps) if (fps > 0.0 and n > 0) else 0.0
        aspect = (w / h) if h > 0 else 0.0

        text = _readout(w, h, n, fps, duration, aspect)
        print(f"[PLS] MediaInfo: {text} ({'video' if n > 1 else 'image'})")

        payload = {
            "text": text,
            "width": w, "height": h, "frame_count": n,
            "fps": fps, "duration_sec": duration, "aspect_ratio": aspect,
        }
        # ui -> readout in the node (web/js/ph_media_info.js); result -> outputs.
        return {"ui": {OUTPUT_NODE_TAG: [payload]}, "result": (w, h, n, fps, duration, aspect)}
