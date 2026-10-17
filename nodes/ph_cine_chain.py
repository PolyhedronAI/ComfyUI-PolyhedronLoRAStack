# -*- coding: ascii -*-
"""Polyhedron Cine chain nodes (C3, v1018): Clip Store, Clip Chain Source,
Seam Check.

A chain of MiniMax H3 clips is run clip by clip through the queue, not as a
loop inside one graph: every clip is a finished file, each can be re-rolled on
its own, VRAM does not grow with the chain (Buildkonzept 26.09., section 5).
These three nodes are the plumbing of that chain:

  Clip Store         -- after the Save: writes the clip's AV latent next to
                        the video (<stem>.plsclip.safetensors, cine_clip.py)
                        so the NEXT clip can continue from the latent itself.
  Clip Chain Source  -- the start of the next run: reads the previous clip BY
                        PATH (field lesson 19.09.: a tile picks "some file of
                        the folder", a path picks the file), hands out its
                        last frame, its tail frames, its tail as a VIDEO with
                        sound (for Reference video_1, K4), its tail audio and
                        its stored latent, and says in `info` what it found.
  Seam Check         -- measures the join between clip n and clip n+1 on our
                        own K1 scale (<= 4 holds, >= 15 no anchor) plus the
                        level step per channel. Measures only; the correction
                        is C4.
"""

import os
from fractions import Fraction

import numpy as np
import torch

from . import cine_clip as C
from . import uls_latent_math as M

import uuid

PREVIEW_SUB = "pls_cine"


def _bases():
    """The folders ComfyUI's /view serves, as (type, dir)."""
    try:
        import folder_paths
        return [("output", folder_paths.get_output_directory()),
                ("input", folder_paths.get_input_directory()),
                ("temp", folder_paths.get_temp_directory())]
    except Exception:
        return []


def _strip(frames, tag):
    """Write strip thumbnails into temp/pls_cine; [] if anything fails --
    a preview must never break a run."""
    try:
        import folder_paths
        out = os.path.join(folder_paths.get_temp_directory(), PREVIEW_SUB)
        names = C.write_previews(frames, out, "%s_%s" % (tag, uuid.uuid4().hex[:8]))
        return [{"filename": n, "subfolder": PREVIEW_SUB, "type": "temp"} for n in names]
    except Exception as e:
        print("[PLS] Cine preview skipped (%s: %s)" % (type(e).__name__, e))
        return []


try:
    from comfy_api.input_impl import VideoFromComponents
    from comfy_api.util import VideoComponents
except Exception:          # pragma: no cover - only outside ComfyUI
    VideoFromComponents = VideoComponents = None


def _split_latent(latent):
    """{name: tensor} of a LATENT dict: 'video' + 'audio' for a joint AV
    latent, 'samples' for anything else."""
    s = latent.get("samples") if isinstance(latent, dict) else latent
    if getattr(s, "is_nested", False):
        parts = s.unbind()
        out = {"video": parts[0]}
        if len(parts) > 1:
            out["audio"] = parts[1]
        return out
    return {"samples": s}


def _join_latent(tensors):
    if "video" in tensors:
        import comfy.nested_tensor
        parts = [tensors["video"]]
        if "audio" in tensors:
            parts.append(tensors["audio"])
        return {"samples": comfy.nested_tensor.NestedTensor(tuple(parts))}
    return {"samples": tensors["samples"]}


def _snap_17k5(n):
    """Down to the H3 clip grid 5, 22, 39, ... (1 stays 1)."""
    n = int(n)
    if n < 5:
        return max(1, n)
    while n % 17 != 5:
        n -= 1
    return n


class ULSClipStore:
    """Write the clip's AV latent next to its saved video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip":
                    "The sampler's OUTPUT latent of this clip (before the VAE "
                    "decode). The next clip continues from exactly this -- no "
                    "decode/encode round trip."}),
                "path": ("STRING", {"forceInput": True, "tooltip":
                    "The saved video's path -- wire Polyhedron Save's `path` "
                    "output. The companion lands next to it as "
                    "<stem>.plsclip.safetensors, with the video's md5."}),
            },
            "optional": {
                "recipe": ("STRING", {"forceInput": True, "tooltip":
                    "Anything worth keeping with the clip: the prompt_text of "
                    "the Reference/Keyframes node, an info output. Stored as "
                    "text in the companion's header."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("path", "info")
    FUNCTION = "store"
    OUTPUT_NODE = True
    CATEGORY = "Polyhedron/Cine"
    DESCRIPTION = ("Writes this clip's AV latent next to its saved video "
                   "(<stem>.plsclip.safetensors) so the next clip of a chain "
                   "can continue from the latent itself instead of decoded "
                   "pixels. Wire the sampler's latent and Polyhedron Save's "
                   "path. The companion carries the video's md5, so a later "
                   "re-save of the video cannot be paired with a stale latent.")

    def store(self, latent, path, recipe=None):
        if not path or not os.path.isfile(path):
            raise ValueError("[PLS] Clip Store: %r is not a file -- wire "
                             "Polyhedron Save's `path` output" % (path,))
        tensors = _split_latent(latent)
        meta = {"recipe": recipe or ""}
        size = M.minimax_size_from_latent(
            latent.get("samples") if isinstance(latent, dict) else latent)
        if size is not None:
            meta.update({"model": "minimax_h3_av", "width": size[0],
                         "height": size[1], "frames": size[2],
                         "fps": M.MINIMAX_FPS})
        target, head = C.write_sidecar(path, tensors, meta)
        shapes = ", ".join("%s %s" % (k, tuple(v.shape)) for k, v in tensors.items())
        info = ("Clip Store: %s\n  %s\n  video md5 %s"
                % (os.path.basename(target), shapes, head["video_md5"][:12]))
        print("[PLS] " + info.replace("\n", "\n[PLS] "))
        return {"ui": {"text": [info]}, "result": (target, info)}


class ULSClipChainSource:
    """The previous clip, read by path: last frame, tail, tail video, audio,
    stored latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "", "tooltip":
                    "Full path of the PREVIOUS clip's video. By path, never "
                    "by a tile: the 19.09. run anchored on a control PNG "
                    "because a tile picks 'some file of the folder'."}),
                "tail_frames": ("INT", {"default": 22, "min": 1, "max": 362,
                    "tooltip": "How many frames `tail` carries. Snapped DOWN to "
                    "H3's clip grid 5 / 22 / 39 ... (17k+5) -- the lengths a "
                    "guide clip may have. 22 is the usual motion context."}),
                "ref_seconds": ("FLOAT", {"default": 3.0, "min": 0.5,
                    "max": 15.0, "step": 0.5, "tooltip":
                    "Length of `tail_video` / `tail_audio` -- the previous "
                    "clip's end as a Reference video (K4). MiniMax specifies "
                    "2-15 s for a reference video."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "VIDEO", "AUDIO", "LATENT", "STRING")
    RETURN_NAMES = ("last_frame", "tail", "tail_video", "tail_audio",
                    "clip_latent", "info")
    FUNCTION = "read"
    CATEGORY = "Polyhedron/Cine"
    DESCRIPTION = ("The start of the next clip of a chain. Reads the previous "
                   "clip BY PATH and hands out its last frame (K1 -> Keyframes "
                   "first_frame), its tail frames on the 17k+5 grid, its end as "
                   "a VIDEO with sound (K4 -> Reference video_1 + "
                   "video_audio_1), and the AV latent Clip Store kept next to "
                   "it. `info` says what was found -- including a companion "
                   "whose md5 no longer matches the video.")

    @classmethod
    def IS_CHANGED(cls, path, tail_frames, ref_seconds):
        try:
            st = os.stat(path)
            return "%s|%d|%d" % (path, st.st_mtime_ns, st.st_size)
        except Exception:
            return path

    def read(self, path, tail_frames, ref_seconds):
        path = (path or "").strip().strip('"')
        if not path or not os.path.isfile(path):
            raise ValueError("[PLS] Clip Chain Source: %r is not a file -- put "
                             "the FULL path of the previous clip's video" % (path,))
        want_tail = _snap_17k5(tail_frames)
        fps_guess = 24.0
        n_ref = max(want_tail, int(round(ref_seconds * fps_guess)) + 1)
        frames, fps, total, audio = C.decode_tail(path, n_ref)
        fps = fps or fps_guess
        n_ref_real = min(frames.shape[0], int(round(ref_seconds * fps)))
        imgs = torch.from_numpy(frames.astype(np.float32) / 255.0)
        tail = imgs[-min(want_tail, imgs.shape[0]):]
        last = imgs[-1:]
        ref = imgs[-max(1, n_ref_real):]

        tail_audio = None
        a = C.audio_tail(audio, ref.shape[0] / float(fps))
        if a is not None:
            tail_audio = {"waveform": torch.from_numpy(a)[None], "sample_rate": audio[1]}
        tail_video = None
        if VideoFromComponents is not None:
            tail_video = VideoFromComponents(VideoComponents(
                images=ref, frame_rate=Fraction(fps).limit_denominator(1001),
                audio=tail_audio))

        tensors, meta, note = C.read_sidecar(path)
        clip_latent = _join_latent(tensors) if tensors else None

        lines = ["Clip Chain Source: %s" % os.path.basename(path),
                 "  %d frames @ %.3f fps (%.2f s), %dx%d"
                 % (total, fps, total / float(fps), frames.shape[2], frames.shape[1]),
                 "  last_frame = frame %d of %d" % (total - 1, total),
                 "  tail = last %d frame(s)%s" % (tail.shape[0],
                     "" if tail.shape[0] == want_tail else
                     " (asked %d, the clip is shorter)" % want_tail),
                 "  tail_video / tail_audio = last %.2f s%s"
                 % (ref.shape[0] / float(fps),
                    "" if tail_audio is not None else " -- the video has NO audio track"),
                 "  clip_latent: %s" % note]
        if meta:
            lines.append("  stored with: %s" % ", ".join(
                "%s=%s" % (k, meta[k]) for k in ("width", "height", "frames") if k in meta))
        if clip_latent is None:
            lines.append("  -> clip_latent is EMPTY: continuing from the latent "
                         "needs Clip Store after the previous clip's Save")
        info = "\n".join(lines)
        print("[PLS] " + info.replace("\n", "\n[PLS] "))
        payload = {"kind": "source", "label": "tail %d frame(s) of %s" % (tail.shape[0], os.path.basename(path)),
                   "strip": _strip(frames[-tail.shape[0]:], "src"), "seam_at": None,
                   "reel": _with_posters(C.reel_items(path, _bases())), "current": path}
        return {"ui": {"pls_cine": [payload]},
                "result": (last, tail, tail_video, tail_audio, clip_latent, info)}


SEAM_SHOW = 8     # frames each side of the join in the strip


class ULSSeamCheck:
    """Measure the join between two clips."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prev": ("IMAGE", {"tooltip":
                    "The END of clip n -- Clip Chain Source `tail` (at least "
                    "one frame; 8 or more lets the level trend be fitted)."}),
                "next": ("IMAGE", {"tooltip":
                    "The START of clip n+1 -- the decoded frames of the new "
                    "clip (at least 2 frames for the noise band)."}),
            },
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("seam", "info")
    FUNCTION = "measure"
    OUTPUT_NODE = True
    CATEGORY = "Polyhedron/Cine"
    DESCRIPTION = ("Measures the join between clip n and clip n+1 on the Cine "
                   "scale: mean luma difference last-vs-first frame against "
                   "the new clip's own frame-to-frame band (<= 4: the anchor "
                   "holds, >= 15: no anchor arrived), plus the level step per "
                   "channel against where clip n was heading. Measures only.")

    def measure(self, prev, next):
        p, q = prev.cpu().numpy(), next.cpu().numpy()
        mx = C.seam_metrics(p, q)
        info = "\n".join(["Seam Check:"] + ["  " + l for l in C.seam_report(mx)])
        print("[PLS] " + info.replace("\n", "\n[PLS] "))
        k = SEAM_SHOW
        a, b = p[-k:], q[:k]
        payload = {"kind": "seam", "strip": _strip(np.concatenate([a, b], 0), "seam"),
                   "seam_at": int(a.shape[0]),
                   "label": "seam %.2f -> %s" % (mx["seam"], C.seam_verdict(mx["seam"]))}
        return {"ui": {"text": [info], "pls_cine": [payload]}, "result": (mx["seam"], info)}


class ULSSeamFix:
    """Trim the overlap and even out the level step at the start of clip n+1."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prev": ("IMAGE", {"tooltip":
                    "The END of clip n -- Clip Chain Source `tail` (8 or more "
                    "frames let the trend of clip n be fitted)."}),
                "next": ("IMAGE", {"tooltip":
                    "ALL decoded frames of clip n+1 (VAE decode). The fixed "
                    "frames go on to the Save."}),
                "trim_head": ("INT", {"default": 0, "min": 0, "max": 361,
                    "tooltip": "Frames to drop from the START of clip n+1 -- "
                    "an overlap that was rendered again (0 = none). The audio "
                    "is trimmed by the same time."}),
                "level_fix": ("BOOLEAN", {"default": True, "tooltip":
                    "Even out the brightness/colour step at the join: a gain "
                    "per channel, full at the first frame, back to 1.0 by "
                    "`fix_frames` -- the clip keeps its own grade after that, "
                    "so a long chain accumulates nothing."}),
                "fix_frames": ("INT", {"default": 12, "min": 2, "max": 48,
                    "tooltip": "Frames over which the correction fades out. "
                    "12 is where the measured H3 continuation step ends."}),
                "fade_ms": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 200.0,
                    "step": 1.0, "tooltip": "Audio fade-in at the join, in "
                    "milliseconds, against a click. 0 = off."}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "The sound of clip n+1 (VAE "
                    "decode audio) -- trimmed with the frames and faded in."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = ("frames", "audio", "seam", "info")
    FUNCTION = "fix"
    OUTPUT_NODE = True
    CATEGORY = "Polyhedron/Cine"
    DESCRIPTION = ("Between the VAE decode of clip n+1 and its Save: drops a "
                   "re-rendered overlap, evens out the brightness/colour step "
                   "at the join with a per-channel gain that fades back to "
                   "identity within a few frames (nothing accumulates down a "
                   "chain), fades the audio in against a click, and reports "
                   "the seam before and after on the Cine scale.")

    def fix(self, prev, next, trim_head, level_fix, fix_frames, fade_ms, audio=None):
        p = prev.cpu().numpy()
        q = next.cpu().numpy()
        t = int(trim_head)
        if t >= q.shape[0]:
            raise ValueError("[PLS] Seam Fix: trim_head %d leaves nothing of the "
                             "%d-frame clip" % (t, q.shape[0]))
        q = q[t:]
        before = C.seam_metrics(p, q)
        gains = None
        if level_fix:
            gains = C.level_gains(p, q, n=int(fix_frames))
            q = C.apply_gains(q, gains)
        after = C.seam_metrics(p, q)
        out_audio = None
        if audio is not None:
            wav = audio["waveform"].cpu().numpy()
            sr = int(audio["sample_rate"])
            cut = int(round(t / float(M.MINIMAX_FPS) * sr))
            wav = wav[..., cut:]
            if fade_ms > 0:
                wav = C.fade_in(wav, sr, fade_ms)
            out_audio = {"waveform": torch.from_numpy(wav), "sample_rate": sr}
        lines = ["Seam Fix: %d frame(s) kept%s" % (q.shape[0], (" (trimmed %d)" % t) if t else ""),
                 "  before: seam %.2f -> %s" % (before["seam"], C.seam_verdict(before["seam"])),
                 "  after:  seam %.2f -> %s" % (after["seam"], C.seam_verdict(after["seam"]))]
        if gains is not None:
            g = gains[0]
            lines.append("  level gain at the join R %.3f G %.3f B %.3f, identity by frame %d"
                         % (g[0], g[1], g[2], gains.shape[0]))
        if audio is not None:
            lines.append("  audio: %s%s" % (("trimmed %.3f s, " % (t / float(M.MINIMAX_FPS))) if t else "",
                                           ("faded in over %.0f ms" % fade_ms) if fade_ms > 0 else "no fade"))
        info = "\n".join(lines)
        print("[PLS] " + info.replace("\n", "\n[PLS] "))
        k = SEAM_SHOW
        payload = {"kind": "seam", "strip": _strip(np.concatenate([p[-k:], q[:k]], 0), "fix"),
                   "seam_at": int(min(k, p.shape[0])),
                   "label": "seam %.2f -> %.2f after the fix" % (before["seam"], after["seam"])}
        return {"ui": {"text": [info], "pls_cine": [payload]},
                "result": (torch.from_numpy(q), out_audio, after["seam"], info)}


class ULSClipAssemble:
    """Join the clips of a chain into one video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "first": ("STRING", {"default": "", "tooltip":
                    "Full path of the FIRST clip of the chain. Every video of "
                    "its folder from this one on (oldest first, the order they "
                    "were made in) is joined."}),
                "last": ("STRING", {"default": "", "tooltip":
                    "Full path of the LAST clip to join. Empty = up to the "
                    "newest clip of the folder."}),
                "name": ("STRING", {"default": "chain", "tooltip":
                    "File name stem of the joined video. It lands in the "
                    "subfolder `assembled` of the chain's folder, so it never "
                    "joins the chain's own reel."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("path", "info")
    FUNCTION = "join"
    OUTPUT_NODE = True
    CATEGORY = "Polyhedron/Cine"
    DESCRIPTION = ("Joins the clips of a chain into one video by packet copy "
                   "-- no re-encode, no quality loss, sound included. The "
                   "range is first..last in the folder's making order, the "
                   "same order the Chain Source reel shows.")

    def join(self, first, last, name):
        first = (first or "").strip().strip('"')
        last = (last or "").strip().strip('"')
        if not first or not os.path.isfile(first):
            raise ValueError("[PLS] Clip Assemble: first %r is not a file" % (first,))
        items = C.reel_items(first, [], limit=100000)
        paths = [it["path"] for it in items]
        a = paths.index(os.path.abspath(first)) if os.path.abspath(first) in paths else None
        if a is None:
            raise ValueError("[PLS] Clip Assemble: %s is not a video of its folder" % first)
        b = len(paths) - 1
        if last:
            if os.path.abspath(last) not in paths:
                raise ValueError("[PLS] Clip Assemble: last %r is not a video in the "
                                 "folder of first" % (last,))
            b = paths.index(os.path.abspath(last))
            if b < a:
                raise ValueError("[PLS] Clip Assemble: last was made BEFORE first -- "
                                 "the range runs in making order")
        chosen = paths[a:b + 1]
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (name or "chain")) or "chain"
        out_dir = os.path.join(os.path.dirname(os.path.abspath(first)), "assembled")
        os.makedirs(out_dir, exist_ok=True)
        n = 1
        while os.path.exists(os.path.join(out_dir, "%s_%05d_.mp4" % (safe, n))):
            n += 1
        out = os.path.join(out_dir, "%s_%05d_.mp4" % (safe, n))
        secs = C.concat_videos(chosen, out)
        info = "\n".join(["Clip Assemble: %d clip(s) -> %s (%.2f s)" % (len(chosen), out, secs)]
                         + ["  %2d  %s" % (i + 1, os.path.basename(p)) for i, p in enumerate(chosen)])
        print("[PLS] " + info.replace("\n", "\n[PLS] "))
        ref = C.view_ref(out, _bases())
        payload = {"kind": "source", "label": "assembled: %d clips, %.2f s" % (len(chosen), secs),
                   "strip": [], "seam_at": None,
                   "reel": _with_posters([{"name": os.path.basename(out), "path": out, "mtime": 0,
                                           "latent": False, "current": True, "view": ref}]),
                   "current": out}
        return {"ui": {"text": [info], "pls_cine": [payload]}, "result": (out, info)}


# --------------------------------------------------------------------------
# route: the reel BEFORE a run -- type a path, see the chain around it
# --------------------------------------------------------------------------

def _with_posters(items):
    """Attach a poster still to every reel item (temp/pls_cine, cached)."""
    try:
        import folder_paths
        out = os.path.join(folder_paths.get_temp_directory(), PREVIEW_SUB)
        os.makedirs(out, exist_ok=True)
    except Exception:
        return items
    for it in items:
        name = C.poster_for(it["path"], out)
        it["poster"] = ({"filename": name, "subfolder": PREVIEW_SUB, "type": "temp"}
                        if name else None)
    return items


async def handle_cine_reel(request):
    from aiohttp import web
    path = request.rel_url.query.get("path", "")
    try:
        items = _with_posters(C.reel_items(path, _bases()))
    except Exception as e:
        return web.json_response({"items": [], "error": "%s: %s" % (type(e).__name__, e)})
    return web.json_response({"items": items})


def register_cine_routes():
    """GET /pls/cine/reel?path=... under the bare and the /api path (the
    uls_routes rule: app.router routes get no /api alias of their own)."""
    try:
        from server import PromptServer
        app = PromptServer.instance.app
    except Exception as e:
        print("[PLS] Cine reel route not registered: %s" % e)
        return 0
    n = 0
    for p in ("/pls/cine/reel", "/api/pls/cine/reel"):
        try:
            app.router.add_route("GET", p, handle_cine_reel)
            n += 1
        except Exception as e:
            print("[PLS] could not add route GET %s: %s" % (p, e))
    return n


NODE_CLASS_MAPPINGS = {
    "ULSClipStore": ULSClipStore,
    "ULSClipChainSource": ULSClipChainSource,
    "ULSSeamCheck": ULSSeamCheck,
    "ULSSeamFix": ULSSeamFix,
    "ULSClipAssemble": ULSClipAssemble,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ULSClipStore": "\u2b21 Polyhedron Clip Store",
    "ULSClipChainSource": "\u2b21 Polyhedron Clip Chain Source",
    "ULSSeamCheck": "\u2b21 Polyhedron Seam Check",
    "ULSSeamFix": "\u2b21 Polyhedron Seam Fix",
    "ULSClipAssemble": "\u2b21 Polyhedron Clip Assemble",
}
