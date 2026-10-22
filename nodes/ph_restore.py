# -*- coding: ascii -*-
"""The RESTORE path of Polyhedron Power Upscale -- v1194 (v1197: on Core's LIBRARY level; v1199: the node loads it).

v1199 (Frank 21.10.: "Entweder holen wir uns das Modell mit in die Node oder wir machen das extern wie bei ESRGAN,
dann gibt es aber keinen Check und keinen Download"): the restoration model is picked ON the node ('restore_model'),
loaded here with its own VAE (choices / resolve / load) -- the door on the node offers exactly what the pick lists.
'model' keeps the refine model, so restore and refine run in ONE node (restore_order: after / before the refine).
A restoration model wired into 'model' is refused with the way to the pick.

A restoration model (Core's SeedVR2 family: one-step flow, conditioned on the latent of the picture itself, no
prompt) restores the frames: the pixel path delivers the frames at
the target size as before (ESRGAN / lanczos / RTX VSR, 'upscale_by'), and this module runs the family's recipe
over them -- pad, encode with the model's VAE, split into temporal chunks that fit the card, condition each
chunk on its own latent, ONE sampler step, crossfade the chunks back together, decode, align and colour-match to
the frames that went in.

The recipe is Core's (comfy_extras/nodes_seedvr.py and its template: Preprocess -> VAEEncodeTiled -> TemporalChunk
-> Conditioning + KSampler(1 step, cfg 1, euler/simple, denoise 1) -> TemporalMerge -> VAEDecodeTiled ->
PostProcessing). v1197: it is written here on Core's LIBRARY level -- comfy.sample, comfy.model_management,
comfy.ldm.seedvr.color_fix / constants -- because the house law since v1063 (Frank's audit) forbids importing
Core's NODE modules (nodes, comfy_extras.*): a node module is a UI surface that moves; the library is the
contract. The chunk law (constants), the crossfade (Hann over the middle third) and the colour transfer
functions are Core's own; what is re-stated here is the plumbing between them, step for step as the template.

General by construction: the family is detected on the wired MODEL (Core's model class, not a file name), the VAE
is checked against it (a wrong VAE is refused with the reason), the widgets name no model.
"""
import json
import math
import os
import threading
import time

import torch
import torch.nn.functional as F

RESTORE_COLORS = ("lab", "wavelet", "adain", "none")
RESTORE_ORDERS = ("after refine", "before refine")      # v1199
OFF = "off"
MODEL_FOLDER, VAE_FOLDER = "diffusion_models", "vae"
FAMILY_TAG = "seedvr"                                     # v1199: a restoration checkpoint of the family carries it
TILE = dict(tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8)   # Core's template values
# v1206 (Frank's U4, 21.10.: OOM in the decode at 1056x1856 x141 -- the refine model still held the card, the chunk
# law saw 3.46 GiB free and planned ONE frame per chunk): the ladder the VAE steps down on an out-of-memory, the
# first rung is Core's template; and the floor of the auto plan -- below 5 frames SeedVR2 has no temporal context.
TILE_LADDER = (TILE, dict(tile_size=256, overlap=32, temporal_size=32, temporal_overlap=4),
               dict(tile_size=256, overlap=32, temporal_size=16, temporal_overlap=4),
               dict(tile_size=128, overlap=16, temporal_size=8, temporal_overlap=2))
MIN_AUTO_FRAMES = 5
BEAT_S = 15.0
SAMPLER, SCHEDULER, STEPS, CFG, DENOISE = "euler", "simple", 1, 1.0, 1.0        # the template's KSampler


# ------------------------------------------------------------------ v1199 the pick on the node
def _ww():
    try:
        from . import ph_world_weights as WW
    except ImportError:  # pragma: no cover - direct import in the guards
        import ph_world_weights as WW
    return WW


def _files(folder):
    """{name: path} of every file ComfyUI lists for a models folder (Core's folder_paths), {} without ComfyUI."""
    try:
        import folder_paths
        out = {}
        for n in folder_paths.get_filename_list(folder):
            p = folder_paths.get_full_path(folder, n)
            if p:
                out[n] = p
        return out
    except Exception:
        return {}


def _registry(folder):
    WW = _ww()
    return [n for n, spec in WW.REGISTRY.items() if spec.get("set") == "restore" and spec.get("folder") == folder]


def model_files():
    """{name: path or None}: the family's checkpoints -- the door's registry files (None = not on disk) plus every
    file in diffusion_models whose name carries the family tag (a checkpoint someone placed by hand)."""
    WW = _ww()
    have = _files(MODEL_FOLDER)
    out = {}
    for n in _registry(MODEL_FOLDER):
        out[n] = have.get(n) or WW.local_path(n)
    for n, p in sorted(have.items()):
        if FAMILY_TAG in n.lower() and "vae" not in n.lower():
            out.setdefault(n, p)
    return out


def choices():
    """The combo: 'off' first, then the checkpoints in the house's list form ('name . size', a missing registry file
    'name . size . not installed' -- the Download sits on the node's line)."""
    WW = _ww()
    return [OFF] + [WW.label(n, p) if p else WW.label(n, None) for n, p in model_files().items()]


def vae_path():
    """The family's own VAE on disk: the door's registry VAE, else a vae file carrying the family tag -- or None."""
    WW = _ww()
    have = _files(VAE_FOLDER)
    for n in _registry(VAE_FOLDER):
        p = have.get(n) or WW.local_path(n)
        if p:
            return p
    for n, p in sorted(have.items()):
        if FAMILY_TAG in n.lower():
            return p
    return None


def resolve(pick):
    """'restore_model' -> None for off, else (model path, vae path); a missing file is refused by name with the way
    to the Download."""
    WW = _ww()
    name = WW.strip(pick)
    if not name or name == OFF:
        return None
    files = model_files()
    p = files.get(name)
    if not p:
        raise ValueError("\u2b21 Power Upscale: restore_model %r is not installed -- the Download on the node's line "
                         "fetches it (into models/%s)" % (name, MODEL_FOLDER))
    v = vae_path()
    if not v:
        need = (_registry(VAE_FOLDER) or ["the family's VAE"])[0]
        raise ValueError("\u2b21 Power Upscale: restore_model needs its own VAE (%s) -- the Download on the node's line "
                         "fetches it (into models/%s)" % (need, VAE_FOLDER))
    return p, v


_LOADED = {"key": None, "model": None, "vae": None}


def load(paths):
    """(model path, vae path) -> (MODEL, VAE) through Core's LIBRARY (comfy.sd), resident until another pick.
    The model is checked to BE a restoration model and the VAE to be its own -- refused by name otherwise."""
    key = tuple(paths)
    if _LOADED["key"] == key and _LOADED["model"] is not None:
        return _LOADED["model"], _LOADED["vae"]
    import comfy.sd
    import comfy.utils
    _LOADED.update(key=None, model=None, vae=None)
    model = comfy.sd.load_diffusion_model(paths[0])
    if restore_family(model) is None:
        raise ValueError("\u2b21 Power Upscale: restore_model %s is no restoration model (one-step, self-conditioned) -- "
                         "pick a checkpoint of the family" % paths[0])
    sd, md = comfy.utils.load_torch_file(paths[1], return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=md)
    try:
        vae.throw_exception_if_invalid()
    except AttributeError:
        pass
    ok, what = vae_matches(vae)
    if not ok:
        raise ValueError("\u2b21 Power Upscale: the VAE found for restore_model (%s) is %s, not the family's video VAE"
                         % (paths[1], what))
    _LOADED.update(key=key, model=model, vae=vae)
    return model, vae


def restore_family(model):
    """The restoration family of a wired MODEL, or None. Witness A: Core's model class; witness B: the
    self-conditioning buffers Core requires of the family (positive_conditioning / negative_conditioning)."""
    inner = getattr(model, "model", None)
    if inner is None:
        return None
    try:
        import comfy.model_base as mb
        cls = getattr(mb, "SeedVR2", None)
        if cls is not None and isinstance(inner, cls):
            return "seedvr2"
    except Exception:
        pass
    dm = getattr(inner, "diffusion_model", None)
    if dm is not None and hasattr(dm, "positive_conditioning") and hasattr(dm, "negative_conditioning"):
        return "seedvr2"
    return None


def vae_matches(vae):
    """Is the wired VAE the family's own (Core's SeedVR2 video VAE)? -> (ok, what it is)."""
    fsm = getattr(vae, "first_stage_model", None)
    try:
        import comfy.ldm.seedvr.vae as sv
        if isinstance(fsm, sv.VideoAutoencoderKLWrapper):
            return True, "SeedVR2 video VAE"
    except Exception:
        pass
    return False, type(fsm).__name__ if fsm is not None else "no VAE"


def chunk_plan(frames_per_chunk):
    """0 = auto (Core's measured VRAM law), else a 4n+1 count."""
    n = int(frames_per_chunk)
    if n <= 0:
        return {"chunking_mode": "auto"}
    if (n - 1) % 4 != 0:
        raise ValueError("Power Upscale: restore_frames must be a 4n+1 frame count (1, 5, 9, 13, 17, 21, ...) or 0 = auto; got %d" % n)
    return {"chunking_mode": "manual", "frames_per_chunk": n}


# ---------------------------------------------------------------- the recipe's pieces (Core's, re-stated)
def pad_frames(frames):
    """Core's Preprocess: (N,H,W,3) 0..1 -> (1,T,H',W',3): clamp, pad to /16 with black (bottom/right), the
    frame count to 4n+1 by repeating the last frame."""
    x = frames[..., :3].clamp(0.0, 1.0).permute(0, 3, 1, 2)          # N,3,H,W
    h, w = int(x.shape[-2]), int(x.shape[-1])
    ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="constant", value=0.0)
    t = int(x.shape[0])
    if t > 1 and (t - 1) % 4 != 0:
        x = torch.cat([x, x[-1:].repeat(4 - ((t - 1) % 4), 1, 1, 1)], 0)
    return x.permute(0, 2, 3, 1).unsqueeze(0).contiguous()


def _free_gib():
    import comfy.model_management as mm
    return mm.get_free_memory(mm.get_torch_device()) / (1024 ** 3)


def to_host(x):
    """v1207: a tensor on the card -> the same frames on the CPU (the restore moves what it needs itself)."""
    if torch.is_tensor(x) and x.device.type != "cpu":
        return x.detach().to("cpu")
    return x


def card_report():
    """v1207: who holds the card -- torch allocated / reserved (this process), CUDA free / total (the driver's view,
    every process), and the models ComfyUI still lists. 'other' = total - CUDA free - torch reserved: memory NOT in
    torch's pool (another process, or an allocator outside torch's). -> (line, dict); on a CPU-only run (None, {})."""
    if not torch.cuda.is_available():
        return None, {}
    import comfy.model_management as mm
    dev = mm.get_torch_device()
    g = 1024 ** 3
    try:
        alloc = torch.cuda.memory_allocated(dev) / g
        reserved = torch.cuda.memory_reserved(dev) / g
        free, total = (v / g for v in torch.cuda.mem_get_info(dev))
    except Exception:
        return None, {}
    models = []
    for lm in list(getattr(mm, "current_loaded_models", [])):
        try:
            name = type(lm.model.model).__name__
            size = float(lm.model.loaded_size()) / g if hasattr(lm.model, "loaded_size") else 0.0
            models.append("%s %.2f GiB" % (name, size))
        except Exception:
            models.append("?")
    other = max(0.0, total - free - reserved)
    d = {"allocated": alloc, "reserved": reserved, "cuda_free": free, "total": total, "other": other, "models": models}
    line = ("torch allocated %.2f GiB, torch reserved %.2f GiB, CUDA free %.2f of %.2f GiB, held outside torch's pool "
            "%.2f GiB; models ComfyUI still lists: %s" % (alloc, reserved, free, total, other,
                                                         ", ".join(models) if models else "none"))
    return line, d


def release_dynamic_buffers():
    """v1208: what Core's executor does after every node when Dynamic VRAM (comfy-aimdo) is on -- prefetch queues,
    cast buffers, the virtual-bar watermarks -- in Core's order, each only if this Core has it. -> names of the calls
    that ran ([] when Dynamic VRAM is off)."""
    import comfy.model_management as mm
    try:
        import comfy.memory_management as cmm
        if not getattr(cmm, "aimdo_enabled", False):
            return []
    except Exception:
        return []
    ran = []
    try:
        import comfy.model_prefetch as mp
        if hasattr(mp, "cleanup_prefetch_queues"):
            mp.cleanup_prefetch_queues()
            ran.append("prefetch queues")
    except ImportError:
        pass
    if hasattr(mm, "reset_cast_buffers"):
        mm.reset_cast_buffers()
        ran.append("cast buffers")
    try:
        import comfy_aimdo.model_vbar as vb
        if hasattr(vb, "vbars_reset_watermark_limits"):
            vb.vbars_reset_watermark_limits()
            ran.append("vbar watermarks")
    except ImportError:
        pass
    return ran


def clear_card(why):
    """v1206: unload every model ComfyUI holds on the card and empty the cache BEFORE the restore plans its chunks
    -- the refine model (or anything earlier in the graph) is not needed while the restoration model runs, and the
    chunk law can only plan with what is free. A later refine loads its model again by itself. -> (before, after) GiB.
    v1207: the breakdown of who holds the card is printed after the clear.
    v1208 (Frank's U4b: 11.28 GiB held outside torch, no model listed): with Dynamic VRAM the card also holds the
    allocator's cast buffers of the models that ran -- Core releases them only AFTER a node (execution.py, the
    finally after get_output_data). The restore runs inside the node that just refined, so it releases them itself,
    the same three calls in Core's order."""
    import gc
    import comfy.model_management as mm
    before = _free_gib()
    mm.unload_all_models()
    gc.collect()
    dyn = release_dynamic_buffers()
    mm.soft_empty_cache(True)
    after = _free_gib()
    print("[PLS] Power Upscale: restore clears the card (%s): free %.2f -> %.2f GiB%s"
          % (why, before, after, " (Dynamic VRAM buffers released: %s)" % ", ".join(dyn) if dyn else ""))
    line, _ = card_report()
    if line:
        print("[PLS] Power Upscale:   card after the clear: " + line)
    return before, after


def plan_chunks(n, h, w, frames_per_chunk):
    """v1207: the chunk plan from the FRAMES' size, before the encode -- a run that cannot fit stops at once, not
    after minutes of encoding. The latent shape is what pad_frames + the VAE make of it (/16 pad, /8 spatial, 4n+1
    frames -> (T-1)/4+1 latent frames)."""
    from comfy.ldm.seedvr import constants as C
    ds = C.BYTEDANCE_VAE_SPATIAL_DOWNSAMPLE
    hp, wp = h + (16 - h % 16) % 16, w + (16 - w % 16) % 16
    t = n if n <= 1 or (n - 1) % 4 == 0 else n + 4 - ((n - 1) % 4)
    shape = (1, C.SEEDVR2_LATENT_CHANNELS, (t - 1) // 4 + 1, hp // ds, wp // ds)
    return chunk_frames(torch.empty(shape, device="meta"), frames_per_chunk)


def chunk_frames(latent, frames_per_chunk):
    """Core's TemporalChunk law: pixel frames per chunk from the free memory (auto), else the given 4n+1 count.
    v1206: the auto plan never goes below MIN_AUTO_FRAMES -- when the law cannot give that, the run stops with the
    numbers instead of restoring frame by frame (no temporal context = flicker); a MANUAL count is the user's word."""
    from comfy.ldm.seedvr import constants as C
    t_latent = int(latent.shape[2])
    t_pixel = 4 * (t_latent - 1) + 1
    if int(frames_per_chunk) <= 0:
        free_gb = _free_gib()
        mpx = (int(latent.shape[0]) * int(latent.shape[3]) * int(latent.shape[4])) * (C.BYTEDANCE_VAE_SPATIAL_DOWNSAMPLE ** 2) / 1e6
        reserve = C.SEEDVR2_CHUNK_RESERVED_GIB + C.SEEDVR2_CHUNK_SIGMA_K * C.SEEDVR2_CHUNK_SIGMA_GIB
        budget = free_gb - reserve
        chunk_latent_max = int(budget / (C.SEEDVR2_CHUNK_GIB_PER_MPX_FRAME * mpx)) if budget > 0 else 0
        fpc = min(4 * (chunk_latent_max - 1) + 1, t_pixel) if chunk_latent_max >= 1 else 0
        floor = min(MIN_AUTO_FRAMES, t_pixel)
        if fpc < floor:
            need = reserve + ((floor - 1) // 4 + 1) * C.SEEDVR2_CHUNK_GIB_PER_MPX_FRAME * mpx
            raise RuntimeError(
                "Power Upscale: the restore does not fit the card at this size -- %.2f GiB free after clearing, "
                "%.2f Mpx per frame; Core's chunk law needs about %.1f GiB for %d frames per chunk (the temporal "
                "minimum). Restore at a smaller size (restore_order 'before refine' on the input, or a lower "
                "upscale_by), or set restore_frames to %d to force it at your own risk."
                % (free_gb, mpx, need, floor, floor))
        print("[PLS] Power Upscale: restore chunks (auto, Core's law): free %.2f GiB, %.2f Mpx/frame -> %d frames per chunk (clip %d)"
              % (free_gb, mpx, fpc, t_pixel))
        return fpc
    return min(int(frames_per_chunk), t_pixel)


class _Beat:
    """v1206: a phase that runs as ONE call (encode, decode) still says it is alive -- a line every BEAT_S."""

    class null:
        """v1210: a do-nothing context (no log mute available)."""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def __init__(self, label):
        self.label = label
        self.t0 = time.monotonic()
        self._stop = threading.Event()
        self._th = None

    def __enter__(self):
        print("[PLS] Power Upscale:   restore %s ..." % self.label)

        def run():
            while not self._stop.wait(BEAT_S):
                e = int(time.monotonic() - self.t0)
                print("[PLS] Power Upscale:   restore %s ... %d:%02d elapsed" % (self.label, e // 60, e % 60))
        self._th = threading.Thread(target=run, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._th is not None:
            self._th.join(1.0)
        e = time.monotonic() - self.t0
        print("[PLS] Power Upscale:   restore %s %s in %.1f s" % (self.label, "failed" if exc[0] else "done", e))
        return False


def _prog():
    """v1210: the shared progress instrument (ph_progress) -- None when it cannot be imported (guards, tooling)."""
    try:
        from . import ph_progress as P
    except ImportError:
        try:
            import ph_progress as P
        except ImportError:
            return None
    return P


def _clock_fmt(sec):
    if sec is None:
        return "no rate yet"
    sec = int(round(float(sec)))
    return "~%d:%02d" % (sec // 60, sec % 60)


def _phase(name, est, clock=None, node_id=None, canvas=(0, 0), steps=None):
    """v1210: a phase of the restore as the joint refine runs its phases -- ph_progress._Phase (heartbeat every 15 s
    with elapsed / left of the estimate, the node HUD, the run clock); falls back to the plain _Beat."""
    P = _prog()
    if P is not None and hasattr(P, "_Phase"):
        return P._Phase(name, est, clock, node_id=node_id, canvas=canvas, step=(0 if steps else None), steps=steps,
                        stage="restore", label="restore", prefix="Power Upscale")
    return _Beat(name)


LAST_STEP_S = {}          # v1210: seconds of the rung that RAN, per op (the rate is learned from it, not the failures)


RUNGS_FILE = "restore_rungs.json"


def _rungs_path():
    """v1209: the learned rungs live next to the learned rates (user directory / polyhedron)."""
    try:
        import folder_paths
        return os.path.join(folder_paths.get_user_directory(), "polyhedron", RUNGS_FILE)
    except Exception:
        return None


def _rungs_load():
    path = _rungs_path()
    try:
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            return d if isinstance(d, dict) else {}
    except Exception:
        pass
    return {}


def _card_tag():
    """The card the rungs were learned on: total memory in whole GiB ('cpu' without CUDA)."""
    try:
        if torch.cuda.is_available():
            return "gpu%d" % int(round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)))
    except Exception:
        pass
    return "cpu"


def _pixel_area(op, x):
    """Pixels per frame the VAE call works on: encode gets frames (.., H, W, 3), decode a latent (.., h, w)."""
    try:
        if op == "encode":
            return int(x.shape[-3]) * int(x.shape[-2])
        from comfy.ldm.seedvr import constants as C
        return int(x.shape[-2]) * int(x.shape[-1]) * C.BYTEDANCE_VAE_SPATIAL_DOWNSAMPLE ** 2
    except Exception:
        return 0


def rung_start(op, area, card=None):
    """v1209: the rung to START on. A rung that ran out of memory at a SMALLER (or equal) frame size runs out at this
    one too -- so the start is the largest rung any learned entry of this card and op at area <= this one needed.
    Never skips a rung that could fit. -> (rung, entry or None)."""
    card = card or _card_tag()
    best, src = 0, None
    for e in _rungs_load().get("entries", []):
        try:
            if e["op"] == op and e["card"] == card and int(e["area"]) <= int(area) and int(e["rung"]) > best:
                best, src = int(e["rung"]), e
        except Exception:
            continue
    return min(best, len(TILE_LADDER) - 1), src


def rung_learn(op, area, rung, card=None):
    """v1209: remember the rung that ran (one entry per op / card / area; written only when it is new or changed)."""
    card = card or _card_tag()
    d = _rungs_load()
    entries = [e for e in d.get("entries", []) if isinstance(e, dict)]
    for e in entries:
        if e.get("op") == op and e.get("card") == card and int(e.get("area", -1)) == int(area):
            if int(e.get("rung", -1)) == int(rung):
                return False
            e["rung"], e["when"] = int(rung), time.strftime("%Y-%m-%d %H:%M")
            break
    else:
        entries.append({"op": op, "card": card, "area": int(area), "rung": int(rung),
                        "when": time.strftime("%Y-%m-%d %H:%M")})
    path = _rungs_path()
    try:
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"entries": entries[-64:]}, fh)
        return True
    except Exception:
        return False


def vae_step(vae, op, x, label, est=None, clock=None, node_id=None):
    """v1206: encode_tiled / decode_tiled down TILE_LADDER -- on an out-of-memory the card is cleared and the next,
    smaller rung runs; any other error, or the last rung failing, is raised. -> (result, rung index).
    v1209 (Frank's U4: three failed decode rungs cost 5:39 before 128 px / 8 frames ran): the walk STARTS on the
    rung learned for this card at this size or smaller (rung_start), and the rung that ran is remembered."""
    import comfy.model_management as mm
    area = _pixel_area(op, x)
    start, src = rung_start(op, area)
    if start > 0:
        t = TILE_LADDER[start]
        print("[PLS] Power Upscale: restore %s starts at tiles %d px / %d frames -- the larger rungs ran out of memory "
              "here before (learned %s at %d px per frame; delete %s in the user directory to probe again)"
              % (label, t["tile_size"], t["temporal_size"], src.get("when", "?"), int(src["area"]), RUNGS_FILE))
    for i in range(start, len(TILE_LADDER)):
        t = TILE_LADDER[i]
        name = "%s (tiles %d px, %d frames)" % (label, t["tile_size"], t["temporal_size"])
        ts = time.monotonic()
        try:
            ph = _phase(name, est if i == start else None, clock, node_id)
            with ph:
                fn = vae.encode_tiled if op == "encode" else vae.decode_tiled
                out = fn(x, tile_x=t["tile_size"], tile_y=t["tile_size"], overlap=t["overlap"],
                         tile_t=t["temporal_size"], overlap_t=t["temporal_overlap"])
            LAST_STEP_S[op] = time.monotonic() - ts
            if not isinstance(ph, _Beat):                # the plain _Beat says "done in" itself
                print("[PLS] Power Upscale:   restore %s done in %.1f s" % (name, LAST_STEP_S[op]))
            if i > start or src is not None:
                rung_learn(op, area, i)
            return out, i
        except Exception as e:
            mm.raise_non_oom(e)
            if i == len(TILE_LADDER) - 1:
                raise
            print("[PLS] Power Upscale: restore %s ran out of memory at tiles %d px / %d frames after %.1f s -- clearing "
                  "the card, next rung %d px / %d frames" % (label, t["tile_size"], t["temporal_size"],
                                                            time.monotonic() - ts,
                                                            TILE_LADDER[i + 1]["tile_size"],
                                                            TILE_LADDER[i + 1]["temporal_size"]))
            clear_card("after an out-of-memory in the %s" % label)


def count_chunks(t_latent, frames_per_chunk, overlap):
    """v1210: how many chunks split_latent makes -- known before the encode, for the plan."""
    t_pixel = 4 * (int(t_latent) - 1) + 1
    if t_pixel <= frames_per_chunk:
        return 1
    chunk_latent = (int(frames_per_chunk) - 1) // 4 + 1
    step = chunk_latent - min(int(overlap), chunk_latent - 1)
    n, start = 0, 0
    while True:
        n += 1
        if start + chunk_latent >= t_latent:
            return n
        start += step


def split_latent(samples, frames_per_chunk, overlap):
    """-> (chunks [tensors], effective overlap) -- Core's TemporalChunk."""
    t_latent = int(samples.shape[2])
    t_pixel = 4 * (t_latent - 1) + 1
    if t_pixel <= frames_per_chunk:
        return [samples], 0
    chunk_latent = (frames_per_chunk - 1) // 4 + 1
    overlap = min(int(overlap), chunk_latent - 1)
    step = chunk_latent - overlap
    chunks = []
    for start in range(0, t_latent, step):
        end = min(start + chunk_latent, t_latent)
        chunks.append(samples[:, :, start:end].contiguous())
        if end >= t_latent:
            break
    return chunks, overlap


def _crossfade_weights(overlap, device, dtype):
    """Core's weights: a Hann fade over the middle third of the overlap, flat shoulders outside."""
    ramp = torch.linspace(0.0, 1.0, steps=overlap, device=device, dtype=dtype)
    ramp = ((ramp - 1.0 / 3.0) / (1.0 / 3.0)).clamp(0.0, 1.0)
    return 0.5 + 0.5 * torch.cos(math.pi * ramp)


def merge_latent(chunks, overlap):
    """Core's TemporalMerge: concatenate, crossfading each overlap."""
    first = chunks[0]
    if len(chunks) == 1:
        return first
    if overlap == 0:
        return torch.cat(chunks, dim=2)
    chunk_latent = int(first.shape[2])
    step = chunk_latent - min(int(overlap), chunk_latent - 1)
    t_total = step * (len(chunks) - 1) + int(chunks[-1].shape[2])
    b, c, _, h, w = first.shape
    merged = torch.empty((b, c, t_total, h, w), device=first.device, dtype=first.dtype)
    merged[:, :, :chunk_latent] = first
    filled = chunk_latent
    for i, chunk in enumerate(chunks[1:], start=1):
        start = i * step
        end = start + int(chunk.shape[2])
        fade = min(filled - start, int(chunk.shape[2]))
        if fade > 0:
            w_prev = _crossfade_weights(fade, chunk.device, chunk.dtype).view(1, 1, fade, 1, 1)
            merged[:, :, start:start + fade] = merged[:, :, start:start + fade] * w_prev + chunk[:, :, :fade] * (1.0 - w_prev)
            merged[:, :, start + fade:end] = chunk[:, :, fade:]
        else:
            merged[:, :, start:end] = chunk
        filled = end
    return merged


def conditioning(model, samples):
    """Core's Conditioning: the family's own text buffers + the chunk's latent (with a ones mask) as 'condition'."""
    from comfy.ldm.seedvr import constants as C
    if samples.ndim != 5 or int(samples.shape[1]) != C.SEEDVR2_LATENT_CHANNELS:
        raise ValueError("Power Upscale: the restore path expects a 5-D latent with %d channels, got %s"
                         % (C.SEEDVR2_LATENT_CHANNELS, tuple(samples.shape)))
    dm = model.model.diffusion_model
    cond = samples.movedim(1, -1).contiguous()
    cond = torch.cat((cond, cond.new_ones(cond.shape[:-1] + (1,))), dim=-1).movedim(-1, 1)
    positive = [[dm.positive_conditioning.unsqueeze(0), {"condition": cond}]]
    negative = [[dm.negative_conditioning.unsqueeze(0), {"condition": cond}]]
    return positive, negative


def sample_chunk(model, samples, seed):
    """ONE sampler step on a chunk -- comfy.sample.sample with the template's KSampler values."""
    import comfy.sample
    pos, neg = conditioning(model, samples)
    noise = comfy.sample.prepare_noise(samples, int(seed))
    return comfy.sample.sample(model, noise, STEPS, CFG, SAMPLER, SCHEDULER, pos, neg, samples,
                               denoise=DENOISE, disable_pbar=True, seed=int(seed))   # v1210: no bar per chunk


def color_match(decoded, reference, color):
    """Core's PostProcessing: crop to the reference, colour transfer (lab per frame / wavelet / adain) on the
    VAE device in memory-sized chunks, back to 0..1; 'none' = the crop alone. decoded (T,H',W',3), reference
    (T,H,W,3) -> (T,H,W,3)."""
    t = min(int(decoded.shape[0]), int(reference.shape[0]))
    h = min(int(decoded.shape[1]), int(reference.shape[1]))
    w = min(int(decoded.shape[2]), int(reference.shape[2]))
    dec = decoded[:t, :h, :w, :3]
    if color == "none":
        return dec.clamp(0.0, 1.0)
    import comfy.model_management as mm
    from comfy.ldm.seedvr import constants as C
    from comfy.ldm.seedvr.color_fix import adain_color_transfer, lab_color_transfer, wavelet_color_transfer
    ref = reference[:t, :h, :w, :3]
    dev = mm.vae_device()
    out_dev = dec.device
    d = dec.mul(2.0).sub(1.0).permute(0, 3, 1, 2)                 # T,3,H,W in -1..1
    r = ref.mul(2.0).sub(1.0).permute(0, 3, 1, 2).to(dtype=d.dtype)
    mult = {"lab": C.SEEDVR2_LAB_SCALE_MULTIPLIER, "wavelet": C.SEEDVR2_WAVELET_SCALE_MULTIPLIER, "adain": C.SEEDVR2_ADAIN_SCALE_MULTIPLIER}[color]
    per_frame = h * w * 3 * max(d.element_size(), C.SEEDVR2_DTYPE_BYTES_FLOOR) * mult
    chunk = max(1, min(t, int((mm.get_free_memory(dev) * C.SEEDVR2_COLOR_MEM_HEADROOM) // max(per_frame, 1))))
    while True:
        try:
            result = torch.empty_like(d)
            for s in range(0, t, chunk):
                e = min(s + chunk, t)
                if color == "lab":                                  # lab: one frame at a time (Core does the same)
                    for i in range(s, e):
                        result[i:i + 1] = lab_color_transfer(d[i:i + 1].to(dev).clone(), r[i:i + 1].to(dev).clone()).to(out_dev)
                else:
                    fn = wavelet_color_transfer if color == "wavelet" else adain_color_transfer
                    result[s:e] = fn(d[s:e].to(dev), r[s:e].to(dev)).to(out_dev)
            break
        except Exception as e:
            mm.raise_non_oom(e)
            if chunk <= 1:
                raise
            chunk = max(1, chunk // C.SEEDVR2_OOM_BACKOFF_DIVISOR)
    return result.permute(0, 2, 3, 1).add(1.0).div(2.0).clamp(0.0, 1.0)


def learned_estimates(rates_key, n, w, h):
    """v1210: the learned seconds per phase of this restoration model at this size ({} without rates) -- the node
    posts them on its run clock up front, so the run's ETA covers the restore before it starts."""
    P = _prog()
    if P is None:
        return {}
    try:
        r = P._rates_load("restore:%s" % rates_key)
    except Exception:
        return {}
    mpf = int(n) * int(w) * int(h) / 1e6
    return {k: r[k] * mpf for k in ("enc", "step", "dec") if r.get(k)}


def plan_banner(fam, n, w, h, fpc, n_chunks, rates, rungs):
    """v1210: the restore's plan as the joint refine prints its own -- (lines, estimates in seconds). Pure."""
    mpf = n * w * h / 1e6
    est = {k: (rates.get(k) * mpf if rates.get(k) else None) for k in ("enc", "step", "dec")}
    total = sum(est.values()) if all(v is not None for v in est.values()) else None
    te, td = TILE_LADDER[rungs[0]], TILE_LADDER[rungs[1]]
    lines = ["=== RESTORE (%s) === %d frames %dx%d (%.0f megapixel-frames) -> %d chunk(s) of %d frames, one step each"
             % (fam, n, w, h, mpf, n_chunks, fpc),
             "  phase 1/3 encode   %s (tiles %d px / %d frames%s)" % (_clock_fmt(est["enc"]), te["tile_size"],
                                                                     te["temporal_size"], ", learned" if rungs[0] else ""),
             "  phase 2/3 sample   %d chunk(s) %s" % (n_chunks, _clock_fmt(est["step"])),
             "  phase 3/3 decode   %s (tiles %d px / %d frames%s)" % (_clock_fmt(est["dec"]), td["tile_size"],
                                                                     td["temporal_size"], ", learned" if rungs[1] else ""),
             ("  total %s -- rates learned from %d run(s), last %s; the clock corrects as each phase measures"
              % (_clock_fmt(total), int(rates.get("runs", 0) or 0), rates.get("when") or "?")) if total is not None else
             "  total: no rate learned yet -- this run measures them; a heartbeat ticks every 15 s"]
    return lines, est


def restore(model, vae, frames, seed, color="lab", overlap=1, frames_per_chunk=0, progress=None,
            clock=None, node_id=None, mute=True, rates_key=None):
    """frames (N,H,W,3) 0..1 -> restored frames (N,H,W,3) at the same size, plus an info dict.
    v1210 (Frank 21.10.: "dieses Rumgehopse mag ich nicht ... das sollte sauber zusammengefasst werden ... ETA"):
    runs as the joint refine runs -- a plan with learned estimates up front, the three phases as heartbeats (elapsed /
    left, node HUD, run clock), Core's staging chatter muted (mute_staging_logs), the sampler's bar off, the rates
    learned per restoration model."""
    fam = restore_family(model)
    if fam is None:
        raise ValueError("Power Upscale: the wired model is no restoration model")
    ok, what = vae_matches(vae)
    if not ok:
        raise ValueError("Power Upscale: the restore path needs the restoration model's OWN video VAE wired into 'vae' "
                         "-- got %s. Load the VAE that ships with the model." % what)
    if str(color) not in RESTORE_COLORS:
        raise ValueError("Power Upscale: restore_color %r is not one of %s" % (color, RESTORE_COLORS))
    chunk_plan(frames_per_chunk)
    P = _prog()
    try:
        from .ph_logmute import MuteStagingLogs
    except ImportError:
        try:
            from ph_logmute import MuteStagingLogs
        except ImportError:
            MuteStagingLogs = None
    n, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    t0 = time.monotonic()
    frames = to_host(frames)                             # v1207: nothing of ours on the card
    free_before, free_after = clear_card("%d frames at %dx%d" % (n, w, h))
    fpc = plan_chunks(n, h, w, frames_per_chunk)        # v1207: BEFORE the encode -- a misfit stops at once
    # v1210: the plan -- chunk count, the rungs each VAE call starts on, the learned rates of this restoration model
    t_pix = n if n <= 1 or (n - 1) % 4 == 0 else n + 4 - ((n - 1) % 4)
    t_lat = (t_pix - 1) // 4 + 1
    fpc = min(fpc, t_pix)
    n_plan = count_chunks(t_lat, fpc, int(overlap))
    hp, wp = h + (16 - h % 16) % 16, w + (16 - w % 16) % 16
    rungs = (rung_start("encode", hp * wp)[0], rung_start("decode", hp * wp)[0])
    section = "restore:%s" % (rates_key or fam)
    rates = P._rates_load(section) if P is not None else {}
    lines, est = plan_banner(fam, n, w, h, fpc, n_plan, rates, rungs)
    for ln in lines:
        print("[PLS] Power Upscale: " + ln)
    if clock is not None:
        for key, e in (("renc:restore", est["enc"]), ("rstep:restore", est["step"]), ("rdec:restore", est["dec"])):
            if key in getattr(clock, "posts", {}) and e:
                clock.prior(key, e)
    mute_cm = MuteStagingLogs(bool(mute), label="Power Upscale (restore)") if MuteStagingLogs else _Beat.null()
    with mute_cm:
        padded = pad_frames(frames)
        t_pad = time.monotonic()
        latent, enc_rung = vae_step(vae, "encode", padded, "encode", est=est["enc"], clock=clock, node_id=node_id)
        t_enc = time.monotonic()
        if clock is not None and "renc:restore" in getattr(clock, "posts", {}):
            clock.measure("renc:restore", t_enc - t_pad)
        fpc = min(fpc, 4 * (int(latent.shape[2]) - 1) + 1)
        chunks, ov = split_latent(latent, fpc, int(overlap))
        done = []
        ts0 = time.monotonic()
        with _phase("sample", est["step"], clock, node_id, steps=len(chunks)) as ph:
            for i, ch in enumerate(chunks):
                done.append(sample_chunk(model, ch, seed))
                if hasattr(ph, "step"):
                    ph.step = i + 1
                if progress is not None:
                    progress(i + 1, len(chunks))
        t_sample = time.monotonic() - ts0
        print("[PLS] Power Upscale:   restore sample done: %d chunk(s) in %.1f s (%.1f s per chunk)"
              % (len(chunks), t_sample, t_sample / max(1, len(chunks))))
        if clock is not None and "rstep:restore" in getattr(clock, "posts", {}):
            clock.measure("rstep:restore", t_sample)
        merged = merge_latent(done, ov)
        n_chunks, fpc_eff = len(chunks), (4 * (int(chunks[0].shape[2]) - 1) + 1) if chunks else 0
        del done, chunks
        clear_card("before the decode")                  # v1206: the restoration model is done -- the VAE gets the card
        t_dec0 = time.monotonic()
        decoded, dec_rung = vae_step(vae, "decode", merged, "decode", est=est["dec"], clock=clock, node_id=node_id)
        t_dec = time.monotonic()
        if clock is not None and "rdec:restore" in getattr(clock, "posts", {}):
            clock.measure("rdec:restore", t_dec - t_dec0)
    if P is not None:
        try:
            P._rates_learn(section, n * w * h / 1e6, LAST_STEP_S.get("encode"), t_sample, LAST_STEP_S.get("decode"))
        except Exception:
            pass
    if decoded.dim() == 5:
        decoded = decoded.reshape(-1, *decoded.shape[2:])
    result = color_match(decoded, frames[..., :3], str(color))
    result = result[:n, :h, :w, :3].to(frames.device, frames.dtype).contiguous()
    total = time.monotonic() - t0
    plan_total = sum(est.values()) if all(v is not None for v in est.values()) else None
    print("[PLS] Power Upscale: restore finished in %d:%02d%s -- encode %.0f s, sample %.0f s, decode %.0f s"
          % (int(total) // 60, int(total) % 60,
             (" (plan %s)" % _clock_fmt(plan_total)) if plan_total is not None else " (first run: rates learned now)",
             t_enc - t_pad, t_sample, t_dec - t_dec0))
    info = {"family": fam, "frames": n, "size": (w, h), "chunks": n_chunks, "overlap": int(ov),
            "frames_per_chunk": fpc_eff,
            "latent": tuple(int(x) for x in latent.shape), "color": str(color),
            "latent_out": tuple(int(x) for x in merged.shape),
            "free_gib": (round(free_before, 2), round(free_after, 2)), "rungs": (enc_rung, dec_rung),
            "t_pad": t_pad - t0, "t_encode": t_enc - t_pad, "t_sample": t_sample, "t_decode": t_dec - t_dec0,
            "t_total": total, "plan": est}
    return result, info


def verdict(info, where):
    """The restore's part of the node's line (v1199: `where` = where it ran -- after / before the refine)."""
    return ("restore: %s over %d frame(s) at %dx%d -- %d chunk(s) of %d frames (overlap %d latent), one step each, "
            "colour %s; %s; encode %.1f s, sample %.1f s, decode %.1f s, total %.1f s"
            % (info["family"], info["frames"], info["size"][0], info["size"][1], info["chunks"], info["frames_per_chunk"],
               info["overlap"], info["color"], where, info["t_encode"], info["t_sample"], info["t_decode"], info["t_total"]))
