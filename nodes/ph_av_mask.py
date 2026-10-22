# -*- coding: ascii -*-
"""Polyhedron AV mask (v1178) -- per-stream noise masks on the MiniMax H3 AV latent.

THE FOUNDATION for everything that holds part of an H3 clip while the rest is generated: the hard
continuation of a scene (MiniMax Continue `hold`), a fixed soundtrack (MiniMax Audio Lock), and later the
drift control and the context-aware second pass. One library, every node a handle on it -- no node builds
its own mask.

WHAT A MASK IS HERE: the H3 AV latent is a NestedTensor of two streams, video [B,24,T,H/16,W/16] and audio
[B,32,2,T40]. Its `noise_mask` is a NestedTensor of the same two streams with ONE channel:
video [B,1,T,H/16,W/16], audio [B,1,2,T40]. 1 = generate, 0 = keep what the latent holds, a value between =
that row starts part of the way (Core master runs a row of mask m at sigma m * sigma: it sees the kept rows
as clean context; the pinned Core blends them back after every step, classic latent inpainting).
Core packs both streams with the latent (samplers.py, CFGGuider.sample) -- measured on both Cores with a
tiny H3 on the CPU (tests/test_v1178_av_mask.py, A5): the held rows come out unchanged to 1e-6.

THE TIME AXIS (Core's model.py): the video tokens run on a CYCLIC grid -- 1, 4, 4, 4, 4 frames, again and again
(5 tokens = 17 frames); an audio latent frame is 1/40 s, a pixel frame 5/3 of one. A window of n = 17k+5 frames
is minimax_video_latent_t(n) tokens (22 -> 7, 39 -> 12); its sound round(n * 5/3) audio latent frames.

HOLD + RELEASE (the idea of ComfyUI-H3-Motion-Context / Easy Media's "hard context", GPL: read, not
copied): the held prefix is locked (0) and then released in a linear ramp i / (r + 1), i = 1..r, still
INSIDE the prefix -- so new content can grow into the join instead of meeting a wall.

MERGE: two masks on one latent (a soundtrack lock and a hold) meet by the MINIMUM -- what one node keeps,
the other cannot set free.
"""

import collections
import math

FRAME_RESCALE = 5.0 / 3.0       # audio latent frames per pixel frame (Core model.py)
AUDIO_FPS = 40                  # audio latent frames per second
FPS = 24
VIDEO_CH = 24
AUDIO_CH = 32


# --------------------------------------------------------------------------
# the time axis
# --------------------------------------------------------------------------

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)   # Core model.py: the video token grid is CYCLIC -- 5 tokens = 17 frames


def token_spans(tokens):
    """The pixel frames each of the first `tokens` video tokens covers (1, 4, 4, 4, 4, 1, 4, ...)."""
    return [FRAME_PER_TOKEN[k % 5] for k in range(max(0, int(tokens)))]


def frames_of_tokens(tokens):
    """Pixel frames covered by the first `tokens` video tokens (v1181: on Core's cyclic grid -- 7 tokens = 22
    frames, 12 = 39; the exact inverse of uls_latent_math.minimax_video_latent_t on 17k+5)."""
    return sum(token_spans(tokens))


def video_tokens(frames):
    """Tokens of a window of `frames` pixel frames from frame 0: the fewest tokens that cover them, on Core's
    cyclic grid (22 -> 7, 39 -> 12, 124 -> 37 -- uls_latent_math agrees on every 17k+5)."""
    f, t, have = int(frames), 0, 0
    while have < f:
        have += FRAME_PER_TOKEN[t % 5]
        t += 1
    return t


def tokens_back(tokens, frames):
    """How many of the LAST tokens of a `tokens`-token prefix come closest to `frames` frames counted back from its
    end (a hold's release / a drift's taper is counted from the join; ties take fewer tokens). The grid is cyclic,
    so 16 frames back from a 22-frame join are 5 tokens (17 frames), not 4 (13). 0 frames -> 0; capped at the
    prefix. -> (tokens, frames they cover)."""
    f = int(frames)
    if f <= 0:
        return 0, 0
    best, have, n = (0, 0), 0, 0
    for sp in reversed(token_spans(tokens)):
        have += sp
        n += 1
        if abs(have - f) < abs(best[1] - f):
            best = (n, have)
        if have >= f:
            break
    return best


def audio_steps(frames):
    """Audio latent frames under `frames` pixel frames (Core's shared axis, rounded)."""
    return int(round(int(frames) * FRAME_RESCALE))


def ramp(prefix, release):
    """Mask values of a held prefix of `prefix` rows: locked (0) rows, then a linear release
    i / (r + 1), i = 1..r, INSIDE the prefix. -> (values, locked, r)."""
    prefix = max(0, int(prefix))
    r = max(0, min(int(release), prefix))
    locked = prefix - r
    vals = [0.0] * locked + [i / float(r + 1) for i in range(1, r + 1)]
    return vals, locked, r


# --------------------------------------------------------------------------
# the latent
# --------------------------------------------------------------------------

def _unbind(x):
    if x is None:
        return None
    if getattr(x, "is_nested", False) and hasattr(x, "unbind"):
        return list(x.unbind())
    if isinstance(x, (list, tuple)):
        return list(x)
    return None


def streams(latent, tag="AV mask"):
    """(video, audio) of an H3 AV latent; raises by name for anything else."""
    s = latent.get("samples") if isinstance(latent, dict) else None
    parts = _unbind(s)
    if not parts or len(parts) < 2:
        raise ValueError("%s: not a MiniMax H3 AV latent (Polyhedron Empty Latent, latent_type "
                         "'MiniMax H3 AV' -- video AND sound)" % tag)
    v, a = parts[0], parts[1]
    if getattr(v, "ndim", 0) != 5 or int(v.shape[1]) != VIDEO_CH or getattr(a, "ndim", 0) != 4 \
            or int(a.shape[1]) != AUDIO_CH or int(a.shape[2]) != 2:
        raise ValueError("%s: the latent's streams are %s / %s, an H3 AV latent is video [B,24,T,H,W] + audio "
                         "[B,32,2,T]" % (tag, tuple(getattr(v, "shape", ())), tuple(getattr(a, "shape", ()))))
    return v, a


def masks(latent):
    """The latent's (video, audio) noise masks as full one-channel tensors -- ones where it has none."""
    import torch
    v, a = streams(latent)
    vm = torch.ones((v.shape[0], 1) + tuple(v.shape[2:]), dtype=torch.float32)
    am = torch.ones((a.shape[0], 1) + tuple(a.shape[2:]), dtype=torch.float32)
    nm = latent.get("noise_mask") if isinstance(latent, dict) else None
    old = _unbind(nm)
    if old is None and hasattr(nm, "ndim"):
        old = [nm]                      # v1181: a plain tensor is the VIDEO mask (Core pads the sound with ones)
    if old:
        if len(old) >= 1 and old[0] is not None:
            vm = torch.minimum(vm, _fit(old[0], vm.shape))
        if len(old) >= 2 and old[1] is not None:
            am = torch.minimum(am, _fit(old[1], am.shape))
    return vm, am


def _fit(m, shape):
    """An old mask brought to the stream's mask shape (exact shapes pass; others are interpolated)."""
    import torch
    m = m.float().cpu()
    if tuple(m.shape) == tuple(shape):
        return m
    if m.ndim == len(shape) and tuple(m.shape[2:]) == tuple(shape[2:]):
        m = m[:, :1]
    else:
        mode = "trilinear" if len(shape) == 5 else "bilinear"
        m = m.reshape((-1, 1) + tuple(m.shape[-(len(shape) - 2):]))
        m = torch.nn.functional.interpolate(m, size=tuple(shape[2:]), mode=mode)
    # v1181: the batch -- one mask for every item, or the first `b` of them, or the last one repeated
    b = int(shape[0])
    if m.shape[0] >= b:
        return m[:b].clone()
    return torch.cat([m, m[-1:].expand((b - m.shape[0],) + tuple(m.shape[1:]))], dim=0)


def nested(parts):
    """Core's NestedTensor of the parts (the H3 latent's own wrapper)."""
    import comfy.nested_tensor
    return comfy.nested_tensor.NestedTensor(tuple(parts))


def with_streams(latent, video, audio, vmask, amask):
    """A new latent dict: these streams and these masks; every other key carried over."""
    out = dict(latent)
    out["samples"] = nested((video, audio))
    out["noise_mask"] = nested((vmask, amask))
    return out


def hold_prefix(latent, video=None, audio=None, video_release=4, audio_release=4, tag="AV mask"):
    """Copy `video` (tokens [B,24,t,H,W]) and/or `audio` ([B,32,2,s]) to the START of the latent's streams
    and hold them: locked, then released in a ramp inside the prefix. Masks already on the latent stay
    (minimum). -> (latent, report dict)."""
    import torch
    v, a = streams(latent, tag)
    vm, am = masks(latent)
    v2, a2 = v.clone(), a.clone()
    rep = {"video": None, "audio": None}
    if video is not None:
        t = int(video.shape[2])
        if t < 1 or t >= int(v.shape[2]):
            raise ValueError("%s: the held picture is %d token(s), the clip has %d -- it must be shorter"
                             % (tag, t, int(v.shape[2])))
        if tuple(video.shape[3:]) != tuple(v.shape[3:]) or int(video.shape[1]) != VIDEO_CH:
            raise ValueError("%s: the held picture is %s, the clip's grid is %s -- the same size is needed"
                             % (tag, tuple(video.shape), tuple(v.shape)))
        v2[:, :, :t] = video[:1].to(device=v2.device, dtype=v2.dtype)
        vals, locked, r = ramp(t, video_release)
        row = torch.tensor(vals, dtype=torch.float32).view(1, 1, t, 1, 1)
        vm[:, :, :t] = torch.minimum(vm[:, :, :t], row.expand_as(vm[:, :, :t]))
        rep["video"] = {"tokens": t, "frames": frames_of_tokens(t), "locked": locked, "release": r,
                        "values": [round(x, 3) for x in vals]}
    if audio is not None:
        s = int(audio.shape[-1])
        if s < 1 or s >= int(a.shape[-1]):
            raise ValueError("%s: the held sound is %d audio latent frame(s), the clip has %d -- it must be "
                             "shorter" % (tag, s, int(a.shape[-1])))
        a2[..., :s] = audio[:1].to(device=a2.device, dtype=a2.dtype)
        vals, locked, r = ramp(s, audio_release)
        row = torch.tensor(vals, dtype=torch.float32).view(1, 1, 1, s)
        am[..., :s] = torch.minimum(am[..., :s], row.expand_as(am[..., :s]))
        rep["audio"] = {"steps": s, "seconds": s / float(AUDIO_FPS), "locked": locked, "release": r}
    return with_streams(latent, v2, a2, vm, am), rep


def lock_audio(latent, audio, start=0, strength=1.0, tag="AV mask"):
    """Put `audio` ([B,32,2,s] audio latent) into the latent's sound from audio frame `start` and hold it at
    `strength` (1 = kept exactly, 0 = generated anew): mask 1 - strength over the window. What lies past the
    clip's end is cut (said); the sound before `start` and after the window stays free. Silence or a loop
    after a short sound are made on the WAVEFORM before encoding (the node), never as zeros in the latent --
    a zero latent is not silence. Masks already on the latent stay (minimum). -> (latent, report)."""
    import torch
    v, a = streams(latent, tag)
    vm, am = masks(latent)
    a2 = a.clone()
    total = int(a.shape[-1])
    start = max(0, int(start))
    if start >= total:
        raise ValueError("%s: the sound starts at audio frame %d, the clip's sound has %d" % (tag, start, total))
    src = audio[:1].to(device=a2.device, dtype=a2.dtype)
    n = min(int(src.shape[-1]), total - start)
    a2[..., start:start + n] = src[..., :n]
    keep = float(max(0.0, min(1.0, strength)))
    win = am[..., start:start + n]
    am[..., start:start + n] = torch.minimum(win, torch.full_like(win, 1.0 - keep))
    rep = {"start": start, "steps": n, "total": total, "strength": keep,
           "cut": max(0, int(audio.shape[-1]) - n)}
    return with_streams(latent, v, a2, vm, am), rep


# --------------------------------------------------------------------------
# what the Core does with it
# --------------------------------------------------------------------------

def per_row_in_source(src):
    """Does this H3 model code run masked rows at their own timestep (Core master)?"""
    return "audio_denoise_mask" in src and "mask_row_values" in src


def core_per_row():
    """True on a Core whose H3 model sees held rows as clean context (master); False on the pinned Core
    (classic inpainting: the held rows are blended back after every step); None when unreadable."""
    try:
        import inspect
        import comfy.ldm.minimax.model as mm
        return per_row_in_source(inspect.getsource(mm))
    except Exception:
        return None


def core_line():
    pr = core_per_row()
    if pr is True:
        return "Core: the H3 model runs every held row at its own strength (clean context to the model)"
    if pr is False:
        return ("Core: this Core blends the held rows back after every step (classic latent inpainting) -- it "
                "holds them, but the model does not see them as clean context; Core master does")
    return "Core: the H3 model code could not be read -- how it treats held rows is unknown"


# --------------------------------------------------------------------------
# v1179 -- DRIFT CONTROL: the held prefix's mask recomputed at every step
# --------------------------------------------------------------------------
# The idea (ethanfel/ComfyUI-MiniMaxH3-Contex-Loop "Drift-Control AV", via Easy Media's context_swap; read,
# not copied): a held prefix need not be held still WHILE sampling. At every step its rows get the mask
#     m_k = w_k * sigma_next / sigma
# -- w_k = 1 on the far side of the prefix (its oldest frames), tapering to 0 at the join over `taper`
# rows. During the run the far rows follow the model's own prediction by that share and are pulled back to
# the copied clip by the rest (sigma_next / sigma: the share the step keeps anyway) -- so the NEW frames
# attend to a prefix that may already lean toward the new look, while its motion stays the clip's. The row
# at the join is the copied clip at every step. At the last step the ratio is 0: the prefix comes out as
# the copied clip (measured) -- it is the repeated head the film leaves out anyway. The sound keeps its own
# (static) mask. On Core master the model is told the live mask too (its own token-grid pooling, model_base
# _denoise_mask_values), so the per-row timesteps match what the sampler blends; on the pinned Core only the
# sampler side runs (said).

DRIFT_KEY = "pls_av_drift_control"


def drift_weights(prefix, taper):
    """Weights of a prefix of `prefix` rows: 1 on the far side, then a taper to 0 at the join over `taper`
    rows: (taper-1)/taper ... 0. -> list."""
    prefix = int(prefix)
    taper = max(1, min(int(taper), prefix))
    if prefix < 1:
        raise ValueError("drift control: the held prefix is empty")
    return [1.0] * (prefix - taper) + [(taper - j - 1) / float(taper) for j in range(taper)]


def schedule(sigmas):
    """The sampler's sigmas as a descending list of finite values >= 0 (duplicates dropped)."""
    vals = []
    for v in (sigmas.detach().float().reshape(-1).cpu().tolist() if hasattr(sigmas, "detach") else list(sigmas or ())):
        v = float(v)
        if math.isfinite(v) and v >= 0.0:
            vals.append(v)
    return sorted(set(vals), reverse=True)


def drift_ratio(sigma, sigmas):
    """sigma_next / sigma on the schedule (0 at or below the last step)."""
    s = float(sigma)
    if not math.isfinite(s) or s <= 0.0:
        return 0.0
    tol = max(1e-7, abs(s) * 1e-6)
    for c in schedule(sigmas):
        if c < s - tol:
            return max(0.0, min(1.0, c / s))
    return 0.0


def held_prefix(latent):
    """Leading video tokens whose mask is below 1 (the prefix a hold copied in). 0 = none."""
    vm, _am = masks(latent)
    rows = vm.amax(dim=(0, 1, 3, 4))
    n = 0
    for x in rows.tolist():
        if x >= 1.0 - 1e-6:
            break
        n += 1
    return n


class DriftState:
    """The live mask: the sampler's denoise_mask_function + an apply-model wrapper that tells the model."""

    def __init__(self, prefix, taper):
        self.prefix = int(prefix)
        self.weights = drift_weights(prefix, taper)
        self.log = collections.deque(maxlen=64)   # (sigma, ratio) per call -- the guard reads it (v1181: capped)
        self.live = None              # (packed live mask, latent shapes, inner model, sigma) of the current step
        self.per_row = False          # set at install: does the model read per-row masks (Core master)?

    def mask_function(self, sigma, denoise_mask, extra_options=None):
        import torch
        eo = extra_options or {}
        s = float(torch.as_tensor(sigma).detach().float().reshape(-1)[0])
        ratio = drift_ratio(s, eo.get("sigmas", ()))
        inner = getattr(eo.get("model"), "inner_model", None)
        shapes = getattr(inner, "latent_shapes", None)
        if not shapes or len(shapes) < 2 or len(shapes[0]) != 5:
            return denoise_mask                       # not the H3 AV pack: leave it alone
        vshape = tuple(int(x) for x in shapes[0])
        n_v = int(math.prod(vshape[1:]))
        if denoise_mask.ndim != 3 or int(denoise_mask.shape[-1]) < n_v or self.prefix >= vshape[2]:
            return denoise_mask
        out = denoise_mask.clone()
        vid = out[..., :n_v].reshape(vshape)
        w = torch.tensor(self.weights, dtype=vid.dtype, device=vid.device) * ratio
        # v1181: only the rows the hold copied in (their incoming mask is below 1) -- a free row is never touched,
        # whatever `prefix` says (a longer override, a latent of another run with its own mask)
        incoming = denoise_mask[..., :n_v].reshape(vshape)[:, :, :self.prefix]
        held = incoming.amax(dim=(0, 1, 3, 4)) < 1.0 - 1e-6
        rows = w.view(1, 1, self.prefix, 1, 1).expand_as(vid[:, :, :self.prefix])
        vid[:, :, :self.prefix] = torch.where(held.view(1, 1, -1, 1, 1), rows, vid[:, :, :self.prefix])
        self.log.append((s, ratio))
        self.live = (out, shapes, inner, s)
        return out

    def apply_model_wrapper(self, executor, *args, **kwargs):
        if self.live is not None:
            packed, shapes, inner, s_live = self.live
            # v1181: the live mask belongs to THIS step of THIS run -- the same sigma and the same packed length --
            # or it is dropped (a later run with the same model object and no mask, a second pass at another size)
            x, t = (args[0] if args else None), (args[1] if len(args) > 1 else None)
            try:
                s_now = float(t.reshape(-1)[0]) if t is not None else None
            except Exception:
                s_now = None
            same = (x is not None and s_now is not None and int(x.shape[-1]) == int(packed.shape[-1])
                    and abs(s_now - s_live) <= max(1e-6, abs(s_live) * 1e-5))
            if not same:
                self.live = None
                return executor(*args, **kwargs)
            fn = getattr(inner, "_denoise_mask_values", None)
            if self.per_row and callable(fn):
                vals = fn(packed, shapes)
                if "denoise_mask" in vals:
                    kwargs["denoise_mask"] = vals["denoise_mask"].to(packed.device)
                else:
                    kwargs.pop("denoise_mask", None)
        return executor(*args, **kwargs)


def install_drift(model, prefix, taper):
    """A clone of `model` with the drift's live mask installed. Refuses by name: no clone / model_options,
    another live-mask patch already on the model. -> (model, state)."""
    if model is None or not callable(getattr(model, "clone", None)):
        raise ValueError("drift control: `model` is not a ComfyUI model")
    m = model.clone()
    opts = getattr(m, "model_options", None)
    if not isinstance(opts, dict):
        raise ValueError("drift control: the model has no model_options")
    if callable(opts.get("denoise_mask_function")):
        raise ValueError("drift control: the model already carries a live denoise mask (another drift control, "
                         "or a node like it) -- one per model")
    if not callable(getattr(m, "set_model_denoise_mask_function", None)) or \
            not callable(getattr(m, "add_wrapper_with_key", None)):
        raise ValueError("drift control: this Core has no live-mask hooks on the model patcher")
    from comfy.patcher_extension import WrappersMP
    st = DriftState(prefix, taper)
    st.per_row = core_per_row() is True
    m.set_model_denoise_mask_function(st.mask_function)
    m.add_wrapper_with_key(WrappersMP.APPLY_MODEL, DRIFT_KEY, st.apply_model_wrapper)
    m.model_options[DRIFT_KEY] = st
    return m, st
