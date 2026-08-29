"""
Polyhedron Interpolate  (ph_interpolate)
========================================
Frame interpolation for the Suite. The ENGINE is not ours: nodes/vfi/rife_arch.py
is the proven MIT IFNet (Practical-RIFE via ComfyUI-Frame-Interpolation), vendored
byte-for-byte and NOT touched.

WHAT THIS NODE IS FOR -- measured on 2026-07-14, and two of the four reasons the
first cut gave did not survive the measuring.

  1. THE MODEL IN THE INCUMBENT'S OWN DROPDOWN DOES NOT LOAD. rife426.pth ships
     its training graph inside the checkpoint: 30 `teacher.*` tensors (the
     distillation teacher) and 10 `caltime.*`. rife_arch.py references neither.
     The upstream node calls load_state_dict() with the default strict=True and
     no filter, so it CRASHES on the very model its dropdown offers -- the model
     RIFE's author explicitly recommends for footage out of a diffusion model.
     We strip the training heads BY NAME and keep strict=True on the rest, so a
     genuinely wrong architecture still fails loudly. This alone is the node.

  2. KNOBS THAT LIE. `fast_mode` does nothing from arch 4.5 up (contextnet is
     gone). `ensemble` is silently forced to False on 4.26. Both are declared per
     arch, forced off in code, and named in the console. A knob that shows a
     value it does not honour teaches a false model of the machine; the user then
     reasons correctly from it and gets the wrong answer.

  3. THE ALIGNMENT TAX -- the largest number in this file. IFNet.forward pads to
     mod 64 itself, with ZEROS, bottom-right, and crops back (line 732). A canvas
     that is not a multiple of 64 therefore gets a black ring, and the flow
     estimator tries to track it. Ground truth, controlled 8px translation:

         canvas 1075 (dial 1.40)  ->  49.90 dB whole /  39.80 dB border  [4.7]
         canvas 1088 (mod 64)     ->  59.32 dB whole /  55.90 dB border
                                      +9.42 dB          +16.09 dB

     Free, and worth ~16 dB at the border. So this node does not pad. It WARNS.

  4. WHAT THE FIRST CUT GOT WRONG, kept here because the error is instructive.
     It padded the frames itself -- replicate, centred -- believing the engine did
     not pad at all. It does. And against ground truth our "better" pad was WORSE
     by 4.18 dB: the network was TRAINED with the zeros-bottom-right ring, so a
     centred replicate ring is out of distribution. You do not fight an engine
     that was trained a particular way. You tell the user how to feed it.

  5. MEMORY BY HACK. `clear_cache_after_n_frames` is an empty_cache() on a
     counter: a dial that trades speed for memory and bounds neither. The chunk
     here is computed from mem_get_info; the output tensor is preallocated.

  6. NO GATES. Two identical frames interpolate to noise, not motion. A pair that
     is a scene CUT interpolates to a dissolve -- a certainty at every seam of a
     joined sequence. Both are cheap to detect, free to skip, and neither may ever
     change the OUTPUT COUNT.

THE TIMELINE LAW, confirmed in the field: k frames per pair over N inputs yields
k*(N-1)+1 outputs. The console said "129 frames generated" from 65 inputs at
multiplier 2. 2*(65-1)+1 = 129. Not 130.

THE HONEST HEADLINE: the incumbent spent 6.5s of a 461s prompt. This is not a
speed card. It is a correctness card.
"""
import os
import re

import torch

import comfy.model_management as mm
import folder_paths

from .ph_runclock import _fmt_clock

# v887: ComfyUI's VIDEO type (optional) -- the exact Media Loader / Power
# Upscale pattern. Absent -> the video OUTPUT is simply None; the frames
# output is untouched, so a build without the API loses a convenience, never
# a result.
try:
    from comfy_api.input_impl import VideoFromComponents
    from comfy_api.util import VideoComponents
    from fractions import Fraction
    _HAS_VIDEO_API = True
except Exception:  # pragma: no cover
    VideoFromComponents = VideoComponents = None
    Fraction = None
    _HAS_VIDEO_API = False


# --------------------------------------------------------------------------
# Model registry. Names are the community's, not ours -- a saved workflow that
# stores "rife426.pth" must keep working, so this map is append-only.
# --------------------------------------------------------------------------
CKPT_ARCH = {
    "rife47.pth": "4.7",
    "rife49.pth": "4.7",
    "rife417.pth": "4.17",
    "rife426.pth": "4.26",
}

# v836 (audit B2): where the weights come from, as a ph_weights registry --
# the pack's ONE download door (.part staging, sha256 verify, timeout,
# progress marks). ALL FOUR combo checkpoints are pinned: bytes and sha256
# MEASURED 03.08. from the Fannovel16 release assets themselves (the
# styler00dollar mirror of the old two-mirror list 404s on rife426/rife47
# -- measured in v791 -- so nothing real is lost by the door taking one
# URL). An unpinned name no longer downloads at all: the door only opens
# verified, and the error text names the URL and the folder for a manual
# drop -- the same rule the Cutout engines live by.
_RIFE_BASE = ("https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/"
              "releases/download/models/")
RIFE_WEIGHTS = {
    "rife426.pth": {
        "file": "rife426.pth", "url": _RIFE_BASE + "rife426.pth",
        "sha256": "606421fe2148a9fdeca14e58d94dc339e"
                  "87b87e0ebcc5dec84a50d6f488cfe7b",
        "bytes": 24620531},
    "rife47.pth": {
        "file": "rife47.pth", "url": _RIFE_BASE + "rife47.pth",
        "sha256": "6a8a825ab2750558bdd20dcced386fd82"
                  "b7222c7ba58c11d3b611d9c44f1be63",
        "bytes": 21344827},
    "rife49.pth": {
        "file": "rife49.pth", "url": _RIFE_BASE + "rife49.pth",
        "sha256": "e55fd00f3cc184e3c65961f4bb827a9da"
                  "022e78eed36b055242c0ac30000d533",
        "bytes": 21345274},
    "rife417.pth": {
        "file": "rife417.pth", "url": _RIFE_BASE + "rife417.pth",
        "sha256": "f6b1561354dda3b31606190cba1d1009e"
                  "a001558c1b9c9cef2d03939bb381ee5",
        "bytes": 21497983},
}

# Practical-RIFE ships its TRAINING graph inside the checkpoint. `teacher` is the
# distillation teacher, `caltime` a time-calibration head; neither appears
# anywhere in rife_arch.py, so neither is part of inference. rife426.pth carries
# 40 such tensors -- which is why the upstream node, which loads strict and does
# not filter, CRASHES on the very model its own dropdown offers.
#
# We strip these, and then keep strict=True on what remains. That is the point:
# a checkpoint that is genuinely the wrong architecture must still fail loudly.
# Dropping to strict=False would swallow the real error along with this one.
TRAINING_ONLY_PREFIXES = ("teacher.", "caltime.")

DTYPES = ["float32", "float16", "bfloat16"]
_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

_model_cache = {}


# ==========================================================================
# PURE LAYER -- stdlib only. The guards exec these. Keep it that way.
# ==========================================================================

def _arch_for(ckpt_name):
    """Resolve a checkpoint filename to its architecture generation.

    Unknown names are a hard error, not a guess. A wrong arch loads a partial
    state dict and produces plausible garbage -- the worst failure mode there is.
    """
    if ckpt_name not in CKPT_ARCH:
        raise ValueError(
            "Unknown checkpoint '%s'. Known: %s. An unknown arch would load a "
            "partial state dict and interpolate garbage that still looks like "
            "video, so this refuses instead of guessing."
            % (ckpt_name, ", ".join(sorted(CKPT_ARCH)))
        )
    return CKPT_ARCH[ckpt_name]


def _modulus(_arch_ver=None):
    """64. Unconditionally, for every arch.

    Not our choice and not our number: IFNet.forward pads to ((h-1)//64+1)*64
    itself, regardless of architecture, and crops back at line 732. This function
    exists so the node can TELL the user where the boundary is. It no longer
    moves a single pixel -- see the header, item 4.
    """
    return 64


def _alignment_cost(w, h):
    """How many invented rows/columns the ENGINE's own pad will bolt on.

    Zeros, bottom-right (F.pad's default: `padding = (0, pw-w, 0, ph-h)`). The
    flow estimator then tries to track that black ring, and it pays at the border.

    Returns (pad_w, pad_h, aligned).
    """
    mod = _modulus()
    pw = ((w - 1) // mod + 1) * mod - w
    ph = ((h - 1) // mod + 1) * mod - h
    return pw, ph, (pw == 0 and ph == 0)


def _nearest_aligned(w, h):
    """The canvas a dial should land on. Upward only -- cropping is the Save's
    job, and it already has a law about it."""
    mod = _modulus()
    return ((w - 1) // mod + 1) * mod, ((h - 1) // mod + 1) * mod


def _scale_list(arch_ver, scale_factor):
    """The pyramid. Five levels for 4.26, four for everything before it."""
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive, got %r" % (scale_factor,))
    if arch_ver == "4.26":
        base = [16.0, 8.0, 4.0, 2.0, 1.0]
    else:
        base = [8.0, 4.0, 2.0, 1.0]
    return [b / scale_factor for b in base]


def _inert_knobs(arch_ver):
    """Which widgets this architecture ignores. The anti-lie law.

    Sourced from the engine itself, not from folklore:
      - contextnet was removed at 4.5, so fast_mode is dead from there up.
      - the 4.26 loader hard-assigns ensemble = False.

    A knob in this set is reported, never silently swallowed.
    """
    inert = set()
    try:
        major, minor = arch_ver.split(".")
        num = int(major) * 100 + int(minor)
    except (ValueError, AttributeError):
        return inert
    if num >= 405:
        inert.add("fast_mode")
    if arch_ver == "4.26":
        inert.add("ensemble")
    return inert


def _timeline(n_frames, multiplier):
    """The task list, and the output count it is contractually bound to.

    Returns (tasks, out_count) where tasks is [(pair_index, t), ...] with
    0 < t < 1, and out_count = multiplier*(n_frames-1) + 1.

    THE LAW, confirmed in the field: 65 inputs at multiplier 2 gave 129 frames,
    not 130. Every input frame survives; each of the (n-1) gaps receives
    (multiplier-1) new frames.
    """
    if n_frames < 2:
        raise ValueError("Interpolation needs at least 2 frames, got %d" % n_frames)
    if multiplier < 1:
        raise ValueError("multiplier must be >= 1, got %d" % multiplier)
    tasks = []
    for pair in range(n_frames - 1):
        for step in range(1, multiplier):
            tasks.append((pair, step / float(multiplier)))
    return tasks, multiplier * (n_frames - 1) + 1


def _fps_plan(n_frames, src_fps, dst_fps):
    """target_fps -> the integer multiplier that lands closest, plus the truth.

    A multiplier is a lie told about time: it changes the frame count and leaves
    the fps to whatever the encoder assumes. Here the user names the fps they
    want, and we report the fps they will actually get -- because only integer
    multipliers keep every source frame on its own exact timestamp.
    """
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError("fps must be positive (src=%r dst=%r)" % (src_fps, dst_fps))
    raw = dst_fps / float(src_fps)
    mult = max(1, int(round(raw)))
    actual = src_fps * mult
    _tasks, out_count = _timeline(n_frames, mult)
    return mult, actual, out_count


def _gate_verdict(mae, static_eps, cut_theta):
    """What to do with a pair, decided BEFORE any inference.

    'dup'    -- the two frames are the same picture. Interpolating identical
                pixels cannot invent motion; it only invents noise.
    'hold'   -- the two frames are different SCENES. Interpolating across a cut
                produces a dissolve. At the seam of a concatenated sequence this
                is not an edge case, it is a certainty.
    'interp' -- do the work.

    Thresholds of 0 disable the respective gate. Note what this function does
    NOT do: it never changes how MANY frames come out. A skipped pair still
    yields its (multiplier-1) frames -- they are copies, not absences. Shifting
    the timeline to "save" frames would desynchronise every downstream fps.
    """
    if static_eps > 0 and mae <= static_eps:
        return "dup"
    if cut_theta > 0 and mae >= cut_theta:
        return "hold"
    return "interp"


def _strip_training_heads(keys):
    """Which state_dict keys survive into the inference model.

    Pure and key-only on purpose: the guard can exec it without torch, and it
    can be proven against the real checkpoints without loading a single tensor.
    """
    return [k for k in keys if not k.startswith(TRAINING_ONLY_PREFIXES)]


_PROBE_PAIRS = 4
_CHUNK_CEIL = 64
_GRIND_FACTOR = 2.0


def _chunk_size(free_bytes, peak_per_pair, floor=1, ceil=_CHUNK_CEIL):
    """How many pairs fit in one GPU call, from a MEASURED peak.

    v598 guessed. `in_flight=6` claimed the engine holds six full-size RGB
    tensors per task. The field said otherwise -- same canvas (768), same
    weights (4.7), same precision, same clip, only the batch changed:

        8 pairs  ->  39 ms/task
        16 pairs ->  43 ms/task
        64 pairs -> 296 ms/task

    Flat, flat, cliff. That is not the cost of a bigger batch, that is the
    driver paging to system RAM. And the peak cannot be computed from outside
    the engine: IFBlock.conv0 strides the resolution down by four, block0
    (c=192) only ever sees scale=8, and ensemble doubles the forwards. Any
    constant I put here would be the same guess wearing a bigger number.

    So the caller MEASURES it (probe chunk, allocator peak) and passes it in.
    This function does arithmetic and nothing else.
    """
    n = int(free_bytes * 0.75 // max(1, int(peak_per_pair)))
    return max(floor, min(ceil, n))


# ==========================================================================
# ENGINE LAYER -- torch from here down.
# ==========================================================================

# ---------------------------------------------------------------------------
# v829 -- THE DIAMOND CASE (stage 2 of the v828 sweep). These checkpoints are
# HOUSE picks: the pack ships their definitions, pins a default and fetches
# them itself -- exactly what the cutout's diamond marks. So every entry
# carries "\u25c8 <name>", plus " \u00b7 <size>" from the disk when the file
# is there, else from the measured pin below, else nothing (the cutout's
# three-step size rule, v752).
#
# The strip regex is the SAME shape as ph_basics._SIZE_SUFFIX on purpose,
# and a guard holds the two equal -- one truth kept by measurement instead
# of an import that would drag comfy.sd into this module's load.
_DECO_SIZE = re.compile(r"\s\u00b7\s[\d.,]+\s?(KB|MB|GB|TB)$")

# Measured off the Fannovel16 release (the header comment above, 2026-07),
# not recalled: the two files that release actually serves.
# v836: DERIVED from the registry -- two places holding the same bytes
# would drift (the house mirror doctrine).
_CKPT_BYTES = {k: v["bytes"] for k, v in RIFE_WEIGHTS.items()}


def _strip_deco(value):
    """Diamond and size off, exact suffix only -- the v828 rule."""
    out = _DECO_SIZE.sub("", str(value)).rstrip()
    if out.startswith("\u25c8 "):
        out = out[2:]
    return out


def _fmt_size(n):
    try:
        n = int(n)
    except Exception:
        return ""
    if n <= 0:
        return ""
    if n >= 1 << 30:
        return "%.1f GB" % (n / float(1 << 30))
    if n >= 1 << 20:
        return "%d MB" % round(n / float(1 << 20))
    return "%d KB" % max(1, round(n / float(1 << 10)))


def _ckpt_entry(name):
    """One combo line: '\u25c8 <name> \u00b7 <size>'."""
    size = ""
    try:
        path, _ = _find_ckpt(name)
        if path:
            size = _fmt_size(os.path.getsize(path))
    except Exception:
        pass
    if not size:
        size = _fmt_size(_CKPT_BYTES.get(name, 0))
    out = "\u25c8 " + name
    if size:
        out += " \u00b7 " + size
    return out


def _ckpt_choices():
    return [_ckpt_entry(n) for n in sorted(CKPT_ARCH.keys())]


def _vfi_dir():
    """models/vfi is ours; the incumbent's ckpts dir is where Frank's weights
    already sit. Look in both, say which one answered."""
    roots = []
    try:
        roots.append(os.path.join(folder_paths.models_dir, "vfi"))
    except Exception:
        pass
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    roots.append(os.path.join(os.path.dirname(here),
                              "ComfyUI-Frame-Interpolation", "ckpts", "rife"))
    return roots


def _find_ckpt(ckpt_name):
    for root in _vfi_dir():
        p = os.path.join(root, ckpt_name)
        if os.path.isfile(p):
            return p, root
    return None, None


def _fetch_ckpt(ckpt_name):
    """Pull the weights on first use -- through the pack's ONE download
    door (v836, audit B2). ph_weights stages to .part, verifies sha256
    against the measured pin, times out, prints MB marks, and installs
    atomically; the old urlretrieve here had none of that. On any failure
    the door's error names the URL and the destination for a manual drop."""
    from . import ph_weights
    return ph_weights.ensure_weights("vfi", ckpt_name, RIFE_WEIGHTS,
                                     tag="Interpolate")


def _load_model(ckpt_name, arch_ver, dtype_str, device):
    key = (ckpt_name, dtype_str)
    if key in _model_cache:
        return _model_cache[key]

    from .vfi.rife_arch import IFNet

    path, root = _find_ckpt(ckpt_name)
    if path is None:
        path = _fetch_ckpt(ckpt_name)
        root = os.path.dirname(path)

    # v835 (audit A2), MEASURED before flipping: the pinned Fannovel16
    # release checkpoints (rife426/rife47) load under weights_only=True
    # with all 198 keys and bit-equal tensors -- they are plain state
    # dicts, the training heads are tensors like everything else. The
    # house rule (four engines already load this way) now holds here too.
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    keep = _strip_training_heads(list(sd.keys()))
    dropped = len(sd) - len(keep)
    sd = {k: sd[k] for k in keep}

    model = IFNet(arch_ver=arch_ver)
    # strict=True on what remains. The training heads are gone by NAME, not by
    # loosening the check -- strict=False would have swallowed a genuinely wrong
    # architecture too, and a partial load produces video that looks almost
    # right, which is the one result nobody ever debugs.
    model.load_state_dict(sd, strict=True)
    if dropped:
        print("[PLS] Interpolate: dropped %d training-only tensor(s) (%s) - they are not part "
              "of inference. The upstream node loads these strict and unfiltered, which is why "
              "%s crashes there." % (dropped, ", ".join(TRAINING_ONLY_PREFIXES), ckpt_name))
    model.eval()
    if _DTYPE_MAP[dtype_str] != torch.float32:
        model = model.to(_DTYPE_MAP[dtype_str])
    model = model.to(device)

    _model_cache[key] = model
    print("[PLS] Interpolate: loaded %s (arch %s, %s) from %s"
          % (ckpt_name, arch_ver, dtype_str, root))
    return model


def _frames_from_video(video):
    """Frames and frame rate out of a wired VIDEO object -- the Save's
    _fps_of semantics made LOUD: the Save may fall back silently because
    it always has a frame_rate widget to fall back to; here the video IS
    the source, so an unreadable one is an error, not a default."""
    try:
        comps = video.get_components()
    except Exception as e:
        raise RuntimeError(
            "Polyhedron Interpolate: the wired video cannot be read "
            "(get_components failed: %r)." % (e,)) from e
    imgs = getattr(comps, "images", None)
    if imgs is None or int(imgs.shape[0]) == 0:
        raise RuntimeError(
            "Polyhedron Interpolate: the wired video carries no frames.")
    try:
        fr = float(comps.frame_rate)
    except Exception:
        fr = 0.0
    return imgs, (fr if fr > 0 else 0.0)


# v890: the stretch machinery lives in ph_audio_stretch (ONE source, shared
# with the free ULSAudioStretch node) and the arithmetic in uls_audio_math.
# Imported lazily inside _retime_audio so a broken audio stack can never cost
# a finished interpolation its import.


def _audio_of(video):
    """v887: the sound of a wired video, or None. Never raises -- a clip with
    no audio is the ordinary case, not a fault."""
    if video is None:
        return None
    try:
        return video.get_components().audio
    except Exception:
        return None



def _retime_audio(audio, audio_mode, n_in, src_fps, n_out, out_fps):
    """v890: the soundtrack, made to follow the picture's own timeline.

    Returns (audio_for_video, note). keep -> untouched (v887). mute -> None.
    stretch to output -> the SHARED machinery in ph_audio_stretch runs the
    plan from uls_audio_math: the tempo is (n_in/src_fps)/(n_out/out_fps),
    computed from the very numbers this node just interpolated with -- never
    re-derived elsewhere. Near-1 tempos are trimmed instead of resynthesised,
    and the trim to the output duration retires the 31 ms tail. Refusals and
    failures fall back to the UNCHANGED audio, said out loud -- a soundtrack
    problem must never cost a finished interpolation."""
    if audio is None or audio_mode == "keep":
        return audio, ""
    if audio_mode == "mute":
        return None, "audio muted"
    try:
        try:
            from .ph_audio_stretch import stretch_audio, trim_audio
            from .uls_audio_math import stretch_plan
        except ImportError:  # pragma: no cover - direct-run fallback
            from ph_audio_stretch import stretch_audio, trim_audio
            from uls_audio_math import stretch_plan
        plan = stretch_plan(n_in, src_fps, n_out, out_fps)
        if plan["action"] == "refuse":
            print("[PLS] Interpolate: " + plan["note"])
            return audio, plan["note"]
        if plan["action"] == "trim":
            out = trim_audio(audio, plan["d_out"])
            print("[PLS] Interpolate: " + plan["note"])
            return out, plan["note"]
        out = stretch_audio(audio, plan["tempo"], pitch_mode="preserve",
                            d_target=plan["d_out"])
        print("[PLS] Interpolate: " + plan["note"])
        return out, plan["note"]
    except Exception as exc:
        note = ("audio retime failed (%s: %s) - original audio kept"
                % (type(exc).__name__, exc))
        print("[PLS] Interpolate: " + note)
        return audio, note


def _build_video(frames, audio, frame_rate):
    """v887: the interpolated frames as a VIDEO, carrying the ORIGINAL audio
    and the NEW rate.

    The house already solved this once -- ph_power_upscale._build_video does
    exactly this for the upscaled frames. Interpolate was the only video stage
    in the tree without it, and that gap is the whole sync bug: with no audio
    path THROUGH the node, the sound has to travel AROUND it, and every route
    around it lands in one of three traps (a VIDEO wired to the Save wins over
    both the frame_rate input AND the image input; an unwired fps output
    writes doubled frames at the old rate; target_fps rounds to an integer
    multiple that a hard-typed rate downstream then contradicts).

    Rate and sound ride INSIDE one object -- the v791 principle, one step
    further. Returns None when this build has no VIDEO API, mirroring the
    Media Loader convention."""
    if not _HAS_VIDEO_API or frames is None or int(frames.shape[0]) == 0:
        return None
    rate = frame_rate if frame_rate else 16
    if not isinstance(rate, Fraction):
        rate = Fraction(rate).limit_denominator(1000000)
    try:
        return VideoFromComponents(VideoComponents(images=frames,
                                                   frame_rate=rate,
                                                   audio=audio))
    except Exception as exc:      # never cost a finished interpolation
        print("[PLS] Interpolate: the video output could not be built "
              "(%s: %s) - the frames output is unaffected."
              % (type(exc).__name__, exc))
        return None


def _sync_note(n_in, src_fps, n_out, out_fps):
    """v887: the duration arithmetic, spelled out.

    THE LAW, worth stating because it is the reassuring half: _timeline puts
    source frame k at output index k*multiplier, and at out_fps = src*mult its
    timestamp is k*mult / (mult*src) = k/src -- its ORIGINAL second. So the
    interpolation itself cannot drift against a soundtrack. What it does do is
    end a hair early: it emits mult*(n-1)+1 frames, not mult*n, so the clip is
    short by (1 - 1/mult)/src seconds -- 31 ms at 16 fps doubled. A fixed tail,
    never a drift. Anything bigger than this note says is a WRITTEN RATE that
    does not match out_fps."""
    if src_fps <= 0 or out_fps <= 0:
        return ""
    d_in = n_in / float(src_fps)
    d_out = n_out / float(out_fps)
    return ("duration %.3fs -> %.3fs (%+.0f ms; every source frame keeps its "
            "own timestamp, so this is a tail, not a drift)"
            % (d_in, d_out, 1000.0 * (d_out - d_in)))


class ULSInterpolate:
    """Frame interpolation with a clean pyramid, honest knobs and counted memory."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (_ckpt_choices(), {
                    "default": _ckpt_entry("rife426.pth"),  # v829
                    "tooltip": "Which RIFE checkpoint to run. Prefer the newest for footage that came out of a"
                               " diffusion model; after a run, interp_info names the loaded arch and any knob "
                               "it ignored."}),
                "rate_mode": (["multiplier", "target_fps"], {
                    "default": "multiplier",
                    "tooltip": "A multiplier changes the frame count and leaves the frame rate "
                               "to whatever the encoder assumes. Name the rate you want instead "
                               "and the node reports the rate you will actually get."}),
                "multiplier": ("INT", {"default": 2, "min": 1, "max": 16}),
                "source_fps": ("FLOAT", {"default": 16.0, "min": 0.1, "max": 240.0, "step": 0.1}),
                "target_fps": ("FLOAT", {"default": 32.0, "min": 0.1, "max": 480.0, "step": 0.1}),
                "precision": (DTYPES, {
                    "default": "float32",
                    "tooltip": "float32 is the default on purpose: this network's coarse-to-fine "
                               "flow is sensitive to half precision, and the weights are 23 MB - "
                               "there is no memory to save. Treat the faster modes as a "
                               "measurement, not a setting."}),
                "ensemble": ("BOOLEAN", {"default": True}),
                "fast_mode": ("BOOLEAN", {"default": True}),
                "scale_factor": ([0.25, 0.5, 1.0, 2.0, 4.0], {"default": 1.0}),
                "static_skip": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.2, "step": 0.001,
                    "tooltip": "Pairs whose two frames differ by less than this (mean absolute "
                               "error, 0-1) are duplicated instead of interpolated. Identical "
                               "pixels cannot produce motion, only noise. 0 disables."}),
                "cut_guard": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Pairs that differ by MORE than this are held, not blended. "
                               "Interpolating across a scene cut yields a dissolve; at the seam "
                               "of a joined sequence that is a certainty, not a risk. 0 disables."}),
                  # APPENDED LAST on purpose: widget values restore by
                  # position (guard #577). "keep" preserves the pre-v890
                  # pass-through behaviour bit for bit.
                  "audio_mode": (["keep", "stretch to output", "mute"], {
                      "default": "keep",
                      "tooltip": "What happens to a wired video's soundtrack. keep: passed "
                                 "through untouched - right whenever the duration does not "
                                 "change (plain fps doubling). stretch to output: retimed by "
                                 "exactly the video's own factor and trimmed to the output "
                                 "duration - the slow-motion mode; speech keeps its pitch and "
                                 "stays on the lips. mute: the video output carries no sound."}),
            },
            "optional": {
                # PIN ORDER NOTE: frames stays the FIRST pin (it merely moved
                # from required to optional, so video-only wiring can pass
                # validation); video sits after it. Wire ONE of the two --
                # interpolate() is loud about none and about both.
                "frames": ("IMAGE", {
                    "tooltip": "Frame batch to interpolate. Wire EITHER this "
                               "OR video, not both."}),
                "video": ("VIDEO", {
                    "tooltip": "A video to interpolate. Its own frame rate is "
                               "used as source_fps (the widget is ignored and "
                               "interp_info says so); if the video names no "
                               "rate, the widget steps back in. No fps wire "
                               "needed -- and none should be used: an fps "
                               "wire from a Save behind a de-selected switch "
                               "carries an ExecutionBlocker that silently "
                               "disables this whole node."}),
            },
        }


    @classmethod
    def VALIDATE_INPUTS(cls, **_kw):
        """v829: the combo decorates values (diamond + size), so a workflow
        saved before this cut carries the bare filename -- accepted here,
        stripped and resolved in interpolate(). Unknown checkpoints still
        fail there, loudly and by name (the v823 wound class)."""
        return True

    # v887: `video` is APPENDED, never inserted. A link stores its origin as a
    # SLOT INDEX, so putting the new output anywhere but the end would
    # re-number every wire in every saved workflow -- the output-side twin of
    # the widget serialisation law (#577). frames stays slot 0.
    # v890 APPEND-ONLY on the OUTPUT side too (the v887 twin rule): links
    # store their origin as a slot INDEX, so `audio` sits BEHIND everything
    # that already existed. frames stays slot 0 forever.
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "STRING", "VIDEO", "AUDIO")
    RETURN_NAMES = ("frames", "frame_count", "fps", "interp_info", "video",
                    "audio")
    OUTPUT_TOOLTIPS = (
        "The interpolated frame batch.",
        "How many frames came out: multiplier*(n-1)+1, never multiplier*n.",
        "The rate these frames are meant to be written at. Wire it to the "
        "Save's frame_rate -- or use the video output and wire nothing.",
        "Readout: arch, counts, gates, timing, and the duration arithmetic.",
        "The interpolated frames, the NEW rate and the source clip's ORIGINAL "
        "audio in one object. Wire this to the Save and the sound stays in "
        "sync without an fps wire; audio only rides along when a video was "
        "wired IN (a bare frame batch carries no sound).")
    FUNCTION = "interpolate"
    CATEGORY = "Polyhedron/Video"
    # v599: the two claims this text used to make were both false, and one of
    # them was false BECAUSE v598 measured it and then forgot to edit the prose.
    #   * "Pads the picture to the modulus" -- it does NOT. Our own pad measured
    #     4.18 dB WORSE than the engine's (it trains with the zeros ring). v598
    #     revoked the pad and left the sentence standing. Eleventh case.
    #   * "sizes its own memory from what the card has free" -- it sized it from
    #     a CONSTANT and called the free-VRAM reading a budget. The field paid
    #     7.6x. Now it probes, and the sentence is finally true.
    DESCRIPTION = ("Interpolates new frames between the ones you have. Leaves the canvas alone: "
                   "the network pads to its own modulus with the black ring it was trained on, "
                   "and a ring of ours on top measured worse -- so the node warns about an "
                   "unaligned canvas instead of quietly 'fixing' it. Skips pairs that are already "
                   "identical and holds pairs that are a scene cut, without ever changing how many "
                   "frames come out. Names its target by frame rate rather than by multiplier. "
                   "Measures what one pair of frames really costs on your card before it commits "
                   "to a batch size, watches the clock for driver paging the allocator cannot see, "
                   "and reports every knob the chosen model ignores. Takes a frame batch "
                   "OR a video: a wired video brings its own frame rate, so no fps wire "
                   "is needed anywhere.")

    def interpolate(self, frames=None, ckpt_name="rife426.pth", rate_mode="multiplier",
                    multiplier=2, source_fps=16.0, target_fps=32.0,
                    precision="float32", ensemble=True, fast_mode=True,
                    scale_factor=1.0, static_skip=0.0, cut_guard=0.0,
                    audio_mode="keep",
                    video=None):
        ckpt_name = _strip_deco(ckpt_name)  # v829: diamond+size decoration
        arch_ver = _arch_for(ckpt_name)
        mod = _modulus(arch_ver)
        inert = _inert_knobs(arch_ver)
        device = mm.get_torch_device()
        fps_note = ""
        if frames is not None and video is not None:
            raise RuntimeError(
                "Polyhedron Interpolate: frames AND video are both wired -- "
                "wire exactly one; a silent precedence would hide which "
                "source ran.")
        if frames is None:
            if video is None:
                raise RuntimeError(
                    "Polyhedron Interpolate: neither frames nor video is "
                    "wired -- there is nothing to interpolate.")
            frames, vfps = _frames_from_video(video)
            if vfps > 0:
                source_fps = float(vfps)
                fps_note = ("fps %.6g read from the wired video "
                            "(source_fps widget ignored)" % source_fps)
            else:
                fps_note = ("the wired video names no frame rate: "
                            "source_fps widget %.6g used" % float(source_fps))
            print("[PLS] Interpolate: %s" % fps_note)
        n = int(frames.shape[0])
        h, w = int(frames.shape[1]), int(frames.shape[2])

        # ---- One frame is not an error. It is a still. ----------------------
        # Interpolation fills GAPS. A single frame has no gap, so there is nothing
        # to fill and nothing went wrong. v599 raised ValueError here, which killed
        # the whole queue over a picture the user deliberately fed in -- a node
        # that fails on valid input because it has no work to do is blaming the
        # user for its own idleness.
        #
        # It passes through, at the source rate, and SAYS so. Silent would be worse
        # than loud: a still that quietly claims 48 fps downstream is the same class
        # of lie this node exists to stamp out.
        if n < 2:
            print("[PLS] Interpolate: %d frame(s) in -- there is no gap to interpolate ACROSS, so "
                  "there is nothing to do and nothing wrong. Passing through untouched at %.2f fps. "
                  "The model was not even loaded." % (n, source_fps))
            info = ("passthrough: %d frame(s), no gap to interpolate; fps "
                    "unchanged at %.2f" % (n, source_fps))
            if fps_note:
                info += " | " + fps_note
            _early_audio = _audio_of(video)
            if audio_mode == "mute":
                _early_audio = None
            return (frames, n, float(source_fps), info,
                    _build_video(frames, _early_audio, source_fps),
                    _early_audio)

        # ---- Timeline -----------------------------------------------------
        if rate_mode == "target_fps":
            multiplier, out_fps, _ = _fps_plan(n, source_fps, target_fps)
            if abs(out_fps - target_fps) > 0.01:
                print("[PLS] Interpolate: %.2f fps is not an integer multiple of %.2f fps. "
                      "Landing on %.2f fps (x%d) - every source frame keeps its exact "
                      "timestamp, which a fractional multiplier could not promise."
                      % (target_fps, source_fps, out_fps, multiplier))
        else:
            out_fps = source_fps * multiplier
        tasks, out_count = _timeline(n, multiplier)

        # ---- The knobs that this arch ignores. Said out loud. --------------
        if inert:
            asked = []
            if "fast_mode" in inert and fast_mode:
                asked.append("fast_mode")
            if "ensemble" in inert and ensemble:
                asked.append("ensemble")
            if asked:
                why = {"fast_mode": "contextnet was removed at arch 4.5",
                       "ensemble": "arch 4.26 does not implement it"}
                for kn in asked:
                    print("[PLS] Interpolate: %s is OFF for arch %s (%s). You asked for it; "
                          "the model cannot honour it. Saying so beats pretending."
                          % (kn, arch_ver, why[kn]))
        if "ensemble" in inert:
            ensemble = False
        if "fast_mode" in inert:
            fast_mode = False

        # ---- Gates, decided before a single forward ------------------------
        # The MAE sweep runs ALWAYS, gates on or off. It costs 64 cheap ops on a
        # 65-frame clip, and without it the thresholds are a guess: no default can
        # be right for both a locked-off shot and a whip pan. The node hands you
        # the distribution; you set the dial from a number instead of a feeling.
        maes = [float(torch.mean(torch.abs(frames[p] - frames[p + 1]))) for p in range(n - 1)]
        verdicts = [_gate_verdict(m, static_skip, cut_guard) for m in maes]
        n_dup = verdicts.count("dup")
        n_hold = verdicts.count("hold")
        live = [t for t in tasks if verdicts[t[0]] == "interp"]

        model = _load_model(ckpt_name, arch_ver, precision, device)
        dtype = _DTYPE_MAP[precision]
        scales = _scale_list(arch_ver, scale_factor)

        # ---- Geometry: we do NOT pad. We warn. -----------------------------
        pad_w, pad_h, aligned = _alignment_cost(w, h)

        # ---- Memory: measured, not guessed ---------------------------------
        # No chunk here. v598 sized it up front from a constant and Frank's
        # card paid 7.6x for it. The size is now EARNED below: one small probe
        # chunk runs first, the allocator reports what a pair really costs,
        # and the wall clock keeps watch from there.
        try:
            free_b = torch.cuda.mem_get_info(device)[0]
        except Exception:
            free_b = 2 * 1024 ** 3

        print("[PLS] Interpolate: %df %dx%d -> %df @ %.2f fps | arch %s %s | pyramid %s"
              % (n, w, h, out_count, out_fps, arch_ver, precision,
                 "/".join("%g" % s for s in scales)))
        if not aligned:
            aw, ah = _nearest_aligned(w, h)
            print("[PLS] Interpolate:   NOTE - %dx%d is not a multiple of %d, so the network pads "
                  "it to %dx%d with BLACK (zeros, bottom-right) and crops back. The flow estimator "
                  "then tries to track that ring. Measured against ground truth on an 8px "
                  "translation, arch 4.7: an unaligned canvas cost 9.4 dB overall and 16.1 dB at "
                  "the border. Landing the upscale dial on %dx%d costs nothing and avoids it "
                  "entirely. Your dial stays yours - this is a note, not a change."
                  % (w, h, mod, w + pad_w, h + pad_h, aw, ah))
        else:
            print("[PLS] Interpolate:   canvas is a multiple of %d - the network's own pad is a "
                  "no-op and it sees only real pixels. This is the cheap case, and the good one."
                  % mod)
        srt = sorted(maes)
        print("[PLS] Interpolate:   pair difference (MAE): min %.4f  median %.4f  max %.4f. "
              "Set static_skip just above the min to skip frozen pairs; set cut_guard just below "
              "the max ONLY if that max is a real cut and not just fast motion - the node cannot "
              "tell those apart, and you can."
              % (srt[0], srt[len(srt) // 2], srt[-1]))
        if n_dup or n_hold:
            print("[PLS] Interpolate:   gates: %d pair(s) duplicated (static), %d held (cut), "
                  "%d interpolated. Frame count is unchanged - a skipped pair still fills its "
                  "slots, with copies." % (n_dup, n_hold, len(verdicts) - n_dup - n_hold))
        print("[PLS] Interpolate:   %d task(s), %.1f GB free -- probing with %d pair(s) "
              "to measure what one really costs"
              % (len(live), free_b / 1024 ** 3, min(_PROBE_PAIRS, len(live))))

        # ---- Output tensor: allocated once, written in place ----------------
        out = torch.empty((out_count, h, w, 3), dtype=frames.dtype, device=frames.device)
        for i in range(n):
            out[i * multiplier] = frames[i]
        for pair, t in tasks:
            if verdicts[pair] == "interp":
                continue
            src = frames[pair] if verdicts[pair] == "hold" else frames[pair]
            slot = pair * multiplier + int(round(t * multiplier))
            out[slot] = src

        t_start = _now()
        t_said = t_start
        done = 0
        chw = frames.permute(0, 3, 1, 2)

        # ---- The chunk is EARNED, in the order v566-v573 learned to earn it -
        # TWO JUDGES, and the second one outranks the first:
        #
        #   the allocator PROPOSES. One small probe chunk runs, and
        #   max_memory_allocated says what a pair actually costs. That is a
        #   measurement, not the constant v598 shipped.
        #
        #   the wall clock DISPOSES. WDDM never OOMs -- it spills to system RAM
        #   over PCIe and GRINDS, and torch's allocated counter is blind to
        #   driver-side paging (v570, measured). A chunk that runs 2x slower per
        #   task than the probe IS grinding, whatever the counter says. The clock
        #   is the one detector the driver cannot hide from.
        #
        # And no empty_cache between chunks: handing the pool back makes the NEXT
        # chunk pay the re-commit. That was measured on this very card, on
        # cudaMallocAsync, and it cost four minutes (v572).
        # THE CHUNK CLIMBS, IT DOES NOT LEAP. The first cut of v599 probed, sized
        # the chunk from the measured peak, and jumped straight to it. On a run of
        # 64 tasks that means ONE chunk -- and when it ground, the watch fired
        # into an empty room: there was no "rest of the run" left to protect.
        # 17.9 s instead of 18.9 s. That is v570's exact mistake, and it is
        # written down in v570's own guard: the watch was right and TOO LATE,
        # while the bad chunk donated four minutes to the driver.
        #
        # So the peak is a CEILING, not a destination. The chunk doubles from the
        # probe upward, every step timed. A grind costs ONE step and is never
        # repeated; the last healthy size stands for the rest of the run.
        cuda = (getattr(device, "type", str(device)) == "cuda")
        chunk = min(_PROBE_PAIRS, len(live))
        cap = _CHUNK_CEIL
        good = chunk
        base_ms = None
        probed = False

        with torch.inference_mode():
            pos = 0
            while pos < len(live):
                batch = live[pos:pos + chunk]
                if cuda and not probed:
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    held = torch.cuda.memory_allocated(device)
                t_chunk = _now()

                f0 = torch.cat([chw[p:p + 1] for p, _ in batch]).to(device, dtype=dtype)
                f1 = torch.cat([chw[p + 1:p + 2] for p, _ in batch]).to(device, dtype=dtype)
                # No pad here, on purpose. IFNet pads to mod 64 with zeros and crops
                # back itself; a replicate ring of ours on top measured 4.18 dB WORSE
                # (the network was trained with the zeros ring). Header, item 4.
                ts = torch.tensor([t for _, t in batch], dtype=dtype,
                                  device=device).view(-1, 1, 1, 1)
                mid = model(f0, f1, ts, scales, fast_mode, ensemble)
                mid = mid.float().clamp(0, 1).permute(0, 2, 3, 1).to(out.device, out.dtype)
                for k, (pair, t) in enumerate(batch):
                    slot = pair * multiplier + int(round(t * multiplier))
                    out[slot] = mid[k]
                if cuda:
                    torch.cuda.synchronize(device)
                ms = 1000.0 * (_now() - t_chunk) / max(1, len(batch))
                done += len(batch)
                pos += len(batch)
                del f0, f1, mid          # the dels stay; empty_cache does NOT (v572)

                if not probed:
                    probed = True
                    base_ms = ms
                    good = len(batch)
                    if cuda:
                        peak = max(1, torch.cuda.max_memory_allocated(device) - held)
                        per_pair = peak / float(len(batch))
                        free_now = torch.cuda.mem_get_info(device)[0]
                        cap = _chunk_size(free_now, per_pair)
                        guess = (w + pad_w) * (h + pad_h) * 3 * dtype.itemsize * 6
                        print("[PLS] Interpolate:   probe: %.0f MB/pair MEASURED -- v598's constant "
                              "claimed %.0f MB, and that error is the whole bug. Ceiling %d pair(s) "
                              "from %.1f GB free; baseline %.0f ms/task. Climbing to it, not "
                              "leaping -- anything %.0fx slower is the driver paging, and one step "
                              "is all it gets."
                              % (per_pair / 1024 ** 2, guess / 1024 ** 2, cap,
                                 free_now / 1024 ** 3, base_ms, _GRIND_FACTOR))
                    chunk = min(cap, max(1, chunk * 2))
                elif base_ms and ms > _GRIND_FACTOR * base_ms and len(batch) > good:
                    # The clock says this step ground. The allocator will never admit
                    # it -- WDDM pages to system RAM instead of failing, and torch's
                    # counter cannot see driver-side paging. Fall back to the last
                    # size that ran clean and STOP climbing: the ceiling was wrong.
                    print("[PLS] Interpolate:   GRIND at %d pair(s) -- %.0f ms/task against the "
                          "probe's %.0f. The card is paging, and the allocator will not say so. "
                          "Back to %d for the rest; the measured ceiling (%d) was too brave."
                          % (len(batch), ms, base_ms, good, cap))
                    chunk = good
                    cap = good
                else:
                    good = max(good, len(batch))
                    chunk = min(cap, max(1, chunk * 2))

                if done < len(live) and (_now() - t_said) > 3.0:
                    t_said = _now()
                    print("[PLS] Interpolate:   %d/%d tasks (%.0f ms/task, chunk %d)"
                          % (done, len(live), 1000.0 * (_now() - t_start) / max(1, done), chunk))

        wall = _now() - t_start
        per = 1000.0 * wall / max(1, len(live))
        info = ("arch=%s %s | %df -> %df @ %.2f fps (x%d) | canvas %dx%d %s | "
                "dup %d hold %d interp %d | %.1fs (%.0f ms/frame)"
                % (arch_ver, precision, n, out_count, out_fps, multiplier, w, h,
                   "aligned" if aligned else ("engine pads +%dx%d black" % (pad_w, pad_h)),
                   n_dup, n_hold, len(live), wall, per))
        if inert:
            info += " | inert: %s" % ",".join(sorted(inert))
        print("[PLS] Interpolate: done in %s -> %d frames @ %.2f fps (%.0f ms/frame)"
              % (_fmt_clock(wall), out_count, out_fps, per))
        if fps_note:
            info += " | " + fps_note

        # ---- v887: the duration arithmetic, and the sound ------------------
        sync = _sync_note(n, source_fps, out_count, out_fps)
        if sync:
            info += " | " + sync
            print("[PLS] Interpolate: %s" % sync)
        audio = _audio_of(video)
        audio, retime_note = _retime_audio(audio, audio_mode, n, source_fps,
                                           out_count, out_fps)
        if retime_note:
            info += " | " + retime_note
        out_video = _build_video(out, audio, out_fps)
        if out_video is not None:
            print("[PLS] Interpolate: video output ready at %.2f fps, audio=%s"
                  " - wire IT to the Save and no fps wire is needed; a bare "
                  "'frames' wire still needs Interpolate.fps -> "
                  "Save.frame_rate, or the doubled frames are written at the "
                  "OLD rate."
                  % (out_fps, "carried through" if audio is not None
                     else "none (no video was wired in)"))
        return (out, out_count, out_fps, info, out_video, audio)


def _now():
    import time
    return time.perf_counter()
