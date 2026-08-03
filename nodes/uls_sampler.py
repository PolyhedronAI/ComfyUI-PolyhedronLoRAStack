"""
Polyhedron Sampler  (ULSSampler)
═══════════════════════════════════════
Milestone 0 — the solid, classic base of the Polyhedron Sampler family.

A drop-in KSampler with the standard wiring (MODEL, positive, negative,
latent_image -> LATENT) that intentionally unifies what the stock nodes split:

  - a `denoise` field (0..1) — present on the simple KSampler but MISSING on
    KSampler (Advanced); needed to feed an image/video latent at LOW denoise
    (e.g. 0.25) for img2img / v2v refinement.
  - the FULL sampler and scheduler lists (comfy.samplers.KSampler.SAMPLERS /
    SCHEDULERS) — no per-model subset.
  - an `add_noise` toggle (for leftover-noise / two-phase chaining).

Live preview (v410): an ANIMATED in-node video preview. ComfyUI core decodes only
the first frame of a video latent, so the stock preview is a single still — we
mirror core's Latent2RGB math but decode EVERY frame and stream them to our own
frontend (web/js/uls_live_preview.js), which loops them in the node. Model-free
(uses the model's own latent_rgb_factors), continuous across the MoE HIGH->LOW
handoff, and gated on the global preview method.

Live preview MODES (v413/v417): a serialised `preview_mode` combo on the node selects
the decode path. `Standard (ComfyUI)` (the default, v417) delegates to Core's own
`latent_preview.get_previewer` — the exact previewer the KSampler uses, honouring
--preview-method and decoding via models/vae_approx (TAESD/TAEHV) or latent2rgb. The
other modes force a specific decoder: latent2rgb `smooth` (LANCZOS) / `crisp` (NEAREST)
for the near-free linear projection, or a TAE decode (`taew2_1` image path /
`lighttaew2_1` WAN video TAEHV). The TAE and Standard paths REUSE ComfyUI core's
previewer machinery (latent_preview) and fall back to latent2rgb (smooth) if a decoder
is unavailable, so a missing decoder can never crash a render. The mode applies from the
first step of each render (it is the snapshot the node receives at execute time).

Design: we DELEGATE the denoise->sigma math and the whole sampling loop to
ComfyUI core (comfy.sample.sample with denoise=), mirroring nodes.common_ksampler,
so there is nothing bespoke here to drift out of sync with Core. Classic I/O
only — no WANVIDEOMODEL / block-swap / context-window exotica (out of scope by
design). The WAN 5D video latent flows through Core unchanged, so this slots
straight into the existing WAN T2V/I2V workflows.
"""

import base64
import io
import os
import re
import time

import torch
from PIL import Image

import comfy.samplers

try:  # package load (ComfyUI) vs direct module load (tools)
    from .ph_logmute import MuteStagingLogs as _MuteStagingLogs
except ImportError:  # pragma: no cover
    from ph_logmute import MuteStagingLogs as _MuteStagingLogs
try:  # v576: the shared run clock (nodes/ph_runclock.py, ex ph_power_upscale)
    from .ph_runclock import _fmt_clock, _RunClock
except ImportError:  # pragma: no cover
    from ph_runclock import _fmt_clock, _RunClock
import comfy.sample
import comfy.utils
from comfy.cli_args import args, LatentPreviewMethod


# ─────────────────────────────────────────────────────────────────────────────
# Live VIDEO preview (v410)
# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI core's Latent2RGB/TAEHV previewers decode only the FIRST frame of a 5D
# video latent (latent_preview.Latent2RGBPreviewer slices `x0[0, :, 0]`), so the
# stock preview is a single still. Here we mirror the EXACT core Latent2RGB math
# but decode EVERY temporal frame, then stream the frames to our own frontend
# (web/js/uls_live_preview.js), which loops them as an in-node animation.
# Model-free: it uses the model's own latent_rgb_factors (no extra download).
# Every step here is wrapped so a preview failure can NEVER break sampling, and it
# is gated on the user's global preview method (silent when previews are off).

_PREVIEW_EVENT = "polyhedron.live_preview"

# v576: steps faster than this narrate as ONE folded summary line per stage
# instead of a per-step litany (the exact courtesy the pixel chunks learned in
# v567). Slow steps (WAN video: 40-90+ s) speak every time - each line earned.
_CLOCK_FOLD_S = 5.0

# v415: live (mid-render) preview-mode override, keyed by node id. The frontend POSTs
# here via /pls/sampler/preview_mode (nodes/uls_routes.py) when preview_mode is changed
# WHILE a render runs; the sampling callback reads it each step and re-points the
# previewer. It only ever changes the PREVIEW, never the output latent. Cleared at the
# start of every render so each run begins from the node's serialised preview_mode.
_LIVE_PREVIEW_MODE = {}   # str(node_id) -> requested mode string


def _previews_enabled():
    """Respect ComfyUI's global preview method — the same switch the stock
    KSampler obeys. NoPreviews -> we stay silent."""
    try:
        return args.preview_method != LatentPreviewMethod.NoPreviews
    except Exception:
        return False


def _current_node_id():
    """Id of the node currently executing, so the frontend can attach the preview
    to THIS node. None -> we simply don't send (never a crash)."""
    try:
        from server import PromptServer
        return PromptServer.instance.last_node_id
    except Exception:
        return None


def _send_preview(node_id, frames_b64, fps):
    if node_id is None or not frames_b64:
        return
    try:
        from server import PromptServer
        PromptServer.instance.send_sync(_PREVIEW_EVENT, {
            "node": str(node_id),
            "frames": frames_b64,
            "fps": fps,
        })
    except Exception:
        pass  # a dropped preview frame must never interrupt generation


def _preview_scale(w, h, side):
    """Target (w, h) so a preview frame fits a ``side`` x ``side`` box.

    The long edge maps to exactly ``side`` with aspect kept, scaling up or down as
    needed -- a tiny latent is enlarged to a readable size, a large one is shrunk.
    The caller resamples with a smooth high-quality filter (LANCZOS), which suits
    photorealistic output as well as pixel-art: the real pixel grid is imposed after
    VAE decode and is never visible in this latent preview, so a content-agnostic
    smooth fit reads better than a blocky one at every size. (v412)
    """
    long = max(w, h)
    if long == side:
        return w, h
    r = side / long                       # fit long edge to the box, aspect kept (up or down)
    return max(1, round(w * r)), max(1, round(h * r))


def _norm_tae(name):
    """Normalise an approx-VAE / TAE name for tolerant file matching (v414): lower-
    case and drop every non-alphanumeric character, so separators, case and the file
    extension stop mattering -- e.g. 'TAEW2_1.safetensors', 'taew_2_1.pth' and
    'taew2.1.pt' all normalise to a 'taew21...' string. Matching stays a PREFIX test
    against the exact version token, so a different version ('taew2_2') or an
    unrelated VAE never collides with it."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


class _AnimatedVideoPreviewer:
    """Latent2RGB previewer that keeps ALL frames (core keeps one).

    The per-frame projection is the core math (latent_preview
    .Latent2RGBPreviewer / preview_to_image): a linear map from latent channels to
    RGB via latent_rgb_factors, scaled (x + 1) / 2 -> 0..255. We loop the temporal
    axis instead of slicing it to frame 0, fit each frame to a preview box (small
    latents are upscaled, large ones shrunk; see _preview_scale, v412), and JPEG +
    base64 it. Inert (no output) when the model exposes no latent_rgb_factors."""

    # preview_mode combo values — MUST match nodes INPUT_TYPES and the JS
    # ORDER_V404 widget order (web/js/uls_sampler.js).
    MODE_STANDARD = "Still · ComfyUI"
    MODE_STD_LATENT = "Still · latent2rgb"
    MODE_SMOOTH = "Video · latent2rgb (smooth)"
    MODE_CRISP = "Video · latent2rgb (crisp)"
    MODE_TAE_STD = "Video · TAE (taew2_1)"
    MODE_TAE_LIGHT = "Video · TAE (lighttaew2_1)"
    ALL_MODES = (MODE_STANDARD, MODE_STD_LATENT, MODE_SMOOTH, MODE_CRISP, MODE_TAE_STD, MODE_TAE_LIGHT)

    def __init__(self, latent_format, preview_side=512, max_frames=64,
                 preview_mode="latent2rgb (smooth)", device=None):
        preview_mode = _strip_mode(preview_mode)  # v830: decorated combo
        factors = getattr(latent_format, "latent_rgb_factors", None)
        self.ok = factors is not None
        self.preview_side = preview_side          # target long-edge for the in-node preview (v412)
        self.latent_format = latent_format
        self.device = device

        # ── preview mode (v413/v415): pick the decode path + the resample filter ──
        # NEAREST is used ONLY for the crisp latent2rgb upscale (see _filter_for).
        # Every other path -- smooth latent2rgb AND all TAE shrink-to-box -- resamples
        # LANCZOS (a shrink must never be NEAREST). All of this can be re-pointed at
        # runtime via set_mode() for a live (mid-render) switch (v415).
        self._max_frames_base = max_frames        # latent2rgb frame budget (restored on switch)
        self.mode = preview_mode
        self.use_core = preview_mode == self.MODE_STANDARD   # delegate to Core's get_previewer
        self.use_core_l2rgb = preview_mode == self.MODE_STD_LATENT   # Core latent2rgb, 1 frame
        self.use_tae = preview_mode in (self.MODE_TAE_STD, self.MODE_TAE_LIGHT)
        self.filter = self._filter_for(preview_mode)
        # TAE / Core decode per frame -> heavier than the linear projection, so
        # subsample video harder there; plain latent2rgb keeps the generous budget.
        self.max_frames = 16 if (self.use_tae or self.use_core) else max_frames
        # The explicit TAE modes use LITERAL names matching their combo labels (v418).
        # The v414 "light" + <model-declared-name> composition doubled to
        # 'lightlighttaew2_1' when the model already declared the light decoder, so the
        # light mode silently fell back. Literal names are predictable and match the
        # label; model-derived name selection is the job of Standard (ComfyUI), which
        # goes through Core's get_previewer.
        self.tae_name = "lighttaew2_1" if preview_mode == self.MODE_TAE_LIGHT else "taew2_1"
        self._tae = None          # Core previewer object, lazy-built on first frame
        self._tae_tried = False   # build attempted? -> a miss falls back once, quietly
        self._core = None         # Core get_previewer object (Standard mode), lazy
        self._core_tried = False
        self._cl2 = None          # Core Latent2RGBPreviewer (Standard latent2rgb), lazy
        self._cl2_tried = False

        if not self.ok:
            return
        # mirror core: tensor + transpose(0, 1) -> [C, 3]
        self.factors = torch.tensor(factors, device="cpu").transpose(0, 1)
        bias = getattr(latent_format, "latent_rgb_factors_bias", None)
        self.bias = torch.tensor(bias, device="cpu") if bias is not None else None
        self.reshape = getattr(latent_format, "latent_rgb_factors_reshape", None)

    def _filter_for(self, preview_mode):
        # The ONE place NEAREST is chosen: the crisp latent2rgb upscale. Everything
        # else (smooth, and every TAE shrink-to-box) is LANCZOS.
        return Image.NEAREST if preview_mode == self.MODE_CRISP else Image.LANCZOS

    def set_mode(self, preview_mode):
        preview_mode = _strip_mode(preview_mode)  # v830: decorated combo
        """Re-point the decode path for a LIVE (mid-render) switch (v415). Affects the
        PREVIEW only -- it never touches the sampling or the output latent. Recomputes
        the filter / TAE selection / decoder name and resets the lazy TAE so the new
        decoder is (re)built on the next frame. A no-op if the mode is unchanged or
        unknown; never raises."""
        if preview_mode not in self.ALL_MODES or preview_mode == self.mode:
            return
        self.mode = preview_mode
        self.use_core = preview_mode == self.MODE_STANDARD
        self.use_core_l2rgb = preview_mode == self.MODE_STD_LATENT   # Core latent2rgb, 1 frame
        self.use_tae = preview_mode in (self.MODE_TAE_STD, self.MODE_TAE_LIGHT)
        self.filter = self._filter_for(preview_mode)
        self.max_frames = 16 if (self.use_tae or self.use_core) else self._max_frames_base
        self.tae_name = "lighttaew2_1" if preview_mode == self.MODE_TAE_LIGHT else "taew2_1"
        self._tae = None          # force a (re)build for the newly-selected decoder
        self._tae_tried = False
        self._core = None
        self._core_tried = False
        self._cl2 = None
        self._cl2_tried = False

    def _get_core(self):
        """Lazily build ComfyUI Core's OWN previewer (Standard mode) via
        latent_preview.get_previewer -- the exact machinery the KSampler uses. It
        honours the global --preview-method and picks TAESD/TAEHV from models/vae_approx
        or latent2rgb, by Core's own rules. Returns the previewer or None (-> caller
        falls back to latent2rgb smooth). Never raises."""
        if self._core is not None:
            return self._core
        if self._core_tried:
            return None
        self._core_tried = True
        try:
            import latent_preview as lp
            self._core = lp.get_previewer(self.device, self.latent_format)
            if self._core is None:
                print("[PLS v417 PREVIEW] Core previewer unavailable (preview method "
                      "off / no decoder) -> latent2rgb (smooth) fallback")
            return self._core
        except Exception as e:
            print("[PLS v417 PREVIEW] Core previewer build failed ({}) -> "
                  "latent2rgb (smooth) fallback".format(type(e).__name__))
            self._core = None
            return None

    def _get_core_l2rgb(self):
        """Lazily build ComfyUI Core's Latent2RGBPreviewer DIRECTLY (Standard latent2rgb
        mode, v418). This forces Core's model-free latent2rgb view regardless of the
        global --preview-method and regardless of whether vae_approx files are present --
        the 'normal ComfyUI preview without the approx models'. Returns the previewer or
        None (-> caller falls back to our own latent2rgb). Never raises."""
        if self._cl2 is not None:
            return self._cl2
        if self._cl2_tried:
            return None
        self._cl2_tried = True
        try:
            import latent_preview as lp
            self._cl2 = lp.Latent2RGBPreviewer(
                self.latent_format.latent_rgb_factors,
                getattr(self.latent_format, "latent_rgb_factors_bias", None),
                getattr(self.latent_format, "latent_rgb_factors_reshape", None))
            return self._cl2
        except Exception as e:
            print("[PLS v418 PREVIEW] Core latent2rgb previewer failed ({}) -> "
                  "internal latent2rgb fallback".format(type(e).__name__))
            self._cl2 = None
            return None

    def _get_tae(self):
        """Lazily build a Core TAE previewer for self.tae_name and return it (or
        None on a miss -> caller falls back to latent2rgb smooth). REUSES
        latent_preview's previewer classes + comfy's loaders; the TAE math is
        never reconstructed here, and this never raises (a miss/error just returns
        None and is signalled once)."""
        if self._tae is not None:
            return self._tae
        if self._tae_tried:
            return None
        self._tae_tried = True
        try:
            import latent_preview as lp
            import comfy.utils
            import folder_paths
            # Match tolerant of separator / case / extension drift between sources
            # (v414), while staying anchored to the exact version token so a different
            # version (taew2_2) or an unrelated approx VAE never matches.
            want = _norm_tae(self.tae_name)
            cand = next((fn for fn in folder_paths.get_filename_list("vae_approx")
                         if _norm_tae(fn).startswith(want)), "")
            path = folder_paths.get_full_path("vae_approx", cand) if cand else None
            if not path:
                print("[PLS v413 PREVIEW] TAE '{}' not found in models/vae_approx -> "
                      "latent2rgb (smooth) fallback".format(self.tae_name))
                return None
            # Mirror Core's own video/image split (latent_preview.get_previewer):
            # names in VIDEO_TAES load as a VAE (TAEHV); the rest as image TAESD.
            video_taes = getattr(lp, "VIDEO_TAES", [])
            if self.tae_name in video_taes:
                from comfy.sd import VAE
                taesd = VAE(comfy.utils.load_torch_file(path))
                try:
                    taesd.first_stage_model.show_progress_bar = False
                except Exception:
                    pass
                self._tae = lp.TAEHVPreviewerImpl(taesd)
            else:
                from comfy.taesd.taesd import TAESD
                lc = getattr(self.latent_format, "latent_channels", 16)
                taesd = TAESD(None, path, latent_channels=lc)
                if self.device is not None:
                    taesd = taesd.to(self.device)
                self._tae = lp.TAESDPreviewerImpl(taesd)
            return self._tae
        except Exception as e:
            print("[PLS v413 PREVIEW] TAE '{}' decode unavailable ({}) -> "
                  "latent2rgb (smooth) fallback".format(self.tae_name, type(e).__name__))
            self._tae = None
            return None

    def frames_b64(self, x0):
        """x0: predicted clean latent. [B, C, T, H, W] (video) or [B, C, H, W].

        TAE modes decode through Core's previewer; a missing/failed TAE falls back
        to the latent2rgb (smooth) path for the rest of the run (never a crash)."""
        if not self.ok:
            return []
        if self.use_core:
            core = self._get_core()
            if core is not None:
                try:
                    # v419: single still frame, like Core's KSampler preview -- NOT an
                    # animation over the video's temporal frames.
                    return self._frames_core_single(x0, core)
                except Exception as e:
                    print("[PLS v419 PREVIEW] Core decode failed ({}) -> "
                          "latent2rgb (smooth) fallback".format(type(e).__name__))
            # build missed or decode failed -> latent2rgb smooth for the rest
            self.use_core = False
            self.filter = Image.LANCZOS
        if self.use_core_l2rgb:
            cl2 = self._get_core_l2rgb()
            if cl2 is not None:
                try:
                    return self._frames_core_single(x0, cl2)
                except Exception as e:
                    print("[PLS v418 PREVIEW] Core latent2rgb decode failed ({}) -> "
                          "internal latent2rgb fallback".format(type(e).__name__))
            self.use_core_l2rgb = False
        if self.use_tae:
            tae = self._get_tae()
            if tae is not None:
                try:
                    return self._frames_tae(x0, tae)
                except Exception as e:
                    print("[PLS v413 PREVIEW] TAE decode failed ({}) -> "
                          "latent2rgb (smooth) fallback".format(type(e).__name__))
            # build missed or decode failed -> latent2rgb smooth for the rest
            self.use_tae = False
            self.filter = Image.LANCZOS
        return self._frames_latent2rgb(x0)

    def _encode(self, img):
        """Fit a preview frame to the 512 box and JPEG+base64 it. self.filter is
        NEAREST only in the crisp latent2rgb mode; LANCZOS everywhere else (incl.
        every TAE shrink)."""
        tw, th = _preview_scale(img.width, img.height, self.preview_side)
        if (tw, th) != img.size:
            img = img.resize((tw, th), self.filter)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _frames_latent2rgb(self, x0):
        """All-frame Latent2RGB projection (the v410..v412 path), filter per mode."""
        x = x0
        if self.reshape is not None:
            x = self.reshape(x)
        x = x.detach().to(device="cpu", dtype=torch.float32)
        if x.ndim == 5:
            frames = x[0].movedim(0, 1)          # [C, T, H, W] -> [T, C, H, W]
        else:
            frames = x[0].unsqueeze(0)           # [1, C, H, W]
        total = frames.shape[0]
        if total > self.max_frames:              # evenly subsample very long clips
            idx = torch.linspace(0, total - 1, self.max_frames).round().long()
            frames = frames[idx]
        fac = self.factors.to(dtype=frames.dtype)
        bias = self.bias.to(dtype=frames.dtype) if self.bias is not None else None
        out = []
        for t in range(frames.shape[0]):
            rgb = torch.nn.functional.linear(frames[t].movedim(0, -1), fac, bias=bias)  # [H, W, 3]
            rgb = ((rgb + 1.0) / 2.0).clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
            out.append(self._encode(Image.fromarray(rgb)))
        return out

    def _frames_core_single(self, x0, previewer):
        """SINGLE-frame decode through a Core previewer (both Standard modes, v419). No
        temporal loop -> exactly one still image of the latent as it currently stands,
        the way Core's KSampler shows it; the per-step callback refreshes it, so you
        watch one image emerge through the denoise rather than an animated clip."""
        return [self._encode(previewer.decode_latent_to_preview(x0))]

    def _frames_tae(self, x0, tae):
        """All-frame TAE decode via Core's previewer. We slice the temporal axis to
        one latent frame at a time and let Core decode it -> a hi-res PIL frame,
        which _encode shrinks to the box with LANCZOS."""
        x = x0.detach()
        out = []
        if x.ndim == 5:
            total = x.shape[2]                   # temporal latent frames
            if total > self.max_frames:
                idx = torch.linspace(0, total - 1, self.max_frames).round().long().tolist()
            else:
                idx = list(range(total))
            for t in idx:
                frame_latent = x[:, :, t:t + 1]  # keep 5D, single temporal frame
                out.append(self._encode(tae.decode_latent_to_preview(frame_latent)))
        else:
            out.append(self._encode(tae.decode_latent_to_preview(x)))
        return out


# ---------------------------------------------------------------------------
# v830 -- THE PREVIEW DECODER SAYS WHEN IT IS MISSING (Frank's ask), and the
# one that CAN be fetched goes through the pack's one download door.
#
# Before this cut a missing TAE fell back to latent2rgb with only a console
# line -- correct (a preview must never kill a render, the v413 promise,
# which stands word for word) but silent on screen: the user saw a mushier
# preview and no reason. Now the status is ASKABLE (/pls/sampler/tae_status)
# and the frontend raises the amber bubble; taew2_1 is additionally
# INSTALLABLE (/pls/sampler/tae_install) via ph_weights.ensure_weights --
# the first reuse of the one door outside the cutout engines.
#
# SOURCES, measured 02.08. against the net, not recalled:
#   - taew2_1: madebyollin/taehv ships a SAFETENSORS build (the house rule)
#     at safetensors/taew2_1.safetensors -- 22,642,902 bytes, sha256 pinned
#     below from a full pull + hash in the build sandbox.
#   - lighttaew2_1: does NOT exist in that repo (root and safetensors/
#     checked, both branches listed); the name comes from ComfyUI core's
#     VIDEO_TAES. Frank supplied the source 02.08.: lightx2v/Autoencoders
#     on HF -- pinned below chat-side from the LFS pointer (sandbox egress
#     blocks HF, the runtime machine does not). Both are downloadable now.
TAE_REGISTRY = {
    "taew2_1": {
        "file": "taew2_1.safetensors",
        "url": ("https://raw.githubusercontent.com/madebyollin/taehv/"
                "main/safetensors/taew2_1.safetensors"),
        "sha256": ("04766eac0221b5390b985ae3fdcca652"
                   "cbb4b1e8b82b28ea7ff89dfad1b1a93f"),
        "bytes": 22642902,
        "source": "github.com/madebyollin/taehv (safetensors build)",
    },
    "lighttaew2_1": {
        # Pinned 02.08. from Frank's source. The build sandbox cannot
        # reach huggingface.co (egress allowlist), so the pin was read
        # CHAT-SIDE from the repo's own LFS pointer (oid sha256 + exact
        # size) -- the runtime download happens on Frank's machine, which
        # is not fenced. License apache-2.0 (public-build clean).
        "file": "lighttaew2_1.safetensors",
        "url": ("https://huggingface.co/lightx2v/Autoencoders/"
                "resolve/main/lighttaew2_1.safetensors"),
        "sha256": ("8666e769f449f2d955651564bcadb0c5"
                   "e12461b50480e67b7241f32c15f11a19"),
        "bytes": 45274004,
        "source": "huggingface.co/lightx2v/Autoencoders (apache-2.0)",
    },
}

# The strip mirrors ph_basics._SIZE_SUFFIX character for character (a guard
# holds the parity) -- one truth by measurement, not by an import. NOTE the
# mode strings THEMSELVES contain " \u00b7 " ("Video \u00b7 latent2rgb ..."),
# which is exactly why the suffix regex anchors on a trailing SIZE UNIT and
# nothing broader (the v828 rule): the label's own middots never match.
_MODE_SIZE_SUFFIX = re.compile(r"\s\u00b7\s[\d.,]+\s?(KB|MB|GB|TB)$")


def _strip_mode(value):
    """Decoration off a preview_mode value: one size suffix, one diamond."""
    out = _MODE_SIZE_SUFFIX.sub("", str(value)).rstrip()
    if out.startswith("\u25c8 "):
        out = out[2:]
    return out


def _fmt_tae_size(n):
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


def _norm_tae_name(fn):
    """The v414 tolerant match, shared: separator / case / extension drift."""
    base = str(fn).rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    base = base.rsplit(".", 1)[0] if "." in base else base
    return re.sub(r"[\s_\-]+", "", base).lower()


def tae_lookup(tae_name):
    """(path or None) for a TAE decoder in any configured vae_approx folder.

    The ONE search both the previewer and the status route use -- lifted
    from _get_tae (v414 tolerance: separator/case/extension drift, anchored
    on the exact version token so taew2_2 never answers for taew2_1)."""
    try:
        import folder_paths
        want = _norm_tae_name(tae_name)
        cand = next((fn for fn in folder_paths.get_filename_list("vae_approx")
                     if _norm_tae_name(fn).startswith(want)), "")
        return folder_paths.get_full_path("vae_approx", cand) if cand else None
    except Exception:
        return None


def tae_status(tae_name):
    """Everything the frontend needs to warn, inform, or offer the fetch."""
    name = _strip_mode(tae_name)
    # LONGEST key first -- "taew2_1" is a SUBSTRING of "lighttaew2_1", and
    # matching in dict order sent the light mode to the wrong spec. The
    # same trap Meta's own loader fell into with "detector" in
    # "detector_model" (the SAM-3 reading, 25.07.) -- caught here by this
    # guard's first run.
    for key in sorted(TAE_REGISTRY, key=len, reverse=True):
        if key in name:
            name = key
            break
    spec = TAE_REGISTRY.get(name) or {}
    path = tae_lookup(name)
    size = 0
    if path:
        try:
            size = os.path.getsize(path)
        except Exception:
            size = 0
    return {
        "name": name,
        "found": bool(path),
        "path": path or "",
        "file": spec.get("file", name + ".safetensors"),
        "folder": "models/vae_approx",
        "size": size or int(spec.get("bytes") or 0),
        "source": spec.get("source", ""),
        "downloadable": bool(spec.get("url")),
    }


def tae_install(tae_name):
    """Fetch a pinned TAE through the pack's ONE download door
    (ph_weights.ensure_weights: .part file, sha256 verify, primary folder).
    Raises for the unpinned one -- the route turns that into an error text."""
    name = tae_status(tae_name)["name"]
    spec = TAE_REGISTRY.get(name)
    if not spec or not spec.get("url"):
        raise RuntimeError(
            "no pinned source for '%s' -- %s"
            % (name, (spec or {}).get("source", "unknown model")))
    from . import ph_weights as W
    return W.ensure_weights("vae_approx", name, TAE_REGISTRY, tag="tae")


def _mode_entry(mode, tae_name):
    """One decorated combo line for a TAE mode: diamond (house definition,
    pinned default behaviour and fetch) + size by the cutout's three-step
    rule (disk > pin > nothing)."""
    size = ""
    path = tae_lookup(tae_name)
    if path:
        try:
            size = _fmt_tae_size(os.path.getsize(path))
        except Exception:
            size = ""
    if not size:
        size = _fmt_tae_size(TAE_REGISTRY.get(tae_name, {}).get("bytes", 0))
    out = "\u25c8 " + mode
    if size:
        out += " \u00b7 " + size
    return out


def set_live_preview_mode(node_id, mode):
    mode = _strip_mode(mode)  # v830: decorated combo values
    """Record a LIVE preview-mode request for a node (v415). Called by the
    /pls/sampler/preview_mode route (nodes/uls_routes.py) when preview_mode is changed
    DURING a render; the running sampling callback applies it on the next step.
    Validated against the known modes -> returns True if accepted, False otherwise."""
    if mode in _AnimatedVideoPreviewer.ALL_MODES:
        _LIVE_PREVIEW_MODE[str(node_id)] = mode
        return True
    return False


def _get_live_preview_mode(node_id):
    return _LIVE_PREVIEW_MODE.get(str(node_id))


def _clear_live_preview_mode(node_id):
    _LIVE_PREVIEW_MODE.pop(str(node_id), None)


def _cfg_forwards(cfg):
    """v577 pure: model forwards ONE sampler step costs at this cfg.

    Core skips the uncond pass when cfg is 1.0 (comfy/samplers.py: cond_scale
    close to 1.0 -> uncond_ = None), so such a step is ONE model forward where
    cfg > 1 is TWO. That is not a detail - it is THE cross-expert cost ratio in
    Frank's chain: a lightning HIGH at cfg 1.0 against a LOW at cfg 6 measured
    43.1 s vs 93.4 s per step. Factor 2.17 on the same latent, same resolution,
    same architecture. The cfg IS the physics."""
    try:
        return 1.0 if abs(float(cfg) - 1.0) < 1e-6 else 2.0
    except Exception:
        return 2.0


def _chain_posts(high_units, low_units, weight, cfg_high=None, cfg_low=None):
    """v576/v577 pure: the MoE chain's clock plan.

    Both experts denoise the SAME latent, so the pixel part of the weight is
    identical - but the COST of a step is latent x forwards, and the forward
    count is decided by cfg (see _cfg_forwards). With that factor, rung 2 of
    the estimation ladder (same kind, weight-scaled) hands the still-unmeasured
    expert a rate that is already RIGHT: on Frank's chain the run eta at the
    handoff lands at ~467 s instead of ~216 s, and the first LOW measurement
    confirms rather than doubles it. With equal cfgs the weights are equal and
    this is byte-identical to v576. The experts are still different models
    (offload, per-wire LoRA stacks), so the EMA keeps correcting from the first
    own step and the estimate keeps its '~'.

    Pure floats, no torch - extracted and RUN by the guards."""
    w = float(max(1.0, weight))
    return [("step:high", int(high_units), w * _cfg_forwards(cfg_high)),
            ("step:low", int(low_units), w * _cfg_forwards(cfg_low))]


def _clock_close(clock, parts):
    """v576: the run's closing line - total wall time plus the per-stage split
    (done steps @ EMA-free true average). The node badge adds executor
    overhead OUTSIDE this function, so the two numbers may differ slightly -
    said here once, exactly like the Power Upscale done line says it."""
    try:
        segs = []
        for label, key in parts:
            p = clock.posts.get(key)
            if p and p["done"]:
                avg = p["spent"] / p["done"]
                segs.append(f"{label} {p['done']} steps @ {avg:.1f}s")
        split = " + ".join(segs) if segs else "nothing measured"
        print(f"[PLS SAMPLER] done in {_fmt_clock(clock.elapsed())} ({split}) "
              f"- the node badge adds executor overhead outside sampling")
    except Exception:
        pass  # telemetry must never break sampling


def _make_preview_callback(model, total_steps, node_id, step_offset=0,
                           pbar=None, previewer=None, fps=12, min_interval=0.5,
                           preview_mode="latent2rgb (smooth)",
                           clock=None, clock_key=None):
    """Build the sampling callback: it always advances the (optionally shared)
    progress bar, and -- rate-limited -- streams an animated video preview. For the
    two-stage MoE chain pass a SHARED pbar + previewer plus a step_offset, so bar
    and preview run continuously 0..total across HIGH then LOW (no reset).

    v576: pass a shared _RunClock + this stage's post key and the bar becomes a
    TIME bar (value = wall-clock deciseconds, total = elapsed + eta - the v567
    law; ticks are not time). Each step is measured into the clock (the clock's
    own push() then owns the bar - the step-tick writer below is the clock-less
    fallback only, so one bar never has two writers), and the step SPEAKS: its
    measured duration plus stage and run eta, or - under _CLOCK_FOLD_S - one
    folded line per stage. v577: dt comes from clock.tick(), so the cursor is
    the RUN's, not this closure's. Known softness, said out loud: the HIGH->LOW
    model swap happens lazily inside the first LOW forward, so that step's dt
    carries the load; the 0.5-EMA washes it out one step later and every
    estimate wears its '~'."""
    if pbar is None:
        pbar = comfy.utils.ProgressBar(total_steps)
    want_preview = _previews_enabled() and node_id is not None
    if previewer is None and want_preview:
        try:
            previewer = _AnimatedVideoPreviewer(model.model.latent_format,
                                                preview_mode=preview_mode,
                                                device=model.load_device)
        except Exception:
            previewer = None
    state = {"last": 0.0, "spoke": False}

    def callback(step, x0, x, _phase_total):
        # continuous progress across phases; pass None so core's single-frame
        # preview never overrides our animated one
        if clock is not None and clock_key is not None:
            # v576: the clock owns the bar (measure() -> push(): value = wall
            # clock, total = elapsed + eta). Wrapped like the tick writer was -
            # telemetry must never break sampling.
            # v577: dt comes from clock.tick() - the cursor lives on the RUN,
            # not in this closure. A per-callback cursor was seeded at BUILD
            # time, and the chain builds both stage callbacks before phase 1,
            # so the first LOW step measured the whole HIGH phase (222.7 s
            # where the truth was 93.4 s; the eta doubled at the handoff).
            try:
                dt = clock.tick()
                counted = clock.measure(clock_key, dt)
                p = clock.posts[clock_key]
                tag = clock_key.split(":", 1)[1] if ":" in clock_key else clock_key
                if not counted:
                    # v580: a TAIL callback (RES4LYF fires one past the last
                    # step; measured 37.8 s on 65 frames). The clock kept its
                    # seconds -- sum(spent) still equals the wall -- but it is
                    # not a step, so it does not speak. Saying "step 3/3" twice
                    # with two different numbers is the console lying quietly.
                    pass
                elif dt >= _CLOCK_FOLD_S:
                    state["spoke"] = True
                    seta, reta = clock.eta(tag), clock.eta()
                    print(f"[PLS SAMPLER] step {p['done']}/{p['units']} {tag} "
                          f"{dt:.1f}s (stage eta {_fmt_clock(seta or 0)}, "
                          f"run eta ~{_fmt_clock(reta or 0)})")
                elif p["done"] >= p["units"] and not state["spoke"] and p["units"] > 1:
                    per = p["spent"] / max(1, p["done"])
                    per_s = f"{per * 1000:.0f}ms" if per < 0.1 else f"{per:.1f}s"
                    print(f"[PLS SAMPLER] {tag} steps 1-{p['units']} folded ("
                          f"{p['spent']:.1f}s total, {per_s}/step - "
                          f"too fast to narrate one by one)")
            except Exception:
                pass
        else:
            try:
                pbar.update_absolute(step_offset + step + 1, total_steps, None)
            except Exception:
                pass
        # v415: apply a live (mid-render) preview-mode switch for this node, if one
        # was POSTed. Fully guarded -- a switch can only change the PREVIEW, and any
        # error here is swallowed, so it can never break sampling or the progress bar.
        if previewer is not None and node_id is not None:
            try:
                req = _get_live_preview_mode(node_id)
                if req is not None and req != previewer.mode:
                    previewer.set_mode(req)
            except Exception:
                pass
        if not want_preview or previewer is None or not previewer.ok:
            return
        now = time.time()
        is_last = (step_offset + step + 1) >= total_steps
        if (now - state["last"]) < min_interval and not is_last:
            return
        state["last"] = now
        try:
            frames = previewer.frames_b64(x0)
            if frames:
                _send_preview(node_id, frames, fps)
        except Exception:
            pass  # preview must never break sampling

    return callback


def _fix_empty_latent(model, latent_image, latent):
    """fix_empty_latent_channels with a signature fallback. Recent ComfyUI takes
    the two downscale-ratio hints; older builds take only (model, latent_image).
    The call is at sample time (not import time), so this never blocks loading."""
    try:
        return comfy.sample.fix_empty_latent_channels(
            model, latent_image,
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )
    except TypeError:
        return comfy.sample.fix_empty_latent_channels(model, latent_image)


def _apply_sigma_shift(model, shift):
    """v491: apply a flow-matching sigma shift by patching the model's model_sampling,
    mirroring Core's ModelSamplingSD3.patch (comfy_extras/nodes_model_advanced.py) — the
    exact mechanism the Wan MoE KSampler bakes in. WAN 2.2 flow sampling needs a shift
    (native default 8.0); without it the sigma schedule is mis-scaled and a short (e.g.
    8-step) run cannot converge, leaving residual latent noise (RGB speckle) in the decode.

    shift <= 0 is a NO-OP: the model is returned UNCHANGED, so the node keeps its prior
    behaviour (drive the shift from an upstream ModelSamplingSD3 node, or the model's own
    native model_sampling). Any positive shift clones the model and overrides its
    model_sampling with a DiscreteFlow+CONST advanced sampler; the original model object is
    never mutated. This clone MUST happen before the schedule is built (KSampler reads
    model_sampling at construction), so in High + Low BOTH the sigma schedule AND the
    boundary step-count (which rebuilds sigmas from the HIGH model in _moe_sample) see it."""
    if shift is None or shift <= 0:
        return model
    import comfy.model_sampling as _cms
    m = model.clone()

    class _ModelSamplingAdvanced(_cms.ModelSamplingDiscreteFlow, _cms.CONST):
        pass

    ms = _ModelSamplingAdvanced(model.model.model_config)
    ms.set_parameters(shift=shift, multiplier=1000)
    m.add_object_patch("model_sampling", ms)
    return m


# ── v544: per-expert sampler / scheduler ────────────────────────────────────
# The LOW expert may run its OWN sampling algorithm and (in "Wan MoE parity") its
# OWN sigma schedule. "same as high" reuses the HIGH value -> byte-identical to v543.
SAME_AS_HIGH = "same as high"


def _low_or(value, high):
    """Resolve a LOW-expert override: 'same as high' (or empty) -> the HIGH value."""
    v = str(value or SAME_AS_HIGH).strip()
    return high if v == SAME_AS_HIGH else v


# ---------------------------------------------------------------------------
# v685 -- external NOISE source
# ---------------------------------------------------------------------------
# The sampler makes its starting noise in exactly TWO places (the classic path
# and the sigma path); every MoE variant runs through them. Threading an extra
# argument would have meant ~20 edits across five signatures and fourteen call
# sites in this file -- and ONE forgotten call site means the override works in
# some modes and silently not in others, which is the worst failure this node
# could have.
#
# So the source is parked for the DURATION of one run instead. _sample_impl
# sets it in a try/finally, _initial_noise reads it. The scope is honest: a
# ComfyUI node executes sequentially, the value is set and cleared inside a
# single sample() call, and the finally clause runs even when the sampler
# raises. Guard #108 drives exactly that, exception included.
_ACTIVE_NOISE = None


class _NoiseContext:
    """Park a NOISE source for one run. Reentrant by save/restore, so a nested
    call (there is none today) could not corrupt an outer one."""

    def __init__(self, source):
        self.source = source
        self.prev = None

    def __enter__(self):
        global _ACTIVE_NOISE
        self.prev = _ACTIVE_NOISE
        _ACTIVE_NOISE = self.source
        return self

    def __exit__(self, *exc):
        global _ACTIVE_NOISE
        _ACTIVE_NOISE = self.prev
        return False


def _initial_noise(latent, latent_image, seed, add_noise):
    """The noise the model will actually denoise.

    add_noise off -> zeros (unchanged). A wired NOISE source -> its own field,
    generated against the CORRECTED latent geometry. Otherwise the untouched
    original: comfy.sample.prepare_noise. A gaussian source at strength 1.0
    delegates to that same call, so wiring the default changes no byte."""
    if not add_noise:
        return torch.zeros(latent_image.size(), dtype=latent_image.dtype,
                           layout=latent_image.layout, device="cpu")
    batch_inds = latent["batch_index"] if "batch_index" in latent else None
    src = _ACTIVE_NOISE
    if src is not None:
        return src.generate_noise({"samples": latent_image,
                                   "batch_index": batch_inds})
    return comfy.sample.prepare_noise(latent_image, seed, batch_inds)


def _polyhedron_sample(model, seed, steps, cfg, sampler_name, scheduler,
                       positive, negative, latent, denoise, add_noise,
                       start_step=None, last_step=None, force_full_denoise=False,
                       node_id=None, callback=None, preview_mode="latent2rgb (smooth)"):
    """Local mirror of nodes.common_ksampler (kept here so a Core refactor of the
    node-level helper can't silently change our behaviour). Generates the initial
    noise (or zeros when add_noise is off), wires the standard preview callback,
    and hands everything to comfy.sample.sample — which owns the denoise->sigma
    slicing (denoise<1 => new_steps=int(steps/denoise), keep the last steps+1
    sigmas => lower start sigma => less noise on the input latent) AND the
    start/last-step slice used for the HIGH/LOW split (start_step / last_step,
    force_full_denoise pins the final sigma to 0 on the closing pass)."""
    latent_image = latent["samples"]
    latent_image = _fix_empty_latent(model, latent_image, latent)

    noise = _initial_noise(latent, latent_image, seed, add_noise)

    noise_mask = latent.get("noise_mask", None)

    own_clock = None
    if callback is None:
        # v576: this helper OWNS the run -> it owns the clock. Units = the
        # steps this call will actually execute (the advanced start/last slice,
        # clamped >=1); weight = the latent's true numel (frames included).
        _units = max(1, min(int(steps),
                            int(last_step) if last_step is not None else int(steps))
                     - int(start_step or 0))
        pbar = comfy.utils.ProgressBar(steps)
        own_clock = _RunClock(pbar)
        own_clock.post("step:main", _units,
                       float(max(1, latent_image.numel())) * _cfg_forwards(cfg))
        callback = _make_preview_callback(model, steps, node_id, pbar=pbar,
                                          preview_mode=preview_mode,
                                          clock=own_clock, clock_key="step:main")
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = comfy.sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler, positive, negative,
        latent_image, denoise=denoise, disable_noise=(not add_noise),
        start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise,
        noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar,
        seed=seed,
    )
    if own_clock is not None:
        _clock_close(own_clock, (("main", "step:main"),))

    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return (out,)


def _moe_sample(model_high, model_low, seed, steps, cfg_high, cfg_low,
                sampler_name, scheduler, positive, negative, latent, denoise,
                add_noise, boundary, node_id=None, preview_mode="latent2rgb (smooth)",
                handoff_mode="Continuous", sampler_low=None, scheduler_low=None):
    """Wan 2.2 MoE (mixture-of-experts) two-stage denoising in ONE node.

    Wan 2.2 A14B is two expert denoisers — a HIGH-noise expert (early, high-sigma
    steps: layout/motion) and a LOW-noise expert (late, low-sigma steps: detail) —
    switched at a signal-to-noise BOUNDARY (T2V 0.875, I2V 0.900). We build the
    real sigma schedule (denoise-aware; any sigma_shift comes from the model's
    upstream model_sampling), count the steps whose starting sigma is >= boundary
    (= the HIGH stage), then run the manual HIGH/LOW chain automatically: the HIGH
    expert up to the boundary leaving leftover noise, the LOW expert from there to
    a clean finish. This is exactly the two-KSamplerAdvanced setup, collapsed into
    one node, and it delegates every sampling step to Core via _polyhedron_sample."""
    # The schedule the sampling will actually use (KSampler.set_steps owns the
    # denoise->sigma slicing). No model call here — calculate_sigmas is cheap.
    device = model_high.load_device
    ks = comfy.samplers.KSampler(model_high, steps, device,
                                 sampler=sampler_name, scheduler=scheduler, denoise=denoise)
    sigmas = ks.sigmas
    total = max(0, len(sigmas) - 1)                       # number of sampling steps

    # The Wan MoE parity path (pre-v493 value: "Rebase · x0 + renoise") is the Wan MoE KSampler parity
    # path — an ISOLATED branch delegated to _moe_sample_rebase. Detection keys off a stable
    # keyword ("moe" for the current name, "rebase" for the pre-v493 value) so a save from
    # either naming routes correctly; anything else (incl. "Continuous") is the byte-identical
    # Continue path below (same split via sigmas[:-1] >= boundary, HIGH leftover, LOW zeros).
    _hm = str(handoff_mode or "").lower()
    # v544 policy: the LOW expert's SAMPLER is free in every mode (it only consumes the
    # sigma array). Its SCHEDULER *defines* that array, so it is only legal where the two
    # segments are independent -- i.e. "Wan MoE parity" (HIGH ends on a clean x0, LOW
    # re-noises it onto its OWN schedule). In "Continuous" the LOW pass picks up leftover
    # noise at exactly sigma[k] of the HIGH schedule; a different schedule would break that
    # seam, so we IGNORE it and say so instead of silently mis-denoising.
    _sched_low = _low_or(scheduler_low, scheduler)
    if ("moe" not in _hm) and ("rebase" not in _hm) and _sched_low != scheduler:
        print("[PLS] Sampler: scheduler_low is ignored in 'Continuous' (both experts share "
              "ONE schedule; the LOW pass continues on the HIGH sigmas). Switch handoff_mode "
              "to 'Wan MoE parity' for a per-expert schedule.")
        _sched_low = scheduler
    _samp_low = _low_or(sampler_low, sampler_name)
    if ("moe" in _hm) or ("rebase" in _hm):
        return _moe_sample_rebase(model_high, model_low, seed, steps, cfg_high, cfg_low,
                                  sampler_name, scheduler, positive, negative, latent, denoise,
                                  add_noise, boundary, sigmas, total,
                                  node_id=node_id, preview_mode=preview_mode)

    # HIGH stage = steps whose STARTING sigma (sigmas[:-1]) is >= boundary.
    high_steps = int((sigmas[:-1] >= boundary).sum().item()) if total > 0 else 0

    # Degenerate splits -> a single expert, clean finish.
    if high_steps <= 0:
        return _polyhedron_sample(model_low, seed, steps, cfg_low, sampler_name, scheduler,
                                  positive, negative, latent, denoise, add_noise,
                                  start_step=0, last_step=10000, force_full_denoise=True,
                                  node_id=node_id, preview_mode=preview_mode)
    if high_steps >= total:
        return _polyhedron_sample(model_high, seed, steps, cfg_high, sampler_name, scheduler,
                                  positive, negative, latent, denoise, add_noise,
                                  start_step=0, last_step=10000, force_full_denoise=True,
                                  node_id=node_id, preview_mode=preview_mode)

    # ONE shared progress bar + previewer so the live preview and the bar run
    # continuously 0..total across HIGH then LOW (a per-phase callback would reset
    # both at the handoff). Both experts share the latent format, so one previewer
    # built from the HIGH expert serves both. v576: ONE shared clock on that bar -
    # the bar tells TIME across the whole chain, and after the first measured HIGH
    # step the still-unstarted LOW stage already has a rung-2 estimate.
    pbar = comfy.utils.ProgressBar(total)
    clock = _RunClock(pbar)
    for _k, _u, _w in _chain_posts(high_steps, total - high_steps,
                                   latent["samples"].numel(), cfg_high, cfg_low):
        clock.post(_k, _u, _w)
    previewer = None
    if _previews_enabled() and node_id is not None:
        try:
            previewer = _AnimatedVideoPreviewer(model_high.model.latent_format,
                                                preview_mode=preview_mode,
                                                device=model_high.load_device)
        except Exception:
            previewer = None
    cb_high = _make_preview_callback(model_high, total, node_id, step_offset=0,
                                     pbar=pbar, previewer=previewer,
                                     clock=clock, clock_key="step:high")
    cb_low = _make_preview_callback(model_low, total, node_id, step_offset=high_steps,
                                    pbar=pbar, previewer=previewer,
                                    clock=clock, clock_key="step:low")

    # Phase 1 — HIGH expert, steps [0 : high_steps], keep leftover noise.
    (mid,) = _polyhedron_sample(model_high, seed, steps, cfg_high, sampler_name, scheduler,
                                positive, negative, latent, denoise, add_noise,
                                start_step=0, last_step=high_steps, force_full_denoise=False,
                                callback=cb_high)
    # Phase 2 — LOW expert, steps [high_steps : end], no new noise, clean finish.
    out = _polyhedron_sample(model_low, seed, steps, cfg_low, _samp_low, _sched_low,
                             positive, negative, mid, denoise, add_noise=False,
                             start_step=high_steps, last_step=10000, force_full_denoise=True,
                             callback=cb_low)
    _clock_close(clock, (("high", "step:high"), ("low", "step:low")))
    return out


def _moe_sample_rebase(model_high, model_low, seed, steps, cfg_high, cfg_low,
                       sampler_name, scheduler, positive, negative, latent, denoise,
                       add_noise, boundary, sigmas, total,
                       node_id=None, preview_mode="latent2rgb (smooth)", sampler_low=None, scheduler_low=None):
    """v492: the "Wan MoE parity" handoff (pre-v493 value: "Rebase · x0 + renoise") — exact
    parity with the Wan MoE KSampler
    (stduhpf/ComfyUI-WanMoeKSampler). Three mechanical deltas vs. the Continue path, each
    verified against ComfyUI Core 0.27.0 and the reference node:

      Delta 1 — split in TIMESTEP space, ONE step earlier. The reference compares each
        step's TARGET timestep t = model_sampling.timestep(sigma)/1000 against the boundary
        and switches at the first t < boundary MINUS ONE (switch = j-1). For a FLOW model
        timestep(sigma) = sigma*multiplier (model_sampling.py: ModelSamplingDiscreteFlow),
        multiplier=1000, so t == sigma and our Continue count(sigmas[:-1] >= boundary) = k is
        the reference's j; hence switch = k-1. The boundary-CROSSING step runs on LOW here,
        not HIGH.
      Delta 2 — Phase 1 (HIGH) ends on a clean x0 estimate. last_step=switch with
        force_full_denoise=True makes KSampler.sample slice sigmas[:switch+1] and set the
        final sigma to 0 (comfy/samplers.py L1428-1431), so the last HIGH step lands on x0
        instead of leaving leftover noise on sigma_switch.
      Delta 3 — Phase 2 (LOW) re-noises x0 with the ORIGINAL seed noise. CONST.noise_scaling
        (comfy/model_sampling.py L94-97) is sigma*noise + (1-sigma)*x0 (noise_scale=1.0), so
        feeding the full seed noise (add_noise pass-through) re-noises x0 up to sigma_switch,
        then LOW denoises to 0.

    No noise loop-through is needed: comfy.sample.prepare_noise(latent, seed, batch_inds) is
    deterministic in (seed, shape, batch_index), and _polyhedron_sample packs out=latent.copy()
    (preserving batch_index), so Phase 2 regenerates the IDENTICAL original noise tensor from
    the same seed — exactly the reference's "same noise passed to both sample calls". (Its
    Phase-1 disable_noise=True is inert: comfy.sample.sample never forwards disable_noise in
    Core 0.27.0 — comfy/sample.py L71/L74 — so both phases receive the full noise, consistent
    with the clean reference images.)

    The Continue path (in _moe_sample) is untouched by this and stays the clean, mathematically
    reine euler-continuation (the inverse/forward noise_scaling roundtrip cancels exactly)."""
    # Delta 1 — timestep-space split (reference 1:1): switch one step earlier than Continue.
    sampling = model_high.get_model_object("model_sampling")
    ts = [float(sampling.timestep(s)) / 1000.0 for s in sigmas.tolist()]
    switch = total
    for j in range(1, len(ts)):
        if ts[j] < boundary:
            switch = j - 1
            break

    # Degenerate splits -> a single expert, clean finish (mirrors the reference's
    # start_with_high / end_wth_low edges and the Continue path's clamps).
    if switch <= 0:
        return _polyhedron_sample(model_low, seed, steps, cfg_low, sampler_name, scheduler,
                                  positive, negative, latent, denoise, add_noise,
                                  start_step=0, last_step=10000, force_full_denoise=True,
                                  node_id=node_id, preview_mode=preview_mode)
    if switch >= total:
        return _polyhedron_sample(model_high, seed, steps, cfg_high, sampler_name, scheduler,
                                  positive, negative, latent, denoise, add_noise,
                                  start_step=0, last_step=10000, force_full_denoise=True,
                                  node_id=node_id, preview_mode=preview_mode)

    # ONE shared progress bar + previewer so the live preview runs continuously 0..total
    # across HIGH then LOW; the LOW callback offset is the switch index.
    # v576: one shared clock on the bar (see _moe_sample) - posts split at `switch`.
    pbar = comfy.utils.ProgressBar(total)
    clock = _RunClock(pbar)
    for _k, _u, _w in _chain_posts(switch, total - switch,
                                   latent["samples"].numel(), cfg_high, cfg_low):
        clock.post(_k, _u, _w)
    previewer = None
    if _previews_enabled() and node_id is not None:
        try:
            previewer = _AnimatedVideoPreviewer(model_high.model.latent_format,
                                                preview_mode=preview_mode,
                                                device=model_high.load_device)
        except Exception:
            previewer = None
    cb_high = _make_preview_callback(model_high, total, node_id, step_offset=0,
                                     pbar=pbar, previewer=previewer,
                                     clock=clock, clock_key="step:high")
    cb_low = _make_preview_callback(model_low, total, node_id, step_offset=switch,
                                    pbar=pbar, previewer=previewer,
                                    clock=clock, clock_key="step:low")

    # Phase 1 — HIGH expert, steps [0 : switch], finish on a clean x0 (force_full_denoise=True).
    (mid,) = _polyhedron_sample(model_high, seed, steps, cfg_high, sampler_name, scheduler,
                                positive, negative, latent, denoise, add_noise,
                                start_step=0, last_step=switch, force_full_denoise=True,
                                callback=cb_high)
    # Phase 2 — LOW expert, steps [switch : end], re-noise x0 with the SAME seed noise
    # (add_noise pass-through), then denoise to a clean finish.
    # v544: the LOW expert may run its own sampler AND its own schedule here -- the rebase
    # makes the segments independent (x0 + renoise onto the LOW schedule's own sigma at
    # `switch`), which is exactly what the dual-sigma docstring calls "legal per-expert
    # curve design".
    out = _polyhedron_sample(model_low, seed, steps, cfg_low,
                             _low_or(sampler_low, sampler_name),
                             _low_or(scheduler_low, scheduler),
                             positive, negative, mid, denoise, add_noise=add_noise,
                             start_step=switch, last_step=10000, force_full_denoise=True,
                             callback=cb_low)
    _clock_close(clock, (("high", "step:high"), ("low", "step:low")))
    return out


def _polyhedron_sample_sigmas(model, seed, cfg, sampler_name, sigmas, positive, negative,
                              latent, add_noise=True, node_id=None, callback=None,
                              preview_mode="latent2rgb (smooth)"):
    """v416: like _polyhedron_sample but driven by an EXTERNAL sigma schedule (a SIGMAS
    input) instead of steps/scheduler/denoise. Builds a sampler object from sampler_name
    and runs comfy.sample.sample_custom with the explicit sigma array -- exactly Core's
    SamplerCustom path. The array IS the schedule: its length sets the step count, and
    its last value decides a clean (->0) vs. a leftover-noise finish. Preview/callback,
    noise prep, noise_mask and out-packing mirror _polyhedron_sample."""
    latent_image = latent["samples"]
    latent_image = _fix_empty_latent(model, latent_image, latent)

    noise = _initial_noise(latent, latent_image, seed, add_noise)

    noise_mask = latent.get("noise_mask", None)
    steps = max(1, int(sigmas.shape[-1]) - 1)
    own_clock = None
    if callback is None:
        # v576: sigma-driven single run - the array IS the schedule, so its
        # length is the unit count; weight = the latent's true numel.
        pbar = comfy.utils.ProgressBar(steps)
        own_clock = _RunClock(pbar)
        own_clock.post("step:main", steps,
                       float(max(1, latent_image.numel())) * _cfg_forwards(cfg))
        callback = _make_preview_callback(model, steps, node_id, pbar=pbar,
                                          preview_mode=preview_mode,
                                          clock=own_clock, clock_key="step:main")
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    sampler = comfy.samplers.sampler_object(sampler_name)
    samples = comfy.sample.sample_custom(
        model, noise, cfg, sampler, sigmas, positive, negative, latent_image,
        noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed,
    )
    if own_clock is not None:
        _clock_close(own_clock, (("main", "step:main"),))

    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return (out,)


def _moe_sample_sigmas(model_high, model_low, seed, cfg_high, cfg_low, sampler_name,
                       sigmas_high, sigmas_low, positive, negative, latent,
                       add_noise, node_id=None, preview_mode="latent2rgb (smooth)",
                       handoff_mode="Continuous", sampler_low=None):
    """v416: WAN 2.2 MoE chain driven by TWO external sigma schedules, one per expert
    (e.g. the two outputs of the Polyhedron Dual Sigma Curve). No boundary maths -- the
    split is explicit: HIGH runs sigmas_high (leftover noise inherent if it does not
    reach 0), LOW continues on sigmas_low from there to a clean finish. The shared
    continuous progress bar + previewer match _moe_sample.

    v495: the handoff_mode decides the seam. 'Continuous' (default, byte-identical to the
    v416 behaviour): continuity is enforced defensively -- the LOW segment's first sigma is
    snapped to the HIGH segment's last, so the noise level is continuous across the handoff
    (no re-noising jump), and Phase 2 continues with NO new noise. 'Wan MoE parity': the two
    segments are fully INDEPENDENT (no snap) -- the HIGH segment finishes on a clean x0 (its
    final sigma is forced to 0, the exact mirror of force_full_denoise) and the LOW segment
    re-noises that x0 with the ORIGINAL seed noise up to ITS OWN first sigma
    (CONST.noise_scaling; prepare_noise is deterministic in seed/shape/batch_index, so the
    add_noise pass-through regenerates the identical noise tensor). This makes per-expert
    curve design legal: shapes, step counts and sigma ranges no longer have to meet."""
    sigmas_high = sigmas_high.clone()
    sigmas_low = sigmas_low.clone()
    high_steps = max(0, int(sigmas_high.shape[-1]) - 1)
    low_steps = max(0, int(sigmas_low.shape[-1]) - 1)

    # Degenerate: one segment is empty -> run the other expert alone.
    if high_steps <= 0:
        return _polyhedron_sample_sigmas(model_low, seed, cfg_low, _low_or(sampler_low, sampler_name), sigmas_low,
                                         positive, negative, latent, add_noise=add_noise,
                                         node_id=node_id, preview_mode=preview_mode)
    if low_steps <= 0:
        return _polyhedron_sample_sigmas(model_high, seed, cfg_high, sampler_name, sigmas_high,
                                         positive, negative, latent, add_noise=add_noise,
                                         node_id=node_id, preview_mode=preview_mode)

    # v495: the handoff decides the seam (same keyword detection as _moe_sample).
    _hm = str(handoff_mode or "").lower()
    is_rebase = ("moe" in _hm) or ("rebase" in _hm)

    if is_rebase:
        # Wan MoE parity: NO seam snap (segment independence is the point). HIGH finishes
        # on a clean x0: force its final sigma to 0 (mirror of force_full_denoise).
        if float(sigmas_high[-1]) != 0.0:
            sigmas_high[-1] = 0.0
    else:
        # Continuous: defensive continuity -- snap LOW[0] to HIGH[-1] if they diverge
        # (clean handoff, byte-identical v416 behaviour).
        hi_end = float(sigmas_high[-1])
        lo_start = float(sigmas_low[0])
        if abs(hi_end - lo_start) > 1e-4:
            print("[PLS v416 SIGMAS] handoff continuity: snapping sigmas_low[0] "
                  "{:.4f} -> {:.4f} (= sigmas_high[-1])".format(lo_start, hi_end))
            if abs(lo_start - float(sigmas_high[0])) <= 1e-4:
                print("[PLS v495 SIGMAS] hint: sigmas_low starts at sigma_max -- this "
                      "looks like a FULL curve, but these inputs expect SEGMENTS. For a "
                      "full curve feed ONE array into the single `sigmas` input; the "
                      "sampler splits it at the Handoff.")
            sigmas_low[0] = sigmas_high[-1]

    total = high_steps + low_steps
    # v576: one shared clock on the bar (see _moe_sample) - the explicit
    # per-expert step counts ARE the posts.
    pbar = comfy.utils.ProgressBar(total)
    clock = _RunClock(pbar)
    for _k, _u, _w in _chain_posts(high_steps, low_steps,
                                   latent["samples"].numel(), cfg_high, cfg_low):
        clock.post(_k, _u, _w)
    previewer = None
    if _previews_enabled() and node_id is not None:
        try:
            previewer = _AnimatedVideoPreviewer(model_high.model.latent_format,
                                                preview_mode=preview_mode,
                                                device=model_high.load_device)
        except Exception:
            previewer = None
    cb_high = _make_preview_callback(model_high, total, node_id, step_offset=0,
                                     pbar=pbar, previewer=previewer,
                                     clock=clock, clock_key="step:high")
    cb_low = _make_preview_callback(model_low, total, node_id, step_offset=high_steps,
                                    pbar=pbar, previewer=previewer,
                                    clock=clock, clock_key="step:low")

    # Phase 1 — HIGH expert on sigmas_high (leftover noise if it does not reach 0).
    (mid,) = _polyhedron_sample_sigmas(model_high, seed, cfg_high, sampler_name, sigmas_high,
                                       positive, negative, latent, add_noise=add_noise,
                                       callback=cb_high)
    # Phase 2 — LOW expert on sigmas_low. Continuous: continue the SAME denoise, no new
    # noise. Wan MoE parity: re-noise the x0 with the SAME seed noise (add_noise
    # pass-through; prepare_noise regenerates the identical tensor from the seed).
    if is_rebase:
        out = _polyhedron_sample_sigmas(model_low, seed, cfg_low, _low_or(sampler_low, sampler_name), sigmas_low,
                                        positive, negative, mid, add_noise=add_noise,
                                        callback=cb_low)
    else:
        out = _polyhedron_sample_sigmas(model_low, seed, cfg_low, _low_or(sampler_low, sampler_name), sigmas_low,
                                        positive, negative, mid, add_noise=False,
                                        callback=cb_low)
    _clock_close(clock, (("high", "step:high"), ("low", "step:low")))
    return out


def _moe_sample_split_sigmas(model_high, model_low, seed, cfg_high, cfg_low, sampler_name,
                             sigmas, positive, negative, latent, add_noise, boundary,
                             node_id=None, preview_mode="latent2rgb (smooth)",
                             handoff_mode="Continuous"):
    """v495: WAN 2.2 MoE driven by ONE external sigma curve (the single SIGMAS input in
    High + Low). The sampler splits the curve at the Handoff (boundary) and delegates the
    two slices to _moe_sample_sigmas, so the handoff semantics are shared:

      Continuous      split k = count(sigmas[:-1] >= boundary); HIGH = sigmas[:k+1],
                      LOW = sigmas[k:]. The slices SHARE the seam sigma, so the handoff is
                      continuous by construction (the defensive snap never fires).
      Wan MoE parity  split = k - 1 (one step earlier: for flow models the reference's
                      timestep test t = timestep(sigma)/1000 equals sigma, so the count on
                      the raw array IS the reference's j and switch = j-1). The delegate
                      then forces the HIGH slice onto a clean x0 and re-noises the LOW
                      slice to sigma_switch with the seed noise.

    This is the natural consumer for a FULL custom curve (e.g. the Polyhedron Sigma
    Curve): one descending curve in, the sampler does the MoE split, and the boundary
    stays the single source of truth for the model switch. The split is clamped to
    [0, total]; a degenerate split (empty slice) falls through to _moe_sample_sigmas'
    own single-expert handling, so exactly one expert runs the whole curve as-is."""
    sigmas = sigmas.clone()
    total = max(0, int(sigmas.shape[-1]) - 1)
    k = int((sigmas[:-1] >= boundary).sum().item()) if total > 0 else 0
    _hm = str(handoff_mode or "").lower()
    split = (k - 1) if (("moe" in _hm) or ("rebase" in _hm)) else k
    split = max(0, min(split, total))
    return _moe_sample_sigmas(model_high, model_low, seed, cfg_high, cfg_low, sampler_name,
                              sigmas[:split + 1], sigmas[split:],
                              positive, negative, latent, add_noise,
                              node_id=node_id, preview_mode=preview_mode,
                              handoff_mode=handoff_mode)


class ULSSampler:

    @classmethod
    def VALIDATE_INPUTS(cls, preview_mode=None):
        """v830: preview_mode combo values carry decoration (diamond + size),
        so a workflow saved before this cut holds the bare label -- a string
        the list no longer offers for the two TAE modes. Naming ONLY
        preview_mode here exempts ONLY it from the built-in membership check
        (the ph_mask_editor form); sampler_name, scheduler and every other
        combo keep core's validation. The previewer strips and resolves;
        an unknown mode falls back to latent2rgb (smooth) as ever."""
        return True

    """Polyhedron Sampler — one node for every sampling job.

    SINGLE mode is a superset of the stock KSampler AND KSampler (Advanced): the
    simple `denoise` field (which Advanced lacks) plus the advanced step-slice
    controls `start_at_step` / `end_at_step` / `return_with_leftover_noise` (which
    the simple KSampler lacks).

    DUAL EXPERT mode — flipped on with the Single/Dual pill — runs a noise-split
    mixture-of-experts (MoE) in this single node: `model` is the HIGH-noise expert,
    the revealed `model_low` is the LOW-noise expert, and the node switches between
    them at the SNR `boundary` (with its own `cfg_low`). It does the manual HIGH/LOW
    two-KSamplerAdvanced chain automatically, so the whole split collapses to one
    node — sampler/scheduler pickers built in. Used today by Wan 2.2, and applicable
    to any model family that ships a high/low expert pair."""

    CATEGORY = "Polyhedron/Sampling"
    DESCRIPTION = ("One sampler for everything. SINGLE = superset of KSampler + KSampler "
                   "(Advanced): a denoise field plus start/end_at_step + "
                   "return_with_leftover_noise, full sampler/scheduler lists, add_noise. "
                   "DUAL EXPERT pill = noise-split two-expert (MoE) denoising in one node "
                   "(model = HIGH expert, model_low = LOW expert, switched at the boundary, "
                   "with cfg_low) — used by Wan 2.2. Image and video latents; live preview.")
    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    SEARCH_ALIASES = ["sampler", "sample", "ksampler", "denoise", "img2img", "v2v", "wan", "polyhedron"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used to denoise the latent."}),
                "positive": ("CONDITIONING", {"tooltip": "Conditioning to include."}),
                "negative": ("CONDITIONING", {"tooltip": "Conditioning to exclude."}),
                "latent_image": ("LATENT", {"tooltip": "The latent to denoise (image or video)."}),
                # ── Mode — the fundamental Single/Dual decision, kept at the top ──
                "dual_moe": ("BOOLEAN", {"default": False,
                                         "label_on": "High + Low", "label_off": "Single",
                                         "tooltip": "Sampling architecture. Single: one model denoises every step. High + Low: a "
                                                    "noise-split mixture-of-experts (MoE) -- the HIGH-noise expert ('model') "
                                                    "handles the early, high-sigma steps (structure/motion); the LOW-noise expert "
                                                    "('model_low') the later, low-sigma steps (detail); they hand off at the "
                                                    "Handoff noise level. Used by Wan 2.2 (Handoff 0.875 T2V / 0.900 I2V) and by "
                                                    "any model family that ships a high/low expert pair. In High + Low the manual "
                                                    "start/end_at_step controls are replaced by the automatic Handoff split."}),
                "boundary": ("FLOAT", {"default": 0.875, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001,
                                       "tooltip": "Handoff — the noise level where the run passes from the HIGH-noise "
                                                  "expert to the LOW-noise expert. Steps whose sigma is >= Handoff run "
                                                  "on the HIGH expert ('model'), the rest on the LOW expert "
                                                  "('model_low'). Higher = the HIGH expert covers fewer (only the "
                                                  "loudest) steps. Set it by hand for your task — "
                                                  "Text→Video 0.875, Image→Video 0.900 (Wan 2.2 standard)."}),
                "cfg_low": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                                      "tooltip": "High + Low: CFG for the LOW-noise expert pass ('cfg' applies to "
                                                 "the HIGH-noise expert). Distilled / Lightning LoRAs use 1.0."}),
                # ── Universal sampling controls ──
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True,
                                 "tooltip": "Seed for the initial noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000,
                                  "tooltip": "Number of denoising steps."}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                                  "tooltip": "Classifier-Free Guidance. Flow models (e.g. WAN) usually want a lower "
                                             "value than SD; distilled / Lightning LoRAs use 1.0. In High + Low "
                                             "this is the HIGH-noise expert's CFG."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,
                                 {"tooltip": "Sampling algorithm — the full list, no per-model subset."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,
                              {"tooltip": "Sigma schedule — the full list."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "1.0 = full generation from noise. Lower keeps the input "
                                                 "latent's structure for img2img / v2v (e.g. 0.25). This is "
                                                 "the field the stock KSampler (Advanced) does not expose."}),
                "add_noise": ("BOOLEAN", {"default": True,
                                          "tooltip": "Add initial noise. Turn off for the LOW pass of a manual "
                                                     "HIGH/LOW split (denoise an already-noised latent)."}),
                # ── Manual step-slice — Single mode only (Dual uses the boundary) ──
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000,
                                          "tooltip": "First sigma step to run. The LOW pass of a manual split "
                                                     "starts where the HIGH pass ended (e.g. 19)."}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000,
                                        "tooltip": "Last sigma step to run (clamped to the schedule). The HIGH "
                                                   "pass of a manual split ends here (e.g. 19 of 36)."}),
                "return_with_leftover_noise": ("BOOLEAN", {"default": False,
                                                           "tooltip": "Leave the latent partially noised for a "
                                                                      "following pass — ON for the HIGH pass, OFF "
                                                                      "for the LOW / final pass (OFF forces a clean "
                                                                      "final denoise). Ignored in High + Low mode "
                                                                      "(the boundary drives the split there)."}),
                # ── Live preview decoder (v413) — kept LAST for serialised index
                #    stability; applies to both modes (always live, never disabled) ──
                # v830: the two TAE entries carry the diamond + size (the
                # v828/v829 form); the model-free latent2rgb modes stay bare.
                "preview_mode": (["Still · ComfyUI", "Still · latent2rgb",
                                  "Video · latent2rgb (smooth)", "Video · latent2rgb (crisp)",
                                  _mode_entry("Video · TAE (taew2_1)", "taew2_1"),
                                  _mode_entry("Video · TAE (lighttaew2_1)", "lighttaew2_1")],
                                 {"default": "Still · ComfyUI",
                                  "tooltip": "In-node live preview decoder — visualisation only, it NEVER "
                                             "changes the output latent. 'Still ·' modes show a SINGLE frame "
                                             "that refreshes each step (like the KSampler / standard ComfyUI); "
                                             "'Video ·' modes ANIMATE every frame of the predicted latent (only "
                                             "visible when generating video — a single image has one frame). "
                                             "'Still · ComfyUI' uses ComfyUI's own previewer (honours "
                                             "--preview-method; models/vae_approx TAESD/TAEHV or latent2rgb). "
                                             "'Still · latent2rgb' forces the model-free latent2rgb view "
                                             "(ignores vae_approx even when present). 'Video · latent2rgb' is "
                                             "the same projection animated ('smooth' LANCZOS / 'crisp' NEAREST). "
                                             "'Video · TAE' animates a TAE decode: 'lighttaew2_1' is the WAN "
                                             "video TAE (TAEHV); 'taew2_1' the image-TAE path. Anything "
                                             "unavailable falls back to latent2rgb (smooth). Applies from the "
                                             "start of each render."}),
                # ── Flow-matching sigma shift (v491) — kept LAST in required for
                #    serialised index stability (appended after preview_mode). 0 = OFF:
                #    model_sampling is left untouched (upstream ModelSamplingSD3 / native).
                #    WAN 2.2 wants 8.0 (matches the Wan MoE KSampler's built-in shift). ──
                "sigma_shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.01, "round": 0.01,
                                          "tooltip": "Flow-matching sigma shift (ModelSamplingSD3). 0 = OFF: the "
                                                     "model_sampling is left as-is — use an upstream ModelSamplingSD3 "
                                                     "node, or the model's native shift. WAN 2.2 flow sampling NEEDS a "
                                                     "shift: set 8.0 to match the Wan MoE KSampler. Without it a short "
                                                     "run cannot converge and leaves residual noise (RGB speckle). "
                                                     "Applied to BOTH experts in High + Low, before the schedule and "
                                                     "the boundary split are computed. When driving with an external "
                                                     "SIGMAS curve, leave this at 0 (the curve owns the schedule). "
                                                     "Leave at 0 for non-flow models (SD/SDXL) — a flow shift only "
                                                     "makes sense for flow-matching models like WAN."}),
                # ── MoE handoff mode (v492; renamed v493) — kept LAST in required for
                #    serialised index stability (after sigma_shift). "Continuous" IS the v491
                #    behaviour (byte-identical); "Wan MoE parity" matches the Wan MoE KSampler
                #    exactly. High + Low only; inert in the external-SIGMAS path. ──
                "handoff_mode": (["Continuous", "Wan MoE parity"],
                                 {"default": "Continuous",
                                  "tooltip": "High + Low: how the HIGH-noise expert hands the latent to the "
                                             "LOW-noise expert at the Handoff. 'Continuous' (default) runs both "
                                             "experts as ONE unbroken denoise — the HIGH expert stops leaving "
                                             "leftover noise and the LOW expert picks it straight up (this node's "
                                             "original behaviour, the mathematically clean euler-continuation). "
                                             "'Wan MoE parity' reproduces the Wan MoE KSampler exactly: the switch "
                                             "happens ONE step earlier, the HIGH expert finishes on a clean x0 "
                                             "estimate, and the LOW expert re-noises that x0 with the SAME seed "
                                             "noise before denoising to the end — use this to match Wan 2.2 res_2s "
                                             "output. Applies to the built-in schedule AND to external SIGMAS in "
                                             "High + Low (two segments on sigmas_high/sigmas_low, or ONE curve on "
                                             "sigmas that the sampler splits at the Handoff). Single mode ignores "
                                             "this."}),
                # ── v544: per-expert sampler / scheduler — kept LAST in required for
                #    serialised index stability (appended after handoff_mode). Default
                #    "same as high" = the LOW expert reuses the HIGH values, so a run is
                #    byte-identical to v543. High + Low only. ──
                "sampler_low": ([SAME_AS_HIGH] + list(comfy.samplers.KSampler.SAMPLERS),
                                {"default": SAME_AS_HIGH,
                                 "tooltip": "High + Low: sampling ALGORITHM for the LOW-noise "
                                            "expert ('sampler_name' drives the HIGH expert). "
                                            "Free in every handoff mode — a sampler only "
                                            "CONSUMES the sigma schedule, it does not define "
                                            "it. Note that ancestral / SDE samplers (*_a, "
                                            "*_sde) inject fresh noise every step, which "
                                            "changes the character of the LOW pass."}),
                "scheduler_low": ([SAME_AS_HIGH] + list(comfy.samplers.KSampler.SCHEDULERS),
                                  {"default": SAME_AS_HIGH,
                                   "tooltip": "High + Low: sigma SCHEDULE for the LOW-noise "
                                              "expert. Only honoured with handoff_mode = 'Wan "
                                              "MoE parity', where the two segments are "
                                              "independent (the HIGH expert finishes on a clean "
                                              "x0 and the LOW expert re-noises it onto ITS OWN "
                                              "schedule). In 'Continuous' both experts share ONE "
                                              "schedule by construction (the LOW pass picks up "
                                              "the leftover noise at the Handoff sigma), so this "
                                              "is IGNORED and the console says so. Inert when an "
                                              "external SIGMAS curve drives the run — the curve "
                                              "owns the schedule."}),
            },
            "optional": {
                "model_low": ("MODEL", {"tooltip": "High + Low: the LOW-noise expert. The 'model' input is the "
                                                   "HIGH-noise expert. (Wan 2.2 ships these as a high/low pair.) "
                                                   "Connect this and set the mode pill to High + Low."}),
                # ── External sigma schedules (v416) — optional. When connected they
                #    OVERRIDE the built-in schedule: scheduler / steps / denoise (and
                #    the split: boundary in High+Low, start/end in Single) go inert; the
                #    sigma node owns the whole curve. Single uses `sigmas`; High + Low
                #    uses BOTH `sigmas_high` and `sigmas_low` (e.g. the Polyhedron Dual
                #    Sigma Curve's two outputs, one per expert). ──
                "sigmas": ("SIGMAS", {"tooltip": "An external sigma schedule (e.g. from the Polyhedron Sigma "
                                                 "Curve). Overrides scheduler/steps/denoise — the array IS the "
                                                 "schedule (length = steps; last value decides a clean vs. "
                                                 "leftover-noise finish). High + Low mode: the sampler splits "
                                                 "this ONE curve at the Handoff and runs both experts on the "
                                                 "slices (boundary stays active; handoff_mode decides the seam). "
                                                 "Ignored if sigmas_high + sigmas_low are both connected."}),
                "sigmas_high": ("SIGMAS", {"tooltip": "High + Low mode: the HIGH-noise expert's sigma SEGMENT "
                                                      "(NOT a full curve — it should END where the LOW segment "
                                                      "begins; for a full curve use the single `sigmas` input). "
                                                      "Needs sigmas_low connected too. Together they drive the "
                                                      "split, so boundary is ignored."}),
                "sigmas_low": ("SIGMAS", {"tooltip": "High + Low mode: the LOW-noise expert's sigma SEGMENT. "
                                                     "Continuous: its first sigma is snapped to the HIGH segment's "
                                                     "last (continuous handoff). Wan MoE parity: fully independent "
                                                     "of the HIGH segment — HIGH ends on x0 and this segment "
                                                     "re-noises it to its own first sigma."}),
                # v685: the noise the model actually denoises. Unwired = the
                # untouched original (comfy.sample.prepare_noise). Wired with a
                # gaussian source at strength 1.0 = bit identical to that. Any
                # other character is a deliberate excursion -- see the Polyhedron
                # Seed node's noise output.
                "noise": ("NOISE", {"tooltip": "Optional noise SOURCE (e.g. the Polyhedron Seed's "
                                               "`noise` output, or Core's RandomNoise). It replaces the "
                                               "sampler's own gaussian starting noise -- this is the "
                                               "noise the model denoises, so its character (brown, "
                                               "fractal, ...) reaches the result. Unwired, nothing "
                                               "changes. Ignored while add_noise is off."}),
            },
        }

    def sample(self, **kwargs):
        """v558: the ENTIRE run - including every preview decode - inside the
        staging-log mute. The TAEHV/TAESD previewer stages its model on EVERY
        preview step, so a long run buried the console in identical INFO lines.

        The mute is a SCALPEL (nodes/ph_logmute.py): it drops ONLY the known
        staging lines, so a real warning still reaches the console, and it
        reports how many it swallowed. Our own telemetry is print(), so it
        always passes.

        No new widget, on purpose: uls_sampler.js is index-based over 17
        serialised names WITH length heuristics (the file warns about this
        itself), and a stability-first house does not shift that layout for a
        logging tweak. Ask for a switch and it gets its own careful cut."""
        with _MuteStagingLogs(True, label="Sampler"):
            return self._sample_impl(**kwargs)

    def _sample_impl(self, model, positive, negative, latent_image, seed, steps, cfg,
               sampler_name, scheduler, denoise, add_noise,
               start_at_step, end_at_step, return_with_leftover_noise,
               dual_moe, boundary, cfg_low, preview_mode="latent2rgb (smooth)",
               sigma_shift=0.0, handoff_mode="Continuous",
               sampler_low=SAME_AS_HIGH, scheduler_low=SAME_AS_HIGH,
               model_low=None, sigmas=None, sigmas_high=None, sigmas_low=None,
               noise=None):
        # v685: park the NOISE source for the whole run. EVERY path below --
        # single, sigma-driven, and all three MoE variants -- reaches its noise
        # through _initial_noise, so one context covers them all, and the
        # finally clause inside _NoiseContext puts the module back even if the
        # sampler raises.
        with _NoiseContext(noise):
            return self._run(model, positive, negative, latent_image, seed, steps, cfg,
                             sampler_name, scheduler, denoise, add_noise,
                             start_at_step, end_at_step, return_with_leftover_noise,
                             dual_moe, boundary, cfg_low, preview_mode=preview_mode,
                             sigma_shift=sigma_shift, handoff_mode=handoff_mode,
                             sampler_low=sampler_low, scheduler_low=scheduler_low,
                             model_low=model_low, sigmas=sigmas,
                             sigmas_high=sigmas_high, sigmas_low=sigmas_low)

    def _run(self, model, positive, negative, latent_image, seed, steps, cfg,
             sampler_name, scheduler, denoise, add_noise,
             start_at_step, end_at_step, return_with_leftover_noise,
             dual_moe, boundary, cfg_low, preview_mode="latent2rgb (smooth)",
             sigma_shift=0.0, handoff_mode="Continuous",
             sampler_low=SAME_AS_HIGH, scheduler_low=SAME_AS_HIGH,
             model_low=None, sigmas=None, sigmas_high=None, sigmas_low=None):
        # Id of THIS node, so the live preview attaches to it (None -> no preview).
        node_id = _current_node_id()
        # v415: each render starts from the node's serialised preview_mode; drop any
        # stale live override so a mid-render switch only affects the run it was made in.
        _clear_live_preview_mode(node_id)
        # v491: flow-matching sigma shift. Patch model_sampling BEFORE any schedule is
        # built, so BOTH the sampling sigmas AND the High + Low boundary step-count (which
        # rebuilds sigmas from the HIGH model in _moe_sample) see the shift. 0 => untouched
        # (upstream ModelSamplingSD3 / native). Clones only; the input models are not mutated.
        if sigma_shift and sigma_shift > 0:
            # v495: an ENGAGED external sigma path owns the whole schedule, and the shift
            # only affects schedule GENERATION (timestep(sigma) is shift-independent for
            # flow models) -- patching here would be a runtime no-op that misleads (the
            # widget cannot shift an external curve; shape it upstream). Skip; the UI
            # greys the widget in this state.
            _ext_sigmas = ((sigmas_high is not None and sigmas_low is not None)
                           or (sigmas is not None)) if dual_moe else (sigmas is not None)
            if not _ext_sigmas:
                model = _apply_sigma_shift(model, sigma_shift)
                if model_low is not None:
                    model_low = _apply_sigma_shift(model_low, sigma_shift)
        if dual_moe:
            # High + Low: the boundary drives the HIGH/LOW split internally, so the
            # manual start/end_at_step + leftover controls are not used here.
            if model_low is None:
                raise ValueError(
                    "Polyhedron Sampler: High + Low mode needs the 'model_low' input "
                    "(the LOW-noise expert). Connect it, or switch the pill to Single.")
            # v416: with BOTH external sigma schedules connected, the split is explicit
            # (one array per expert) -> the boundary maths is bypassed entirely.
            if sigmas_high is not None and sigmas_low is not None:
                return _moe_sample_sigmas(model, model_low, seed, cfg, cfg_low,
                                          sampler_name, sigmas_high, sigmas_low,
                                          positive, negative, latent_image, add_noise,
                                          node_id=node_id, preview_mode=preview_mode,
                                          handoff_mode=handoff_mode, sampler_low=sampler_low)
            # v495: ONE external curve in High + Low -> the sampler splits it at the
            # Handoff (boundary) and runs the two experts on the slices. handoff_mode
            # decides the seam. The segment pair (BOTH sigmas_high+sigmas_low) takes
            # precedence over the single curve.
            if sigmas is not None:
                return _moe_sample_split_sigmas(model, model_low, seed, cfg, cfg_low,
                                                sampler_name, sigmas, positive, negative,
                                                latent_image, add_noise, boundary,
                                                node_id=node_id, preview_mode=preview_mode,
                                                handoff_mode=handoff_mode)
            return _moe_sample(model, model_low, seed, steps, cfg, cfg_low,
                               sampler_name, scheduler, positive, negative, latent_image,
                               denoise, add_noise, boundary, node_id=node_id,
                               preview_mode=preview_mode, handoff_mode=handoff_mode,
                               sampler_low=sampler_low, scheduler_low=scheduler_low)
        # v416: Single with an external sigma schedule -> run it directly (the array is
        # the whole schedule, so steps/scheduler/denoise and the manual slice are unused).
        if sigmas is not None:
            return _polyhedron_sample_sigmas(model, seed, cfg, sampler_name, sigmas,
                                             positive, negative, latent_image,
                                             add_noise=add_noise, node_id=node_id,
                                             preview_mode=preview_mode)
        # Single: KSampler + KSampler (Advanced) superset. leftover noise OFF ->
        # force the final sigma to 0 (clean); ON -> stop early WITH noise.
        force_full_denoise = not return_with_leftover_noise
        return _polyhedron_sample(model, seed, steps, cfg, sampler_name, scheduler,
                                  positive, negative, latent_image, denoise, add_noise,
                                  start_step=start_at_step, last_step=end_at_step,
                                  force_full_denoise=force_full_denoise, node_id=node_id,
                                  preview_mode=preview_mode)
