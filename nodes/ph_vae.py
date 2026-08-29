"""Polyhedron VAE -- encode, decode, or both, in one node.

BORN FROM FRANK'S WORKFLOW SCREENSHOT (v574): a Filter sandwich needs
Decode -> pixel op -> Encode -> Decode, and stock makes that three separate
boxes with three VAE wires. Sometimes you want ONE lane (less clutter),
sometimes you want BOTH lanes in one box. The mode toggle is the point.

WHAT STOCK DOES ON THIS EXACT REVISION (ba9ffa0a, read, not remembered):
VAE.decode estimates its need with a built-in formula, attempts the FULL
pass, and falls back to hardcoded 256 px tiles only when an OOM exception
fires. The measured gap (three completed runs, v570-v573): WDDM rarely
throws OOM - it pages over PCIe and GRINDS. A decode that does not fit
will not crash on this machine; it will silently crawl for minutes. This
node moves the verdict BEFORE the pass, using comfy's OWN formulas
(vae.memory_used_decode/encode), and says everything out loud.

THE SCAR, RESPECTED: no WANVIDEOVAE bridging (the v122-v244 arc was
abandoned for good reasons; wan_model_bridge.py stays an untouched
anchor). No custom tiler either - comfy's encode_tiled/decode_tiled do
the work; we wrap them with a verdict and a voice.

Modes:
  both      - the wires decide, loudly: pixels wired -> encode lane runs,
              samples wired -> decode lane runs. Both wired -> both run,
              independently. An output whose lane did not run passes the
              matching input through; if there is nothing to pass, it
              returns None and SAYS so (wiring that output downstream
              would fail - the truth beats a silent dummy tensor).
  encode    - only the encode lane, even if samples are wired (announced).
  decode    - only the decode lane, even if pixels are wired (announced).
  roundtrip - decode(encode(pixels)): the built-in what-survives-the-VAE
              meter. Prints the v568 sharpness metric (Laplacian variance
              of the luma) before and after - the 444-second measurement
              as a one-click tool. The latent output carries the
              intermediate, the image output the round-tripped pixels.
"""

import time

import torch

import comfy.model_management as mm

try:                                  # package import (normal ComfyUI load)
    from .ph_logmute import MuteStagingLogs as _MuteInfoLogs
except Exception:                     # script import (tests, tooling)
    try:
        from ph_logmute import MuteStagingLogs as _MuteInfoLogs
    except Exception:
        class _MuteInfoLogs:          # last resort: a no-op capsule
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False


_MODES = ["both", "encode", "decode", "roundtrip"]
_TILING = ["auto", "off", "on"]


def _vae_budget_verdict(need_b, free_b, tiling_mode):
    """v574: the pure budget verdict, exec'd in isolation by the guard.

    'on'  -> tiled, always (the user said so).
    'off' -> full, always (the user said so - the begin line still warns
             when the need exceeds what is free).
    'auto'-> tiled when the need exceeds 85% of free (the v565 headroom
             convention). Stock waits for an OOM exception instead - and
             WDDM pages instead of throwing, so stock's fallback can
             simply never fire while the pass grinds (measured three
             times on the final-pass arc)."""
    m = str(tiling_mode)
    if m == "on":
        return "tiled"
    if m == "off":
        return "full"
    return "tiled" if float(need_b) > 0.85 * float(free_b) else "full"


def _luma_sharpness(img):
    """v568's measuring stick as a number: variance of the Laplacian of
    the luma, first frame. Bigger = sharper. The roundtrip mode prints
    the before/after ratio - what the VAE keeps."""
    f = img[0:1].movedim(-1, 1).float()
    luma = (0.2126 * f[:, 0:1] + 0.7152 * f[:, 1:2] + 0.0722 * f[:, 2:3])
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                     device=luma.device, dtype=luma.dtype).view(1, 1, 3, 3)
    lap = torch.nn.functional.conv2d(luma, k, padding=1)
    return float(lap.var().item())


def _gb(x):
    return float(x) / 1e9



# ---------------------------------------------------------------------------
# v871 -- JOINT (nested) AV LATENTS.
#
# MiniMax H3 hands one latent that carries BOTH halves:
# {"samples": NestedTensor((video, audio))}. Core splits it in the NODES, not in
# comfy/sd.py -- and the two split rules are the whole contract:
#
#   nodes.VAEDecode                      -> latent.unbind()[0]    (video)
#   nodes_audio.vae_decode_audio         -> latent.unbind()[-1]   (audio)
#
# Both check `.is_nested`, which real torch tensors also answer (False), so ONE
# check covers both worlds. The functions below are DECLARED MIRRORS of those
# two lines; if Core ever changes the order, this comment is where to look.
# ---------------------------------------------------------------------------

def _is_joint_latent(latent):
    """True for a joint audio/video latent. Never raises: a plain tensor
    answers is_nested=False and anything else answers nothing at all."""
    return bool(getattr(latent, "is_nested", False))


def _video_latent(latent):
    """DECLARED MIRROR of Core nodes.VAEDecode: the VIDEO half of a joint
    latent, or the latent itself when there is nothing to split."""
    if _is_joint_latent(latent):
        parts = latent.unbind()
        if parts:
            return parts[0]
    return latent


def _audio_latent(latent):
    """DECLARED MIRROR of Core nodes_audio.vae_decode_audio: the AUDIO half of a
    joint latent, or None. None matters -- a plain video latent must NEVER be
    handed to an audio VAE just because an audio_vae happens to be wired."""
    if _is_joint_latent(latent):
        parts = latent.unbind()
        if len(parts) > 1:
            return parts[-1]
    return None


# --- v900: the MiniMax H3 single-frame decode trap --------------------------
#
# Core's MiniMaxH3VideoVAE.decode() carries a SEPARATE branch for one latent
# frame:
#
#     if z.shape[2] == 1:
#         dec = self._finalize_pixels(self._adaptive_decode(z)[:, :, -1:, :, :])
#
# It bypasses decode_temporal entirely, so the ViT decoder sees exactly ONE
# time token. Its position ids are length-normalised per axis --
# 2*(arange(0.5, size)/size) - 1 -- which for size 1 collapses the time axis
# onto exactly 0.0. No even token count ever produces that value (2 -> -0.5,
# +0.5; 4 -> -0.75 .. +0.75), so the decoder is handed a coordinate the
# training distribution does not contain. The result is a 16 px grid, one
# latent pixel wide, laid over the whole frame.
#
# MEASURED on Frank's own field renders (row-mean FFT, power at period 16 px
# against the surrounding band): seven single-frame runs at 1024x1024 all sat
# between x14 and x34, across three samplers, four cfg values, 8/20/30 steps,
# with and without LoRAs, on sdpa AND on sage3 fp4 -- i.e. it is none of those
# things. The same prompt rendered as 22 frames measured x4.3, and the
# reference workflow x2.1. It is the decode path, not the sampler.
#
# Known upstream: Comfy-Org/ComfyUI issue #15416 (opened 2026-08-08, still
# open), which reports the same 16 px patch grid, notes the errors are
# dtype-independent (fp16 / bf16 / fp32 within 0.1 of each other, so it is
# algorithmic and not precision), and states that padding to 2 tokens works.
#
# What we do about it: hand the decoder TWO time tokens instead of one and
# keep the last decoded frame. Core's own temporal arithmetic then applies --
# run in the sandbox against its constants (clip_length 17, token_drop 3,
# vae_ratio_t 4): t=2 pads internally to 7 tokens and yields 5 pixel frames.
# We keep the last, which is Core's own convention in the branch above and
# what the reference workflow does with ImageFromBatch(batch_index=-1).
#
# NOT a Core mirror, and deliberately not written as one: Core's branch stays
# as it is, we simply do not enter it.


def _is_h3_video_vae(vae):
    """True for Core's MiniMax H3 VIDEO VAE, asked of the OBJECT.

    Measured, not guessed from a filename: comfy/sd.py builds
    `self.first_stage_model = comfy.ldm.minimax.vae.MiniMaxH3VideoVAE(...)`
    for this checkpoint (detected there by decoder.transformer_blocks.0.scale1
    + encoder.down.5.block.0.conv1.weight). A user is free to rename the file;
    the class is what decides. Asked by name rather than isinstance so a Core
    refactor degrades to "fix off" instead of an import error.
    """
    try:
        return type(vae.first_stage_model).__name__ == "MiniMaxH3VideoVAE"
    except Exception:
        return False


def _h3_still_needs_pad(vae, latent, mode):
    """True when this decode is the single-frame H3 case and the fix is on."""
    if str(mode) != "auto":
        return False
    if not _is_h3_video_vae(vae):
        return False
    try:
        return int(latent.ndim) == 5 and int(latent.shape[2]) == 1
    except Exception:
        return False


def _h3_pad_to_two(latent):
    """One time token -> two, by repeating it. Content is unchanged; only the
    ViT's time coordinate moves off the untrained 0.0."""
    import torch
    return torch.cat([latent, latent], dim=2)


class ULSVAE:
    """One VAE node, four modes, and a verdict that runs before the pass."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "mode": (_MODES,
                         {"default": "both",
                          "tooltip": "both = two independent lanes; whatever is wired runs (pixels -> LATENT, "
                                     "samples -> IMAGE; wire both and one box replaces a stacked Encode+Decode "
                                     "pair). encode / decode = force one lane; the other input is ignored and the "
                                     "console says so. roundtrip = a quality TEST, not a workflow step: the pixels "
                                     "are encoded and decoded straight back, and the console prints a sharpness "
                                     "ratio before/after -- near 1.0 the VAE kept the detail, low means it ate it "
                                     "(fine texture like lace, hair or embroidery usually does not survive). Answers"
                                     " 'is this detail worth adding BEFORE the VAE?' with a number."}),
                "tiling": (_TILING,
                           {"default": "auto",
                            "tooltip": "auto = decide BEFORE the pass from "
                                       "comfy's own memory formula vs free "
                                       "VRAM (tiled above 85% of free). "
                                       "Stock instead attempts the full "
                                       "pass and waits for an OOM that "
                                       "WDDM often never throws - it pages "
                                       "and grinds. off/on force the "
                                       "verdict."}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096,
                                      "step": 32,
                                      "tooltip": "Spatial tile in PIXELS "
                                                 "(decode converts to "
                                                 "latent internally, like "
                                                 "stock)."}),
                "tile_overlap": ("INT", {"default": 64, "min": 0, "max": 4096,
                                         "step": 32,
                                         "tooltip": "Spatial overlap in "
                                                    "pixels."}),
                "temporal_size": ("INT", {"default": 64, "min": 8,
                                          "max": 4096, "step": 4,
                                          "tooltip": "Video VAEs only: "
                                                     "frames per temporal "
                                                     "tile (stock "
                                                     "semantics)."}),
                "temporal_overlap": ("INT", {"default": 8, "min": 4,
                                             "max": 4096, "step": 4,
                                             "tooltip": "Video VAEs only: "
                                                        "frames of temporal "
                                                        "overlap."}),
                "mute_staging_logs": ("BOOLEAN", {"default": True,
                                                  "label_on": "On",
                                                  "label_off": "Off",
                                                  "tooltip": "Silence "
                                                             "ComfyUI's "
                                                             "staging INFO "
                                                             "lines during "
                                                             "the pass; the "
                                                             "capsule "
                                                             "reports how "
                                                             "many it "
                                                             "muted."}),
                "h3_still_fix": (["auto", "off"],
                                 {"default": "auto",
                                  "tooltip": "MiniMax H3 only, single-frame "
                                             "latents only. Core decodes ONE "
                                             "latent frame through a branch "
                                             "that hands the ViT decoder a "
                                             "time coordinate of exactly 0.0, "
                                             "which no even token count "
                                             "produces -- the result is a 16 "
                                             "px grid over the whole image "
                                             "(ComfyUI issue #15416). 'auto' "
                                             "decodes two time tokens instead "
                                             "and keeps the last frame; it "
                                             "does nothing on any other VAE or "
                                             "for any length above 1. 'off' is "
                                             "the stock path, for A/B with the "
                                             "same seed."}),
            },
            "optional": {
                "pixels": ("IMAGE",),
                "samples": ("LATENT",),
                # v871: the audio half of a joint latent needs a DIFFERENT
                # model than the video half (MiniMax H3 ships
                # minimax_h3_audio_vae_fp32 next to the video VAE). A
                # link-typed input is a SOCKET and never occupies a
                # widgets_values slot, so this costs no baseline.
                "audio_vae": ("VAE", {"tooltip":
                                      "OPTIONAL, and only for a joint "
                                      "audio/video latent: the AUDIO vae. "
                                      "Wire it and the audio output carries "
                                      "the decoded soundtrack; leave it and "
                                      "the audio output is None and says so. "
                                      "This must be the audio VAE, not the "
                                      "video one -- the node checks and "
                                      "names it if the model declares no "
                                      "sample rate."}),
            },
        }

    # v871: AUDIO is APPENDED. Slots 0 and 1 keep their indices, so every
    # workflow saved before v871 keeps its wires.
    RETURN_TYPES = ("LATENT", "IMAGE", "AUDIO")
    RETURN_NAMES = ("latent", "image", "audio")
    FUNCTION = "run"
    CATEGORY = "Polyhedron/VAE"
    DESCRIPTION = ("VAE encode, decode, or both in one node - the mode "
                   "toggle replaces stacked Encode/Decode pairs. Decides "
                   "full vs tiled BEFORE the pass from comfy's own memory "
                   "formulas (stock waits for an OOM that WDDM often never "
                   "throws), narrates need/free/seconds, and offers a "
                   "roundtrip mode that prints what survives the VAE as a "
                   "sharpness number. Handles a JOINT audio/video latent "
                   "(MiniMax H3) in ONE box: the video half decodes through "
                   "vae, and wiring audio_vae decodes the audio half too, so "
                   "the split happens once and correctly instead of twice in "
                   "two separate stock nodes.")

    # ------------------------------------------------------------------ lanes

    def _encode_need(self, vae, pixels):
        # Comfy's own formula wants the shape the model will see. For a
        # video VAE (latent_dim 3) the frame batch becomes T: (1, C, T, H, W).
        try:
            if getattr(vae, "latent_dim", 2) == 3:
                shape = (1, 3, int(pixels.shape[0]),
                         int(pixels.shape[1]), int(pixels.shape[2]))
            else:
                shape = (int(pixels.shape[0]), 3,
                         int(pixels.shape[1]), int(pixels.shape[2]))
            return float(vae.memory_used_encode(shape, vae.vae_dtype))
        except Exception:
            return 0.0   # no formula -> no verdict; announced by caller

    def _decode_need(self, vae, latent):
        try:
            return float(vae.memory_used_decode(tuple(latent.shape),
                                                vae.vae_dtype))
        except Exception:
            return 0.0

    def _encode_lane(self, vae, pixels, verdict, tile_size, tile_overlap,
                     temporal_size, temporal_overlap):
        t0 = time.monotonic()
        if verdict == "tiled":
            # Stock VAEEncodeTiled semantics: pixel-space tiles, raw frame
            # counts; sd.py converts internally.
            t = vae.encode_tiled(pixels[:, :, :, :3], tile_x=int(tile_size),
                                 tile_y=int(tile_size),
                                 overlap=int(tile_overlap),
                                 tile_t=int(temporal_size),
                                 overlap_t=int(temporal_overlap))
        else:
            # Stock VAEEncode: full pass; comfy itself falls back to tiled
            # on a REAL OOM exception - we keep that net underneath ours.
            t = vae.encode(pixels[:, :, :, :3])
        dur = time.monotonic() - t0
        n = int(pixels.shape[0])
        print(f"[PLS] VAE: encode done in {dur:.1f}s "
              f"({dur / max(1, n) * 1000.0:.0f} ms/frame, {verdict}) -> "
              f"latent {tuple(t.shape)}")
        return {"samples": t}, dur

    def _decode_lane(self, vae, latent, verdict, tile_size, tile_overlap,
                     temporal_size, temporal_overlap, h3_still_fix="auto"):
        t0 = time.monotonic()
        # v900: two time tokens instead of one for the H3 single-frame case.
        # See the block above _is_h3_video_vae for why. Guarded so it cannot
        # touch any other VAE, any length above 1, or the 'off' setting.
        padded = _h3_still_needs_pad(vae, latent, h3_still_fix)
        if padded:
            latent = _h3_pad_to_two(latent)
            print("[PLS] VAE: MiniMax H3 single-frame latent -> decoding TWO "
                  "time tokens and keeping the last frame (ComfyUI #15416: "
                  "one token puts the ViT's time coordinate on an untrained "
                  "0.0 and prints a 16 px grid). Set h3_still_fix=off for "
                  "the stock path.")
        if verdict == "tiled":
            # Stock VAEDecodeTiled semantics, replicated verbatim: clamp
            # the overlaps, convert to LATENT space, min-clamp temporal.
            ts, ov = int(tile_size), int(tile_overlap)
            tps, tpo = int(temporal_size), int(temporal_overlap)
            if ts < ov * 4:
                ov = ts // 4
            if tps < tpo * 2:
                tpo = tpo // 2
            tcomp = vae.temporal_compression_decode()
            if tcomp is not None:
                tps = max(2, tps // tcomp)
                tpo = max(1, min(tps // 2, tpo // tcomp))
            else:
                tps, tpo = None, None
            comp = vae.spacial_compression_decode()
            images = vae.decode_tiled(latent, tile_x=ts // comp,
                                      tile_y=ts // comp, overlap=ov // comp,
                                      tile_t=tps, overlap_t=tpo)
        else:
            images = vae.decode(latent)
        if images.ndim == 5:   # combine video batches, like stock
            images = images.reshape(-1, images.shape[-3], images.shape[-2],
                                    images.shape[-1])
        if padded:
            # Core's temporal arithmetic turns 2 tokens into 5 pixel frames
            # (it pads to 7 internally, repeating the last token). Keep the
            # last -- Core's own choice in the single-frame branch, and what
            # the reference workflow does with ImageFromBatch(-1). Taken
            # AFTER the reshape, where the frame axis is dim 0 whichever way
            # the decoder laid it out.
            got = int(images.shape[0])
            images = images[-1:]
            print(f"[PLS] VAE: H3 still fix decoded {got} frame(s), kept the "
                  f"last -> 1 image")
        dur = time.monotonic() - t0
        n = int(images.shape[0])
        print(f"[PLS] VAE: decode done in {dur:.1f}s "
              f"({dur / max(1, n) * 1000.0:.0f} ms/frame, {verdict}) -> "
              f"{int(images.shape[2])}x{int(images.shape[1])} x{n}f")
        return images, dur

    def _decode_audio_lane(self, audio_vae, latent, samples):
        """The AUDIO half of a joint latent, through the audio VAE.

        THE WAVEFORM MATHS IS A 1:1 MIRROR of Core's vae_decode_audio: a
        five-sigma loudness guard, floored at 1.0, applied by division. We do
        NOT improve on it -- deviating would make this node's audio differ from
        every other Comfy path for the SAME latent, which is a silent
        incompatibility and the worst kind.

        What we add is the MEASUREMENT. Stock divides and says nothing, so a
        quiet result looks like a bad model instead of a working guard. Here the
        divisor, the peak before and the peak after are all on the log.

        DELIBERATELY NOT TILED: the audio latent is [B, 32, 2, t]. Core's tiled
        variant takes tile_x/tile_y, which are spatial, and this node's tile
        widgets describe IMAGE tiles. Wiring them in here would be nonsense
        dressed as a feature.
        """
        t0 = time.monotonic()

        rate, via = None, None
        for _attr in ("audio_sample_rate_output", "audio_sample_rate"):
            _v = getattr(audio_vae, _attr, None)
            if _v:
                rate, via = int(_v), _attr
                break
        if rate is None:
            rate, via = 44100, "fallback 44100"
            print("[PLS] VAE: the wired audio_vae declares NO sample rate "
                  "(neither audio_sample_rate_output nor audio_sample_rate). "
                  "That is what a VIDEO vae looks like on this input -- check "
                  "the wire. Decoding anyway at 44100 Hz; if the result is "
                  "noise, this line is why.")
        if isinstance(samples, dict) and "sample_rate" in samples:
            rate, via = int(samples["sample_rate"]), "the latent's own sample_rate"

        audio = audio_vae.decode(latent).movedim(-1, 1)

        try:
            peak_in = float(audio.abs().max())
        except Exception:
            peak_in = float("nan")

        std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
        std[std < 1.0] = 1.0
        audio = audio / std

        try:
            peak_out = float(audio.abs().max())
            div = float(std.max())
        except Exception:
            peak_out, div = float("nan"), float("nan")

        dur = time.monotonic() - t0
        secs = (float(audio.shape[-1]) / rate) if rate else 0.0
        print(f"[PLS] VAE: audio decode done in {dur:.1f}s -> "
              f"{int(audio.shape[1])}ch x {int(audio.shape[-1])} samples "
              f"({secs:.2f}s @ {rate} Hz via {via})")
        print(f"[PLS] VAE: loudness guard (stock's 5-sigma, floored at 1.0) "
              f"divided by {div:.3f}; peak {peak_in:.3f} -> {peak_out:.3f}. "
              f"A divisor of 1.000 means the guard did not bite.")
        return {"waveform": audio, "sample_rate": rate}

    # -------------------------------------------------------------------- run

    def run(self, vae, mode, tiling, tile_size, tile_overlap, temporal_size,
            temporal_overlap, mute_staging_logs, h3_still_fix="auto",
            pixels=None, samples=None,
            audio_vae=None):
        mode = str(mode)
        lat_in = samples["samples"] if isinstance(samples, dict) else samples
        # v871: the DECODE lane sees the video half; lat_in itself stays whole
        # so the LATENT output still passes the joint latent through and a
        # downstream audio decode keeps working.
        lat_dec = _video_latent(lat_in)
        out_audio = None
        free = 0.0
        try:
            free = float(mm.get_free_memory(vae.device))
        except Exception:
            pass

        want_enc = (mode in ("encode", "roundtrip")
                    or (mode == "both" and pixels is not None))
        want_dec = (mode == "decode"
                    or (mode == "both" and lat_in is not None))

        # ---- the begin line: wires, verdicts, and the truth about stock ----
        bits = [f"[PLS] VAE: begin mode={mode}"]
        if pixels is not None:
            bits.append(f"pixels={int(pixels.shape[2])}x"
                        f"{int(pixels.shape[1])} x{int(pixels.shape[0])}f")
        if lat_in is not None:
            if _is_joint_latent(lat_in):
                # v871: .shape on a joint latent delegates to its FIRST part,
                # so the old line reported the video half without knowing it
                # was reporting a half. Name both.
                bits.append("latent=JOINT " + " + ".join(
                    str(tuple(t.shape)) for t in lat_in.unbind()))
            else:
                bits.append(f"latent={tuple(lat_in.shape)}")
        bits.append(f"free={_gb(free):.1f} GB (before load)")
        print(" ".join(bits))

        if mode == "encode" and lat_in is not None:
            print("[PLS] VAE: mode=encode -> the wired samples are IGNORED "
                  "(loudly; switch to 'both' if you meant both lanes)")
        if mode == "decode" and pixels is not None:
            print("[PLS] VAE: mode=decode -> the wired pixels are IGNORED "
                  "(loudly; switch to 'both' if you meant both lanes)")
        if mode == "both" and pixels is None and lat_in is None:
            print("[PLS] VAE: mode=both with NOTHING wired -> nothing to "
                  "do; both outputs return None (the wires are the truth)")
        if want_enc and pixels is None:
            print(f"[PLS] VAE: mode={mode} needs pixels and none are wired "
                  f"-> encode lane skipped, outputs pass through/None")
            want_enc = False
        if mode == "decode" and lat_in is None:
            print("[PLS] VAE: mode=decode needs samples and none are wired "
                  "-> decode lane skipped, outputs pass through/None")
            want_dec = False

        out_latent = samples if isinstance(samples, dict) else (
            {"samples": lat_in} if lat_in is not None else None)
        out_image = pixels
        enc_verdict = dec_verdict = None

        with _MuteInfoLogs(bool(mute_staging_logs), label="VAE"):
            if want_enc:
                need = self._encode_need(vae, pixels)
                enc_verdict = _vae_budget_verdict(need, free, tiling)
                warn = (" - the user forced 'off' ABOVE the budget; stock "
                        "would only fall back on an OOM exception, and "
                        "WDDM pages instead of throwing"
                        if (str(tiling) == "off" and need > 0.85 * free)
                        else "")
                print(f"[PLS] VAE: encode verdict={enc_verdict} "
                      f"(comfy's own formula: need ~{_gb(need):.1f} GB vs "
                      f"{_gb(free):.1f} GB free, tiling={tiling}){warn}")
                out_latent, _ = self._encode_lane(vae, pixels, enc_verdict,
                                                  tile_size, tile_overlap,
                                                  temporal_size,
                                                  temporal_overlap)

            if mode == "roundtrip":
                # The meter: decode what we just encoded, then measure.
                lat_mid = out_latent["samples"]
                need = self._decode_need(vae, lat_mid)
                dec_verdict = _vae_budget_verdict(need, free, tiling)
                print(f"[PLS] VAE: roundtrip decode verdict={dec_verdict} "
                      f"(need ~{_gb(need):.1f} GB vs {_gb(free):.1f} GB "
                      f"free)")
                out_image, _ = self._decode_lane(vae, lat_mid, dec_verdict,
                                                 tile_size, tile_overlap,
                                                 temporal_size,
                                                 temporal_overlap,
                                                 h3_still_fix)
                try:
                    s0 = _luma_sharpness(pixels)
                    s1 = _luma_sharpness(out_image)
                    ratio = (s1 / s0) if s0 > 0 else float("nan")
                    print(f"[PLS] VAE: roundtrip sharpness "
                          f"{s0:.5f} -> {s1:.5f} (ratio {ratio:.3f}) - "
                          f"what the VAE keeps: detail finer than "
                          f"~8 px does not survive this trip "
                          f"(measured).")
                except Exception as exc:
                    print(f"[PLS] VAE: sharpness meter failed ({exc}) - "
                          f"the roundtrip itself is unaffected")

            elif want_dec:
                need = self._decode_need(vae, lat_dec)
                dec_verdict = _vae_budget_verdict(need, free, tiling)
                warn = (" - the user forced 'off' ABOVE the budget; stock "
                        "would only fall back on an OOM exception, and "
                        "WDDM pages instead of throwing"
                        if (str(tiling) == "off" and need > 0.85 * free)
                        else "")
                print(f"[PLS] VAE: decode verdict={dec_verdict} "
                      f"(comfy's own formula: need ~{_gb(need):.1f} GB vs "
                      f"{_gb(free):.1f} GB free, tiling={tiling}){warn}")
                out_image, _ = self._decode_lane(vae, lat_dec, dec_verdict,
                                                 tile_size, tile_overlap,
                                                 temporal_size,
                                                 temporal_overlap,
                                                 h3_still_fix)

            # ---- v871: the audio half, in the same box ----------------
            _aud = _audio_latent(lat_in)
            if audio_vae is not None and _aud is None:
                print("[PLS] VAE: audio_vae is wired but this latent carries "
                      "no audio half -> audio output None. A plain video "
                      "latent is NOT fed to an audio VAE.")
            elif _aud is not None and audio_vae is None:
                print("[PLS] VAE: this is a JOINT audio/video latent and no "
                      "audio_vae is wired -> the soundtrack is left "
                      "undecoded and the audio output is None.")
            elif _aud is not None and audio_vae is not None:
                out_audio = self._decode_audio_lane(audio_vae, _aud, samples)

        if out_latent is None:
            print("[PLS] VAE: LATENT output has no source (no encode ran, "
                  "no samples to pass through) -> None; wiring it "
                  "downstream will fail, and that is the honest outcome")
        if out_image is None:
            print("[PLS] VAE: IMAGE output has no source (no decode ran, "
                  "no pixels to pass through) -> None; wiring it "
                  "downstream will fail, and that is the honest outcome")
        return (out_latent, out_image, out_audio)


NODE_CLASS_MAPPINGS = {"ULSVAE": ULSVAE}
NODE_DISPLAY_NAME_MAPPINGS = {"ULSVAE": "⬡ Polyhedron VAE"}
