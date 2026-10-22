# -*- coding: ascii -*-
"""Polyhedron MiniMax Audio Lock (v1178) -- a given sound held in the H3 AV latent, the picture made to it.

WHAT IT DOES: encodes `audio` with the H3 audio VAE and puts it into the SOUND stream of the run's AV latent,
from `start_frame` on, held by the stream's noise mask at `strength` (1 = kept exactly, 0 = generated anew;
between = the model may rework it that far). The picture stream stays free: H3 generates it to the held
sound -- a music video on a given track, a voice the mouth has to follow. The library is ph_av_mask (the
same masks as MiniMax Continue's hold; two locks on one latent meet by the minimum).

The idea: Easy Media's "lock audio" (GPL: read, not copied). Different from MiniMax Reference's audio
reference: a reference is heard as style / timbre / beat; a lock IS the soundtrack.

SHORT SOUND: `after` decides what follows it -- "free" (H3 continues the sound), "silence" (the waveform is
padded with silence BEFORE encoding and the silence is held too -- a zero latent would not be silence),
"loop" (the waveform repeats to the clip's end, then encoded). LONG SOUND: what lies past the clip's end is
cut and said.

`audio` OUT: the sound as it lies in the clip -- start_frame of silence, then the sound, cut to the clip's
length (and padded / looped as `after` says). Wire it into the Save to deliver the ORIGINAL sound instead of
the decoded one (at strength 1 they differ only by the VAE's round trip).

CORE: master runs the held audio rows as clean context to the model; the pinned Core blends them back after
every step (classic latent inpainting) -- both hold the sound, the info says which.
"""

import torch

from . import ph_av_mask as AM
from . import uls_latent_math as M

AFTERS = ["free", "silence", "loop"]


def place_waveform(wav, sr, start_s, clip_s, after):
    """The waveform as it lies in the clip: `start_s` of silence, the sound, then per `after` until `clip_s`
    (free: nothing more; silence: zeros; loop: the sound again). -> (waveform [B,C,n], cut seconds)."""
    n_clip = int(round(clip_s * sr))
    n0 = int(round(start_s * sr))
    body = wav
    room = max(0, n_clip - n0)
    # v1182: what the sound ITSELF loses past the clip's end, in every mode -- a long sound with "loop" was cut
    # without a word (the loop only fills a sound shorter than the room)
    cut = max(0, int(wav.shape[-1]) - room) / float(sr)
    if after == "loop" and 0 < body.shape[-1] < room:
        reps = -(-room // int(body.shape[-1]))
        body = torch.cat([body] * reps, dim=-1)
    body = body[..., :room]
    if after == "silence" and body.shape[-1] < room:
        body = torch.cat([body, torch.zeros(body.shape[:-1] + (room - body.shape[-1],), dtype=body.dtype)], dim=-1)
    lead = torch.zeros(body.shape[:-1] + (n0,), dtype=body.dtype)
    return torch.cat([lead, body], dim=-1), cut


class ULSMiniMaxAudioLock:
    """A given sound held in the H3 AV latent; the picture is generated to it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "The run's H3 AV latent (Polyhedron Empty Latent 'MiniMax H3 AV', or "
                                                 "MiniMax Continue's latent). Wire the output into the sampler."}),
                "audio_vae": ("VAE", {"tooltip": "The H3 audio VAE."}),
                "audio": ("AUDIO", {"tooltip": "The sound the clip is made to: a track, a voice, a sound bed."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip":
                    "How hard the sound is held. 1 = kept as it is (the picture follows it); 0.5 = the model may "
                    "rework it halfway; 0 = generated anew (no lock)."}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 9999, "tooltip":
                    "Where the sound begins in the clip, in frames (24 per second). Before it the sound is free. "
                    "A scene that continues the one before (22 frames of pinned head): 22 starts the sound at the "
                    "join."}),
                "after": (AFTERS, {"default": AFTERS[0], "tooltip":
                    "A sound shorter than the clip: free = H3 carries on after it; silence = silence is held after "
                    "it; loop = it repeats to the clip's end."}),
            },
        }

    RETURN_TYPES = ("LATENT", "AUDIO", "STRING")
    RETURN_NAMES = ("latent", "audio", "info")
    FUNCTION = "lock"
    CATEGORY = "Polyhedron/Cine"
    DESCRIPTION = ("Holds a given sound in the sound stream of an H3 AV latent (its noise mask), so the picture is "
                   "generated to it -- a music video, a voice to follow. strength 1 keeps it exactly; `audio` out is "
                   "the original sound laid into the clip, for the Save.")

    def lock(self, latent, audio_vae, audio, strength, start_frame, after):
        tag = "[PLS] MiniMax Audio Lock"
        v, a = AM.streams(latent, tag)
        frames = M.minimax_frames_from_latent_t(int(v.shape[2]))
        clip_s = frames / float(AM.FPS)
        start = int(start_frame)
        if start >= frames:
            raise ValueError(tag + ": start_frame %d is past the clip's %d frames" % (start, frames))
        wav = audio["waveform"]
        sr = int(audio["sample_rate"])
        sound_s = wav.shape[-1] / float(sr)
        placed, cut = place_waveform(wav[:1], sr, start / float(AM.FPS), clip_s, after)
        body = placed[..., int(round(start / float(AM.FPS) * sr)):]
        from .ph_minimax_ref import encode_ref_audio
        z, _zt = encode_ref_audio(audio_vae, {"waveform": body, "sample_rate": sr})
        a0 = AM.audio_steps(start)
        out, rep = AM.lock_audio(latent, z, start=a0, strength=float(strength), tag=tag)
        lines = ["MiniMax Audio Lock: clip %d frames (%.2f s), sound %.2f s from frame %d (%.2f s)"
                 % (frames, clip_s, sound_s, start, start / float(AM.FPS)),
                 "  held: audio latent frames %d..%d of %d (%.2f s) at strength %.2f -- %s"
                 % (rep["start"], rep["start"] + rep["steps"] - 1, rep["total"], rep["steps"] / float(AM.AUDIO_FPS),
                    rep["strength"], "kept exactly" if rep["strength"] >= 1.0 else
                    ("generated anew (no lock)" if rep["strength"] <= 0.0 else "the model may rework it"))]
        if cut > 0:
            lines.append("  the sound is %.2f s longer than the room in the clip -- the rest is cut" % cut)
        if after == "free" and start / float(AM.FPS) + sound_s < clip_s - 0.05:
            lines.append("  after the sound (%.2f s on) the clip's sound is free -- H3 carries on"
                         % (start / float(AM.FPS) + sound_s))
        elif after == "silence" and start / float(AM.FPS) + sound_s < clip_s - 0.05:
            lines.append("  after the sound: silence, held")
        elif after == "loop" and sound_s < clip_s - start / float(AM.FPS) - 0.05:
            lines.append("  the sound loops to the clip's end")
        if start:
            lines.append("  before frame %d the sound is free (a continued scene's head carries its own)" % start)
        lines.append("  " + AM.core_line())
        lines.append("  `audio` out: the original sound laid into the clip -- into the Save to deliver it as it is")
        info = "\n".join(lines)
        print("[PLS] " + info.replace("\n", "\n[PLS] "))
        return (out, {"waveform": placed, "sample_rate": sr}, info)


NODE_CLASS_MAPPINGS = {"ULSMiniMaxAudioLock": ULSMiniMaxAudioLock}
NODE_DISPLAY_NAME_MAPPINGS = {"ULSMiniMaxAudioLock": "\u2b21 Polyhedron MiniMax Audio Lock"}
