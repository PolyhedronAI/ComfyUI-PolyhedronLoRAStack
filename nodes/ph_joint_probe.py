"""
ph_joint_probe.py -- "does this model build ONE latent stream or TWO?"

WHY ITS OWN MODULE (v889): the answer is needed in two unrelated places now.
ph_power_upscale asks it to drop the refine stages for a joint model (v883);
ph_basics asks it so Load VAE stops refusing a 24ch video VAE against an H3
model that reports 32 (see below). Two copies of a three-witness probe would
drift, and the house rule is that a mirror is either avoided or driven by the
guard from both ends. ph_power_upscale keeps re-exporting both names, so every
existing caller and test_v880 stay untouched -- the same shape as the v576
ph_runclock re-export.

WHY ph_basics MUST NOT import ph_power_upscale directly: ph_basics is a base
module (the loaders); ph_power_upscale is a heavy leaf that imports uls_sampler
and uls_tile_math. A base importing a leaf is an inversion, and it would drag
the sampler into every Load VAE. This module imports nothing of ours.

WHY NOT uls_latent_math: that module is deliberately torch-free so it can be
unit-tested outside ComfyUI. Witness B here needs comfy.model_base. Putting it
there would break a documented property to save a file.
"""

def _joint_latent_parts(model):
    """How many latent STREAMS this model's format builds -- 1 for an ordinary
    image/video model, 2 for MiniMax H3 (video + audio).

    MEASURED, not name-matched. `LatentFormat.fix_empty_latent` returns its
    argument unchanged; `MiniMaxH3AV` overrides it to hand back a NestedTensor
    of (video, audio). We run it on a throwaway latent and count what comes
    back, so the next joint model is caught without anyone editing a list of
    names here.

    Returns 1 when the question cannot be answered (no format, probe raised).
    Failing OPEN is deliberate: a false refusal would block a legitimate
    upscale, which is worse than the crash this guards against.

    v882: THREE WITNESSES, because one was not enough in the field. The v880
    probe measured only `fix_empty_latent` -- correct against Core MASTER, but
    Frank's field Core (~v0.31) predates that override: MiniMaxH3AV there
    inherits the base method, the probe saw one stream, failed open, and the
    same ten-minute crash came back. A measurement is only valid for the
    build it was taken on. So the probe now asks three independent questions,
    any one of which settles it:

      A  run `fix_empty_latent` and count the streams that come back
         (master and future Cores);
      B  Core's own taxonomy: `model_type is ModelType.FLOW_AV`. This is not
         a name match on a model -- FLOW_AV is the CATEGORY "joint
         audio-video flow", the very property the tile refine cannot serve,
         and Core dispatches ModelSamplingAV off exactly this value
         (model_base.py, since v0.30);
      C  the separate audio clock: only `ModelSamplingAV` carries an
         `audio_shift` attribute (model_sampling.py, since v0.30).
    """
    # -- witness A: run the format and count what comes back --
    try:
        lf = model.model.latent_format
        fix = lf.fix_empty_latent
        ch = int(getattr(lf, "latent_channels", 4) or 4)
        import torch as _t
        out = fix(_t.zeros((1, ch, 1, 8, 8)))
        parts = getattr(out, "tensors", None)
        if isinstance(parts, (tuple, list)) and len(parts) > 1:
            return len(parts)
    except Exception:
        pass
    # -- witness B: Core's own category for joint audio-video flow --
    try:
        import comfy.model_base as _mb
        flow_av = getattr(_mb.ModelType, "FLOW_AV", None)
        if flow_av is not None and \
                getattr(model.model, "model_type", None) is flow_av:
            return 2
    except Exception:
        pass
    # -- witness C: the separate audio clock on the sampling object --
    try:
        if hasattr(model.model.model_sampling, "audio_shift"):
            return 2
    except Exception:
        pass
    return 1


def _joint_streams(model, model_low):
    """v880/v883: DETECT a joint audio/video model before any work is done.

    v880 refused outright; v883 turns the refusal into a NAMED DOWNGRADE in
    upscale(): the refine stages are dropped (they cannot run), the pixel
    path still delivers a file. Returns ("model"|"model_low", n_streams) or
    None. The three-witness measurement lives in _joint_latent_parts.

    THE WOUND, from Frank's field log (26.08.): MiniMax H3 wired into 'model'
    ran the full pixel pass -- 4:46 sampling upstream, 275s of ESRGAN -- and
    then died inside CORE with `IndexError: list index out of range` at
    comfy/ldm/minimax/model.py:493, `audio_src = x[1]`. That line is the FIRST
    statement of the H3 forward: the model reads the audio half of a joint
    latent unconditionally. A tile refine hands it a plain image latent, which
    has no [1].

    It is not a setting anyone can turn off: the model would also need the
    `minimax_payload` token geometry the Reference node builds, and it would
    denoise an audio stream that does not exist in a tile. So this is a hard
    NO, said early and by name -- ten minutes of wall clock is exactly what a
    late refusal costs (the house rule: refusals are written LOUD and NAMED).
    """
    for tag, m in (("model", model), ("model_low", model_low)):
        if m is None:
            continue
        n = _joint_latent_parts(m)
        if n > 1:
            return (tag, n)
    return None
