# -*- coding: ascii -*-
"""Polyhedron MiniMax Drift Control (v1179) -- carry the motion of the scene before, let its look change.

WHAT IT DOES: MiniMax Continue's `hold` copies the end of clip n into the start of this run's latent and
holds it (locked, then released). Drift Control turns that still hold into a LIVE one: at every sampler
step the held picture rows get the mask  w_k x sigma_next / sigma  -- w = 1 on the far side of the
prefix, tapering to 0 at the join over `taper_frames`. While the run samples, the far rows follow the
model's own prediction by that share and are pulled back to the copied clip by the rest: the NEW frames
see a prefix that may already lean toward the new look while its motion stays the clip's -- they are bound
to the movement, less to the old appearance. The row at the join is the copied clip at every step, and at
the last step the whole prefix comes back as the copied clip (it is the repeated head the film leaves out).
Use it when the next scene keeps the movement but changes the subject, the costume, the light -- Easy
Media's "character swap context". The sound keeps the hold's own mask (unchanged).

WIRING: Continue (`hold` = guide + hold in the latent) -> latent -> HERE and the model -> HERE; the model
output goes to the sampler, the latent output (unchanged, passed on) too. The prefix is read from the
latent's mask (the rows the hold copied in); `prefix_frames` can shorten it, never lengthen it (v1181).

CORE: master is told the live mask at every step (its own token-grid pooling), so the model's per-row
timesteps match the blend; the pinned Core runs the sampler side only (said). Refuses a model that already
carries a live mask. A latent with nothing held (the first scene) passes the model unpatched, said (v1181).

The idea: ethanfel/ComfyUI-MiniMaxH3-Contex-Loop "Drift-Control AV" via Easy Media (read, not copied).
The library is ph_av_mask (drift_weights, drift_ratio, DriftState, install_drift).
"""

from . import ph_av_mask as AM


class ULSMiniMaxDriftControl:
    """Live mask on the held prefix: the motion carries, the look may change."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The H3 model that samples this run (after LoRAs / engine). The "
                                               "output goes to the sampler instead."}),
                "latent": ("LATENT", {"tooltip": "MiniMax Continue's latent with `hold` on (the end of the scene "
                                                 "before copied into its start). Passed on unchanged."}),
                "taper_frames": ("INT", {"default": 16, "min": 4, "max": 64, "step": 4, "tooltip":
                    "Over how many frames before the join the drift fades out (rounded to whole tokens of 1 or 4 frames): the join "
                    "itself stays the scene before at every step, the frames further back may lean toward the new "
                    "look while the run samples."}),
                "prefix_frames": ("INT", {"default": 0, "min": 0, "max": 362, "tooltip":
                    "How much of the start drifts, in frames. 0 = all the latent holds (the rows MiniMax Continue's "
                    "hold copied in); a value can only shorten it -- a row that is not held is never touched."}),
            },
        }

    RETURN_TYPES = ("MODEL", "LATENT", "STRING")
    RETURN_NAMES = ("model", "latent", "info")
    FUNCTION = "patch"
    CATEGORY = "Polyhedron/Cine"
    DESCRIPTION = ("Drift control for a continued scene: the held start (MiniMax Continue `hold`) is masked anew at "
                   "every step -- while the run samples, its far frames may lean toward the new look, so the new "
                   "frames follow the motion more than the old appearance; the join stays the scene before. "
                   "Patches the model; wire its model output into the sampler.")

    def patch(self, model, latent, taper_frames, prefix_frames):
        tag = "[PLS] MiniMax Drift Control"
        v, _a = AM.streams(latent, tag)
        held = AM.held_prefix(latent)
        if held < 1:
            # v1181: one workflow for every scene -- the first scene (and any scene not carried) holds nothing; the
            # model passes unpatched, said (as MiniMax Continue and Second Pass Hold do)
            info = ("MiniMax Drift Control: nothing is held at the start of the latent (the first scene, or a scene "
                    "that is not carried) -- the model passes unpatched")
            print("[PLS] " + info)
            return (model, latent, info)
        prefix, src = held, "the latent's hold"
        if int(prefix_frames) > 0:
            want = AM.video_tokens(int(prefix_frames))
            prefix = min(want, held)   # v1181: the drift never reaches past the hold (a free row stays free)
            src = "prefix_frames %d -> %d token(s)%s" % (int(prefix_frames), want,
                                                       ", cut to the hold's %d" % held if want > held else "")
        if prefix >= int(v.shape[2]):
            raise ValueError(tag + ": a prefix of %d token(s) fills the whole clip (%d)" % (prefix, int(v.shape[2])))
        taper, taper_f = AM.tokens_back(prefix, taper_frames)
        taper = max(1, taper)
        m, st = AM.install_drift(model, prefix, taper)
        w = st.weights
        lines = ["MiniMax Drift Control: prefix %d token(s) = frames 0..%d (from %s)"
                 % (prefix, AM.frames_of_tokens(prefix) - 1, src),
                 "  weights far -> join: %s; at each step x sigma_next/sigma -- the join stays the scene before, "
                 "the prefix comes out as it (the head the film leaves out)"
                 % " ".join("%.2f" % x for x in w)]
        if held != prefix:
            lines.append("  note: the latent holds %d token(s), drift runs on its first %d" % (held, prefix))
        lines.append("  taper: the last %d token(s) before the join (%d frame(s))" % (taper, AM.frames_of_tokens(prefix)
                     - AM.frames_of_tokens(prefix - taper)))
        lines.append("  " + ("Core: the model is told the live mask at every step (per-row timesteps)" if st.per_row else
                             "Core: this Core blends the live mask in the sampler only -- the model does not see it "
                             "(Core master does)"))
        lines.append("  the sound keeps the hold's own mask; wire `model` into the sampler")
        info = "\n".join(lines)
        print("[PLS] " + info.replace("\n", "\n[PLS] "))
        return (m, latent, info)


NODE_CLASS_MAPPINGS = {"ULSMiniMaxDriftControl": ULSMiniMaxDriftControl}
NODE_DISPLAY_NAME_MAPPINGS = {"ULSMiniMaxDriftControl": "\u2b21 Polyhedron MiniMax Drift Control"}
