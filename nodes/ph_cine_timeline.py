# -*- coding: ascii -*-
"""Polyhedron Cine Timeline (C6, v1023) -- one clip laid out on a time axis.

The model and the compiler live in cine_timeline.py (pure, guarded); this
file is the ComfyUI side: the node, the anchor images, and the preview
route the editor calls (POST /pls/cine/timeline) -- so the prompt shown in
the node is computed by the SAME code the run uses, never a second copy in
JavaScript.

Wiring:
    Reference Board refs ----> timeline refs      (anchors by @tag)
    images -------------------> image_1..9        (anchors without a board; v1051: 9)
    the stack's clips --------> @clip:K:last|first|S (v1051: a frame of a
                                 rendered clip, resolved here at run time)
    prompt -------------------> MiniMax Keyframes prompt (or CLIP Text Encode)
    first_frame / last_frame -> MiniMax Keyframes first_frame / last_frame
    anchors ------------------> MiniMax Continue anchors (Core master)
    seconds ------------------> Empty Latent duration_seconds
    Cine Storyboard storyboard -> storyboard  (or any LLM node's answer)
    clip_prefix --------------> Save Video filename_prefix  (OPTIONAL -- v1152:
                                 %date:yyyy-MM-dd%/PH_%date:hhmmss%_<project>_whole
                                 | _scene_k, the day folder; unwired, the run's
                                 saved clips join the stack by path, see v1034)

v1028 -- the cutting room (Cine S6 / E1): the timeline may hold scenes; the
run renders the whole timeline or ONE scene (the user's active choice in the
node), the scene clips join the project's stack (v1152: the list in
input/cine_media/<project>/stack/, the clips where the Saves write them --
output/ holds only day folders); v1036 the stack and the join became the video track (POST /pls/cine/edit,
POST /pls/cine/export). v1182: the v1028 routes GET /pls/cine/scenes and POST /pls/cine/assemble were no longer
called by the editor (and the join ran ffmpeg on the event loop) -- removed; the Clip Assemble node keeps the join.
"""

import hashlib
import math
import json
import collections
import os
import re

from . import cine_assemble as A
from . import cine_edit as E
from . import cine_storyboard as SB
from . import cine_timeline as T
from . import h3_prompt as HP


def _board_images(refs):
    out = {}
    for it in (refs or {}).get("images", []) or []:
        if it.get("tag"):
            out["@" + it["tag"]] = it.get("image")
    return out


def _sources(refs, images):
    s = dict(_board_images(refs))
    for k, v in images.items():
        if v is not None:
            s[k] = v
    return s


CTE_CLASS = "ULSCLIPTextEncode"
CTE_FILL = "fill {pos_N}"


def form_fill(prompt, unique_id):
    """v1150: the segments of the CLIP Text Encode that fills this timeline's form -- the one whose pos_external
    is our prompt output and whose external_mode is fill {pos_N} -- read from the running graph. -> list or None
    (no such CTE, or a segment wired instead of typed)."""
    if not isinstance(prompt, dict) or unique_id is None:
        return None
    me = str(unique_id)
    for nd in prompt.values():
        if not isinstance(nd, dict) or nd.get("class_type") != CTE_CLASS:
            continue
        ins = nd.get("inputs") or {}
        src = ins.get("pos_external")
        if not (isinstance(src, list) and len(src) == 2 and str(src[0]) == me and int(src[1]) == 0):
            continue
        if ins.get("external_mode") != CTE_FILL:
            continue                    # v1154 (audit): another CTE on our output may be the filling one
        try:
            n = max(1, min(int(ins.get("segments", 1)), 6))
        except (TypeError, ValueError):
            return None
        segs = [ins.get("pos_%d" % i, "") for i in range(1, n + 1)]
        if not all(isinstance(x, str) for x in segs):
            return None
        return segs
    return None


def cut_seconds(target, frames):
    """v1211: the times (s) of the cuts INSIDE the clip this run renders --
    every shot after the first, as the compiled target places it (a motion
    carry has already moved them back). These are the times the prompt ASKS
    for; the model paints a cut near them, not on the frame. The Interpolate
    looks for the real one around each time."""
    dur = int(frames) / float(T.FPS)
    out = []
    for s in target.get("shots") or []:
        try:
            at = float(s.get("at", 0))
        except (TypeError, ValueError):
            continue
        if 0.0 < at < dur:
            out.append(round(at, 4))
    return sorted(set(out))


def run(model, sources, prev=None, fill=None):
    """Validate + compile + check. -> (result dict, errors). Pure but for
    the image lookup; the route and the node both call this. v1028: compiles
    what the run renders -- the whole timeline or the active scene.
    v1049: `prev` = the newest clip of the scene before (None = none found):
    a scene that carries begins on its last frame."""
    errs = T.validate(model, set(sources) if sources is not None else None)
    if errs:
        return None, errs
    target, k, n = T.resolve(model)
    # v1219: two anchors on ONE frame of the place this run renders (a 'both scenes' anchor on the boundary and the
    # scene's own last anchor) -- one of them would go silently
    tf, seen = T.frames_for(target["seconds"]), {}
    for a in target.get("anchors") or []:
        f = T.frame_of(a.get("at", 0), tf)
        if f in seen:
            return None, ["anchors %s and %s land on the same frame %d of %s -- move one of them"
                          % (seen[f], a.get("src"), f, "scene %d" % k if k else "the clip")]
        seen[f] = a.get("src")
    want = T.carry_of(model, k)
    carry_lines, carried = [], False
    motion = model.get("carry_mode", "pin") == "motion"
    if want and prev:
        carried = True
        if motion:
            target, notes = T.with_motion_head(target, head=T.motion_head(model))   # v1185: the timeline's head length
            carry_lines += notes
        elif model.get("carry_mode", "pin") == "keyframe":
            target, added = T.with_carry_anchor(target)
            if not added:
                carried = False
                carry_lines.append("carry: scene %d has its own first-frame anchor -- that one wins" % k)
    r = T.compile_prompt(target)
    r["cuts"] = cut_seconds(target, r["frames"])      # v1211: the cuts this clip is asked for
    r["scene"], r["scenes"], r["clip_prefix"] = k, n, T.clip_prefix(model, k)
    # v1083: in pin mode too, an own first-frame anchor wins (the info said
    # "carry", the output was the own anchor, and the frame was decoded for nothing)
    if carried and r["first"] is not None and model.get("carry_mode", "pin") == "pin":
        carried = False
        carry_lines.append("carry: scene %d has its own first-frame anchor -- that one wins" % k)
    if want and prev and not carried and r["first"] is not None:
        # v1219 (one policy for the join): an own first-frame anchor wins over the carry -- then the scene before
        # must END on that picture, or the join jumps. Said, with the way to tie both ends to it.
        src = r["first"].get("src")
        both = any(a.get("both") and a.get("src") == src for a in model.get("anchors") or [])
        carry_lines.append("carry: scene %d ends on the same picture (%s marked 'both scenes') -- the join is the anchor"
                           % (k - 1, src) if both else
                           "WARN: carry: scene %d starts on %s, but scene %d ends free -- the join jumps unless %s is "
                           "also scene %d's last frame: mark the anchor 'both scenes' (it must sit on the boundary)"
                           % (k, src, k - 1, src, k - 1))
    r["carry"] = prev if carried else None
    ce, ce_note = T.carry_end(model, k - 1, prev) if carried else (None, "")
    if ce_note:
        carry_lines.append(ce_note)
    r["carry_cut"] = ce is not None
    r["carry_index"] = (ce if ce is not None else T.film_last_index(model, k - 1)) if carried else None
    r["motion"] = T.motion_head(model) if (carried and motion) else 0   # v1185: 5 / 22 / 39
    if carried and motion:
        hd = r["motion"]
        # v1185: the grid's real cost -- the scene grows by the head, then up to the next 17k+5
        old_g, new_g = T.frames_for(T.scene_model(model, k)["seconds"]), r.get("frames") or T.frames_for(target["seconds"])
        carry_lines.append("carry: scene %d CONTINUES scene %d -- its last %d film frames (to frame %d of %s) and their "
                           "sound -> tail / tail_audio -> MiniMax Continue K2 (context_frames >= %d, audio_context %d; "
                           "Core master). The scene is rendered %d frames longer (head %d, the grid adds %d: %d -> %d "
                           "frames): they are its pinned head, every shot after the first moves back by %.2f s; the film "
                           "leaves the repeated frames out"
                           % (k, k - 1, hd, r["carry_index"] + 1, os.path.basename(prev), hd, min(T.MOTION_AUDIO, hd),
                              new_g - old_g, hd, new_g - old_g - hd, old_g, new_g, hd / float(T.FPS)))
        if hd > T.MOTION_FRAMES:
            carry_lines.append("carry: head %d -- MiniMax Continue caps the window at ITS context_frames (22 in older "
                               "workflows): set it to %d or more, else the clip is held for 22 and %d frames of the "
                               "head render free" % (hd, hd, hd - T.MOTION_FRAMES))
    elif carried:
        carry_lines.append("carry: scene %d begins on the frame the film shows LAST of scene %d (frame %d of %s%s) -> first_frame%s"
                           % (k, k - 1, r["carry_index"] + 1, os.path.basename(prev),
                              ", where your cut ends it" if r["carry_cut"] else "",
                              " (the first-frame anchor: alignment line in the prompt, <Picture 1> in MiniMax Keyframes)"
                              if model.get("carry_mode") == "keyframe" else
                              " (pinned only -- MiniMax Reference's first_frame; the prompt is unchanged)"))
    elif want and not prev:
        carry_lines.append("carry: scene %d should begin on the last frame of scene %d, but scene %d has no clip "
                           "in the stack of %s yet (not rendered, or its run's register has not arrived) -- this run starts free"
                           % (k, k - 1, k - 1, T.project_name(model)))
    if k:
        t0, t1 = T.scene_span(model, k)
        r["lines"].insert(0, "render: scene %d of %d -- %s to %s of the timeline (%.2f s); the run's saved clip joins "
                             "the stack of scene %d where your Save writes it (clip_prefix %s only if wired)"
                          % (k, n, T.time_label(t0), T.time_label(t1), t1 - t0, k, r["clip_prefix"]))
        if model.get("render") == "all":
            r["lines"].insert(1, "  all scenes, one run each: the editor queues scene 1..%d as %d runs "
                                 "(a run queued from elsewhere renders the active scene only)" % (n, n))
    elif n > 1:
        r["lines"].insert(0, "render: the whole timeline in one run, %d scenes inside it (hard cuts); the run's saved "
                             "clip joins the stack where your Save writes it (clip_prefix %s only if wired)"
                          % (n, r["clip_prefix"]))
    bf = T.boundary_frames(model)
    for a in model.get("anchors") or []:
        try:
            off = a.get("both") and int(round(float(a.get("at", 0)) * T.FPS)) not in bf
        except (TypeError, ValueError):
            off = False
        if off:
            carry_lines.append("note: anchor %s is marked 'both scenes' but sits at %s, not on a scene boundary -- it "
                               "serves one scene" % (a.get("src"), T.time_label(float(a.get("at", 0)))))
    # v1178: the handoff rules of a continued / continuing scene -- checked, said, never rewritten
    outgoing = bool(motion and k and k < n and T.carry_of(model, k + 1))
    if (carried and motion) or outgoing:
        carry_lines += T.context_notes(target, carried and motion, outgoing)
    r["lines"][0:0] = carry_lines
    n_pics = 0 if target.get("target", "fl2va") == "ref2va" else \
        (1 if r["first"] is not None else 0) + (1 if r["last"] is not None else 0)
    # v1150: a form -- the CLIP Text Encode fills the places; the check and the preview read it filled
    checked = r["prompt"]
    pl = T.places(r["prompt"])
    r["places"] = pl
    if pl and fill is not None:
        checked = T.fill_text(r["prompt"], fill)
        r["filled"] = checked
        r["lines"].append("  form: %s filled from the CLIP Text Encode (%d segment(s)) -- the check reads the whole "
                          "prompt; the encoder fills them the same way at the run" % (", ".join("{%s}" % x for x in pl), len(fill)))
    elif pl:
        r["lines"].append("  form: %s -- the CLIP Text Encode fills them (external_mode fill {pos_N}); no such CTE "
                          "is wired to the prompt output, so the check reads the timeline's part only"
                          % ", ".join("{%s}" % x for x in pl))
    # v1084: the check reads the prompt as the MiniMax nodes encode it -- the
    # // rubrics stripped (they strip them too) -- so the head warns no more
    stripped, _n_rub = HP.strip_rubrics(checked)
    report = HP.check(stripped, {"Picture": n_pics, "Video": 0, "Audio": 0}, r["seconds"])
    # @tags in shot text are resolved DOWNSTREAM (Keyframes / Reference put
    # the live labels in) -- here they are expected, not an error
    tags = [msg for lvl, msg in report if msg.startswith("unresolved tag")]
    r["check"] = ["%s: %s" % (lvl, msg) for lvl, msg in report if not msg.startswith("unresolved tag")]
    if tags:
        board = [m for m in tags if not (m.endswith("@first_frame reaches the model") or m.endswith("@last_frame reaches the model"))]
        r["lines"].append("  %d @tag(s) in the text -- MiniMax Reference resolves board tags and its slots, "
                          "MiniMax Keyframes only @first_frame / @last_frame%s"
                          % (len(tags), (" (%d other tag(s) here: an error at Keyframes)" % len(board)) if board else ""))
    return r, []


def adopt(model, storyboard, sources):
    """-> (model, adopted, notes). A storyboard whose fingerprint differs
    from the one the timeline holds replaces it (length and all); the same
    one leaves the timeline -- and the user's edits -- alone. Refuses an
    unreadable storyboard by name rather than running the old timeline."""
    text = "" if storyboard is None else str(storyboard)
    if not text.strip():
        return model, False, []
    fp = SB.digest(text)
    if model.get("from") == fp:
        return model, False, []
    try:
        m, notes = SB.parse(text, sources=set(sources) if sources is not None else None,
                           fallback_seconds=model.get("seconds"))
    except ValueError as e:
        raise ValueError("[PLS] Cine Timeline: the storyboard does not work:\n  %s"
                         % str(e).replace("; ", "\n  "))
    m["from"] = fp
    # v1084: a storyboard brings shots, anchors, length, style and sound --
    # the project's fields are the timeline's and stay (measured: the take-
    # over reset the project to "cine", dropped the LoRA triggers, the target,
    # the carry mode, the render choice and the transitions)
    kept = []
    for key in ADOPT_KEEPS:
        if key in model and model[key] != T.DEFAULT.get(key):
            m[key] = model[key]
            kept.append(key)
    if model.get("transitions"):
        n_new, n_old = len(T.scenes(m)), len(T.scenes(model))
        if n_new == n_old:
            m["transitions"] = model["transitions"]
            kept.append("transitions")
        else:
            notes.append("the transitions of the old timeline were dropped (%d scenes before, %d now)" % (n_old, n_new))
    if kept:
        notes.append("kept from the timeline: " + ", ".join(kept))
    return m, True, notes


ADOPT_KEEPS = ("project", "triggers", "target", "carry_mode", "head_frames", "render", "active")


class ULSCineTimeline:
    """Shots, anchors and sound of one clip on a time axis -> the official
    H3 prompt, the keyframes, the guides and the length."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("STRING", {"default": json.dumps(T.DEFAULT), "multiline": True,
                    "tooltip": "The timeline's state (JSON), edited by the timeline "
                               "above -- not by hand."}),
            },
            "optional": {
                "refs": ("PLS_REFS", {"tooltip": "Reference Board refs -- anchors "
                    "can use its images by @tag."}),
                "image_1": ("IMAGE", {"tooltip": "Anchor source 'image_1': an image without a board "
                    "(a Media Loader, a frame of an earlier clip, any IMAGE). Up to 9 -- "
                    "the pins grow with what is wired."}),
                "image_2": ("IMAGE", {"tooltip": "Anchor source 'image_2'."}),
                "image_3": ("IMAGE", {"tooltip": "Anchor source 'image_3'."}),
                "image_4": ("IMAGE", {"tooltip": "Anchor source 'image_4'."}),
                "image_5": ("IMAGE", {"tooltip": "Anchor source 'image_5'."}),
                "image_6": ("IMAGE", {"tooltip": "Anchor source 'image_6'."}),
                "image_7": ("IMAGE", {"tooltip": "Anchor source 'image_7'."}),
                "image_8": ("IMAGE", {"tooltip": "Anchor source 'image_8'."}),
                "image_9": ("IMAGE", {"tooltip": "Anchor source 'image_9'."}),
                "storyboard": ("STRING", {"forceInput": True, "tooltip": "From Cine "
                    "Storyboard, or the answer of any LLM node asked with its "
                    "`system` + `request`. A NEW storyboard replaces the timeline "
                    "once; your edits after that stay until the next new one."}),
            },
            # v1150: the running graph -- to find the CLIP Text Encode that fills this timeline's form, so the
            # info's check reads the whole prompt (hidden: no widget, both baselines stand)
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    # v1211: `cuts` APPENDED (links store their origin as a slot index)
    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE", "PLS_ANCHORS", "FLOAT", "INT", "STRING", "STRING", "INT",
                    "IMAGE", "AUDIO", "PLS_CUTS")
    RETURN_NAMES = ("prompt", "first_frame", "last_frame", "anchors", "seconds", "frames", "info",
                    "clip_prefix", "scene", "tail", "tail_audio", "cuts")
    FUNCTION = "build"
    CATEGORY = "Polyhedron/Cine"
    DESCRIPTION = ("One clip on a time axis: shots with their cut times, anchor "
                   "images and the sound, compiled into the official MiniMax H3 "
                   "prompt ([Shot N] At MM:SS.mmm, the alignment line for first "
                   "and last frame). An anchor on the first or last frame goes "
                   "to MiniMax Keyframes; anchors in between go to MiniMax "
                   "Continue as guides (Core master). The length is snapped to "
                   "the 17k+5 frame grid; `seconds` feeds Empty Latent. A cut "
                   "inside the clip can carry an H3 transition (blur to white, "
                   "fade through black, cross-dissolve, or your own sentences): "
                   "the node writes the formula and a sentence on each side of "
                   "the cut, and H3 paints the blend. Scenes: "
                   "mark shots as scene starts, then choose in the node "
                   "whether a run renders the whole timeline, one scene, or all "
                   "scenes one run each (one Run is then queued as one run per "
                   "scene). Every clip a Save node of the run writes joins the "
                   "project's stack BY PATH, wherever it lies under output/ -- "
                   "no copy, no second folder. The stack list and the anchor "
                   "frames live in input/cine_media/<project>/stack/; an export "
                   "goes to the day folder (output/<yyyy-MM-dd>/PH_<hhmmss>_"
                   "<project>_film). Wiring `clip_prefix` to a Save's "
                   "filename_prefix is optional and names that Save's files "
                   "<day>/PH_<hhmmss>_<project>_whole | _scene_k. The clip lane "
                   "under the shots shows the rendered clips on their places "
                   "(takes, transitions on the seams, fine cut by dragging); "
                   "Play previews the film in the node, export writes it. "
                   "A scene that carries in 'motion context' mode hands out the last 22 film frames of the "
                   "scene before and their sound (tail / tail_audio -> MiniMax Continue K2): the clip continues the "
                   "motion and the sound, and the film leaves its repeated first frames out.")

    @classmethod
    def IS_CHANGED(cls, timeline, **kwargs):
        """v1049: a carrying scene depends on the scene before's NEWEST clip --
        a new take there must re-run this node (its timeline JSON is the same).
        v1150: the info's check reads the CLIP Text Encode that fills the form -- its segments are part of the
        answer (Core's cache key covers only this node's own inputs)."""
        try:
            form = form_fill(kwargs.get("prompt"), kwargs.get("unique_id"))
        except Exception:
            form = None
        tail = "" if form is None else "|form:" + hashlib.md5(json.dumps(form).encode("utf-8")).hexdigest()
        r = cls._changed(timeline)
        return r if isinstance(r, float) else r + tail

    @classmethod
    def _changed(cls, timeline):
        try:
            model = T.load(timeline)
            if T.validate(model, None):
                return ""               # v1154 (audit): the run refuses it by name -- nothing of the stack to follow
            _t, k, _n = T.resolve(model)
            # v1051: @clip anchors follow their clip too
            clips = "|".join("%s=%s" % (src, ("%s@%s" % (p, os.path.getmtime(p))) if p else "none")
                             for src, p in sorted(clip_anchor_paths(model).items()))
            if not T.carry_of(model, k):
                return clips
            p = carry_source(model, k)
            return ("%s|%s" % (p, os.path.getmtime(p)) if p else "none") + (("|" + clips) if clips else "")
        except Exception as e:
            # v1084: a broken stack list or an unreadable clip is said, not swallowed
            print("[PLS] Cine Timeline: IS_CHANGED could not read the stack (%s) -- the node re-runs" % e)
            return float("nan")

    def build(self, timeline, refs=None, storyboard=None, prompt=None, unique_id=None, **images):
        tag = "[PLS] Cine Timeline"
        try:
            model = T.load(timeline)
        except ValueError as e:
            raise ValueError("%s: %s" % (tag, e))
        sources = _sources(refs, {s: images.get(s) for s in T.IMAGE_SLOTS})
        model, adopted, sb_notes = adopt(model, storyboard, sources)
        # v1051: frames of rendered clips (@clip:K:...) -- decoded here, kept
        # as PNG in input/cine_media/<project>/stack/frames/ so they can be reused anywhere
        clip_lines = []
        try:
            _t, k0, _n = T.resolve(model) if not T.validate(model, None) else (None, 0, 0)
        except Exception:
            k0 = 0
        # v1088 (audit C8): only the @clip anchors THIS run uses are decoded (the whole
        # timeline, or the active scene) -- the others keep their name for the check
        try:
            used = set(str(a.get("src", "")) for a in (T.place_model(model, k0).get("anchors") or []))
        except Exception:
            used = None
        for src, path in sorted(clip_anchor_paths(model).items()):
            if path is None:
                continue                        # validate() names it
            if used is not None and src not in used:
                sources[src] = None             # known for the check, not decoded, not saved
                continue
            k, which = T.clip_src(src)
            # v1083: "last" is the frame the FILM shows last of that place, a
            # time is the frame the editor's labels mean (frame index, not a seek)
            idx = T.film_last_index(model, k) if which == "last" else (0 if which == "first" else T.asked_frames(which))
            if which == "last" and k:
                ce, _n = T.carry_end(model, k, path)        # v1219: where the cut ends scene k, on this take
                idx = ce if ce is not None else idx
            sidx, sfps, sn = source_index(path, idx)        # v1168: the clip's own frame for that film frame
            img = frame_index(path, sidx)
            sources[src] = img
            png = save_frame(model, src, img)
            clip_lines.append("clip frame %s: %s (film frame %d = frame %d of %s%s) -> %s" % (
                src, which if isinstance(which, str) else "%.2f s" % which, idx + 1, sidx + 1,
                os.path.basename(path), " at %g fps" % sfps if abs(sfps - E.FPS) > 0.01 else "", png or "(not saved)"))
        prev = carry_source(model, k0) if T.carry_of(model, k0) else None
        if T.carry_of(model, k0) and prev is None:
            # v1220 (audit #13): the browser registers a run's clip; a tab in the background or closed sent nothing
            # and scene k started free. A Save wired to clip_prefix names its file <project>_scene_<k-1> in the day
            # folder -- found there and listed as the browser would have
            found = day_clip(model, k0 - 1)
            if found:
                try:
                    A.register(project_folder(model.get("project", "")), k0 - 1, found)
                    prev = carry_source(model, k0) or found
                except Exception as e:
                    prev = found
                    print("[PLS] Cine Timeline: %s could not be listed (%s) -- carrying from it anyway" % (found, e))
                print("[PLS] Cine Timeline: scene %d's clip found in the day folder (no register from the browser): %s"
                      % (k0 - 1, os.path.basename(found)))
        if T.carry_of(model, k0) and prev is None:
            # v1083: in "all" mode the run before registers its clip from the
            # browser (the Save's "executed" event) while this run is already
            # queued -- give that register a moment before starting free
            import time
            for _i in range(REGISTER_WAIT_TRIES):
                time.sleep(REGISTER_WAIT_STEP)
                prev = carry_source(model, k0)
                if prev:
                    print("[PLS] Cine Timeline: scene %d's clip registered after %.1f s -- carrying"
                          % (k0 - 1, REGISTER_WAIT_STEP * (_i + 1)))
                    break
            else:
                print("[PLS] Cine Timeline: scene %d has no clip in the stack of %s yet -- scene %d starts free"
                      % (k0 - 1, T.project_name(model), k0))
        r, errs = run(model, sources, prev, form_fill(prompt, unique_id))
        if errs:
            raise ValueError(tag + ":\n  " + "\n  ".join(errs))
        carry_img = None
        tail = tail_audio = None
        if r.get("carry") and r.get("motion"):
            # v1176: the motion context -- the last frames the FILM shows of scene k-1 (behind its own repeated head
            # when it continued scene k-2) and their sound
            k = int(r["scene"])
            paths = [carry_source(model, j + 1) for j in range(1, k)]
            idx = clip_last_shown(model, k - 1, paths)
            idx = r["carry_index"] if idx is None else idx
            tail = tail_frames(r["carry"], idx, r["motion"])
            tail_audio = tail_sound(r["carry"], idx, r["motion"])
            r["lines"].insert(1, "  tail: %d frame(s) of %s up to its film frame %d%s -- %s" % (
                int(tail.shape[0]), os.path.basename(r["carry"]), idx + 1,
                " (behind its repeated head)" if idx != r["carry_index"] else "",
                "with its sound" if tail_audio is not None else "the clip has no sound"))
            if int(tail.shape[0]) < int(r["motion"]):
                # v1182: the scene was compiled r["motion"] frames longer with every later shot moved back by them;
                # a shorter tail snaps Continue's guide down (5) and the film leaves out fewer -- said loudly
                r["lines"].insert(2, "  WARN: the scene before has only %d frame(s) to carry (%d asked) -- MiniMax "
                                     "Continue snaps its guide down, and the shots after the first sit up to %d frame(s) "
                                     "off their places; render the scene before longer or carry 'first-frame anchor'"
                                  % (int(tail.shape[0]), int(r["motion"]), int(r["motion"]) - int(tail.shape[0])))
        elif r.get("carry"):
            sidx, sfps, sn = source_index(r["carry"], r["carry_index"])
            if abs(sfps - E.FPS) > 0.01:
                r["lines"].insert(1, "carry: %s runs at %g fps (%s frames) -- film frame %d is its frame %d"
                                  % (os.path.basename(r["carry"]), sfps, sn if sn else "?", r["carry_index"] + 1, sidx + 1))
            carry_img = frame_index(r["carry"], sidx)
            sources = dict(sources)
            sources[T.CARRY_SRC] = carry_img
        if r["first"] is not None:
            first = sources[r["first"]["src"]][:1]
        else:
            first = carry_img
        last = sources[r["last"]["src"]][:1] if r["last"] is not None else None
        anchors = [{"src": src, "frame": f, "image": sources[src][:1]} for src, f in r["guides"]]
        # v1216: an anchor whose source is the Media Loader's placeholder (nothing loaded) stops the run by name --
        # before a single step is sampled on a checkerboard (Frank's run PH_093901, 23.10.)
        try:
            from .uls_placeholder import refuse
        except ImportError:  # flat import in the guards
            from uls_placeholder import refuse
        for which, a in (("first frame", r["first"]), ("last frame", r["last"])):
            if a is not None:
                refuse("Cine Timeline", "the %s %s" % (which, a["src"]), sources[a["src"]])
        for a in anchors:
            refuse("Cine Timeline", "the anchor %s" % a["src"], a["image"])
        head = []
        if adopted:
            head = ["storyboard %s taken over -- edit freely, a new storyboard replaces it" % model["from"]]
            head += ["  note: " + n for n in sb_notes]
        elif model.get("from") and storyboard is not None and str(storyboard).strip():
            head = ["storyboard %s already taken over -- the edited timeline runs" % model["from"]]
        info = "\n".join(head + clip_lines + r["lines"] + guide_lines(r)
                          + (["check:"] + ["  " + c for c in r["check"]] if r["check"] else []))
        print("[PLS] " + info.replace("\n", "\n[PLS] "))
        ui = {"prompt": r["prompt"], "info": info, "scene": r["scene"], "scenes": r["scenes"],
              "clip_prefix": r["clip_prefix"], "project": T.project_name(model)}
        if adopted:
            ui["adopt"] = json.dumps(model, ensure_ascii=True)
        return {"ui": {"pls_timeline": [ui]},
                "result": (r["prompt"], first, last, anchors, float(r["seconds"]),
                           int(r["frames"]), info, r["clip_prefix"], int(r["scene"]), tail, tail_audio,
                           {"seconds": list(r.get("cuts") or []), "fps": float(T.FPS), "frames": int(r["frames"])})}


async def handle_timeline_preview(request):
    """POST {timeline, sources: [..]} -> {prompt, lines, check, errors}."""
    from aiohttp import web
    try:
        body = await request.json()
        model = T.load(body.get("timeline", ""))
        srcs = body.get("sources")
        prev = None
        if not T.validate(model, None):
            _t, k0, _n = T.resolve(model)
            prev = carry_source(model, k0) if T.carry_of(model, k0) else None
        fill = body.get("fill")
        fill = [str(x or "") for x in fill] if isinstance(fill, list) else None
        r, errs = run(model, preview_sources(model, srcs), prev, fill)
    except Exception as e:
        return web.json_response({"errors": ["%s" % e]})
    if errs:
        return web.json_response({"errors": errs})
    return web.json_response({"prompt": r.get("filled", r["prompt"]), "form": r["prompt"], "places": r["places"], "lines": r["lines"] + guide_lines(r), "check": r["check"],
                              "frames": r["frames"], "seconds": r["seconds"], "errors": [],
                              "scene": r["scene"], "scenes": r["scenes"], "clip_prefix": r["clip_prefix"]})


async def handle_to_form(request):
    """v1150 ('-> CTE'): POST {timeline} -> {timeline, segments} -- Shot 1's fixed parts as CTE segments, the
    timeline with their places (T.split_to_form); the editor writes both. Nothing is changed on the server."""
    from aiohttp import web
    try:
        body = await request.json()
        model = T.load(body.get("timeline", ""))
        new, segs = T.split_to_form(model)
    except Exception as e:
        return web.json_response({"error": "%s" % e})
    return web.json_response({"timeline": json.dumps(new, ensure_ascii=True), "segments": segs})


# --------------------------------------------------------------- v1049 carry
def carry_source(model, k):
    """The newest clip of scene k-1 in the project's stack (the list and the
    stack list), or None."""
    if not k or int(k) < 2:
        return None
    try:
        takes = A.scene_takes(project_folder(model.get("project", "")), int(k) - 1)
    except Exception:
        return None
    lst = takes.get(int(k) - 1) or []
    return lst[0] if lst else None


def last_frame(path):
    """The clip's LAST picture as an IMAGE [1, H, W, 3] (ffmpeg seeks to the
    end and keeps overwriting one PNG -- the last decoded frame stays)."""
    import subprocess
    import tempfile
    import numpy as np
    import torch
    from PIL import Image
    exe = A.find_ffmpeg()
    if not exe:
        raise ValueError("[PLS] Cine Timeline: carry needs ffmpeg (PATH or imageio-ffmpeg) to read the last frame")
    d = tempfile.mkdtemp(prefix="pls_carry_")
    out = os.path.join(d, "last.png")
    try:
        for seek in ("-1.0", "-3.0", None):
            cmd = [exe, "-v", "error", "-y"] + (["-sseof", seek] if seek else []) + ["-i", path, "-update", "1", "-an", out]
            subprocess.run(cmd, capture_output=True, timeout=120)
            if os.path.isfile(out) and os.path.getsize(out) > 0:
                break
        if not os.path.isfile(out):
            raise ValueError("[PLS] Cine Timeline: could not read the last frame of %s" % os.path.basename(path))
        a = np.asarray(Image.open(out).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(a)[None]
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def source_index(path, film_idx):
    """v1168 (Frank 18.10., the neon film: every scene began on the MIDDLE of the scene before): the editor counts
    film frames at 24 fps, but the clip it reads may run at another rate -- the interpolated final (48 fps, 281
    frames) is the newest take of a scene, and film frame 140 is its middle, not its end. The film frame -> the
    clip's own frame number: film_idx * clip_fps / 24, never past its last frame. -> (index, clip fps, clip frames)."""
    fps, n = float(E.FPS), None
    try:
        ci = clip_info(path)
        fps = float(ci.get("fps") or E.FPS) or float(E.FPS)
        n = int(ci.get("src_frames") or ci.get("frames") or 0) or None
    except Exception:
        pass
    idx = int(math.floor(max(0, int(film_idx)) * fps / float(E.FPS) + 0.5))
    if n:
        idx = min(idx, n - 1)
    return idx, fps, n


def frame_index(path, idx):
    """v1083: frame number idx (0-based) of a clip as an IMAGE [1, H, W, 3] --
    selected by its NUMBER (ffmpeg select=eq(n,idx)), not by a seek time
    (-ss landed one frame off at 2.500 s, measured). Past the end: the last
    frame. idx counts the CLIP's own frames -- a film frame goes through source_index() first (v1168)."""
    import subprocess
    import tempfile
    import numpy as np
    import torch
    from PIL import Image
    exe = A.find_ffmpeg()
    if not exe:
        raise ValueError("[PLS] Cine Timeline: a clip frame needs ffmpeg (PATH or imageio-ffmpeg)")
    idx = max(0, int(idx))
    d = tempfile.mkdtemp(prefix="pls_frame_")
    out = os.path.join(d, "frame.png")
    try:
        subprocess.run([exe, "-v", "error", "-y", "-i", path, "-vf", "select=eq(n\\,%d)" % idx, "-vsync", "0",
                        "-frames:v", "1", "-an", out], capture_output=True, timeout=180)
        if not os.path.isfile(out) or os.path.getsize(out) == 0:
            return last_frame(path)
        a = np.asarray(Image.open(out).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(a)[None]
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------- v1172 frozen frames at the cuts
HELD_DIFF = 1.5      # mean grey difference (0..255, 96x54) of two frames that are one picture (measured: 0.02-0.9
                     # for the frozen pairs of the neon film, 4.5 the least motion between two moving frames)
CARRY_DIFF = 2.5     # the next scene's first picture vs the last one shown before it: the carried start
EDGE_W, EDGE_H = 96, 54
_EDGES = collections.OrderedDict()   # v1183: (path, mtime, size) -> uint8 greys, least recently used out
EDGES_BYTES = 96 * 1024 * 1024        # v1183: the cache holds this much (uint8; it held up to 64 float32 clips)


def film_greys(path):
    """The clip's frames as the FILM counts them (24 fps, as the export resamples), small and grey ->
    float32 [n, 54, 96]. Cached by (path, mtime, size) -- v1183: kept as uint8 (a quarter of the memory), at most
    EDGES_BYTES, the least recently used out."""
    import subprocess
    import numpy as np
    st = os.stat(path)
    key = (os.path.realpath(path), st.st_mtime, st.st_size)
    if key in _EDGES:
        _EDGES.move_to_end(key)
        return _EDGES[key].astype(np.float32)
    exe = A.find_ffmpeg()
    if not exe:
        return None
    r = subprocess.run([exe, "-v", "error", "-i", path, "-an", "-vf", "fps=%d,scale=%d:%d:flags=area,format=gray"
                        % (E.FPS, EDGE_W, EDGE_H), "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, timeout=300)
    if r.returncode != 0 or not r.stdout:
        return None
    a = np.frombuffer(r.stdout, np.uint8)
    a = a[:len(a) // (EDGE_W * EDGE_H) * EDGE_W * EDGE_H].reshape(-1, EDGE_H, EDGE_W).copy()
    _EDGES[key] = a
    while len(_EDGES) > 1 and sum(v.nbytes for v in _EDGES.values()) > EDGES_BYTES:
        _EDGES.popitem(last=False)
    return a.astype(np.float32)


def held_runs(g):
    """(head, tail): how many frames after the first are the first picture again, and before the last the last."""
    import numpy as np
    if g is None or len(g) < 4:
        return 0, 0
    d = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))
    head = 0
    while head < len(d) // 2 and d[head] < HELD_DIFF:
        head += 1
    tail = 0
    while tail < len(d) // 2 and d[len(d) - 1 - tail] < HELD_DIFF:
        tail += 1
    return head, tail


REPEAT_WINDOWS = (5, 22, 39, 56)   # v1176: a pinned head is a guide clip -- 17k+5 frames (Continue snaps to them)
REPEAT_DIFF = 6.0    # mean grey difference (0..255, 96x54) of a re-rendered head against the frames it repeats: the
                     # pinned head comes back through the VAE and the codec, not bit for bit -- measured on the guard's
                     # re-encoded neon clips 0.6-2.1; a wrong alignment of a moving shot 9-40


REPEAT_STILL = 1.0   # v1182: mean grey difference between a head's own consecutive frames below which it is a STILL


def repeat_head(prev, last_shown, g):
    """v1176 (carry_mode "motion"): how many of a clip's FIRST frames repeat the end of the scene before -- the
    frames up to `last_shown` of `prev` (grey film frames) -- the pinned head of a motion context. Tries the guide
    lengths (5, 22, 39, 56); every length below REPEAT_DIFF fits. v1182: the LONGEST fitting length wins -- a
    shorter window is a subset of the same aligned frames, so between 5 and 22 the minimum was a coin flip of codec
    noise that could leave 17 repeated frames in the film. Only a STILL head (its own frames do not move) fits every
    length for no reason -- there the shortest is left out, as before. 0 = no repeated head.
    -> (frames, mean difference or None)."""
    import numpy as np
    if prev is None or g is None:
        return 0, None
    fits = []
    for m in REPEAT_WINDOWS:
        if m > len(g) - 1 or m > last_shown + 1:
            continue
        d = float(np.abs(g[:m].astype(np.float32) - prev[last_shown - m + 1:last_shown + 1].astype(np.float32)).mean())
        if d < REPEAT_DIFF:
            fits.append((m, d))
    if not fits:
        return 0, None
    m, d = max(fits)
    head = g[:m].astype(np.float32)
    if m > 1 and float(np.abs(head[1:] - head[:-1]).mean()) < REPEAT_STILL:
        return min(fits)
    return m, d


def clip_last_shown(model, k, paths):
    """v1176: the film frame (24 fps, of its own clip) the film shows LAST of scene k (1-based) -- the asked end of
    the scene, behind its repeated head when scene k is itself a motion context of scene k-1 (measured, recursively).
    paths[j] = the clip of scene j+1 (None: none). -> index or None."""
    if not k or k > len(paths) or not paths[k - 1]:
        return None
    g = film_greys(paths[k - 1])
    # v1219: where the CUT ends scene k (the editor's carry_ends, on this take) -- a clip frame already
    ce, _note = T.carry_end(model, k, paths[k - 1])
    if ce is not None:
        return min(ce, len(g) - 1) if g is not None else ce
    idx = T.film_last_index(model, k)
    if model.get("carry_mode") == "motion" and T.carry_of(model, k) and k >= 2 and paths[k - 2]:
        pl = clip_last_shown(model, k - 1, paths)
        if pl is not None:
            idx += repeat_head(film_greys(paths[k - 2]), pl, g)[0]
    return min(idx, len(g) - 1) if g is not None else idx


def motion_seams(model):
    """v1176: the scenes (0-based) that continue the one before as a motion context."""
    if model.get("carry_mode") != "motion":
        return set()
    return set(k - 1 for k in range(2, len(T.scenes(model)) + 1) if T.carry_of(model, k))


def tail_frames(path, last_idx, n):
    """v1176: the film frames last_idx-n+1 .. last_idx of a clip (counted at 24 fps, as the film does) as an IMAGE
    [n, H, W, 3] -- the motion context of the scene after it."""
    import subprocess
    import numpy as np
    import torch
    exe = A.find_ffmpeg()
    if not exe:
        raise ValueError("[PLS] Cine Timeline: motion context needs ffmpeg (PATH or imageio-ffmpeg)")
    ci = clip_info(path)
    w, h = int(ci["w"]), int(ci["h"])
    a = max(0, int(last_idx) - int(n) + 1)
    r = subprocess.run([exe, "-v", "error", "-i", path, "-an", "-vf", "fps=%d,select=between(n\\,%d\\,%d)" % (E.FPS, a, last_idx),
                        "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, timeout=300)
    if r.returncode != 0 or not r.stdout:
        raise ValueError("[PLS] Cine Timeline: could not read the last frames of %s" % os.path.basename(path))
    arr = np.frombuffer(r.stdout, np.uint8)
    arr = arr[:len(arr) // (w * h * 3) * w * h * 3].reshape(-1, h, w, 3)
    return torch.from_numpy(arr.astype(np.float32) / 255.0)


def tail_sound(path, last_idx, n, rate=32000):
    """v1176: the sound under the film frames last_idx-n+1 .. last_idx of a clip -> AUDIO {waveform [1, 2, L],
    sample_rate}, or None (a clip without sound). 32 kHz, the H3 audio VAE's rate (no second resample)."""
    import subprocess
    import numpy as np
    import torch
    exe = A.find_ffmpeg()
    if not exe or not clip_info(path).get("audio"):
        return None
    t1 = (int(last_idx) + 1) / float(E.FPS)
    t0 = max(0.0, t1 - int(n) / float(E.FPS))
    r = subprocess.run([exe, "-v", "error", "-i", path, "-vn", "-af", "atrim=start=%.6f:end=%.6f,asetpts=PTS-STARTPTS"
                        % (t0, t1), "-ac", "2", "-ar", str(rate), "-f", "f32le", "-"], capture_output=True, timeout=300)
    if r.returncode != 0 or not r.stdout:
        return None
    a = np.frombuffer(r.stdout, np.float32)
    a = a[:len(a) // 2 * 2].reshape(-1, 2).T.copy()
    return {"waveform": torch.from_numpy(a)[None], "sample_rate": rate}


def held_at_cuts(paths, motion=(), prev_last=None):
    """Per scene (head, tail, n[, repeated]) frames to leave out at a cut (None: no clip): its own frozen frames, and at the head
    one more when its first picture is the last one shown before it (the carried start); n = its frames as decoded
    at 24 fps (the tail counts back from them). v1176: scene k (0-based) in `motion` -- a motion context -- leaves
    out its repeated head (repeat_head) instead, measured against prev_last[k], the frame of scene k-1's clip its
    tail ended on (clip_last_shown)."""
    import numpy as np
    gs = []
    for p in paths:
        try:
            gs.append(film_greys(p) if p else None)
        except Exception:
            gs.append(None)
    runs = [held_runs(g) if g is not None else None for g in gs]
    out = []
    for k, g in enumerate(gs):
        if g is None:
            out.append(None)
            continue
        head, tail = runs[k]
        prev = gs[k - 1] if k > 0 else None
        if k in motion and prev is not None and runs[k - 1] is not None:
            pl = len(prev) - 1 - runs[k - 1][1]
            if prev_last is not None and k < len(prev_last) and prev_last[k] is not None:
                pl = min(int(prev_last[k]), len(prev) - 1)
            rep_n, _d = repeat_head(prev, pl, g)
            if rep_n:
                out.append((rep_n, tail, len(g), True))     # v1181: True = a repeated head, out at every seam kind
                continue
        if prev is not None and runs[k - 1] is not None:
            last_shown = prev[len(prev) - 1 - runs[k - 1][1]]
            if float(np.abs(g[head] - last_shown).mean()) < CARRY_DIFF:
                head += 1
        out.append((head, tail, len(g)))
    return out


REGISTER_WAIT_TRIES, REGISTER_WAIT_STEP = 6, 0.5     # v1083: 3 s for the run before's register
DAY_CLIP_DAYS = 2                                     # v1220: today's and yesterday's day folder (a run past midnight)


def day_clip(model, k):
    """v1220 (audit #13): the newest clip of scene k (0 = the whole run) in the day folders, by the name a Save wired
    to clip_prefix gives it (<project>_scene_<k> / _whole) -- or None. Read only; the caller lists it."""
    import time
    proj = re.escape(T.project_name(model))
    rx = re.compile(r"_%s_%s(?:_|\.)" % (proj, "scene_%d" % int(k) if k else "whole"), re.I)
    best = None
    for d in range(DAY_CLIP_DAYS):
        folder = os.path.join(_output_dir(), time.strftime("%Y-%m-%d", time.localtime(time.time() - d * 86400)))
        try:
            names = os.listdir(folder)
        except OSError:
            continue
        for nm in names:
            if not (VIDEO_EXT.search(nm) and rx.search(nm)):
                continue
            full = os.path.join(folder, nm)
            try:
                mt = os.path.getmtime(full)
            except OSError:
                continue
            if best is None or mt > best[0]:
                best = (mt, full)
    return best[1] if best else None


# --------------------------------------------------------------- v1051 clip frames
def clip_anchor_paths(model):
    """{src: lead clip path or None} for every @clip:K:... anchor."""
    out = {}
    for a in model.get("anchors") or []:
        src = str(a.get("src", "")).strip()
        cs = T.clip_src(src)
        if cs is None or src in out:
            continue
        k = cs[0]
        try:
            takes = A.scene_takes(project_folder(model.get("project", "")), max(1, k))
        except Exception:
            takes = {}
        lst = takes.get(k) or []
        out[src] = lst[0] if lst else None
    return out


def preview_sources(model, srcs):
    """The preview's source set: what the editor says is wired (None = not
    knowable, the check is skipped) plus every @clip anchor whose clip exists
    -- the editor never sends those, the backend knows the stack."""
    if not isinstance(srcs, list):
        return None
    out = {s: None for s in srcs}
    for src, path in clip_anchor_paths(model).items():
        if path:
            out[src] = None
    return out


def save_frame(model, src, img):
    """Keep a used clip frame as input/cine_media/<project>/stack/frames/<scene>_<which>.png
    (overwritten each run; v1152: in input/, not output/) -- the picture the run anchored on, reusable in a
    Media Loader or a Reference Board. -> the path relative to input/, or ''."""
    try:
        import numpy as np
        from PIL import Image
        k, which = T.clip_src(src)
        name = "%s_%s.png" % ("whole" if k == 0 else "scene%02d" % k,
                              which if isinstance(which, str) else ("%.2fs" % which).replace(".", "_"))
        folder = os.path.join(project_folder(model.get("project", "")), "frames")
        os.makedirs(folder, exist_ok=True)
        arr = (img[0].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(folder, name))
        import folder_paths
        return "input/" + os.path.relpath(os.path.join(folder, name), folder_paths.get_input_directory()).replace(os.sep, "/")
    except Exception:
        return ""


def guide_lines(r):
    """v1051: what happens to the anchors between first and last frame -- said
    by the timeline, before MiniMax Continue meets them."""
    g = r.get("guides") or []
    if not g:
        return []
    where = ", ".join("%s @ %.2f s" % (src, f / float(T.FPS)) for src, f in g)
    try:                       # the same test as MiniMax Continue (the model code, v1057/v1061)
        from .ph_minimax_continue import core_has_guides
        has = core_has_guides()
    except Exception:
        has = False
    if has:
        return ["guides: %d anchor(s) between first and last frame (%s) -> `anchors` -> MiniMax "
                "Continue (this Core has MiniMaxH3AddGuide)" % (len(g), where)]
    return ["guides: %d anchor(s) between first and last frame (%s) -- WARN: this Core has no "
            "MiniMaxH3AddGuide, it places H3 keyframes only on the first and the last frame. "
            "MiniMax Continue will refuse them; update Core (master) or move them to 0 s / the end"
            % (len(g), where)]


# --------------------------------------------------------------- v1028 stack
def _output_dir():
    import folder_paths
    return folder_paths.get_output_directory()


def project_folder(project):
    """v1152 (Frank 17.10.: "everything in the folder of the day -- no task folders"): the project's working
    folder input/cine_media/<project>/stack/ -- the stack list (paths relative to output/) and the anchor
    frames. output/ holds only day folders now: the clips lie where the Saves write them, the films go to the
    day folder (day_folder). An old output/<project>/ is read once into the new list (A.migrate), never
    written again, never deleted."""
    import folder_paths
    safe = T.project_name({"project": project})
    A.OUTPUT_ROOT = _output_dir()
    folder = os.path.join(folder_paths.get_input_directory(), "cine_media", safe, "stack")
    try:
        A.migrate(os.path.join(_output_dir(), safe), folder)
    except Exception as e:
        print("[PLS] Cine Timeline: the old folder output/%s/ was not taken over (%s)" % (safe, e))
    return folder


def day_folder():
    """v1152: output/<yyyy-MM-dd>/ -- the day folder the Saves' %date:yyyy-MM-dd% names."""
    import time
    return os.path.join(_output_dir(), time.strftime("%Y-%m-%d"))


def film_stem(model, what="film"):
    """v1152: PH_<hhmmss>_<project>_<what> -- the Saves' naming (PH_%date:hhmmss%_...)."""
    import time
    return "PH_%s_%s_%s" % (time.strftime("%H%M%S"), T.project_name(model), what)


def register_clip(project, scene, ref, primary=False, run_files=None, sweep=None):
    """v1034: the editor saw a run of scene `scene` (0 = the whole timeline)
    write the clip `ref` ({filename, subfolder, type}) -- list it in the
    project's stack. Only files of type "output" under output/. -> rel path."""
    if not isinstance(ref, dict) or str(ref.get("type", "output")) != "output":
        raise ValueError("only clips in output/ can join the stack")
    name = os.path.basename(str(ref.get("filename", "")))
    if not name:
        raise ValueError("no filename")
    sub = str(ref.get("subfolder", "") or "")
    full = os.path.join(_output_dir(), sub, name)
    return A.register(project_folder(project), int(scene), full, primary=bool(primary), run_files=run_files, sweep=sweep)


async def handle_register(request):
    """POST {project, scene, file: {filename, subfolder, type}} -> {rel} or {error}."""
    from aiohttp import web
    try:
        body = await request.json()
        rel = register_clip(body.get("project", ""), body.get("scene", 0), body.get("file"),
                            primary=bool(body.get("primary")), run_files=body.get("run_files"),
                            sweep=body.get("sweep") if isinstance(body.get("sweep"), (int, float)) else None)
    except Exception as e:
        return web.json_response({"error": "%s" % e})
    print("[PLS] Cine stack: scene %s <- %s%s" % (body.get("scene"), rel,
          "  (the run's clip: it leads the takes)" if body.get("primary") else ""))
    return web.json_response({"rel": rel})


async def handle_renders(request):
    """v1198: POST {project, count} -> the project's renders, newest first (cine_assemble.render_groups)."""
    from aiohttp import web
    try:
        body = await request.json()
        return web.json_response(A.render_groups(project_folder(body.get("project", "")), int(body.get("count") or 1)))
    except Exception as e:
        return web.json_response({"error": "%s" % e})


# --------------------------------------------------------------- v1036 the video track
_PROBES = {}


def clip_info(path):
    """{frames, audio, w, h, fps} of a clip, cached by (path, mtime, size).
    frames = the video stream's own count (PyAV), else duration x fps."""
    st = os.stat(path)
    key = (os.path.realpath(path), st.st_mtime, st.st_size)
    if key in _PROBES:
        return _PROBES[key]
    import av
    with av.open(path) as c:
        vs = next((x for x in c.streams if x.type == "video"), None)
        if vs is None:
            raise ValueError("%s has no video stream" % os.path.basename(path))
        fps = float(vs.average_rate or vs.guessed_rate or E.FPS)
        n = int(vs.frames or 0)
        if n <= 0:
            dur = float(vs.duration * vs.time_base) if vs.duration and vs.time_base else (float(c.duration) / 1e6 if c.duration else 0.0)
            n = int(round(dur * fps))
        info = {"frames": n, "audio": any(x.type == "audio" for x in c.streams),
                "w": int(vs.codec_context.width), "h": int(vs.codec_context.height), "fps": fps}
        if abs(fps - E.FPS) > 0.01 and fps > 0:
            # v1043: a clip from disk may run at 25/30/60 fps -- the edit counts
            # film frames at 24 (the export resamples with fps=24 first)
            info["src_frames"] = n
            info["frames"] = E.film_frames(n, fps)      # v1195: the one law (half up, as ffmpeg's fps filter counts)
    if len(_PROBES) > 256:
        _PROBES.clear()
    _PROBES[key] = info
    return info


STRIP_TILES, STRIP_H = 24, 96


def strip_step(frames, tiles=STRIP_TILES):
    """Every step-th frame is a picture of the strip (tile i = frame i*step)."""
    return max(1, int(frames) // tiles)


def _strip(path, frames, tiles=STRIP_TILES, height=STRIP_H, film_fps=None):
    """A film strip of the clip (one jpg, `tiles` pictures `height` px high)
    in temp/, cached by path + mtime + size. -> the full path, or None (no
    ffmpeg / it failed). v1038: 24 pictures, 96 px -- the lane can be pulled
    taller and shows them in their aspect (web/js tilePlan).
    v1064 (Frank 08.10.: the render lane of a 48 fps clip showed only its first
    2.4 s, shot 3 never): film_fps resamples the clip to the film's rate FIRST,
    so `frames` (film frames) and the strip count the same frames -- the
    render lane maps a tile by film frames. None = the file's own frames (the
    layer rows map by the file's frames, web/js `nat`)."""
    import hashlib
    import subprocess
    import folder_paths
    st = os.stat(path)
    h = hashlib.md5(("%s|%s|%s|%s|%s" % (os.path.realpath(path), st.st_mtime, tiles, height, film_fps or "")).encode("utf-8")).hexdigest()[:16]
    folder = os.path.join(folder_paths.get_temp_directory(), "pls_cine_strips")
    out = os.path.join(folder, h + ".jpg")
    if os.path.isfile(out):
        return out
    exe = A.find_ffmpeg()
    if not exe:
        return None
    os.makedirs(folder, exist_ok=True)
    step = strip_step(frames, tiles)
    cmd = [exe, "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-vf",
           ("fps=%d," % int(film_fps) if film_fps else "")
           + "select='not(mod(n\\,%d))',scale=-2:%d,tile=%dx1" % (step, height, tiles), "-frames:v", "1", "-vsync", "0", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return out if r.returncode == 0 and os.path.isfile(out) else None


def _wave(path, width=1200):
    """v1037: the clip's waveform (one png, mono, 64 px high; v1153 normalised) in temp/, cached
    by path + mtime. -> the full path, or None (no audio / no ffmpeg)."""
    import hashlib
    import subprocess
    import folder_paths
    st = os.stat(path)
    h = hashlib.md5(("%s|%s|w%d|v1153" % (os.path.realpath(path), st.st_mtime, width)).encode("utf-8")).hexdigest()[:16]
    folder = os.path.join(folder_paths.get_temp_directory(), "pls_cine_strips")
    out = os.path.join(folder, h + "_w.png")
    if os.path.isfile(out):
        return out
    exe = A.find_ffmpeg()
    if not exe:
        return None
    os.makedirs(folder, exist_ok=True)
    cmd = [exe, "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-filter_complex",
           # v1153 (Frank: "the sound tracks with their waveforms"): a quiet clip (breath, linen) was a flat line --
           # the level is normalised (dynaudnorm) and drawn on a sqrt scale, so the shape shows at any level
           "aformat=channel_layouts=mono,dynaudnorm=f=150:g=15:p=0.95,showwavespic=s=%dx64:colors=0x80cbc4:scale=sqrt" % width,
           "-frames:v", "1", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return out if r.returncode == 0 and os.path.isfile(out) else None


def edit_plan(model, edit, strips=True):
    """The video track of a timeline: the clips on their places (the takes,
    the plan, the refs). whole = the render choice says whole."""
    from . import ph_cine_chain as CH
    from . import cine_clip as C
    whole = model.get("render", "whole") == "whole"
    n = len(T.scenes(model))
    folder = project_folder(model.get("project", ""))
    takes = A.scene_takes(folder, n)
    root = os.path.realpath(_output_dir())   # v1084: the same root register() uses (v1152: output/, named)
    ed = E.clean_edit(edit)
    places = [0] if whole else list(range(1, n + 1))
    bases = CH._bases()
    clips, info = [], []
    for k in places:
        cands = takes.get(k, [])
        rels = [os.path.relpath(x, root).replace(os.sep, "/") for x in cands]
        # v1046 (Frank 30.09.: "show me the last clip"): V1 is the newest run --
        # a take is only pinned while "new render -> new track" is on
        want = (ed.get("takes") or {}).get(str(k)) if ed.get("auto_track", True) is not False else None   # v1080: on by default
        i = rels.index(want) if want in rels else 0
        item = {"k": k, "takes": rels, "take": i, "path": None, "view": None, "strip": None, "name": None,
                # v1165: V1 shows a clip that replaced an earlier run's; v1190: the green NEW is "came in with the
                # latest render" -- the places that render has not reached yet keep their clip WITHOUT the mark
                "replaced": bool(i == 0 and rels and A.replaced(folder, k, rels, rels[0],
                                                                 stem=os.path.basename(T.clip_prefix(model, k)))),
                "new": bool(i == 0 and rels and A.is_new(folder, k, rels[0]))}
        if cands:
            path = cands[i]
            try:
                ci = clip_info(path)
            except Exception as e:
                ci = None
                item["error"] = "%s" % e
            if ci:
                item.update(ci)
                item.update({"path": path, "name": os.path.basename(path), "view": C.view_ref(path, bases)})
                if strips:
                    # v1064: the render lane maps tiles by FILM frames -- a clip at
                    # another rate (the 48 fps interpolation) is resampled first
                    sp = _strip(path, ci["frames"], film_fps=E.FPS if "src_frames" in ci else None)
                    item["strip"] = C.view_ref(sp, bases) if sp else None
                    item["strip_tiles"] = STRIP_TILES if sp else 0
                    item["strip_step"] = strip_step(ci["frames"]) if sp else 1
                    wv = _wave(path) if ci.get("audio") else None
                    item["wave"] = C.view_ref(wv, bases) if wv else None
            clips.append(ci)
        else:
            clips.append(None)
        info.append(item)
    starts = [] if whole else [T.scene_span(model, k)[0] for k in range(2, n + 1)]
    # v1083: a place whose prompt reaches into the grid overhang is shown to its end
    tails = set()
    for k in places:
        try:
            pm = T.place_model(model, k)
            if T.film_frames(pm) >= T.frames_for(pm["seconds"]):
                tails.add(k)
        except Exception:
            pass
    held = None
    if not whole:
        cps = [it.get("path") if c else None for it, c in zip(info, clips)]
        mot = motion_seams(model)
        pls = [None] + [clip_last_shown(model, k, cps) if k in mot else None for k in range(1, len(cps))]
        held = held_at_cuts(cps, mot, pls)   # v1172; v1176 the motion heads
    # v1219: the take each place shows now -- a cut list made on another take moves to this one's head
    now = {it["k"]: it["takes"][it["take"]] for it in info if it.get("path") and it.get("takes")}
    p = E.plan(starts, float(model.get("seconds", 6)), clips, model.get("transitions") or [],
               T.TRANSITIONS, edit=ed, whole=whole, tails=tails, held=held, takes=now)
    sizes = set((c["w"], c["h"]) for c in clips if c)
    if len(sizes) > 1:
        p["warnings"].append("the clips differ in size %s -- the export scales to the first" % sorted(sizes))
    out = {"whole": whole, "fps": E.FPS, "plan": p, "clips": info, "folder": folder, "lines": E.describe(p)}
    film = stack_of(ed, p, info, strips)
    if film is not None:
        out["film"] = film
        out["lines"] = out["lines"] + E.describe_stack(film)
    return out


# --------------------------------------------------------------- v1043 the layers
VIDEO_EXT = re.compile(r"\.(mp4|webm|mov|mkv|m4v|avi)$", re.I)


def _lan_locked():
    """v1160: the ONE LAN lock of the pack (uls_routes._lan_browse_locked; ph_media_routes in a build without
    the shared route file) -- True when ComfyUI is bound beyond localhost and ULS_ALLOW_LAN_BROWSE is not set.
    Fail-open like the lock itself: no lock found = local behaviour."""
    for mod in ("uls_routes", "ph_media_routes"):
        try:
            m = __import__(__package__ + "." + mod, fromlist=["_lan_browse_locked"]) if __package__ else __import__(mod)
            return bool(m._lan_browse_locked())
        except Exception:
            continue
    return False


def layer_path(src):
    """The file of a layer's source -- READ only, never written. "render":
    relative to output/ (and inside it); "file": an absolute path (a clip
    read where it lies) or relative to input/ (a dropped copy)."""
    import folder_paths
    path = str((src or {}).get("path") or "")
    if not path:
        raise ValueError("the layer has no file")
    if src.get("kind") == "render":
        root = os.path.realpath(_output_dir())
        full = os.path.realpath(os.path.join(root, path))
        if not (full == root or full.startswith(root + os.sep)):
            raise ValueError("a render layer must lie in output/")
    elif os.path.isabs(path):
        # v1160: a clip read WHERE IT LIES names any path on this machine -- the Media Loader's LAN lock
        # applies: bound beyond localhost (--listen) it is refused unless ULS_ALLOW_LAN_BROWSE=1 opens it
        if _lan_locked():
            raise ValueError("reading a clip where it lies is locked while ComfyUI is bound beyond localhost "
                             "(--listen) -- drop it as a copy (upper half), or set ULS_ALLOW_LAN_BROWSE=1")
        full = os.path.realpath(path)
    else:
        # v1160: a dropped copy lies in input/ -- a relative path that climbs out of it ("../..") is refused
        root = os.path.realpath(folder_paths.get_input_directory())
        full = os.path.realpath(os.path.join(root, path))
        if not full.startswith(root.rstrip(os.sep) + os.sep):
            raise ValueError("a copied clip must lie in input/")
    if not os.path.isfile(full):
        raise ValueError("not found: %s" % path)
    if not VIDEO_EXT.search(full):
        raise ValueError("not a video: %s" % os.path.basename(full))
    return full


def _layer_clip(path, strips):
    from . import ph_cine_chain as CH
    from . import cine_clip as C
    ci = dict(clip_info(path))
    ci.update({"path": path, "name": os.path.basename(path), "view": C.view_ref(path, CH._bases())})
    if ci["view"] is None:
        # read where it lies (outside output/input/temp): the Media Loader's
        # range route plays it (/uls/media/file)
        ci["file_url"] = {"folder": os.path.dirname(path), "file": os.path.basename(path)}
    if strips:
        sp = _strip(path, ci.get("src_frames", ci["frames"]))
        ci["strip"] = C.view_ref(sp, CH._bases()) if sp else None
        ci["strip_tiles"] = STRIP_TILES if sp else 0
        ci["strip_step"] = strip_step(ci.get("src_frames", ci["frames"])) if sp else 1
        wv = _wave(path) if ci.get("audio") else None
        ci["wave"] = C.view_ref(wv, CH._bases()) if wv else None
    return ci


def stack_of(ed, p, base_clips, strips=True):
    """The stack of the edit -> the film (cine_edit.stack_film) with every
    layer's plan and clip, or None when the edit is the plain v1037 film (one
    render layer, no gap, no span): then nothing changes -- same plan, same
    export, byte for byte."""
    layers = ed.get("layers") or []
    gaps = any(E.is_gap(x) for x in p["pieces"])
    if not layers and not gaps and "span" not in ed:
        return None
    if not layers:
        layers = [{"id": "base", "src": {"kind": "base"}, "video": True, "audio": True, "lock": False}]
    by_k = {c["k"]: c for c in base_clips}
    entries, view = [], []
    for L in layers:
        row = {"id": L["id"], "name": L.get("name") or ("render" if L["id"] == "base" else L["id"]),
               "src": L["src"], "video": L["video"], "audio": L["audio"], "lock": L["lock"], "solo": bool(L.get("solo"))}
        if L["id"] == "base":
            paths = [None if s.get("gap") else by_k.get(s["k"], {}).get("path") for s in p["segments"]]
            has_audio = all(by_k.get(s["k"], {}).get("audio") for s in p["segments"] if not s.get("gap") and not s["missing"])
            entries.append({"id": "base", "name": row["name"], "video": L["video"], "audio": L["audio"], "solo": row["solo"],
                            "has_audio": has_audio, "track": p, "paths": paths})
            row["base"] = True
        elif L["src"].get("kind") == "empty":
            tr = E.track([], {0: None}, T.TRANSITIONS, [])
            row.update({"plan": tr, "empty": True})
            entries.append({"id": L["id"], "name": row["name"], "video": L["video"], "audio": L["audio"], "solo": row["solo"],
                            "has_audio": False, "track": tr, "paths": []})
        else:
            warns = []
            # v1198: a track's files -- k 0 = src, k i = more[i-1] (a render's scenes on one track); each read alone,
            # a missing one names itself and leaves the others playing
            srcs = [L["src"]] + list(L.get("more") or [])
            cis, paths_k, errs = [], [], []
            for i, sx in enumerate(srcs):
                try:
                    pth = layer_path(sx)
                    cis.append(_layer_clip(pth, strips))
                    paths_k.append(pth)
                except Exception as e:
                    cis.append(None)
                    paths_k.append(None)
                    errs.append(("%s" % e) if i == 0 else ("file %d: %s" % (i + 1, e)))
            tr = E.track(L["pieces"], {i: (c["frames"] if c else None) for i, c in enumerate(cis)}, T.TRANSITIONS, warns)
            ci = cis[0]
            if ci:
                row["clip"] = ci
            if len(srcs) > 1:
                row["clips"] = cis
                row["paths_k"] = paths_k
            if errs:
                row["error"] = "; ".join(errs)
            tr = E.shift_sound(tr, L.get("aoff", 0))     # v1153: the pair unlinked, the sound shifted
            row["plan"] = tr
            row["link"] = L.get("link", True) is not False
            row["aoff"] = int(L.get("aoff", 0) or 0)
            used = sorted({int(s["k"]) for s in tr["segments"] if not s.get("gap")})
            has_a = bool(used) and all(cis[k] and cis[k].get("audio") for k in used if k < len(cis))
            entries.append({"id": L["id"], "name": row["name"], "video": L["video"], "audio": L["audio"], "solo": row["solo"],
                            "has_audio": has_a if used else bool(ci and ci.get("audio")), "track": tr,
                            "paths": [None if s.get("gap") else (paths_k[s["k"]] if 0 <= s["k"] < len(paths_k) else None) for s in tr["segments"]]})
        view.append(row)
    span = E.fr(ed["span"]) if "span" in ed else p["length"]
    film = E.stack_film(entries, span)
    for i, r in enumerate(view):
        r["has_audio"] = entries[i]["has_audio"]
        if "error" in r:
            film["warnings"].append("layer %s: %s" % (r["name"], r["error"]))
    return {"length": film["length"], "span": ed.get("span"), "flat": film["flat"], "complete": film["complete"] and not any(
        "error" in r for r in view if r["video"] or r["audio"]), "warnings": film["warnings"], "layers": view}


def snap_edit(model, edit, which="all"):
    """v1219 (Frank 24.10.: "eine Moeglichkeit, die Clips wieder korrekt an die Schneidegrenze zurueckzusetzen, wenn
    ich -- auch aus Versehen -- was verschoben habe"): every clip whose place is known (V1's own, a track's render
    file named ..._scene_<k>_ / ..._whole_) goes back to where the PROMPT cuts that place: its film start, its source
    window with the repeated head (a motion context) or the frozen frames at a cut left out, measured on THAT file.
    Files with no place (a clip dropped from disk) stay where they lie. Locked tracks are not touched.
    which = "all" (V1 + every track) or a track id. -> {"base": reset V1?, "layers": {id: patch}, "lines": [..]}."""
    ed = E.clean_edit(edit)
    whole = model.get("render", "whole") == "whole"
    plain = {k: v for k, v in ed.items() if k not in ("pieces", "pieces_n", "cuts", "end", "slip", "keep_tail")}
    d = edit_plan(model, plain, strips=False)
    p = d["plan"]
    leads = {it["k"]: it.get("path") for it in d["clips"]}
    places, acc = {}, 0
    for pc in p["pieces"]:
        n = E._length(pc)
        if not E.is_gap(pc) and pc["k"] not in places:
            places[pc["k"]] = {"at": acc, "len": n, "a": pc["a"], "b": pc["b"]}
        acc += n
    n_sc = len(T.scenes(model))
    cps = [leads.get(k) for k in range(1, n_sc + 1)]
    mot = motion_seams(model)
    trans = model.get("transitions") or []
    lines = []
    out = {"base": which in ("all", "base"), "layers": {}, "lines": lines}
    if out["base"]:
        lines.append("V1: the render's cut list goes -- the film cuts where the prompt cuts")
    for L in ed.get("layers") or []:
        if L["id"] == "base" or L["src"].get("kind") not in ("render", "file") or (which not in ("all", L["id"])):
            continue
        name = L.get("name") or L["id"]
        if L.get("lock"):
            lines.append("track %s is locked -- left as it is" % name)
            continue
        srcs = [L["src"]] + list(L.get("more") or [])
        files = []
        for sx in srcs:
            k = E.scene_of(sx.get("path"), whole)
            if k is None or k not in places:
                files.append(None)
                continue
            try:
                path = layer_path(sx)
                frames = int(clip_info(path)["frames"])
            except Exception as e:
                lines.append("track %s: %s -- %s" % (name, os.path.basename(str(sx.get("path"))), e))
                files.append(None)
                continue
            if leads.get(k) and os.path.realpath(leads[k]) == os.path.realpath(path):
                a, b = places[k]["a"], places[k]["b"]
            else:
                a = 0
                if k >= 2:
                    prev = leads.get(k - 1)
                    pl = clip_last_shown(model, k - 1, cps) if (k - 1) in mot else None
                    try:
                        hv = held_at_cuts([prev, path], {1} if (k - 1) in mot else set(), [None, pl])[1] if prev else None
                    except Exception:
                        hv = None
                    if hv:
                        rep = len(hv) > 3 and bool(hv[3])
                        cut = E._seam_kind(trans, T.TRANSITIONS, k - 2) == "cut"
                        a = int(hv[0]) if (rep or cut) else 0
                a = min(a, max(0, frames - E.MIN_SEG))
                b = min(frames, a + places[k]["len"])
            files.append({"place": k, "a": a, "b": b})
        pcs, done = E.snap_layer(L.get("pieces") or [], files, places)
        if not done:
            lines.append("track %s: no clip with a known place (scene_<k> in its name) -- left as it is" % name)
            continue
        patch = {"pieces": pcs, "src": dict(L["src"])}
        at_of = {i: at for i, _k, at in done}
        if 0 in at_of:
            patch["src"]["at"] = at_of[0]
        if L.get("more"):
            patch["more"] = [dict(m, at=at_of[i + 1]) if (i + 1) in at_of else dict(m) for i, m in enumerate(L["more"])]
        out["layers"][L["id"]] = patch
        lines.append("track %s: %s" % (name, ", ".join("%s at %s" % ("the whole run" if k == 0 else "scene %d" % k,
                                                                     T.time_label(at / float(T.FPS))) for _i, k, at in done)))
    return out


async def handle_snap(request):
    """v1219: POST {timeline, edit, which} -> snap_edit's patches (nothing is saved on the server)."""
    import asyncio
    from aiohttp import web
    try:
        body = await request.json()
        model = T.load(body.get("timeline", ""))
        loop = asyncio.get_running_loop()
        d = await loop.run_in_executor(None, snap_edit, model, body.get("edit"), str(body.get("which") or "all"))
    except Exception as e:
        return web.json_response({"error": "%s: %s" % (type(e).__name__, e)})
    return web.json_response(d)


async def handle_edit(request):
    """POST {timeline, edit} -> the video track {whole, plan, clips, lines} or {error}."""
    import asyncio
    from aiohttp import web
    try:
        body = await request.json()
        model = T.load(body.get("timeline", ""))
        loop = asyncio.get_running_loop()
        d = await loop.run_in_executor(None, edit_plan, model, body.get("edit"))
    except Exception as e:
        return web.json_response({"error": "%s: %s" % (type(e).__name__, e)})
    return web.json_response(d)


def _stamps(paths):
    """v1154 (audit): the preview cache's key names each file WITH its time and size -- a clip overwritten in place
    (a dropped file read where it lies) no longer returns the stale preview."""
    out = []
    for p in paths:
        try:
            st = os.stat(p)
            out.append([p, st.st_mtime, st.st_size])
        except (OSError, TypeError):
            out.append([p, None, None])
    return out


def export_film(model, edit, preview=False):
    """The film from the plan -> (out_path, lines). preview: small, fast, temp/."""
    import subprocess
    d = edit_plan(model, edit, strips=False)
    p = d["plan"]
    if d.get("film") is not None:
        return _export_stack(model, d, preview)
    if not p["complete"]:
        raise ValueError("; ".join(w for w in p["warnings"] if "not rendered" in w) or "a clip is missing")
    exe = A.find_ffmpeg()
    if not exe:
        raise ValueError("ffmpeg not found (PATH or imageio-ffmpeg) -- the export needs it")
    # one input per PIECE (v1037: a clip split in two is two inputs)
    by_k = {c["k"]: c for c in d["clips"]}
    paths = [by_k[sg["k"]]["path"] for sg in p["segments"]]
    audio = all(c.get("audio") for c in d["clips"])
    first = d["clips"][0]
    size = None
    if preview:
        h = 360
        size = (int(round(first["w"] * h / float(first["h"]) / 2)) * 2, h)
    elif len(set((c["w"], c["h"]) for c in d["clips"])) > 1:
        size = (first["w"], first["h"])
    if preview:
        import folder_paths
        import hashlib
        key = hashlib.md5(json.dumps([_stamps(paths), p["segments"], p["seams"]], sort_keys=True).encode("utf-8")).hexdigest()[:16]
        folder = os.path.join(folder_paths.get_temp_directory(), "pls_cine_preview")
        os.makedirs(folder, exist_ok=True)
        out = os.path.join(folder, key + ".mp4")
        if os.path.isfile(out):
            return out, ["exact preview (cached): " + out] + d["lines"]
    else:
        out = A.next_free(day_folder(), film_stem(model))   # v1152: the day folder, PH_<hhmmss>_<project>_film
    cmd = E.ffmpeg_command(exe, paths, out, p, audio, preview=preview, size=size)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(out):
        # the console gets everything; the node gets the lines that name the
        # cause (ffmpeg repeats the whole filter string before them)
        print("[PLS] Cine export: ffmpeg failed\n[PLS]   " + " ".join(cmd) + "\n" + (r.stderr or ""))
        why = [ln for ln in (r.stderr or "").splitlines() if ln.strip() and "[v0]" not in ln and len(ln) < 300]
        raise ValueError("ffmpeg failed: %s" % (" | ".join(why[-3:]) or "no output file"))
    head = "%s: %s (%.3f s%s)" % ("exact preview" if preview else "Export", out, p["length"] / float(E.FPS),
                                  "" if audio else ", video only -- a clip had no audio")
    return out, [head] + d["lines"]


def _export_stack(model, d, preview):
    """v1043: the film of a stack of layers (cine_edit.stack_command)."""
    import subprocess
    film = d["film"]
    if not film["complete"]:
        raise ValueError("; ".join(film["warnings"]) or "a layer's clip is missing")
    exe = A.find_ffmpeg()
    if not exe:
        raise ValueError("ffmpeg not found (PATH or imageio-ffmpeg) -- the export needs it")
    p = d["plan"]
    entries = []
    for r in film["layers"]:
        if r.get("base"):
            by_k = {c["k"]: c for c in d["clips"]}
            has_a = all(by_k.get(s["k"], {}).get("audio") for s in p["segments"] if not s.get("gap"))
            entries.append({"id": "base", "name": r["name"], "video": r["video"], "audio": r["audio"], "solo": r.get("solo"), "has_audio": has_a,
                            "track": p, "paths": [None if s.get("gap") else by_k[s["k"]]["path"] for s in p["segments"]]})
        else:
            pk = r.get("paths_k") or [(r.get("clip") or {}).get("path")]     # v1198: one path per file of the track
            entries.append({"id": r["id"], "name": r["name"], "video": r["video"], "audio": r["audio"], "solo": r.get("solo"),
                            "has_audio": bool(r.get("has_audio")), "track": r["plan"],
                            "paths": [None if s.get("gap") else (pk[s["k"]] if 0 <= s["k"] < len(pk) else None) for s in r["plan"]["segments"]]})
    stack = E.stack_film(entries, film["length"])
    # the film's size: the render's first clip, else the first layer with one
    first = next((c for c in d["clips"] if c.get("w")), None) or next(
        ((r.get("clip") or {}) for r in film["layers"] if (r.get("clip") or {}).get("w")), None)
    if not first:
        raise ValueError("no clip in the stack")
    w, h = int(first["w"]), int(first["h"])
    size = (int(round(w * 360 / float(h) / 2)) * 2, 360) if preview else (w // 2 * 2, h // 2 * 2)
    audio = any(x["has_audio"] and x["audio"] for x in entries)
    if preview:
        import folder_paths
        import hashlib
        key = hashlib.md5(json.dumps([[_stamps(x["paths"]), x["track"]["segments"], x["track"]["seams"], x["video"], x["audio"]] for x in entries]
                                     + [film["length"]], sort_keys=True).encode("utf-8")).hexdigest()[:16]
        folder = os.path.join(folder_paths.get_temp_directory(), "pls_cine_preview")
        os.makedirs(folder, exist_ok=True)
        out = os.path.join(folder, "s" + key + ".mp4")
        if os.path.isfile(out):
            return out, ["exact preview (cached): " + out] + d["lines"]
    else:
        out = A.next_free(day_folder(), film_stem(model))   # v1152: the day folder, PH_<hhmmss>_<project>_film
    cmd = E.stack_command(exe, stack, out, size, audio, preview=preview)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(out):
        print("[PLS] Cine export: ffmpeg failed\n[PLS]   " + " ".join(cmd) + "\n" + (r.stderr or ""))
        why = [ln for ln in (r.stderr or "").splitlines() if ln.strip() and "[c0]" not in ln and len(ln) < 300]
        raise ValueError("ffmpeg failed: %s" % (" | ".join(why[-3:]) or "no output file"))
    head = "%s: %s (%.3f s, %d layer(s)%s)" % ("exact preview" if preview else "Export", out, film["length"] / float(E.FPS),
                                               len(entries), "" if audio else ", no sound")
    return out, [head] + d["lines"]


def media_dir(project):
    """v1044: input/cine_media/<project>/ -- where a dropped clip is COPIED
    (Frank: "its own folder where all the cutting work lies"). Made on demand."""
    import folder_paths
    d = os.path.join(folder_paths.get_input_directory(), "cine_media", T.project_name({"project": project}))
    os.makedirs(d, exist_ok=True)
    return d


async def handle_media_dir(request):
    """GET ?project= -> {dir, rel} (rel to input/)."""
    from aiohttp import web
    try:
        d = media_dir(request.rel_url.query.get("project", ""))
        import folder_paths
        return web.json_response({"dir": d, "rel": os.path.relpath(d, folder_paths.get_input_directory()).replace(os.sep, "/")})
    except Exception as e:
        return web.json_response({"error": "%s" % e})


def probe_layer(src):
    """v1044: a clip picked for a layer -> {frames (24 fps), fps, w, h, audio,
    name} -- refused like layer_path (never written)."""
    full = layer_path(src)
    ci = clip_info(full)
    return {"frames": ci["frames"], "fps": ci["fps"], "w": ci["w"], "h": ci["h"], "audio": ci["audio"],
            "name": os.path.basename(full)}


async def handle_probe(request):
    """POST {src: {kind, path}} -> the clip's facts, or {error}."""
    import asyncio
    from aiohttp import web
    try:
        body = await request.json()
        loop = asyncio.get_running_loop()
        d = await loop.run_in_executor(None, probe_layer, body.get("src") or {})
    except Exception as e:
        return web.json_response({"error": "%s" % e})
    return web.json_response(d)


async def handle_export(request):
    """POST {timeline, edit, preview} -> {path, view, lines} or {error}."""
    import asyncio
    from aiohttp import web
    try:
        body = await request.json()
        model = T.load(body.get("timeline", ""))
        loop = asyncio.get_running_loop()
        out, lines = await loop.run_in_executor(None, export_film, model, body.get("edit"), bool(body.get("preview")))
    except Exception as e:
        return web.json_response({"error": "%s" % e})
    from . import ph_cine_chain as CH
    from . import cine_clip as C
    print("[PLS] " + "\n[PLS] ".join(lines))
    return web.json_response({"path": out, "view": C.view_ref(out, CH._bases()), "lines": lines})


def register_timeline_routes():
    try:
        from server import PromptServer
        app = PromptServer.instance.app
    except Exception as e:
        print("[PLS] Cine timeline route not registered: %s" % e)
        return 0
    n = 0
    for method, path, fn in (("POST", "/pls/cine/timeline", handle_timeline_preview),
                             ("POST", "/pls/cine/register", handle_register),
                             ("POST", "/pls/cine/renders", handle_renders),
                             ("POST", "/pls/cine/edit", handle_edit),
                             ("POST", "/pls/cine/snap", handle_snap),          # v1219
                             ("POST", "/pls/cine/export", handle_export),
                             ("GET", "/pls/cine/media_dir", handle_media_dir),
                             ("POST", "/pls/cine/probe", handle_probe),
                             ("POST", "/pls/cine/to_form", handle_to_form)):
        for p in (path, "/api" + path):
            try:
                app.router.add_route(method, p, fn)
                n += 1
            except Exception as e:
                print("[PLS] could not add route %s %s: %s" % (method, p, e))
    return n


NODE_CLASS_MAPPINGS = {"ULSCineTimeline": ULSCineTimeline}
NODE_DISPLAY_NAME_MAPPINGS = {"ULSCineTimeline": "\u2b21 Polyhedron Cine Timeline"}
