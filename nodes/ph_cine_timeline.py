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
output/ holds only day folders), the editor shows the stack (GET /pls/cine/scenes) and joins it with transitions
(POST /pls/cine/assemble -> nodes/cine_assemble.py, ffmpeg xfade).
"""

import hashlib
import json
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
    want = T.carry_of(model, k)
    carry_lines, carried = [], False
    if want and prev:
        carried = True
        if model.get("carry_mode", "pin") == "keyframe":
            target, added = T.with_carry_anchor(target)
            if not added:
                carried = False
                carry_lines.append("carry: scene %d has its own first-frame anchor -- that one wins" % k)
    r = T.compile_prompt(target)
    r["scene"], r["scenes"], r["clip_prefix"] = k, n, T.clip_prefix(model, k)
    # v1083: in pin mode too, an own first-frame anchor wins (the info said
    # "carry", the output was the own anchor, and the frame was decoded for nothing)
    if carried and r["first"] is not None and model.get("carry_mode", "pin") != "keyframe":
        carried = False
        carry_lines.append("carry: scene %d has its own first-frame anchor -- that one wins" % k)
    r["carry"] = prev if carried else None
    r["carry_index"] = T.film_last_index(model, k - 1) if carried else None
    if carried:
        carry_lines.append("carry: scene %d begins on the frame the film shows LAST of scene %d (frame %d of %s) -> first_frame%s"
                           % (k, k - 1, r["carry_index"] + 1, os.path.basename(prev),
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


ADOPT_KEEPS = ("project", "triggers", "target", "carry_mode", "render", "active")


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

    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE", "PLS_ANCHORS", "FLOAT", "INT", "STRING", "STRING", "INT")
    RETURN_NAMES = ("prompt", "first_frame", "last_frame", "anchors", "seconds", "frames", "info",
                    "clip_prefix", "scene")
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
                   "Play previews the film in the node, export writes it.")

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
            img = frame_index(path, idx)
            sources[src] = img
            png = save_frame(model, src, img)
            clip_lines.append("clip frame %s: %s (frame %d) of %s -> %s" % (
                src, which if isinstance(which, str) else "%.2f s" % which, idx + 1,
                os.path.basename(path), png or "(not saved)"))
        prev = carry_source(model, k0) if T.carry_of(model, k0) else None
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
        if r.get("carry"):
            carry_img = frame_index(r["carry"], r["carry_index"])
            sources = dict(sources)
            sources[T.CARRY_SRC] = carry_img
        if r["first"] is not None:
            first = sources[r["first"]["src"]][:1]
        else:
            first = carry_img
        last = sources[r["last"]["src"]][:1] if r["last"] is not None else None
        anchors = [{"src": src, "frame": f, "image": sources[src][:1]} for src, f in r["guides"]]
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
                           int(r["frames"]), info, r["clip_prefix"], int(r["scene"]))}


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


def frame_index(path, idx):
    """v1083: frame number idx (0-based) of a clip as an IMAGE [1, H, W, 3] --
    selected by its NUMBER (ffmpeg select=eq(n,idx)), not by a seek time
    (-ss landed one frame off at 2.500 s, measured). Past the end: the last
    frame."""
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


REGISTER_WAIT_TRIES, REGISTER_WAIT_STEP = 6, 0.5     # v1083: 3 s for the run before's register


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


def stack_items(project, count):
    """The stack as the editor shows it: one entry per scene 1..count (None
    when not rendered yet) + the newest whole render, with /view refs and
    posters (the Clip Chain's own helpers)."""
    from . import ph_cine_chain as CH
    from . import cine_clip as C
    folder = project_folder(project)
    files = A.scene_files(folder, count)
    bases = CH._bases()

    def item(k):
        f = files.get(k)
        if not f:
            return None
        it = {"scene": k, "name": f["name"], "path": f["path"], "mtime": f["mtime"], "runs": f["runs"],
              "view": C.view_ref(f["path"], bases), "latent": False, "current": False}
        return CH._with_posters([it])[0]
    return {"folder": folder, "scenes": [item(k) for k in range(1, int(count) + 1)], "whole": item(0)}


async def handle_scenes(request):
    """GET /pls/cine/scenes?project=..&count=N -> {folder, scenes: [item|null], whole}."""
    from aiohttp import web
    q = request.rel_url.query
    try:
        count = max(1, min(64, int(q.get("count", "1"))))
        return web.json_response(stack_items(q.get("project", ""), count))
    except Exception as e:
        return web.json_response({"folder": "", "scenes": [], "whole": None, "error": "%s: %s" % (type(e).__name__, e)})


def register_clip(project, scene, ref, primary=False, run_files=None):
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
    return A.register(project_folder(project), int(scene), full, primary=bool(primary), run_files=run_files)


async def handle_register(request):
    """POST {project, scene, file: {filename, subfolder, type}} -> {rel} or {error}."""
    from aiohttp import web
    try:
        body = await request.json()
        rel = register_clip(body.get("project", ""), body.get("scene", 0), body.get("file"),
                            primary=bool(body.get("primary")), run_files=body.get("run_files"))
    except Exception as e:
        return web.json_response({"error": "%s" % e})
    print("[PLS] Cine stack: scene %s <- %s%s" % (body.get("scene"), rel,
          "  (the run's clip: it leads the takes)" if body.get("primary") else ""))
    return web.json_response({"rel": rel})


async def handle_assemble(request):
    """POST {project, count, transitions, name} -> {path, view, lines} or {error}."""
    from aiohttp import web
    try:
        body = await request.json()
        folder = project_folder(body.get("project", ""))
        out, lines = A.assemble(folder, int(body.get("count", 0)), body.get("transitions") or [],
                                T.TRANSITIONS, out_dir=day_folder(),
                                name=film_stem({"project": body.get("project", "")}, str(body.get("name", "") or "film")))
    except Exception as e:
        return web.json_response({"error": "%s" % e})
    from . import ph_cine_chain as CH
    from . import cine_clip as C
    print("[PLS] " + "\n[PLS] ".join(lines))
    ref = C.view_ref(out, CH._bases())
    poster = CH._with_posters([{"path": out}])[0].get("poster")
    return web.json_response({"path": out, "view": ref, "poster": poster, "lines": lines})


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
            info["frames"] = max(1, int(round(n * E.FPS / fps)))
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
                # v1165: V1 shows a clip that replaced an earlier run's (the green NEW)
                "replaced": bool(i == 0 and rels and A.replaced(folder, k, rels, rels[0],
                                                                 stem=os.path.basename(T.clip_prefix(model, k))))}
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
    p = E.plan(starts, float(model.get("seconds", 6)), clips, model.get("transitions") or [],
               T.TRANSITIONS, edit=ed, whole=whole, tails=tails)
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
            try:
                path = layer_path(L["src"])
                ci = _layer_clip(path, strips)
                tr = E.track(L["pieces"], {0: ci["frames"]}, T.TRANSITIONS, warns)
                row["clip"] = ci
            except Exception as e:
                path, ci = None, None
                tr = E.track(L["pieces"], {0: None}, T.TRANSITIONS, warns)
                row["error"] = "%s" % e
            tr = E.shift_sound(tr, L.get("aoff", 0))     # v1153: the pair unlinked, the sound shifted
            row["plan"] = tr
            row["link"] = L.get("link", True) is not False
            row["aoff"] = int(L.get("aoff", 0) or 0)
            entries.append({"id": L["id"], "name": row["name"], "video": L["video"], "audio": L["audio"], "solo": row["solo"],
                            "has_audio": bool(ci and ci.get("audio")), "track": tr,
                            "paths": [None if s.get("gap") else path for s in tr["segments"]]})
        view.append(row)
    span = E.fr(ed["span"]) if "span" in ed else p["length"]
    film = E.stack_film(entries, span)
    for i, r in enumerate(view):
        r["has_audio"] = entries[i]["has_audio"]
        if "error" in r:
            film["warnings"].append("layer %s: %s" % (r["name"], r["error"]))
    return {"length": film["length"], "span": ed.get("span"), "flat": film["flat"], "complete": film["complete"] and not any(
        "error" in r for r in view if r["video"] or r["audio"]), "warnings": film["warnings"], "layers": view}


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
            path = (r.get("clip") or {}).get("path")
            entries.append({"id": r["id"], "name": r["name"], "video": r["video"], "audio": r["audio"], "solo": r.get("solo"),
                            "has_audio": bool(r.get("has_audio")), "track": r["plan"],
                            "paths": [None if s.get("gap") else path for s in r["plan"]["segments"]]})
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
                             ("GET", "/pls/cine/scenes", handle_scenes),
                             ("POST", "/pls/cine/register", handle_register),
                             ("POST", "/pls/cine/assemble", handle_assemble),
                             ("POST", "/pls/cine/edit", handle_edit),
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
