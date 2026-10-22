# -*- coding: ascii -*-
"""Cine timeline -- the model and the compiler (C6, v1023). Pure: no ComfyUI.

One timeline = one clip. It holds:

  seconds     the wanted length; the clip gets Core's frame count for it
              (align up to 17k+5 at 24 fps -- the same rule as Empty Latent)
  style       the style words that follow [Shot 1] (official section 3)
  shots       [{"at": s, "cut": phrase, "text": ...}], shot 1 at 0.0
  anchors     [{"src": "@tag" | "image_1..9" | "@clip:K:first|last|S", "at": s}]
  soundscape  overall_soundscape
  music       non_diegetic_music ("N/A" when empty -- else H3 adds some)
  from        the fingerprint of the storyboard this timeline took over
              ("" = drawn by hand); a new storyboard replaces it once (C7)

  v1028 (the cutting room, Cine S6 / E1) -- a timeline may hold SCENES:
  render      "whole"  = the timeline is ONE clip, one run (as before)
              "scene"  = the run renders ONE scene of it (the active one);
                         the choice is the user's, made in the node -- never
                         guessed from the length
              "all"    = v1034: every scene, one run each -- the editor splits
                         ONE queued run into one run per scene (active = 1..n);
                         a run that arrives unsplit renders the active scene
  active      1-based index of the scene a "scene" (or "all") run renders
  project     the project's name: its stack (input/cine_media/<project>/stack/) and the names
              of its clips (`clip_prefix`: <day>/PH_<hhmmss>_<project>_whole | _scene_k, v1152)
  transitions per scene boundary: [{"kind": ..., "seconds": ...}] -- kind
              is one of TRANSITIONS; used by the assemble route (ffmpeg xfade)
  shots[i].scene   true = this shot begins a new scene (shot 1 always does)

  v1049 -- scene continuity (Frank 01.10.: "a connection from clip 1 to 2"):
  shots[i].carry   on a scene-start shot: scene k BEGINS on the last frame of
                   scene k-1's newest clip (the project's stack). The timeline
                   hands that frame out as `first_frame`.
  triggers    v1050: the LoRA trigger words -- with `style` the head of [Shot 1]
              in every run (// TRIGGERS + // LOOK), so scenes rendered one by
              one share them
  carry_mode  "pin"      = the frame is only pinned (MiniMax Reference's
                           first_frame, ref2va) -- the prompt is unchanged
              "keyframe" = it is the first-frame anchor of the scene: the
                           official alignment line goes into the prompt and
                           MiniMax Keyframes presents it as <Picture 1> (fl2va)
              "motion"   = v1176 (Cine context continuation): the last
                           MOTION_FRAMES film frames of scene k-1 and their
                           sound go out as tail / tail_audio (-> MiniMax
                           Continue K2, Core master): the scene is rendered
                           with that window as its pinned HEAD -- MOTION_FRAMES
                           longer, every shot after the first and every anchor
                           moved back by it -- and continues the motion and the
                           sound. The clip on disk keeps the head; the film
                           leaves it out (measured: its first frames that
                           repeat the end of scene k-1).

  v1035 -- H3 transitions, written as text (Frank's field test 29.09.):
  shots[i].h3t     {"kind": one of H3T_NAMES, "seconds": 0.25..2,
                    "out"/"in"/"formula": only for "H3 custom"}
              a transition H3 PAINTS inside one clip. It sets the cut phrase
              (the official formula) AND two sentences: one at the end of
              the shot before ("During the last half second of the shot,
              <out>."), one behind the first sentence of the shot after
              ("<In> within half a second."). The formula alone does not do
              it (the wipe probe: a hard cut); formula + description does
              (clip PH_103202: blur to white peaks on the cut's frame, fade
              through black lies 0.3 s before to 1.5 s after the cut).
              A shot that starts a SCENE ignores its h3t (two clips).

  A scene is compiled like a clip of its own: its shots re-based to 0, its
  length snapped to the 17k+5 grid, its anchors the ones inside its span.
  Style, soundscape and music are the timeline's (E3 may give a scene its
  own).

and compiles to the OFFICIAL H3 form (prompting package, 10_H3_OFFIZIELL.md
sections 2 and 3, 60_VORLAGEN.md forms A1-A4):

  [alignment line]                       only with a first and/or last frame

  integrated_multimodal_description: [Shot 1] <style> <text>

  [Shot 2] At 00:02.500, the camera cuts to <text>

  overall_soundscape: ...

  non_diegetic_music: ...

ANCHORS, SPLIT BY WHAT THE MODEL CAN SEE (measured at Core's source, 26.09.):
  - an anchor on frame 0 or on the last frame is a KEYFRAME -> MiniMax
    Keyframes, which shows it to the text encoder as <Picture n>; only
    these get the official alignment line (I2VA / FL2VA / L2VA wording,
    verbatim);
  - any other anchor is a GUIDE -> MiniMax Continue (Core master's Add Guide
    semantics). Guides are NOT shown to the text encoder, so the prompt
    cannot name them; the info line says so instead of inventing a label.

Whether H3 keeps the shot times is field test A (open). The shot track works
either way: the times are written in the form H3 was trained on.
"""

import json
import math
import re

FPS = 24
MIN_S, MAX_S = 4.0, 15.0
CUTS = ["the camera cuts to", "the shot cuts to", "the shot transitions to",
        "the shot changes to", "the shot switches to", "(none)"]
N_IMAGE_SLOTS = 9          # v1051: was 3 -- as many as H3 takes as references
IMAGE_SLOTS = tuple("image_%d" % n for n in range(1, N_IMAGE_SLOTS + 1))

# v1051: a frame of a RENDERED clip as an anchor -- "@clip:K:last" (K = scene,
# 0 = the whole timeline), "@clip:K:first", "@clip:K:2.5" (seconds into it).
# The clip is scene K's lead take in the project's stack (the run's own clip
# first, else the newest), resolved by the node at run time.
CLIP_SRC_RE = re.compile(r"^@clip:(\d{1,3}):(first|last|\d+(?:\.\d+)?)$")


def clip_src(src):
    """'@clip:K:which' -> (K, which) with which 'first' | 'last' | seconds
    (float), or None for any other source."""
    m = CLIP_SRC_RE.match(str(src or "").strip())
    if not m:
        return None
    w = m.group(2)
    return int(m.group(1)), (w if w in ("first", "last") else float(w))
RENDER_MODES = ("whole", "scene", "all")
# editor name -> ffmpeg xfade transition; "cut" is a one-frame fade
TRANSITIONS = {"cut": "fade", "dissolve": "fade", "fade to black": "fadeblack",
               "fade to white": "fadewhite", "wipe left": "wipeleft",
               "wipe right": "wiperight", "slide left": "slideleft"}
TRANSITION_NAMES = ("cut", "dissolve", "fade to black", "fade to white",
                    "wipe left", "wipe right", "slide left")
DEFAULT_TRANSITION = {"kind": "cut", "seconds": 0.5}
# v1035: transitions H3 paints inside the clip (formula + description).
# Measured: blur to white and fade through black (29.09.); cross-dissolve
# is the official formula with a description, not measured yet. No wipe:
# measured a hard cut (P1_D, 28.09.).
H3T_PRESETS = {
    "H3 blur to white": {
        "formula": "the shot cross-dissolves to",
        "out": "the whole image slowly drifts out of focus into a soft glowing blur while the light swells until the frame turns almost white",
        "in": "The image resolves out of the white glow and comes back into sharp focus"},
    "H3 fade through black": {
        "formula": "the shot fades to black, then fades in on",
        "out": "the light drains from the scene and the whole image sinks into darkness until the frame is black",
        "in": "Brightness returns and the image rises out of the dark"},
    "H3 cross-dissolve": {
        "formula": "the shot cross-dissolves to",
        "out": "the image begins to dissolve softly while the next image shows through it",
        "in": "The previous image melts away over the new one"},
}
H3T_CUSTOM = "H3 custom"
H3T_NAMES = tuple(H3T_PRESETS) + (H3T_CUSTOM,)
H3T_FORMULAS = ("the shot cross-dissolves to", "the shot fades to black, then fades in on",
                "the camera cuts to", "(none)")
H3T_MIN, H3T_MAX, H3T_DEFAULT = 0.25, 2.0, 0.5
DEFAULT = {
    "v": 1, "seconds": 6.0, "style": "Live-action, cinematic,",
    "shots": [{"at": 0.0, "cut": CUTS[0], "text": ""}],
    "anchors": [], "soundscape": "", "music": "N/A", "from": "",
    "render": "whole", "active": 1, "project": "cine", "transitions": [],
    "carry_mode": "pin", "triggers": "", "target": "fl2va", "head_frames": 22,
}
CARRY_MODES = ("pin", "keyframe", "motion")
MOTION_FRAMES = 22            # v1176: the pinned head (a 17k+5 window: 0.92 s; Continue's context_frames) -- the default
MOTION_AUDIO = 18             # v1176: how far back its sound reaches (Continue's audio_context; 3-frame grid)
# v1185 (Frank 20.10., the render times): the head's length is the timeline's choice -- 17k+5 windows, because
# Continue snaps a guide to them. 5 = 0.21 s (the scene grows by 17 frames: 141 + 5 = 146 -> 158), 22 = 0.92 s
# (141 + 22 = 163 -> 175, the v1176 behaviour), 39 = 1.63 s (141 + 39 = 180 -> 192). Which length carries the
# motion is UNMEASURED (Prompting 67, F5); the default stays 22 so every saved timeline runs as before.
HEAD_FRAMES = (5, 22, 39)


def motion_head(model):
    """v1185: the pinned head's length in frames for carry_mode "motion" (model key head_frames; 22 when absent)."""
    try:
        h = int(model.get("head_frames", MOTION_FRAMES) or MOTION_FRAMES)
    except (TypeError, ValueError):
        return MOTION_FRAMES
    return h if h in HEAD_FRAMES else MOTION_FRAMES
# v1051: which H3 model the run feeds -- it decides what a first/last-frame
# anchor IS. fl2va: MiniMax Keyframes shows it as <Picture n> and the prompt
# gets the official alignment line. ref2va: MiniMax Reference pins it at frame
# 0 (first_frame) -- NOT a picture of the prompt (the references keep their
# numbers), so no alignment line; ref2va has no last frame.
TARGETS = ("fl2va", "ref2va")
CARRY_SRC = "@carry"          # the anchor source of a carried frame (never on a board)


# --------------------------------------------------------------- the grid
def align_frames(n):
    """Core's align_frame_count: up to the next 17k+5."""
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def asked_frames(seconds):
    """The frames a length asks for, before the grid -- half up like the
    browser's Math.round (v1083: Python's round() goes to the even number;
    the editor and this counted 158.5 one frame apart)."""
    return int(math.floor(float(seconds) * FPS + 0.5))


def frames_for(seconds):
    return align_frames(asked_frames(seconds))


# --------------------------------------------------------------- v1083 the film's truth
# A render is longer than asked: the grid lifts 6.0 s (144 frames) to 158. The
# film cut that overhang -- and the carry took the CLIP's last frame, the last
# keyframe sat in the overhang, a shot there was valid but never shown (the
# audit of 10.10.). ONE rule now: a place is shown to its grid end when the
# prompt puts something there (a shot at or after the asked length, an anchor
# on a frame at or after it); otherwise the film cuts at the asked length. The
# carry and @clip:K:last take the frame the FILM shows last, never another.
def film_frames(target):
    """The frames the film shows of a compiled place (the whole model or a
    scene_model): the asked length, or the grid when the prompt reaches into
    the overhang. Pure."""
    grid = frames_for(target["seconds"])
    asked = max(1, min(grid, asked_frames(target["seconds"])))
    if asked >= grid:
        return grid
    for s in target.get("shots") or []:
        try:
            if frame_of(s.get("at", 0), grid) >= asked:
                return grid
        except Exception:
            continue
    for a in target.get("anchors") or []:
        try:
            if frame_of(a.get("at", 0), grid) >= asked:
                return grid
        except Exception:
            continue
    return asked


def place_model(model, k):
    """Place k as the compiler sees it: 0 = the whole timeline, else scene k."""
    return model if not k else scene_model(model, int(k))


def film_last_index(model, k):
    """The index of the frame the film shows last of place k (0 = whole)."""
    return film_frames(place_model(model, k)) - 1


def time_label(sec):
    """MM:SS.mmm (official section 3)."""
    ms = int(round(float(sec) * 1000.0))
    return "%02d:%02d.%03d" % (ms // 60000, (ms // 1000) % 60, ms % 1000)


def mark(sec):
    """S.SS -- exactly two decimals (official section 2)."""
    return "%.2f" % float(sec)


def frame_of(sec, frames):
    """The pixel frame a time lands on, inside the clip."""
    return max(0, min(frames - 1, int(round(float(sec) * FPS))))


# --------------------------------------------------------------- the model
def load(text):
    """The timeline JSON -> a model dict (defaults filled). Raises by name."""
    try:
        d = json.loads(text) if text and str(text).strip() else {}
    except Exception as e:
        raise ValueError("the timeline state is not JSON (%s)" % e)
    if not isinstance(d, dict):
        raise ValueError("the timeline state is not an object")
    m = json.loads(json.dumps(DEFAULT))
    m.update({k: d[k] for k in d if k in DEFAULT})
    return m


def shot_at(model, sec):
    """1-based number of the shot that contains time `sec`."""
    n = 1
    for i, s in enumerate(sorted(model["shots"], key=lambda s: float(s.get("at", 0))), start=1):
        if float(s.get("at", 0)) <= float(sec) + 1e-9:
            n = i
    return n


def validate(model, sources=None):
    """Errors by name ([] = compiles). `sources`: the anchor sources that
    exist (tags and wired slots); None = do not check sources."""
    errs = []
    try:
        seconds = float(model.get("seconds", 0))
    except Exception:
        seconds = 0.0
    if not seconds > 0:
        errs.append("the length is not a positive number of seconds")
        return errs
    frames = frames_for(seconds)
    dur = frames / float(FPS)
    shots = model.get("shots") or []
    if not shots:
        errs.append("there is no shot -- a clip has at least [Shot 1]")
    ats = []
    for i, s in enumerate(shots, start=1):
        try:
            ats.append(float(s.get("at", 0)))
        except Exception:
            errs.append("shot %d: 'at' is not a number" % i)
            ats.append(float("nan"))
        if s.get("cut", CUTS[0]) not in CUTS:
            errs.append("shot %d: cut %r is not one of the official phrases" % (i, s.get("cut")))
        errs.extend("shot %d: %s" % (i, e) for e in h3t_errors(s.get("h3t")))
    if ats and not any(math.isnan(a) for a in ats):
        if abs(ats[0]) > 1e-9:
            errs.append("shot 1 starts at %.3f s -- [Shot 1] starts at 0 and carries no time" % ats[0])
        for i in range(1, len(ats)):
            if ats[i] <= ats[i - 1]:
                errs.append("shot %d at %.3f s is not after shot %d (%.3f s) -- cuts go forward"
                            % (i + 1, ats[i], i, ats[i - 1]))
            if ats[i] >= dur:
                errs.append("shot %d at %.3f s starts after the clip ends (%.3f s)" % (i + 1, ats[i], dur))
    if model.get("target", "fl2va") not in TARGETS:
        errs.append("target %r is not one of %s" % (model.get("target"), "/".join(TARGETS)))
    if model.get("carry_mode", "pin") not in CARRY_MODES:
        errs.append("carry_mode %r is not one of %s" % (model.get("carry_mode"), "/".join(CARRY_MODES)))
    if "head_frames" in model and model.get("head_frames") is not None:
        try:
            hf = int(model.get("head_frames"))
        except (TypeError, ValueError):
            hf = None
        if hf not in HEAD_FRAMES:
            errs.append("head_frames %r is not one of %s (17k+5 windows)"
                        % (model.get("head_frames"), "/".join(str(h) for h in HEAD_FRAMES)))
    if model.get("render", "whole") not in RENDER_MODES:
        errs.append("render %r is not one of %s" % (model.get("render"), "/".join(RENDER_MODES)))
    elif model.get("render") in ("scene", "all"):
        n_sc = len(scenes(model))
        try:
            k = int(model.get("active", 1))
        except Exception:
            k = 0
        if k < 1 or k > n_sc:
            errs.append("active scene %r does not exist -- the timeline has %d scene(s)" % (model.get("active"), n_sc))
    for i, t in enumerate(model.get("transitions") or [], start=1):
        if not isinstance(t, dict) or t.get("kind", "cut") not in TRANSITIONS:
            errs.append("transition %d: %r is not one of %s" % (i, (t or {}).get("kind") if isinstance(t, dict) else t, ", ".join(TRANSITION_NAMES)))
            continue
        try:
            d = float(t.get("seconds", DEFAULT_TRANSITION["seconds"]))
        except Exception:
            d = -1.0
        if t.get("kind", "cut") != "cut" and not (0.05 <= d <= 5.0):
            errs.append("transition %d: %.2f s is outside 0.05 .. 5 s" % (i, d))
    seen = {}
    for i, a in enumerate(model.get("anchors") or [], start=1):
        src = str(a.get("src", "")).strip()
        if not src:
            errs.append("anchor %d has no source (@tag from the board, image_1..%d, or @clip:K:last)" % (i, N_IMAGE_SLOTS))
            continue
        if sources is not None and src not in sources:
            cs = clip_src(src)
            if cs is not None:
                errs.append("anchor %d: %s -- %s has no rendered clip in the stack of %s yet (render it first, "
                            "or pick another source)" % (i, src, "the whole timeline" if cs[0] == 0 else "scene %d" % cs[0],
                                                         project_name(model)))
            elif src.startswith("@clip:"):
                errs.append("anchor %d: %s is not a clip frame -- write @clip:K:first, @clip:K:last or "
                            "@clip:K:<seconds> (K = scene, 0 = the whole timeline)" % (i, src))
            else:
                errs.append("anchor %d: %s is not on the board and not wired" % (i, src))
        try:
            at = float(a.get("at", 0))
        except Exception:
            errs.append("anchor %d: 'at' is not a number" % i)
            continue
        if at < 0 or at > dur + 1e-9:
            errs.append("anchor %d at %.3f s is outside the clip (0 .. %.3f s)" % (i, at, dur))
            continue
        f = frame_of(at, frames)
        if f in seen:
            errs.append("anchors %d and %d land on the same frame %d" % (seen[f], i, f))
        seen[f] = i
    return errs


# --------------------------------------------------------------- scenes
def sorted_shots(model):
    return sorted(model.get("shots") or [], key=lambda s: float(s.get("at", 0)))


def scenes(model):
    """[(first_shot_index, end_shot_index_exclusive)] over the sorted shots.
    Shot 1 begins scene 1; a shot flagged `scene` begins the next one."""
    shots = sorted_shots(model)
    if not shots:
        return []
    starts = [0] + [i for i in range(1, len(shots)) if shots[i].get("scene")]
    return [(a, b) for a, b in zip(starts, starts[1:] + [len(shots)])]


def carry_of(model, k):
    """v1049: does scene k (1-based, >= 2) begin on scene k-1's last frame?"""
    if not k or int(k) < 2:
        return False
    sc = scenes(model)
    if int(k) > len(sc):
        return False
    return bool(sorted_shots(model)[sc[int(k) - 1][0]].get("carry"))


def with_carry_anchor(target):
    """The scene's model with the carried frame as its first-frame anchor
    (carry_mode "keyframe"); an own anchor on frame 0 wins (-> unchanged)."""
    frames = frames_for(target["seconds"])
    first, _last, _g = split_anchors(target, frames)
    if first is not None:
        return target, False
    m = json.loads(json.dumps(target))
    m["anchors"] = [{"src": CARRY_SRC, "at": 0.0}] + list(m.get("anchors") or [])
    return m, True


def with_motion_head(target, head=MOTION_FRAMES):
    """v1176: the scene's model with a pinned head of `head` frames (carry_mode "motion"): the length grows by
    it, Shot 1 still opens at 0 (its first frames ARE the head -- the end of the scene before), every later shot
    and every anchor moves back by it; the last-frame anchor stays the last frame of the longer grid; an own
    anchor on frame 0 is dropped (the head owns those frames). -> (model, notes)."""
    hs = int(head) / float(FPS)
    old = frames_for(target["seconds"])
    m = json.loads(json.dumps(target))
    m["seconds"] = float(target["seconds"]) + hs
    m["head"] = int(head)
    new = frames_for(m["seconds"])
    notes = []
    shots = sorted(m.get("shots") or [], key=lambda s: float(s.get("at", 0)))
    for i, s in enumerate(shots):
        if i:
            s["at"] = float(s.get("at", 0)) + hs
    m["shots"] = shots
    an = []
    for a in m.get("anchors") or []:
        f = frame_of(a.get("at", 0), old)
        if f == 0:
            notes.append("motion: the anchor %s on the first frame is left out -- the pinned head owns frame 0"
                         % a.get("src"))
            continue
        a["at"] = (new - 1) / float(FPS) if f == old - 1 else float(a.get("at", 0)) + hs
        an.append(a)
    m["anchors"] = an
    return m, notes


CONTEXT_HOLD_S = 2.0          # v1178: no cut before this many seconds after the join (unmeasured -- Prompting 67)
CONTEXT_TAIL_S = 1.0          # v1178: no cut in the last second before a join the next scene continues
FINISH_WORDS = ("finishes", "finished", "arrives", "arrived", "settles", "settled", "comes to rest", "comes to a stop",
                "stops", "stopped", "slows", "slowing", "centered", "centred", "lands", "sits down")


def context_notes(target, incoming, outgoing):
    """v1178: the handoff rules of a scene that continues the one before (incoming: it has a pinned head) or is
    continued by the next (outgoing) -- checked against the scene's shots, said in the info, never rewritten.
    The rules (Easy Media's notes on the Motion Context chaining: hold, then change; end in motion) are
    UNMEASURED: Prompting 67 keeps them as open checks. -> list of lines."""
    out = []
    shots = sorted(target.get("shots") or [], key=lambda s: float(s.get("at", 0)))
    head = int(target.get("head", 0) or 0) / float(FPS)
    end = float(target.get("seconds", 0))
    if incoming:
        lim = head + CONTEXT_HOLD_S
        for i, s in enumerate(shots[1:], 2):
            at = float(s.get("at", 0))
            if at < lim - 1e-6:
                out.append("context rule: [Shot %d] cuts at %s, %.2f s after the join -- a continued scene opens with a "
                           "hold; a cut before %.1f s tends to come out ON TOP of the pinned head (both framings). "
                           "Unmeasured rule (Prompting 67)" % (i, time_label(at), at - head, CONTEXT_HOLD_S))
        if shots and len(str(shots[0].get("text", "")).strip()) < 12:
            out.append("context rule: [Shot 1] says almost nothing -- a held head with nothing happening tends to "
                       "freeze; give the performer a breath, a weight shift, a glance (unmeasured, Prompting 67)")
    if outgoing:
        for i, s in enumerate(shots[1:], 2):
            at = float(s.get("at", 0))
            if at > end - CONTEXT_TAIL_S + 1e-6:
                out.append("context rule: [Shot %d] cuts at %s, inside the last %.1f s -- the next scene continues "
                           "from these frames; a cut there hands it a fresh framing to hold (unmeasured, Prompting 67)"
                           % (i, time_label(at), CONTEXT_TAIL_S))
        if shots:
            last = str(shots[-1].get("text", "")).lower()
            hit = [w for w in FINISH_WORDS if re.search(r"\b%s\b" % re.escape(w), last)]
            if hit:
                out.append("context rule: the last shot ends in '%s' -- the next scene continues the motion; end IN "
                           "motion, not at rest (unmeasured, Prompting 67)" % "', '".join(hit[:3]))
    return out


def scene_span(model, k):
    """(t0, t1) seconds of scene k (1-based) inside the whole timeline."""
    sc = scenes(model)
    shots = sorted_shots(model)
    a, b = sc[k - 1]
    t0 = float(shots[a].get("at", 0))
    t1 = float(shots[b].get("at", 0)) if b < len(shots) else frames_for(model["seconds"]) / float(FPS)
    return t0, t1


def project_name(model):
    """A folder-safe project name ('cine' when empty)."""
    raw = str(model.get("project", "") or "").strip()
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw).strip("_")
    return safe or "cine"


DAY_PREFIX = "%date:yyyy-MM-dd%/PH_%date:hhmmss%_"


def clip_prefix(model, k):
    """The Save node's filename_prefix for scene k (0 = the whole timeline). v1152 (Frank 17.10.: "everything
    in the folder of the day -- no task folders"): %date:yyyy-MM-dd%/PH_%date:hhmmss%_<project>_scene_<k> or
    ..._whole -- the Save expands the tokens, so a wired Save writes into the day folder like every other."""
    return "%s%s_%s" % (DAY_PREFIX, project_name(model), "scene_%d" % k if k else "whole")


def scene_model(model, k):
    """Scene k as a timeline of its own: shots re-based to 0, the length =
    the scene's span (snapped up to the grid by the compiler), the anchors
    inside the span re-based and clamped to the scene's last frame."""
    t0, t1 = scene_span(model, k)
    a, b = scenes(model)[k - 1]
    shots = sorted_shots(model)
    m = json.loads(json.dumps(model))
    m["seconds"] = t1 - t0
    m["shots"] = []
    for i, s in enumerate(shots[a:b]):
        s2 = dict(s)
        s2["at"] = 0.0 if i == 0 else float(s.get("at", 0)) - t0
        s2.pop("scene", None)
        m["shots"].append(s2)
    # v1152 (Frank 17.10.): the form's places of the timeline's Shot 1 ({pos_1} .. -- the fixed parts the CLIP
    # Text Encode fills) head the [Shot 1] of EVERY run, as the triggers and the look do (head_block); a scene
    # that does not begin with the timeline's Shot 1 would else get them appended OUTSIDE the H3 description
    if a > 0:
        fh = places(shots[0].get("text", ""))
        if fh:
            m["form_head"] = "\n".join("{%s}" % p for p in fh)
    last_scene = k == len(scenes(model))
    whole_frames = frames_for(model["seconds"])
    grid = frames_for(m["seconds"])
    last_t = (grid - 1) / float(FPS)
    asked = max(1, min(grid, asked_frames(m["seconds"])))
    m["anchors"] = []
    for an in model.get("anchors") or []:
        at = float(an.get("at", 0))
        if at < t0 - 1e-9 or at > t1 + 1e-9 or (at >= t1 - 1e-9 and not last_scene):
            continue
        a2 = dict(an)
        # the timeline's LAST-frame keyframe stays the last frame of the last
        # scene, although the scene's own grid may end a few frames later.
        # v1083: the same for EVERY scene -- an anchor on the scene's last
        # asked frame (or in the overhang) is the scene's LAST keyframe, not a
        # guide a few frames before the grid's end (the audit: whether it was
        # keyframe or guide depended on the span lying on the grid)
        if (last_scene and frame_of(at, whole_frames) == whole_frames - 1) or frame_of(at - t0, grid) >= asked - 1:
            a2["at"] = last_t
        else:
            a2["at"] = min(last_t, max(0.0, at - t0))
        m["anchors"].append(a2)
    m["render"] = "whole"
    return m


def resolve(model):
    """What a run renders: (the model to compile, scene index k or 0 for the
    whole timeline, scene count). Call validate() first."""
    n = len(scenes(model))
    if model.get("render", "whole") in ("scene", "all"):
        k = max(1, min(n, int(model.get("active", 1))))
        return scene_model(model, k), k, n
    return model, 0, n


# --------------------------------------------------------------- v1035 H3 transitions
def h3t_errors(t):
    """Errors of one shot's h3t ([] = fine or absent)."""
    if t is None:
        return []
    if not isinstance(t, dict) or t.get("kind") not in H3T_NAMES:
        return ["transition %r is not one of %s" % (t.get("kind") if isinstance(t, dict) else t, ", ".join(H3T_NAMES))]
    errs = []
    try:
        d = float(t.get("seconds", H3T_DEFAULT))
    except Exception:
        d = -1.0
    if not (H3T_MIN - 1e-9 <= d <= H3T_MAX + 1e-9):
        errs.append("%s: %.2f s is outside %.2f .. %.0f s" % (t["kind"], d, H3T_MIN, H3T_MAX))
    if t["kind"] == H3T_CUSTOM and t.get("formula", H3T_FORMULAS[0]) not in H3T_FORMULAS:
        errs.append("H3 custom: formula %r is not one of %s" % (t.get("formula"), " / ".join(H3T_FORMULAS)))
    return errs


def h3t_of(shot):
    """The shot's H3 transition as {kind, seconds, formula, out, in}, or None
    (none set, or the shot starts a scene -- then two clips meet there)."""
    t = shot.get("h3t")
    if not isinstance(t, dict) or t.get("kind") not in H3T_NAMES or shot.get("scene"):
        return None
    p = dict(H3T_PRESETS.get(t["kind"]) or {})
    if t["kind"] == H3T_CUSTOM:
        p = {"formula": t.get("formula", H3T_FORMULAS[0]),
             "out": str(t.get("out", "")).strip(), "in": str(t.get("in", "")).strip()}
    p["kind"] = t["kind"]
    p["seconds"] = min(H3T_MAX, max(H3T_MIN, float(t.get("seconds", H3T_DEFAULT))))
    return p


def span_words(sec, last=False):
    """The asked length as words. within: 0.5 -> 'half a second', 1 -> 'one
    second'; last=True (after 'During the last'): 'half second', 'second'."""
    d = round(float(sec) * 20) / 20.0
    named = {0.25: ("a quarter of a second", "quarter second"), 0.5: ("half a second", "half second"),
             1.0: ("one second", "second")}
    for v, w in named.items():
        if abs(d - v) < 1e-9:
            return w[1 if last else 0]
    return ("%.2f" % d).rstrip("0").rstrip(".") + " seconds"


def _closed(text):
    """The text with a closing mark, so a sentence can follow."""
    t = text.rstrip()
    if t and t[-1] not in ".!?":
        t += "."
    return t


def _cap(text):
    return text[:1].upper() + text[1:] if text else text


def with_out(text, p):
    """Shot text + the sentence that leads OUT of it."""
    if not p or not p.get("out"):
        return text
    tail = "During the last %s of the shot, %s." % (span_words(p["seconds"], last=True), p["out"].rstrip(" ."))
    return (_closed(text) + " " + tail) if text.strip() else _cap(tail)


def with_in(text, p):
    """Shot text with the sentence that leads IN put behind its first
    sentence (the formula names the new image, then it arrives)."""
    if not p or not p.get("in"):
        return text
    lead = "%s within %s." % (_cap(p["in"].rstrip(" .")), span_words(p["seconds"]))
    t = text.strip()
    if not t:
        return lead
    for k, ch in enumerate(t):
        if ch in ".!?" and (k + 1 == len(t) or t[k + 1] == " "):
            # v1084: "Dr. Smith enters." -- the dot of an abbreviation or an
            # initial does not end the sentence (the lead landed after "Dr.")
            word = t[:k].rsplit(" ", 1)[-1]
            if ch == "." and (word.lower() in ABBREVIATIONS or (len(word) == 1 and word.isalpha())):
                continue
            rest = t[k + 1:].strip()
            return t[:k + 1] + " " + lead + ((" " + rest) if rest else "")
    return t + ". " + lead


ABBREVIATIONS = frozenset(("dr", "mr", "mrs", "ms", "prof", "st", "vs", "etc", "no", "jr", "sr", "mt", "ft", "e.g", "i.e", "approx", "fig", "col", "gen", "lt", "sgt", "capt"))


# --------------------------------------------------------------- compile
def split_anchors(model, frames):
    """-> (first, last, guides): the anchor dicts on frame 0 / on the last
    frame, and the rest as (src, frame) in time order."""
    first = last = None
    guides = []
    for a in sorted(model.get("anchors") or [], key=lambda a: float(a.get("at", 0))):
        f = frame_of(a.get("at", 0), frames)
        if f == 0:
            first = a
        elif f == frames - 1:
            last = a
        else:
            guides.append((a["src"], f))
    return first, last, guides


def alignment_line(model, frames, first, last):
    """The official alignment sentence, or "" (T2VA). Wording verbatim from
    10_H3_OFFIZIELL.md section 2; S.SS = frame / 24, two decimals."""
    if first is not None and last is None:
        return ("For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced.")
    t_last = (frames - 1) / float(FPS)
    n_last = shot_at(model, t_last)
    if first is not None and last is not None:
        return ("How the reference pictures align with the target video -- "
                "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target "
                "video; Picture 2 (from Shot %d) aligns with the %s-second mark of the "
                "target video." % (n_last, mark(t_last)))
    if last is not None:
        return ("How the reference pictures align with the target video -- "
                "<Picture 1> (from [Shot %d]) aligns with the %s-second mark of the "
                "target video." % (n_last, mark(t_last)))
    return ""


def _comma(t):
    t = str(t or "").strip()
    return t if not t or t.endswith(",") else t + ","


# v1150 (Frank 17.10.): the parts of a prompt that never change live in the CLIP Text Encode; the timeline's text
# carries their places -- {pos_1} .. {pos_6}, {prompt} -- and the CTE (external_mode "fill {pos_N}") puts its
# segments in. This copy of the CTE's fill is held equal to it by tests/test_v1150_timeline_form.py (the node
# modules stay import-independent, as everywhere in the pack).
PLACE_RX = re.compile(r"\{(prompt|pos_[1-6])\}")


def places(text):
    """The places in a compiled prompt, in order of first appearance."""
    out = []
    for k in PLACE_RX.findall(str(text or "")):
        if k not in out:
            out.append(k)
    return out


def fill_text(text, parts):
    """The CTE's fill: {pos_N} = segment N as typed, {prompt} = all non-empty segments one under the other."""
    raw = {"pos_%d" % (i + 1): str(p or "") for i, p in enumerate(parts)}
    whole = "\n".join(x for x in (str(p or "") for p in parts) if x.strip())
    return PLACE_RX.sub(lambda mo: whole if mo.group(1) == "prompt" else raw.get(mo.group(1), ""), str(text or ""))


STEER_RUBRICS = ("SHOT", "CAMERA", "MOTION", "ACTION", "MOVE", "MOVEMENT", "TRANSITION", "AUDIO", "SOUND",
                 "DIALOGUE")
_RUBRIC_RX = re.compile(r"^\s*//\s*([A-Za-z][A-Za-z0-9_/-]*)")


def split_to_form(model, max_seg=6):
    """v1150 ('-> CTE'): Shot 1's head (triggers, look) and every rubric that does not steer the shot (SCENE,
    LIGHT, SUBJECT, HAIR, WARDROBE, FACE, ...) move into CTE segments; the timeline keeps the steering rubrics
    (STEER_RUBRICS) and a place {pos_k} where each run of moved rubrics stood -- so the filled text is byte for
    byte the old one. -> (new model, [segment texts]). Raises ValueError beyond max_seg runs."""
    shots = sorted(model.get("shots") or [], key=lambda s: float(s.get("at", 0)))
    if not shots:
        return json.loads(json.dumps(model)), []
    hb = head_block(model)
    text = str(shots[0].get("text", "")).strip()
    full = hb + text if hb.endswith("\n") else " ".join(x for x in (hb, text) if x)
    blocks, cur, name = [], [], None
    for ln in full.split("\n"):
        mo = _RUBRIC_RX.match(ln)
        if mo and cur:
            blocks.append((name, cur))
            cur = []
        if mo:
            name = mo.group(1).upper()
        cur.append(ln)
    if cur:
        blocks.append((name, cur))
    pieces, segs, run = [], [], []
    for nm, ls in blocks:
        # v1154 (audit): text before any // rubric (a shot written without rubrics) stays in the timeline -- it is
        # what the shot does; it moved whole into the CTE before and left Shot 1 a bare {pos_1}
        if nm in STEER_RUBRICS or nm is None:
            if run:
                segs.append("\n".join(run))
                pieces.append("{pos_%d}" % len(segs))
                run = []
            pieces.append("\n".join(ls))
        else:
            run += ls
    if run:
        segs.append("\n".join(run))
        pieces.append("{pos_%d}" % len(segs))
    if len(segs) > max_seg:
        raise ValueError("Shot 1 has %d runs of fixed rubrics between steering ones -- the CLIP Text Encode has "
                         "%d segments" % (len(segs), max_seg))
    new = json.loads(json.dumps(model))
    first = min(range(len(new["shots"])), key=lambda i: float(new["shots"][i].get("at", 0)))
    new["shots"][first]["text"] = "\n".join(pieces)
    new["triggers"] = ""
    new["style"] = ""
    return new, segs


def head_block(model):
    """v1050 (Frank 01.10.: "two separate areas, one for the triggers, one for
    the style -- consistency over e.g. three clips rendered one by one"): the
    head of [Shot 1] in EVERY run -- the triggers and the look, each under its
    own // rubric (the CLIP Text Encode strips the rubrics, 0 tokens; every line
    ends on a comma, 40 section 36). Without triggers the old form stays: the
    style alone, as before."""
    trig = _comma(model.get("triggers", ""))
    style = str(model.get("style", "") or "").strip()
    if not trig:
        return style
    return "// TRIGGERS\n" + trig + ("\n// LOOK\n" + _comma(style) if style else "") + "\n"


def compile_prompt(model):
    """-> dict(prompt, frames, seconds, first, last, guides, lines).
    Call validate() first; this assumes a valid model."""
    frames = frames_for(model["seconds"])
    dur = frames / float(FPS)
    shots = sorted(model["shots"], key=lambda s: float(s.get("at", 0)))
    first, last, guides = split_anchors(model, frames)
    parts = []
    ref2va = model.get("target", "fl2va") == "ref2va"
    align = "" if ref2va else alignment_line(model, frames, first, last)
    if align:
        parts.append(align)
    body = []
    # v1035: shot i (0-based, >= 1) may carry an H3 transition; it writes
    # into shot i-1's end and shot i's start, and sets shot i's formula
    h3 = [None] + [h3t_of(s) for s in shots[1:]]
    for i, s in enumerate(shots, start=1):
        text = str(s.get("text", "")).strip()
        p_in = h3[i - 1]
        p_out = h3[i] if i < len(shots) else None
        text = with_out(with_in(text, p_in) if i > 1 else text, p_out)
        if i == 1:
            fh = str(model.get("form_head", "") or "")
            if fh and not places(text):
                text = fh + ("\n" + text if text else "")
            hb = head_block(model)
            body.append("[Shot 1] " + (hb + text if hb.endswith("\n") else " ".join(x for x in (hb, text) if x)))
        else:
            cut = p_in["formula"] if p_in else s.get("cut", CUTS[0])
            lead = "" if cut == "(none)" else cut + " "
            body.append("[Shot %d] At %s, %s%s" % (i, time_label(s["at"]), lead, text))
    parts.append("integrated_multimodal_description: " + "\n\n".join(body))
    sound = str(model.get("soundscape", "")).strip()
    if sound:
        parts.append("overall_soundscape: " + sound)
    music = str(model.get("music", "")).strip() or "N/A"
    parts.append("non_diegetic_music: " + music)
    prompt = "\n\n".join(parts)

    lines = ["Cine Timeline: %.2f s asked -> %d frames = %.3f s (17k+5 at %d fps), %d shot(s)"
             % (float(model["seconds"]), frames, dur, FPS, len(shots))]
    trig = [x for x in (t.strip() for t in str(model.get("triggers", "") or "").split(",")) if x]
    if trig:
        lines.append("  head of [Shot 1]: %d trigger(s) + the look -- the same in every run of this timeline "
                     "(scene runs share them)" % len(trig))
    if model.get("form_head"):
        lines.append("  head of [Shot 1]: the form's places %s of the timeline's Shot 1 -- the CLIP Text Encode "
                     "fills them (every run carries them)" % " ".join("{%s}" % p for p in places(model["form_head"])))
    if dur < MIN_S or dur > MAX_S:
        lines.append("WARN: %.2f s is outside the official 4-15 s" % dur)
    for i, s in enumerate(shots, start=1):
        end = shots[i]["at"] if i < len(shots) else dur
        lines.append("  [Shot %d] %s - %s  (%.2f s)" % (i, time_label(s["at"]), time_label(end),
                                                      float(end) - float(s["at"])))
        p = h3[i - 1]
        if p:
            lines.append("    <- %s at the cut, %.2f s asked on each side (H3 paints it and decides the exact length)"
                         % (p["kind"], p["seconds"]))
        elif i > 1 and s.get("scene") and isinstance(s.get("h3t"), dict):
            lines.append("    note: this shot starts a scene -- its %s is not used (two clips meet here; "
                         "the scene transition applies)" % s["h3t"].get("kind"))
    if first is not None and ref2va:
        lines.append("  first frame: %s at frame 0 -> first_frame -> MiniMax Reference first_frame (ref2va: "
                     "pinned, not a picture of the prompt -- no alignment line)" % first["src"])
    elif first is not None:
        lines.append("  keyframe first: %s at frame 0 -> Keyframes first_frame (<Picture 1>)" % first["src"])
    if last is not None and ref2va:
        lines.append("  WARN: last frame %s -- ref2va has no last-frame input; switch the timeline's model to "
                     "fl2va (MiniMax Keyframes) or move the anchor off the last frame" % last["src"])
    elif last is not None:
        lines.append("  keyframe last: %s at frame %d -> Keyframes last_frame (<Picture %d>)"
                     % (last["src"], frames - 1, 2 if first is not None else 1))
    for src, f in guides:
        lines.append("  guide: %s at frame %d (%.2f s) -> Continue anchors (Core master); "
                     "not shown to the text encoder, so the prompt does not name it" % (src, f, f / float(FPS)))
    if not sound:
        lines.append("WARN: no overall_soundscape -- H3 then invents the sound")
    return {"prompt": prompt, "frames": frames, "seconds": dur,
            "first": first, "last": last, "guides": guides, "lines": lines}
