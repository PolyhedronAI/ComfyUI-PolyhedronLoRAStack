# -*- coding: ascii -*-
"""cine_edit -- the cutting room's EDIT PLAN (v1036 Cine V1-V4, v1037 the
PIECES: split, heal, stretch). Pure: no ComfyUI, no files; the caller hands
in what the clips are.

Two tracks, one time axis (Frank, 29.09.):
  the PROMPT track  -- shots, cut times, H3 transitions: goes INTO the run;
  the VIDEO track   -- the rendered clips under the shots, ffmpeg transitions
                       on their seams, trims, splits, tempo: AFTER the run.

THE FILM IS A LIST OF PIECES (v1037). A piece plays source frames [a, b) of
one clip (k: 0 = the whole run, 1..n = scene k) at a speed:

    {"k", "a", "b", "speed", "tr"}     film length L = round((b - a) / speed)

Pieces follow each other; the film's cuts are their running sums. A piece
can be SPLIT at any frame (two pieces of the same clip, [a, s) + [s, b)),
HEALED again (the two touch: same clip, b == a, same speed), TRIMMED (a seam
moves: one piece gives what the other takes), STRETCHED (its speed; the
pieces after it move), SLIPPED (a and b move together).

THE SOUND (v1038, Frank 29.09.): linked by default -- a piece's sound plays
where its picture plays. A seam can carry an AUDIO OFFSET ("ao" of the piece
after it, film frames): the sound's cut sits at the picture's cut + ao. ao > 0
= the sound of the piece before runs on under the next picture (L-cut), ao < 0
= the sound of the piece after starts under the picture before (J-cut). The
sound stays in sync with its own picture (same source window, just longer or
earlier) -- the NLE split edit, not a slide. The audio crossfade of a
transition sits centred on the SOUND's cut. The plan returns "asegments"
(one per piece, like "segments", with pad_head/pad_tail = silence where the
material ends) and each seam's "ao" / "a_at".

A seam carries a transition ("tr" of the piece after it; none = a cut). The
blend sits CENTRED on the cut: the piece before plays on past its b, the one
after starts before its a -- material the 17k+5 grid leaves anyway (a 2.00 s
scene comes back as 56 frames). Where the material ends, the picture is
held (named). Without pieces in the edit, they come from the prompt: one per
scene (or the whole run), the v1036 edit (cuts, end, slip, keep_tail) taken
over -- the plan returns them, so the editor always edits the effective list.

Everything counts FRAMES at 24 fps -- the browser's player and the ffmpeg
export read the same numbers.

THE EDIT (node.properties.pls_cine_edit -- NOT the timeline widget: a fine
cut after a render must not change the node's input, or ComfyUI renders the
whole graph again):
  pieces     {"whole": [...], "scenes": [...]}  per render kind (v1037)
  takes      {k: rel}  the take of scene k (rel to output/); absent = newest
  cuts / end / keep_tail / slip   the v1036 form, read when there are no pieces
"""

import math

FPS = 24


def film_frames(n, fps):
    """v1195: how many FILM frames (at FPS) a clip of n frames at `fps` becomes -- the number ffmpeg's fps filter
    emits, which the export runs on every clip (fps=24,trim...). MEASURED against ffmpeg on 25 clips (48, 25, 30,
    60, 120, 96, 16, 12, 23.976 fps, odd and even counts): floor(n * FPS / fps + 0.5), i.e. round HALF UP.
    Python's round() rounds half to EVEN -- and an interpolated clip (2(N-1)+1 frames at 2x fps) lands on .5 every
    time: 281 at 48 fps gave 140 instead of 141, the export held a picture at every seam of Frank's neon film
    (20.10.). One law, one place; clip_info and every reader count by it."""
    n = int(n)
    fps = float(fps) if fps else float(FPS)
    if n <= 0 or fps <= 0:
        return 0
    if abs(fps - FPS) < 0.01:
        return n
    return max(1, int(math.floor(n * FPS / fps + 0.5)))


MIN_SEG = 2          # frames: the shortest piece
SPEED_MIN, SPEED_MAX = 0.25, 4.0
PIECE_MAX = FPS * 60 * 60     # v1082: a piece longer than an hour of frames is a stored error, not a cut
# v1043 -- EMPTINESS IS A PIECE (the layers, Frank 30.09.: "cut areas can be
# deleted and are empty then; the others can be moved left and right"). A gap
# piece plays nothing: black and silence on the bottom, the layers below show
# through everywhere else. It keeps every layer a gapless chain, so split,
# heal, trim, stretch and slip work on a layer exactly as on the film. Its a
# sits far from 0 so a trim can grow it at both ends; norm_layer() puts it
# back (a = GAP_A, speed 1) after every op.
SPAN_MAX = 3600.0            # v1047: the longest film (seconds) -- JS SPAN_MAX
GAP = -1
GAP_A = 100000


def rnd(x):
    """Half up, as the browser's Math.round -- Python's round() goes to the
    even number (22.5 -> 22, JS 23): the editor and the plan counted a piece
    at 50% / 200% one frame apart (found by the v1038 parity guard)."""
    return int(math.floor(float(x) + 0.5))


def fr(sec):
    return rnd(float(sec) * FPS)


def _num(x, default=None):
    try:
        v = float(x)
        return v if v == v else default
    except Exception:
        return default


def _clean_piece(p):
    if not isinstance(p, dict):
        return None
    try:
        k, a, b = int(p["k"]), int(p["a"]), int(p["b"])
    except Exception:
        return None
    sp = _num(p.get("speed"), 1.0)
    if k == GAP and b - a >= 1:
        return {"k": GAP, "a": GAP_A, "b": GAP_A + _length({"a": a, "b": b, "speed": min(SPEED_MAX, max(SPEED_MIN, sp or 1.0))}),
                "speed": 1.0}
    if k < 0 or a < 0 or b - a < 1 or sp is None:
        return None
    out = {"k": k, "a": a, "b": b, "speed": min(SPEED_MAX, max(SPEED_MIN, sp))}
    ao = _num(p.get("ao"), 0.0)
    if ao:
        out["ao"] = rnd(ao)
    tr = p.get("tr")
    if isinstance(tr, dict) and isinstance(tr.get("kind"), str):
        out["tr"] = {"kind": tr["kind"], "seconds": _num(tr.get("seconds"), 0.5)}
    return out


def clean_edit(e):
    """The edit dict with only known, well-typed keys."""
    e = e if isinstance(e, dict) else {}
    out = {}
    if isinstance(e.get("cuts"), list) and all(_num(c) is not None for c in e["cuts"]):
        out["cuts"] = [float(c) for c in e["cuts"]]
    if _num(e.get("end")) is not None and float(e["end"]) > 0:
        out["end"] = float(e["end"])
    if e.get("keep_tail"):
        out["keep_tail"] = True
    for key in ("slip", "takes"):
        if isinstance(e.get(key), dict):
            d = {}
            for k, v in e[key].items():
                try:
                    kk = int(k)
                except Exception:
                    continue
                if key == "slip" and _num(v) is not None and float(v) > 0:
                    d[str(kk)] = float(v)
                elif key == "takes" and isinstance(v, str) and v:
                    d[str(kk)] = v
            if d:
                out[key] = d
    if isinstance(e.get("pieces"), dict):
        pc = {}
        for mode in ("whole", "scenes"):
            lst = e["pieces"].get(mode)
            if isinstance(lst, list) and lst:
                ps = [_clean_piece(p) for p in lst]
                if all(ps):
                    pc[mode] = ps
        if pc:
            out["pieces"] = pc
    if e.get("auto_track") is False:
        out["auto_track"] = False           # v1080: OFF is the explicit state; a missing key means ON
    elif e.get("auto_track"):
        out["auto_track"] = True            # v1046: new render -> new track (V1's take pinned)
    # v1198 (Frank 21.10.: "die anderen Teile dann auch auf dem EINEN neuen Track, damit sie alle zusammenbleiben"):
    # the third choice of "new render ->" -- every scene of ONE render lands on ONE new track (render_tracks = the
    # render's sweep stamp -> its track id)
    if e.get("auto_mode") == "render" and out.get("auto_track") is not False:
        out["auto_mode"] = "render"
    if isinstance(e.get("render_tracks"), dict):
        rt = {}
        for k, v in list(e["render_tracks"].items())[-RENDER_TRACKS_MAX:]:
            if isinstance(k, str) and k and isinstance(v, str) and v:
                rt[k[:40]] = v[:40]
        if rt:
            out["render_tracks"] = rt
    if _num(e.get("span")) is not None and float(e["span"]) > 0:
        out["span"] = min(float(e["span"]), SPAN_MAX)   # v1047: the film's length, up to an hour
    lay = clean_layers(e.get("layers"))
    if lay:
        out["layers"] = lay
    return out


# --------------------------------------------------------------- v1043 the layers
LAYER_KINDS = ("base", "file", "render", "empty")
MORE_MAX = 64               # v1198: the extra files of one track (a render's scenes on one track)
RENDER_TRACKS_MAX = 32


def _clean_src(x):
    """v1198: one extra file of a track -> {kind, path, at} or None (kind file / render, at = film frame it was laid)."""
    if not isinstance(x, dict) or x.get("kind") not in ("file", "render"):
        return None
    path = x.get("path")
    if not isinstance(path, str) or not path:
        return None
    o = {"kind": x["kind"], "path": path}
    at = _num(x.get("at"))
    if at is not None and at >= 0:
        o["at"] = int(at)
    return o


def clean_layers(v):
    """The layer stack of the edit, TOP FIRST (the head row on screen is the
    top one). Exactly one "base" entry: the render layer, whose pieces stay
    in edit.pieces (the v1037 form -- an old edit is a stack of one). Every
    other layer is one source file and its own pieces (k 0 = that file, GAP
    = empty); v1198: plus optional MORE files (k 1.. = more[0..]). -> a list, or [] when there is nothing but the base."""
    if not isinstance(v, list):
        return []
    out, ids, base = [], set(), False
    for x in v:
        if not isinstance(x, dict):
            continue
        src = x.get("src") if isinstance(x.get("src"), dict) else {}
        kind = src.get("kind") if src.get("kind") in LAYER_KINDS else ("base" if x.get("id") == "base" else None)
        if kind is None:
            continue
        lid = "base" if kind == "base" else str(x.get("id") or "")
        if not lid or lid in ids or (kind == "base" and base):
            continue
        L = {"id": lid, "src": {"kind": kind}, "video": x.get("video", True) is not False,
             "audio": x.get("audio", True) is not False, "lock": bool(x.get("lock"))}
        if x.get("solo"):
            L["solo"] = True                # v1045: only the solo tracks sound
        # v1153 (Frank 17.10.: "V and A held together ... unlinked, each can be moved alone"): the pair's clamp.
        # Unlinked, the layer's sound may stand shifted against its picture (aoff, film frames; + = later)
        if x.get("link") is False and kind != "base":
            L["link"] = False
        aoff = _num(x.get("aoff"), 0)
        if aoff and kind not in ("base", "empty"):
            L["aoff"] = max(-AOFF_MAX, min(AOFF_MAX, int(round(aoff))))
        if isinstance(x.get("name"), str) and x["name"]:
            L["name"] = x["name"][:120]
        if kind == "base":
            base = True
        elif kind == "empty":
            L["pieces"] = []            # v1044: opened by hand, a clip comes later
        else:
            path = src.get("path")
            if not isinstance(path, str) or not path:
                continue
            L["src"]["path"] = path
            ps = [_clean_piece(p) for p in (x.get("pieces") or [])]
            if not all(ps):
                continue
            # v1198: a track may carry MORE files -- piece k 0 = src, k i = more[i-1] (a render's scenes on one
            # track); a piece naming a file the track does not have becomes empty
            more = [m for m in (_clean_src(m) for m in (x.get("more") or [])[:MORE_MAX]) if m] if isinstance(x.get("more"), list) else []
            if more:
                L["more"] = more
            if _num(src.get("at")) is not None and _num(src.get("at")) >= 0:
                L["src"]["at"] = int(_num(src.get("at")))
            ps = [p if is_gap(p) or 0 <= p["k"] <= len(more) else dict(p, k=GAP, a=GAP_A, b=GAP_A + max(1, _length(p))) for p in ps]
            # v1044: every piece deleted -- the layer stays (its row, its file;
            # the reset brings the clip back), it just shows nothing
            L["pieces"] = norm_layer(ps)
        ids.add(lid)
        out.append(L)
    if not base:
        out.append({"id": "base", "src": {"kind": "base"}, "video": True, "audio": True, "lock": False})
    # v1045: the render is V1 / A1 -- the bottom of the stack
    out = [x for x in out if x["id"] != "base"] + [x for x in out if x["id"] == "base"]
    return out if len(out) > 1 or not (out[0]["video"] and out[0]["audio"] and not out[0]["lock"] and not out[0].get("solo")) else []


AOFF_MAX = 3600 * FPS


def shift_sound(tr, aoff):
    """v1153: a layer's track with its sound shifted by aoff film frames against the picture (the pair unlinked
    and the sound dragged alone). The asegments move as one, the seams' a_at with them; sound pushed before
    frame 0 is cut off (the export trims it the same way). -> a new track dict; aoff 0 = the track itself."""
    d = int(aoff or 0)
    if not d:
        return tr
    out = dict(tr)
    out["aoff"] = d
    asegs = []
    for q in tr.get("asegments") or []:
        q2 = dict(q)
        q2["film_in"] = int(q["film_in"]) + d
        q2["film_out"] = int(q["film_out"]) + d
        asegs.append(q2)
    out["asegments"] = asegs
    seams = []
    for sm in tr.get("seams") or []:
        s2 = dict(sm)
        s2["a_at"] = int(sm["a_at"] if sm.get("a_at") is not None else sm["at"]) + d
        seams.append(s2)
    out["seams"] = seams
    return out


def layer_ranges(pieces):
    """[(film_in, film_out)] of a layer's material (its non-gap pieces)."""
    out, acc = [], 0
    for p in pieces:
        n = _length(p)
        if not is_gap(p):
            out.append((acc, acc + n))
        acc += n
    return out


def gap_piece(n):
    """n film frames of nothing."""
    return {"k": GAP, "a": GAP_A, "b": GAP_A + max(1, int(n)), "speed": 1.0}


def is_gap(p):
    return int(p.get("k", 0)) == GAP


def norm_layer(pieces):
    """After every op: gaps back to a = GAP_A at speed 1 (their film length
    kept), two gaps side by side become one, a gap at the END goes (a layer
    ends where its last material ends). Pieces are copied."""
    out = []
    for p in pieces:
        if is_gap(p):
            if int(p.get("b", 0)) - int(p.get("a", 0)) <= 0:
                continue                    # v1154: a gap closed to nothing goes (the JS twin does the same)
            n = _length(p)
            if out and is_gap(out[-1]):
                out[-1] = gap_piece(_length(out[-1]) + n)
            else:
                out.append(gap_piece(n))
        else:
            out.append(dict(p))
    while out and is_gap(out[-1]):
        out.pop()
    return out


def place_clip(frames, at):
    """A new layer: the whole clip (frames at 24 fps), starting at film frame
    `at` (a gap before it)."""
    ps = ([gap_piece(at)] if int(at) > 0 else []) + [{"k": 0, "a": 0, "b": max(1, int(frames)), "speed": 1.0}]
    return norm_layer(ps)


def delete_piece(pieces, j):
    """Piece j becomes empty for its film length -- the layers below show
    through (Frank: "cut areas can be deleted and are empty then")."""
    out = [dict(p) for p in pieces]
    if 0 <= j < len(out) and not is_gap(out[j]):
        out[j] = gap_piece(_length(out[j]))
    return norm_layer(out)


def move_piece(pieces, j, d):
    """Piece j moved d film frames along its layer: the gap on one side gives
    what the gap on the other takes -- up to the next piece (never over it).
    At the head of the layer a gap is made; at the end the layer grows."""
    out = [dict(p) for p in pieces]
    if not (0 <= j < len(out)) or is_gap(out[j]):
        return norm_layer(out)
    d = int(d)
    before = _length(out[j - 1]) if j > 0 and is_gap(out[j - 1]) else 0
    after_gap = j + 1 < len(out) and is_gap(out[j + 1])
    last = j == len(out) - 1
    if d < 0:
        d = max(d, -before)
    elif d > 0 and not last:
        d = min(d, _length(out[j + 1]) if after_gap else 0)
    if d == 0:
        return norm_layer(out)
    n = [] + out[:j]
    if j > 0 and is_gap(out[j - 1]):
        g = before + d
        n = out[:j - 1] + ([gap_piece(g)] if g > 0 else [])
    elif d > 0:
        n = out[:j] + [gap_piece(d)]
    n.append(out[j])
    rest = out[j + 1:]
    if rest and is_gap(rest[0]):
        g = _length(rest[0]) - d
        rest = ([gap_piece(g)] if g > 0 else []) + rest[1:]
    elif rest and d < 0:
        rest = [gap_piece(-d)] + rest
    return norm_layer(n + rest)


def _cover(segs, length, key_in="film_in", key_out="film_out"):
    """[(a, b)] film frames a segment list covers inside [0, length)."""
    out = []
    for s in segs:
        if s.get("gap") or s.get("missing"):
            continue
        a, b = max(0, int(s[key_in])), min(int(length), int(s[key_out]))
        if b > a:
            out.append((a, b))
    return out


def _runs(owner, length):
    """Per-frame owners -> [{film_in, film_out, layer}] (layer None = nothing)."""
    out = []
    for f in range(int(length)):
        o = owner[f]
        if out and out[-1]["layer"] == o:
            out[-1]["film_out"] = f + 1
        else:
            out.append({"film_in": f, "film_out": f + 1, "layer": o})
    return out


def flatten(layers, length):
    """THE STACK -> what the film shows. layers: TOP FIRST, each {video,
    audio, has_audio, track} (track = a plan: segments + asegments). For every
    film frame the TOPMOST layer that is switched on and has material there
    wins -- picture and sound each on their own (a layer's picture off, its
    sound on: the sound plays over the picture below). Nothing there: black /
    silence (layer None). -> {vruns, aruns, vcover, acover} (cover per layer,
    its own material regardless of what lies above -- the export's overlay)."""
    n = int(length)
    vo, ao = [None] * n, [None] * n
    vcov, acov = [], []
    solo = any(L.get("solo") for L in layers)       # v1045: solo = only the solo tracks sound
    for L in layers:
        tr = L.get("track") or {}
        vc = _cover(tr.get("segments") or [], n) if L.get("video", True) else []
        ac = []
        if L.get("audio", True) and L.get("has_audio", True) and (not solo or L.get("solo")):
            for s in tr.get("asegments") or []:
                if s.get("gap") or s.get("missing"):
                    continue
                a = max(0, int(s["film_in"]) + int(s.get("pad_head", 0) or 0))
                b = min(n, int(s["film_out"]) - int(s.get("pad_tail", 0) or 0))
                if b > a:
                    ac.append((a, b))
        vcov.append(vc)
        acov.append(ac)
    for i in range(len(layers) - 1, -1, -1):      # bottom first, the top paints last
        for a, b in vcov[i]:
            for f in range(a, b):
                vo[f] = i
        for a, b in acov[i]:
            for f in range(a, b):
                ao[f] = i
    return {"vruns": _runs(vo, n), "aruns": _runs(ao, n), "vcover": vcov, "acover": acov}


# --------------------------------------------------------------- the pieces
def _seam_kind(transitions, kinds, j):
    """The kind of the prompt's transition at boundary j (0-based: between scene j+1 and j+2): 'cut' unless a
    blend the editor knows is set."""
    t = transitions[j] if 0 <= j < len(transitions) and isinstance(transitions[j], dict) else {}
    kind = str(t.get("kind", "cut") or "cut")
    return kind if kind in kinds else "cut"


def default_pieces(scene_starts, asked_seconds, frames, transitions, kinds, e, whole, warns, tails=(), held=None,
                   notes=None):
    """The pieces the prompt gives (v1036 form of the edit included): one per
    scene, or the whole run. A piece's a is where it shows AT its cut -- the
    blend into it takes the frames before (the v1036 placement).
    v1172 (Frank 19.10.: the videos "hackeln" at the transitions): held = per scene (head, tail[, n]) -- the frozen
    frames a render begins / ends with (measured on the clip; the carried start picture counted into head). At a
    CUT they are left out: the scene before ends on its first frozen frame, the next begins on its first moving
    one. Under a blend nothing is trimmed (the blend hides them). n = the frames as decoded (a container's own count
    can be one off -- the tail is counted back from the frames that are really there)."""
    n = len(frames)
    P = [fr(s) for s in ([] if whole else scene_starts)]
    end = fr(e["end"]) if "end" in e else fr(asked_seconds)
    C = list(P)
    if not whole and "cuts" in e:
        if len(e["cuts"]) == len(P):
            C = [fr(c) for c in e["cuts"]]
        else:
            warns.append("the film's cuts (%d) do not match the scenes (%d boundaries) -- the prompt's cuts are used"
                         % (len(e["cuts"]), len(P)))
    lo = 0
    for j in range(len(C)):
        C[j] = max(lo + MIN_SEG, C[j])
        lo = C[j]
    end = max(end, lo + MIN_SEG)
    bounds = [0] + C + [end]
    pieces = []
    for k in range(n):
        slip = max(0, fr((e.get("slip") or {}).get(str(0 if whole else k + 1), 0)))
        if frames[k] is not None:
            slip = min(slip, max(0, frames[k] - MIN_SEG))
        h1 = 0
        tr = None
        if k > 0:
            t = transitions[k - 1] if k - 1 < len(transitions) and isinstance(transitions[k - 1], dict) else {}
            kind = str(t.get("kind", "cut") or "cut")
            tr = {"kind": kind, "seconds": _num(t.get("seconds"), 0.5)}
            if kind in kinds and kind != "cut":
                d = max(2, fr(tr["seconds"]))
                d = min(d, bounds[k] - bounds[k - 1], bounds[k + 1] - bounds[k])
                h1 = d - (d + 1) // 2 if d >= 2 else 0
        th = tt = 0
        if held and not whole and held[k]:
            # v1181: a REPEATED head (motion context, held[k][3]) is left out at every seam kind -- under a dissolve
            # the same 0.9 s would otherwise be blended over itself; a frozen head only at a plain cut, as before
            repeat = len(held[k]) > 3 and bool(held[k][3])
            if k > 0 and (repeat or _seam_kind(transitions, kinds, k - 1) == "cut"):
                th = int(held[k][0])
            if k < n - 1 and _seam_kind(transitions, kinds, k) == "cut":
                tt = int(held[k][1])
            nk = int(held[k][2]) if len(held[k]) > 2 and held[k][2] else frames[k]
            if nk is not None:                           # a piece keeps at least MIN_SEG frames of its own
                th = max(0, min(th, nk - tt - MIN_SEG - slip))
        a = slip + h1 + th
        L = bounds[k + 1] - bounds[k]
        if k == n - 1 and e.get("keep_tail") and frames[k] is not None:
            L = max(L, frames[k] - a)
        # v1083: the place's prompt reaches into the grid overhang (a last
        # keyframe, a shot after the asked length): the film shows the clip to
        # its end, so what the prompt put there is in the film
        elif (0 if whole else k + 1) in tails and frames[k] is not None and frames[k] - a > L:
            L = frames[k] - a
            if k < n - 1:
                warns.append("scene %d is shown to its last frame (its prompt reaches into the grid overhang) -- "
                             "the film's cut after it lies %d frames after the prompt's" % (k + 1, frames[k] - a - (bounds[k + 1] - bounds[k])))
        b = a + L
        if (th or tt) and frames[k] is not None:
            nk = int(held[k][2]) if len(held[k]) > 2 and held[k][2] else frames[k]
            b = max(a + MIN_SEG, min(b, nk - tt, frames[k]))
            if notes is not None:
                notes.append("scene %d: %s left out (frozen frames at a cut)" % (k + 1, " and ".join(
                    x for x in ("the first %d" % th if th else "", "the last %d" % tt if tt else "") if x)))
        p = {"k": 0 if whole else k + 1, "a": a, "b": b, "speed": 1.0}
        if tr:
            p["tr"] = tr
        pieces.append(p)
    return pieces, P


def _length(p):
    return max(1, rnd((p["b"] - p["a"]) / float(p["speed"])))


def ao_range(pieces, C, seams, fk, j, prev_A):
    """v1038: where the SOUND's cut of seam j (pieces j-1 | j) may sit, as an
    offset from the picture's cut -> (lo, hi) film frames, lo <= 0 <= hi.
    Inside the material (the sound before runs on only as far as its clip
    goes, the sound after starts only as early as its clip begins), and the
    sound between two cuts keeps at least max(MIN_SEG, both blends).
    prev_A = where the sound's cut before sits (0 for the first seam). The
    browser mirrors this (ph_cine_timeline.js aoRange) -- the guard drives
    both."""
    at = C[j - 1]
    d_here = seams[j - 1]["d"]
    d_prev = seams[j - 2]["d"] if j >= 2 else 0
    d_next = seams[j]["d"] if j < len(seams) else 0
    nxt = C[j] if j < len(C) else at
    lo = prev_A + max(MIN_SEG, d_prev, d_here) - at
    hi = nxt - max(MIN_SEG, d_here, d_next) - at
    pa, pb = pieces[j - 1], pieces[j]
    fa = fk.get(pa["k"])
    if fa is not None:
        hi = min(hi, int(math.floor((fa - pa["b"]) / float(pa["speed"]))))
    lo = max(lo, -int(math.floor(pb["a"] / float(pb["speed"]))))
    return min(lo, 0), max(hi, 0)


def audio_segments(pieces, starts, C, seams, fk, warns):
    """v1038: the sound of every piece -- where it plays in the film (its
    picture's place moved by the offsets of the seams beside it, plus the
    halves of the blends), and which source window that is. Where the
    material ends: silence (pad_head / pad_tail), never a held sound."""
    m = len(pieces)
    A = [0]
    for j in range(1, m):
        want = int(pieces[j].get("ao", 0) or 0)
        if is_gap(pieces[j - 1]) or is_gap(pieces[j]):
            want, lo, hi = 0, 0, 0          # v1043: silence has no split edit
        else:
            lo, hi = ao_range(pieces, C, seams, fk, j, A[-1])
        got = max(lo, min(hi, want))
        if got != want:
            warns.append("seam %d: the sound's cut can move %.2f .. %.2f s from the picture's cut -- %.2f s"
                         % (j, lo / float(FPS), hi / float(FPS), got / float(FPS)))
        seams[j - 1]["ao"] = got
        seams[j - 1]["a_at"] = C[j - 1] + got
        A.append(C[j - 1] + got)
    A.append(C[-1])
    out = []
    for j, p in enumerate(pieces):
        h1 = seams[j - 1]["h1"] if j > 0 else 0
        h2 = seams[j]["h2"] if j < m - 1 else 0
        film_in, film_out = A[j] - h1, A[j + 1] + h2
        if is_gap(p):
            out.append({"j": j, "k": GAP, "film_in": film_in, "film_out": film_out, "src_in": 0, "src_out": 0,
                        "speed": 1.0, "pad_head": 0, "pad_tail": 0, "frames": None, "missing": False, "gap": True})
            continue
        sp = float(p["speed"])
        f = fk.get(p["k"])
        src_in = p["a"] + (film_in - starts[j]) * sp
        src_out = src_in + (film_out - film_in) * sp
        pad_head = pad_tail = 0
        if src_in < 0:
            pad_head = int(math.ceil(-src_in / sp - 1e-9))
            src_in = 0.0
        if f is not None and src_out > f:
            pad_tail = int(math.ceil((src_out - f) / sp - 1e-9))
            src_out = float(f)
        if sp == 1.0:
            src_in, src_out = int(round(src_in)), int(round(src_out))
        out.append({"j": j, "k": p["k"], "film_in": film_in, "film_out": film_out, "src_in": src_in, "src_out": src_out,
                    "speed": sp, "pad_head": pad_head, "pad_tail": pad_tail, "frames": f, "missing": f is None})
    return out


def plan(scene_starts, asked_seconds, clips, transitions, kinds, edit=None, whole=False, tails=(), held=None):
    """The film.

    scene_starts   seconds of the scene boundaries in the PROMPT (scene 2..n);
                   [] for one scene or a whole render
    asked_seconds  the timeline's length (the default end)
    clips          one per place: {"frames": int, ...} or None (not rendered);
                   whole=True -> exactly one (the whole run)
    transitions    per scene boundary {"kind", "seconds"} (the prompt's)
    kinds          editor name -> ffmpeg xfade name ("cut" -> no blend)

    -> {"length", "pieces", "segments", "asegments", "seams", "cuts", "cuts_prompt",
        "place_cuts", "frames", "warnings", "complete", "source"} -- frames;
    segment (one per piece): j, k, film_in, film_out (with the blends),
    src_in / src_out (source frames), speed, hold_head / hold_tail / hold
    (film frames of a held picture), frames, missing; seam: j (1-based),
    kind, xfade, d, h1, h2, at, max_d (without a held picture), asked_d,
    ao / a_at (v1038: the sound's cut); asegment (one per piece): film_in,
    film_out, src_in, src_out, speed, pad_head / pad_tail (silence)."""
    e = clean_edit(edit)
    warns = []
    n = 1 if whole else len(scene_starts) + 1
    if len(clips) != n:
        raise ValueError("the plan needs %d clip(s), got %d" % (n, len(clips)))
    frames = [int(c["frames"]) if c else None for c in clips]
    fk = {(0 if whole else i + 1): frames[i] for i in range(n)}
    mode = "whole" if whole else "scenes"
    pieces = (e.get("pieces") or {}).get(mode)
    source = "edit"
    if pieces:
        # v1082: a piece past its clip's end is clamped to it (the "keep tail" of a
        # clip that was not there yet stored 1e9); a piece longer than PIECE_MAX
        # with no clip to measure it against is bad like a piece off its clip
        pieces = [dict(p, b=min(p["b"], fk[p["k"]])) if not is_gap(p) and fk.get(p["k"]) is not None and p["b"] > fk[p["k"]] else p
                  for p in pieces]
        bad = [p for p in pieces if not is_gap(p) and (p["k"] not in fk or (fk[p["k"]] is not None and p["a"] >= fk[p["k"]])
                                                      or p["b"] - p["a"] > PIECE_MAX)]
        if bad:
            warns.append("the cut list does not fit the clips any more (scene %s) -- the prompt's pieces are used; "
                         "'reset cuts' clears it" % ", ".join(str(p["k"]) for p in bad))
            pieces = None
    P = [fr(s) for s in ([] if whole else scene_starts)]
    if not pieces:
        notes = []
        pieces, P = default_pieces(scene_starts, asked_seconds, frames, transitions, kinds, e, whole, warns, tails=tails,
                                   held=held, notes=notes)
        source = "prompt"
    else:
        notes = []
    t = track(pieces, fk, kinds, warns)
    t.update({"cuts_prompt": P, "frames": {str(k): v for k, v in fk.items()}, "source": source, "held": notes})
    return t


def track(pieces, fk, kinds, warns):
    """v1043: ONE chain of pieces -> its segments, sound, seams (the v1037
    plan without the prompt's part; plan() and every layer use it). fk =
    {k: frames at 24 fps or None}; a GAP piece plays nothing: no material,
    no blend and no split edit on its seams. -> {length, pieces, segments,
    asegments, seams, cuts, end, place_cuts, warnings, complete}."""
    pieces = [dict(p) for p in pieces]
    m = len(pieces)
    if not m:
        return {"length": 0, "pieces": [], "segments": [], "asegments": [], "seams": [], "cuts": [], "end": 0,
                "place_cuts": [], "warnings": warns, "complete": True}
    L = [_length(p) for p in pieces]
    C = []
    acc = 0
    for x in L:
        acc += x
        C.append(acc)
    starts = [0] + C[:-1]

    seams = []
    for j in range(1, m):
        t = pieces[j].get("tr") or {}
        kind = str(t.get("kind", "cut") or "cut")
        if is_gap(pieces[j - 1]) or is_gap(pieces[j]):
            kind = "cut"                    # v1043: nothing to blend with
        if kind not in kinds:
            warns.append("seam %d: %r is not a transition -- cut" % (j, kind))
            kind = "cut"
        asked = 0 if kind == "cut" else max(2, fr(_num(t.get("seconds"), 0.5)))
        span = min(L[j - 1], L[j])
        d = asked
        if d > span:
            warns.append("seam %d: %s of %.2f s is longer than the pieces beside it -- %.2f s"
                         % (j, kind, d / float(FPS), span / float(FPS)))
            d = span if span >= 2 else 0
        h2 = (d + 1) // 2
        h1 = d - h2
        # without a held picture: the piece before has (frames - b) past its
        # end, the piece after has a before its start (both at their speed)
        pa, pb = pieces[j - 1], pieces[j]
        fa, fb = fk.get(pa["k"]), fk.get(pb["k"])
        room = []
        if fa is not None and not is_gap(pa):
            room.append(2 * int(math.floor((fa - pa["b"]) / pa["speed"])))
        if fb is not None and not is_gap(pb):
            room.append(2 * int(math.floor(pb["a"] / pb["speed"])))
        max_d = max(0, min(room + [span]))
        seams.append({"j": j, "kind": kind, "xfade": kinds[kind] if kind != "cut" else None, "asked_d": asked,
                      "d": d, "h1": h1, "h2": h2, "at": C[j - 1], "max_d": max_d})

    segs = []
    for j, p in enumerate(pieces):
        h1 = seams[j - 1]["h1"] if j > 0 else 0
        h2 = seams[j]["h2"] if j < m - 1 else 0
        film_in, film_out = starts[j] - h1, C[j] + h2
        if is_gap(p):
            segs.append({"j": j, "k": GAP, "film_in": film_in, "film_out": film_out, "src_in": 0, "src_out": 0,
                         "speed": 1.0, "hold_head": 0, "hold_tail": 0, "hold": 0, "frames": None, "missing": False,
                         "gap": True})
            continue
        sp = float(p["speed"])
        f = fk.get(p["k"])
        src_in = p["a"] - h1 * sp
        src_out = src_in + (film_out - film_in) * sp
        hold_head = hold_tail = 0
        if src_in < 0:
            hold_head = int(math.ceil(-src_in / sp - 1e-9))
            src_in = 0.0
        if f is not None and src_out > f:
            hold_tail = int(math.ceil((src_out - f) / sp - 1e-9))
            src_out = float(f)
        if hold_head or hold_tail:
            what = "the whole clip" if p["k"] == 0 else "clip %d" % p["k"]
            if any(q["k"] == p["k"] for i, q in enumerate(pieces) if i != j):
                what += " (piece %d)" % (j + 1)
            warns.append("%s: its material ends %.2f s early -- the %s picture is held%s"
                         % (what, (hold_head + hold_tail) / float(FPS), "first" if hold_head and not hold_tail else "last",
                            " (under the blend)" if (hold_tail and hold_tail <= h2) or (hold_head and hold_head <= h1) else ""))
        if sp == 1.0:
            src_in, src_out = int(round(src_in)), int(round(src_out))
        segs.append({"j": j, "k": p["k"], "film_in": film_in, "film_out": film_out, "src_in": src_in, "src_out": src_out,
                     "speed": sp, "hold_head": hold_head, "hold_tail": hold_tail, "hold": hold_head + hold_tail,
                     "frames": f, "missing": f is None})
    asegs = audio_segments(pieces, starts, C, seams, fk, warns)
    for k, f in sorted(fk.items()):
        if f is None and any(p["k"] == k for p in pieces) and k != GAP:
            warns.append("scene %d is not rendered yet" % k if k else "the whole timeline is not rendered yet")
    place_cuts = [C[j - 1] for j in range(1, m) if pieces[j]["k"] != pieces[j - 1]["k"]]
    return {"length": C[-1], "pieces": pieces, "segments": segs, "asegments": asegs, "seams": seams, "cuts": C[:-1], "end": C[-1],
            "place_cuts": place_cuts, "warnings": warns, "complete": not any(s["missing"] for s in segs)}


# --------------------------------------------------------------- ffmpeg
def atempo_chain(speed):
    """atempo takes 0.5 .. 2 per stage -- a chain for the rest. v1156 (audit): a speed of 0 or less (or NaN) would
    loop forever -- refused (the callers clamp to SPEED_MIN, the twin in uls_audio_math guards the same)."""
    out, s = [], float(speed)
    if not s > 0.0:
        raise ValueError("atempo: speed %r is not above 0" % speed)
    while s > 2.0:
        out.append(2.0)
        s /= 2.0
    while s < 0.5:
        out.append(0.5)
        s /= 0.5
    out.append(s)
    return out


def filter_graph(p, audio, size=None):
    """-filter_complex for the plan; input i = segment i's file. Every
    segment ends as EXACTLY its film frames (trim after the tempo and the
    held pictures), so the joins add up to the plan's length."""
    parts = []
    asegs = p.get("asegments") or [dict(s, pad_head=s["hold_head"], pad_tail=s["hold_tail"]) for s in p["segments"]]
    for i, s in enumerate(p["segments"]):
        n = s["film_out"] - s["film_in"]
        sp = float(s["speed"])
        a0, a1 = int(math.floor(s["src_in"])), int(math.ceil(s["src_out"]))
        v = "[%d:v]fps=%d,trim=start_frame=%d:end_frame=%d,setpts=PTS-STARTPTS" % (i, FPS, a0, max(a0 + 1, a1))
        if sp != 1.0:
            # tempo: slower = new frames blended between the old ones
            # (ffmpeg framerate), faster = frames dropped (fps)
            v += ",setpts=PTS/%.6f,%s" % (sp, ("framerate=fps=%d" % FPS) if sp < 1.0 else ("fps=%d" % FPS))
        if s["hold_head"]:
            v += ",tpad=start_mode=clone:start=%d" % s["hold_head"]
        v += ",tpad=stop_mode=clone:stop=%d,trim=end_frame=%d,setpts=PTS-STARTPTS" % (s["hold_tail"] + 4, n)
        if size:
            v += ",scale=%d:%d" % size
        v += ",format=yuv420p,settb=1/%d[v%d]" % (FPS, i)
        parts.append(v)
        if audio:
            # v1038: the sound has its own window (the offsets of its seams)
            q = asegs[i]
            na = q["film_out"] - q["film_in"]
            b0, b1 = int(math.floor(q["src_in"])), int(math.ceil(q["src_out"]))
            a = ("[%d:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                 "atrim=start=%.6f:end=%.6f,asetpts=PTS-STARTPTS" % (i, b0 / float(FPS), max(b0 + 1, b1) / float(FPS)))
            if sp != 1.0:
                a += "," + ",".join("atempo=%.6f" % x for x in atempo_chain(sp))
            if q["pad_head"]:
                a += ",adelay=%d:all=1" % int(round(q["pad_head"] * 1000.0 / FPS))
            a += ",apad=whole_dur=%.6f,atrim=end=%.6f,asetpts=PTS-STARTPTS" % (na / float(FPS), na / float(FPS))
            parts.append(a + "[a%d]" % i)
    vp, ap = "[v0]", "[a0]"
    acc = p["segments"][0]["film_out"] - p["segments"][0]["film_in"]
    for i in range(1, len(p["segments"])):
        s, sm = p["segments"][i], p["seams"][i - 1]
        vo, ao = "[vj%d]" % i, "[aj%d]" % i
        if sm["d"] >= 2 and sm["xfade"]:
            parts.append("%s[v%d]xfade=transition=%s:duration=%.6f:offset=%.6f%s"
                         % (vp, i, sm["xfade"], sm["d"] / float(FPS), (acc - sm["d"]) / float(FPS), vo))
            if audio:
                parts.append("%s[a%d]acrossfade=d=%.6f%s" % (ap, i, sm["d"] / float(FPS), ao))
            acc += (s["film_out"] - s["film_in"]) - sm["d"]
        else:
            # concat hands out its own time base -- set it back, or the next
            # xfade refuses the pair (measured: "Failed to configure output pad")
            parts.append("%s[v%d]concat=n=2:v=1:a=0,settb=1/%d,fps=%d%s" % (vp, i, FPS, FPS, vo))
            if audio:
                parts.append("%s[a%d]concat=n=2:v=0:a=1%s" % (ap, i, ao))
            acc += s["film_out"] - s["film_in"]
        vp, ap = vo, ao
    if audio:
        # v1038: the sound must end where the picture ends -- the offsets
        # telescope (0 at both ends), the blends take the same d
        tot = sum(q["film_out"] - q["film_in"] for q in asegs) - sum(
            sm["d"] for sm in p["seams"] if sm["d"] >= 2 and sm["xfade"])
        if tot != acc:
            raise ValueError("the sound makes %d frames, the picture %d" % (tot, acc))
    return ";".join(parts), vp, ap, acc


def ffmpeg_command(exe, paths, out_path, p, audio, preview=False, size=None):
    graph, vout, aout, total = filter_graph(p, audio, size=size)
    if total != p["length"]:
        raise ValueError("the graph makes %d frames, the plan %d" % (total, p["length"]))
    cmd = [exe, "-y", "-hide_banner", "-loglevel", "error"]
    for x in paths:
        cmd += ["-i", x]
    cmd += ["-filter_complex", graph, "-map", vout]
    if audio:
        cmd += ["-map", aout, "-c:a", "aac", "-b:a", "128k" if preview else "192k"]
    cmd += ["-frames:v", str(p["length"]), "-r", str(FPS), "-c:v", "libx264",
            "-crf", "26" if preview else "17", "-preset", "veryfast" if preview else "medium",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    return cmd


def stack_film(entries, length):
    """v1043: the stack -> the film. entries TOP FIRST: {id, name, video,
    audio, has_audio, track (a plan / track()), paths (one per segment, None
    for a gap)}. length = the edit span in film frames (what is exported;
    material past it is shown dimmed, not exported). -> {length, layers,
    flat, complete, warnings}."""
    L = max(1, int(length))
    flat = flatten(entries, L)
    warns = []
    for i, x in enumerate(entries):
        tl = int((x.get("track") or {}).get("length", 0) or 0)
        if tl > L and x["id"] != "base":
            warns.append("layer %s reaches %.2f s past the edit span -- not exported" % (x.get("name") or x["id"], (tl - L) / float(FPS)))
    holes = sum(r["film_out"] - r["film_in"] for r in flat["vruns"] if r["layer"] is None)
    if holes:
        warns.append("%.2f s of the film show no layer -- black%s" % (holes / float(FPS), " and silent" if all(
            r["layer"] is None for r in flat["aruns"] if r["film_in"] < L) else ""))
    complete = all((x.get("track") or {}).get("complete", True) for x in entries if x.get("video") or x.get("audio"))
    return {"length": L, "layers": entries, "flat": flat, "complete": complete, "warnings": warns}


def describe_stack(film):
    """Info lines for a stack: who shows where."""
    lay = film["layers"]
    nm = lambda i: "nothing" if i is None else (lay[i].get("name") or lay[i]["id"])
    out = ["Stack: %d layer(s), film %.3f s" % (len(lay), film["length"] / float(FPS))]
    for r in film["flat"]["vruns"]:
        out.append("  picture %.3f-%.3f s  <- %s" % (r["film_in"] / float(FPS), r["film_out"] / float(FPS), nm(r["layer"])))
    for r in film["flat"]["aruns"]:
        out.append("  sound   %.3f-%.3f s  <- %s" % (r["film_in"] / float(FPS), r["film_out"] / float(FPS), nm(r["layer"])))
    return out + ["WARN: " + w for w in film["warnings"]]


def _between(iv):
    """ffmpeg timeline expression for frame ranges [(a, b)] (frame f sits at
    t = f / FPS; half a frame of margin on both sides)."""
    return "+".join("between(t,%.6f,%.6f)" % ((a - 0.5) / FPS, (b - 0.5) / FPS) for a, b in iv) or "0"


def _layer_chain(parts, tag, tr, inputs, size, audio, has_audio, video=True):
    """One layer's picture and sound as two streams from film frame 0 to the
    layer's end: every piece fitted into size (aspect kept, black bars), a
    GAP as black / silence; the seams as in filter_graph. -> (vlabel, alabel,
    frames)."""
    W, H = size
    segs, seams = tr["segments"], tr["seams"]
    asegs = tr["asegments"]
    fit = ",scale=%d:%d:force_original_aspect_ratio=decrease,pad=%d:%d:(ow-iw)/2:(oh-ih)/2,setsar=1" % (W, H, W, H)
    for i, s in enumerate(segs):
        n = s["film_out"] - s["film_in"]
        lv, la = "[%sv%d]" % (tag, i), "[%sa%d]" % (tag, i)
        if s.get("gap"):
            parts.append("color=c=black:s=%dx%d:r=%d:d=%.6f,trim=end_frame=%d,setpts=PTS-STARTPTS,setsar=1,format=yuv420p,settb=1/%d%s"
                         % (W, H, FPS, (n + 2) / float(FPS), n, FPS, lv))
        else:
            sp = float(s["speed"])
            a0, a1 = int(math.floor(s["src_in"])), int(math.ceil(s["src_out"]))
            v = "[%d:v]fps=%d,trim=start_frame=%d:end_frame=%d,setpts=PTS-STARTPTS" % (inputs[i], FPS, a0, max(a0 + 1, a1))
            if sp != 1.0:
                v += ",setpts=PTS/%.6f,%s" % (sp, ("framerate=fps=%d" % FPS) if sp < 1.0 else ("fps=%d" % FPS))
            if s["hold_head"]:
                v += ",tpad=start_mode=clone:start=%d" % s["hold_head"]
            v += ",tpad=stop_mode=clone:stop=%d,trim=end_frame=%d,setpts=PTS-STARTPTS" % (s["hold_tail"] + 4, n)
            parts.append(v + fit + ",format=yuv420p,settb=1/%d%s" % (FPS, lv))
        if audio:
            q = asegs[i]
            na = q["film_out"] - q["film_in"]
            if q.get("gap") or not has_audio:
                parts.append("anullsrc=r=48000:cl=stereo,aformat=sample_fmts=fltp:channel_layouts=stereo,atrim=end=%.6f,asetpts=PTS-STARTPTS%s"
                             % (na / float(FPS), la))
            else:
                sp = float(s["speed"])
                b0, b1 = int(math.floor(q["src_in"])), int(math.ceil(q["src_out"]))
                a = ("[%d:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                     "atrim=start=%.6f:end=%.6f,asetpts=PTS-STARTPTS" % (inputs[i], b0 / float(FPS), max(b0 + 1, b1) / float(FPS)))
                if sp != 1.0:
                    a += "," + ",".join("atempo=%.6f" % x for x in atempo_chain(sp))
                if q["pad_head"]:
                    a += ",adelay=%d:all=1" % int(round(q["pad_head"] * 1000.0 / FPS))
                a += ",apad=whole_dur=%.6f,atrim=end=%.6f,asetpts=PTS-STARTPTS" % (na / float(FPS), na / float(FPS))
                parts.append(a + la)
    vp, ap = "[%sv0]" % tag, "[%sa0]" % tag
    acc = segs[0]["film_out"] - segs[0]["film_in"]
    for i in range(1, len(segs)):
        s, sm = segs[i], seams[i - 1]
        vo, ao = "[%svj%d]" % (tag, i), "[%saj%d]" % (tag, i)
        if sm["d"] >= 2 and sm["xfade"]:
            parts.append("%s[%sv%d]xfade=transition=%s:duration=%.6f:offset=%.6f%s"
                         % (vp, tag, i, sm["xfade"], sm["d"] / float(FPS), (acc - sm["d"]) / float(FPS), vo))
            if audio:
                parts.append("%s[%sa%d]acrossfade=d=%.6f%s" % (ap, tag, i, sm["d"] / float(FPS), ao))
            acc += (s["film_out"] - s["film_in"]) - sm["d"]
        else:
            parts.append("%s[%sv%d]concat=n=2:v=1:a=0,settb=1/%d,fps=%d%s" % (vp, tag, i, FPS, FPS, vo))
            if audio:
                parts.append("%s[%sa%d]concat=n=2:v=0:a=1%s" % (ap, tag, i, ao))
            acc += s["film_out"] - s["film_in"]
        vp, ap = vo, ao
    d = int(tr.get("aoff", 0) or 0)
    if audio and d:
        # v1153: the whole sound shifted against the picture (the pair unlinked): later = silence in front,
        # earlier = its head cut off
        lab = "[%sash]" % tag
        if d > 0:
            parts.append("%sadelay=%d:all=1%s" % (ap, int(round(d * 1000.0 / FPS)), lab))
        else:
            parts.append("%satrim=start=%.6f,asetpts=PTS-STARTPTS%s" % (ap, -d / float(FPS), lab))
        ap = lab
    if not video:
        # v1154 (audit): a layer that only SOUNDS (its picture off) -- its picture chain ends in a sink; ffmpeg refused
        # the graph with a dangling label ("has an unconnected output")
        parts.append("%snullsink" % vp)
        vp = None
    return vp, ap, acc


def stack_graph(film, size, audio):
    """v1043: the filter graph of a STACK of layers (film from ph_cine_timeline
    edit_plan: length, layers TOP FIRST with track / video / audio /
    has_audio / paths, the flatten() result). Input order = the layers from
    the top, their non-gap segments in order. The picture: black of the
    film's length, every layer laid over it from the bottom up where it has
    material (overlay, enable = its cover) -- the top one wins, as flatten()
    says. The sound: every layer's sound, silent except where flatten() gives
    it the sound, mixed (only one is ever loud). -> (graph, vout, aout, paths)."""
    W, H = size
    L = int(film["length"])
    fl = film["flat"]
    parts, paths = [], []
    chains = {}
    need_v = [i for i, x in enumerate(film["layers"]) if fl["vcover"][i]]
    own_a = {}
    for r in fl["aruns"]:
        if r["layer"] is not None:
            own_a.setdefault(r["layer"], []).append((r["film_in"], r["film_out"]))
    need_a = sorted(own_a) if audio else []
    for i, lay in enumerate(film["layers"]):
        if i not in need_v and i not in need_a:
            continue
        inputs = {}
        for j, s in enumerate(lay["track"]["segments"]):
            if not s.get("gap"):
                inputs[j] = len(paths)
                paths.append(lay["paths"][j])
        chains[i] = _layer_chain(parts, "l%d" % i, lay["track"], inputs, size, i in need_a, lay.get("has_audio", True),
                                 video=i in need_v)
    parts.append("color=c=black:s=%dx%d:r=%d:d=%.6f,setsar=1,format=yuv420p,settb=1/%d[c0]" % (W, H, FPS, (L + 2) / float(FPS), FPS))
    cur, n = "[c0]", 0
    for i in sorted(need_v, reverse=True):            # the bottom layer first
        n += 1
        nxt = "[c%d]" % n
        parts.append("%s%soverlay=eof_action=pass:enable='%s'%s" % (cur, chains[i][0], _between(fl["vcover"][i]), nxt))
        cur = nxt
    parts.append("%strim=end_frame=%d,setpts=PTS-STARTPTS[vout]" % (cur, L))
    aout = None
    if audio:
        if not need_a:
            parts.append("anullsrc=r=48000:cl=stereo,aformat=sample_fmts=fltp:channel_layouts=stereo,atrim=end=%.6f[aout]" % (L / float(FPS)))
        else:
            ins = []
            for i in need_a:
                lab = "[m%d]" % i
                parts.append("%svolume=volume=0:enable='not(%s)'%s" % (chains[i][1], _between(own_a[i]), lab))
                ins.append(lab)
            mix = "".join(ins) + ("amix=inputs=%d:normalize=0:duration=longest:dropout_transition=0," % len(ins) if len(ins) > 1 else "anull,")
            parts.append(mix + "apad=whole_dur=%.6f,atrim=end=%.6f,asetpts=PTS-STARTPTS[aout]" % (L / float(FPS), L / float(FPS)))
        aout = "[aout]"
    return ";".join(parts), "[vout]", aout, paths


def stack_command(exe, film, out_path, size, audio, preview=False):
    graph, vout, aout, paths = stack_graph(film, size, audio)
    cmd = [exe, "-y", "-hide_banner", "-loglevel", "error"]
    for x in paths:
        cmd += ["-i", x]
    cmd += ["-filter_complex", graph, "-map", vout]
    if audio:
        cmd += ["-map", aout, "-c:a", "aac", "-b:a", "128k" if preview else "192k"]
    cmd += ["-frames:v", str(int(film["length"])), "-r", str(FPS), "-c:v", "libx264",
            "-crf", "26" if preview else "17", "-preset", "veryfast" if preview else "medium",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    return cmd


def describe(p):
    """Info lines for the node and the console."""
    L = ["Film: %d frames = %.3f s, %d piece(s)%s%s" % (p["length"], p["length"] / float(FPS), len(p["segments"]),
                                                        "" if p["source"] == "edit" else " (from the prompt)",
                                                        "" if p["complete"] else " -- NOT complete")]
    for i, s in enumerate(p["segments"]):
        if i:
            sm = p["seams"][i - 1]
            L.append("    -- %s%s at %.3f s --%s" % (sm["kind"], "" if not sm["d"] else " %.2f s" % (sm["d"] / float(FPS)),
                                                    sm["at"] / float(FPS),
                                                    "" if not sm.get("ao") else "  sound %s at %.3f s (%s)" % (
                                                        "cuts", sm["a_at"] / float(FPS), "L-cut" if sm["ao"] > 0 else "J-cut")))
        L.append("  %s  film %.3f-%.3f s  <- clip %.3f-%.3f s%s%s" % (
            "whole" if s["k"] == 0 else "scene %d" % s["k"], s["film_in"] / float(FPS), s["film_out"] / float(FPS),
            s["src_in"] / float(FPS), s["src_out"] / float(FPS),
            "" if s["speed"] == 1.0 else "  x%.2f" % s["speed"], " + %d held" % s["hold"] if s["hold"] else ""))
    return L + ["  " + h for h in p.get("held") or []] + ["WARN: " + w for w in p["warnings"]]
