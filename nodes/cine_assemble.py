# -*- coding: ascii -*-
"""cine_assemble -- the cutting room's stack and its join (Cine S6 / E1,
v1028). Pure but for the file system and, in assemble(), one ffmpeg call.

THE STACK. A timeline with scenes renders one scene per run; every clip a
Save of the run writes is listed by path (register) in the project's stack
list -- since v1152 in input/cine_media/<project>/stack/, the paths relative
to output/ (OUTPUT_ROOT), the clips themselves where the Saves wrote them (the
day folder). scene_takes() / scene_files() read the list back (and, for an old
project, its output/<project>/ folder): the NEWEST file of every scene -- the
stack the editor shows and the join takes. Nothing is copied.

THE JOIN. Scene clips are separate films: a hard cut between them has no
H3 continuity (that is inside a clip, between its shots), so the boundary
gets a TRANSITION -- the ones every cutting program has. ffmpeg's xfade
does them all on one filter graph, audio with acrossfade beside it:

    [0:v][1:v] xfade=transition=fade:duration=D1:offset=O1 [v1];
    [v1][2:v]  xfade=...:offset=O2 [v2]; ...

O_i = the running length of what was joined so far, minus D_i. A "cut" is
a one-frame fade (CUT_S): the same graph, no second code path, and one
frame of blend is invisible at 24 fps. Every clip of a project shares size,
fps and codec (one Save node, one model), which is what xfade requires.
"""

import os
import re
import shutil
import subprocess
import threading

FPS = 24
CUT_S = 1.0 / FPS
VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv", ".m4v")
_SCENE_RE = re.compile(r"^scene_(\d+)_\d+_?\.(mp4|webm|mov|mkv|m4v)$", re.I)
_WHOLE_RE = re.compile(r"^whole_\d+_?\.(mp4|webm|mov|mkv|m4v)$", re.I)


# --------------------------------------------------------------- the stack
# v1034: the stack LIST. A run whose Save node was not wired to the
# timeline's clip_prefix wrote its clip somewhere else under output/; the
# editor then reports that file (POST /pls/cine/register) and it is listed
# here, per scene. Paths are stored relative to output/ (the folder's
# parent), never outside it; a listed file that is gone is ignored.
STACK_LIST = "stack.json"

# v1152 (Frank 17.10.: "everything in the folder of the day -- no task folders"): the stack list no longer lives
# in output/<project>/ but beside the project's media in input/; its paths stay relative to OUTPUT/. The caller
# (ph_cine_timeline.project_folder) names that root here; unset, the folder's parent is the root (as before).
OUTPUT_ROOT = None


def _root(folder):
    if OUTPUT_ROOT:
        return os.path.realpath(OUTPUT_ROOT)
    return os.path.realpath(os.path.dirname(os.path.abspath(folder)))


def _list_path(folder):
    return os.path.join(folder, STACK_LIST)


def read_list(folder):
    """{k: [rel paths, oldest first]} from folder/stack.json ({} when none)."""
    import json
    try:
        with open(_list_path(folder), "r", encoding="utf-8") as f:
            d = json.load(f)
        return {int(k): [str(x) for x in v if isinstance(x, str)] for k, v in (d.get("scenes") or {}).items()}
    except Exception:
        return {}


def register(folder, k, full_path, keep=20, primary=False):
    """Add a clip to the list for scene k (0 = whole). The clip must be a
    video file INSIDE output/ (_root: OUTPUT_ROOT, else the folder's parent). -> the stored rel path.
    v1050: primary=True marks it THE clip of its run (the editor picks it at
    the run's end: the Save wired to clip_prefix, else the run's last video)
    -- it then leads the scene's takes (see _order)."""
    root = _root(folder)
    full = os.path.realpath(full_path)
    if not (full == root or full.startswith(root + os.sep)):
        raise ValueError("not inside output/: %s" % full_path)
    if not os.path.isfile(full):
        raise ValueError("no such file: %s" % full_path)
    if not full.lower().endswith(VIDEO_EXTS):
        raise ValueError("not a video clip: %s" % os.path.basename(full))
    k = int(k)
    if k < 0 or k > 999:
        raise ValueError("scene %r out of range" % k)
    rel = os.path.relpath(full, root).replace(os.sep, "/")
    with _LIST_LOCK:
        lst = read_list(folder)
        items = [x for x in lst.get(k, []) if x != rel] + [rel]
        lst[k] = items[-int(keep):]
        prim = read_primary(folder)
        if primary:
            import time
            prim[k] = {"rel": rel, "at": time.time()}
        _write_list(folder, lst, prim)
    return rel


# v1154 (audit): register (the aiohttp loop) and migrate (the run's thread) both read, change and write the whole
# list -- one lock around each read..replace, and a temp file of its own per write (a fixed name collided on Windows)
_LIST_LOCK = threading.RLock()


def _write_list(folder, lst, prim):
    import json
    import tempfile
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=STACK_LIST + ".", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"v": 1, "scenes": {str(a): b for a, b in sorted(lst.items())},
                       "primary": {str(a): b for a, b in sorted(prim.items())}}, f, indent=1)
        os.replace(tmp, _list_path(folder))
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def migrate(legacy, folder):
    """v1152: a project's OLD folder output/<project>/ -> the stack list in `folder`. Its list and primaries and
    its scene_k_* / whole_* clips are listed by path (relative to the same root) -- each call adds what the new
    list lacks, so a clip that still lands there later is not lost. Nothing is moved or deleted; the new list is
    written only when something was added. -> True when it was written."""
    if not legacy or not os.path.isdir(legacy):
        return False
    with _LIST_LOCK:
        return _migrate(legacy, folder)


def _migrate(legacy, folder):
    root = _root(folder)
    lst = read_list(folder)
    prim = read_primary(folder)
    have = set(x for v in lst.values() for x in v)
    add = {}
    for k, rels in read_list(legacy).items():
        for rel in rels:
            add.setdefault(k, []).append(rel)
    for name in sorted(os.listdir(legacy)):
        m = _SCENE_RE.match(name)
        k = int(m.group(1)) if m else (0 if _WHOLE_RE.match(name) else None)
        if k is None:
            continue
        full = os.path.realpath(os.path.join(legacy, name))
        if full.startswith(root + os.sep):
            add.setdefault(k, []).append(os.path.relpath(full, root).replace(os.sep, "/"))
    changed = False
    for k, rels in add.items():
        for rel in rels:
            if rel not in have:
                lst.setdefault(k, []).append(rel)
                have.add(rel)
                changed = True
    for k, v in read_primary(legacy).items():
        if k not in prim:
            prim[k] = v
            changed = True
    if changed:
        _write_list(folder, lst, prim)
    return changed


def read_primary(folder):
    """v1050: {k: {"rel", "at"}} -- the clip the editor picked per scene."""
    import json
    try:
        with open(_list_path(folder), "r", encoding="utf-8") as f:
            d = json.load(f)
        out = {}
        for k, v in (d.get("primary") or {}).items():
            if isinstance(v, dict) and isinstance(v.get("rel"), str):
                out[int(k)] = {"rel": v["rel"], "at": float(v.get("at") or 0)}
        return out
    except Exception:
        return {}


# a file of the same run may be written a moment AFTER the pick is stored
PRIMARY_GRACE = 2.0


def _order(folder, k, timed):
    """[(mtime, full)] -> full paths NEWEST first, the scene's PRIMARY clip in
    front when no file is newer than the pick (a later run -- even one the
    editor did not see -- wins by its own time again)."""
    out = [f for _, f in sorted(timed, key=lambda x: (-x[0], x[1]))]
    pr = read_primary(folder).get(int(k))
    if not pr or not out:
        return out
    root = _root(folder)
    pf = os.path.realpath(os.path.join(root, pr["rel"]))
    if pf not in [os.path.realpath(x) for x in out]:
        return out
    newest_other = max((mt for mt, f in timed if os.path.realpath(f) != pf), default=0.0)
    if newest_other > pr["at"] + PRIMARY_GRACE:
        return out
    return [x for x in out if os.path.realpath(x) == pf] + [x for x in out if os.path.realpath(x) != pf]


def _listed(folder, count):
    """{k: [full paths that still exist]} from the list, scenes 1..count + 0."""
    root = _root(folder)
    out = {}
    for k, rels in read_list(folder).items():
        if k > int(count) and k != 0:
            continue
        for rel in rels:
            full = os.path.realpath(os.path.join(root, rel))
            if (full.startswith(root + os.sep)) and os.path.isfile(full):
                out.setdefault(k, []).append(full)
    return out


def scene_takes(folder, count):
    """v1036: {k: [full paths, NEWEST first]} -- every take of scenes
    1..count (+ 0 = whole renders) from the folder and the stack list."""
    listed = _listed(folder, count) if folder and os.path.isdir(folder) else {}
    if not folder or not os.path.isdir(folder):
        return {}
    by = {}
    for name in os.listdir(folder):
        m = _SCENE_RE.match(name)
        k = int(m.group(1)) if m else (0 if _WHOLE_RE.match(name) else None)
        if k is None or (k > int(count) and k != 0):
            continue
        by.setdefault(k, []).append(os.path.join(folder, name))
    for k, extra in listed.items():
        have = set(os.path.realpath(x) for x in by.get(k, []))
        by.setdefault(k, []).extend(x for x in extra if os.path.realpath(x) not in have)
    out = {}
    for k, cands in by.items():
        timed = []
        for full in cands:
            try:
                timed.append((os.path.getmtime(full), full))
            except OSError:
                continue
        if timed:
            out[k] = _order(folder, k, timed)
    return out


def scene_files(folder, count):
    """{k: {"path", "name", "mtime", "runs"}} for the scenes 1..count that
    have a rendered clip in `folder` (newest per scene), plus key 0 for the
    newest whole-timeline render. Missing scenes are simply absent.
    v1034: clips from the stack list (stack.json) count too -- the newest
    file of a scene wins, wherever it lies under output/."""
    out = {}
    listed = _listed(folder, count) if folder and os.path.isdir(folder) else {}
    if not folder or not os.path.isdir(folder):
        return out
    by = {}
    for name in os.listdir(folder):
        m = _SCENE_RE.match(name)
        k = int(m.group(1)) if m else (0 if _WHOLE_RE.match(name) else None)
        if k is None or (k > int(count) and k != 0):
            continue
        by.setdefault(k, []).append(os.path.join(folder, name))
    for k, extra in listed.items():
        have = set(os.path.realpath(x) for x in by.get(k, []))
        by.setdefault(k, []).extend(x for x in extra if os.path.realpath(x) not in have)
    for k, cands in by.items():
        timed = []
        for full in cands:
            try:
                timed.append((os.path.getmtime(full), full))
            except OSError:
                continue
        if timed:
            first = _order(folder, k, timed)[0]       # v1050: the primary clip leads
            mt = dict((f, m) for m, f in timed)[first]
            out[k] = {"path": first, "name": os.path.basename(first), "mtime": mt, "runs": len(cands)}
    return out


# --------------------------------------------------------------- the plan
def xfade_plan(durations, transitions, kinds):
    """durations: seconds of the clips in order; transitions: one dict per
    boundary ({"kind", "seconds"}; missing = cut); kinds: editor name ->
    ffmpeg name. -> (steps, total): steps = [{"name", "kind", "d", "offset"}]
    -- offset in the JOINED stream so far, the xfade rule."""
    if len(durations) < 2:
        raise ValueError("a join needs at least two clips (%d given)" % len(durations))
    steps = []
    total = float(durations[0])
    for i in range(1, len(durations)):
        t = transitions[i - 1] if i - 1 < len(transitions) and isinstance(transitions[i - 1], dict) else {}
        name = str(t.get("kind", "cut") or "cut")
        if name not in kinds:
            raise ValueError("boundary %d: %r is not a transition of the editor" % (i, name))
        d = CUT_S if name == "cut" else float(t.get("seconds", 0.5))
        # a transition cannot be longer than either neighbour
        d = max(CUT_S, min(d, float(durations[i - 1]) - CUT_S, float(durations[i]) - CUT_S))
        offset = max(0.0, total - d)
        steps.append({"name": name, "kind": kinds[name], "d": round(d, 4), "offset": round(offset, 4)})
        total = total + float(durations[i]) - d
    return steps, total


def filter_graph(n, steps, audio):
    """The ffmpeg -filter_complex string for n inputs and the plan."""
    if len(steps) != n - 1:
        raise ValueError("plan has %d steps for %d clips" % (len(steps), n))
    v_parts, a_parts = [], []
    v_prev, a_prev = "[0:v]", "[0:a]"
    for i, s in enumerate(steps, start=1):
        v_out = "[v%d]" % i if i < n - 1 else "[vout]"
        v_parts.append("%s[%d:v]xfade=transition=%s:duration=%.4f:offset=%.4f%s"
                       % (v_prev, i, s["kind"], s["d"], s["offset"], v_out))
        v_prev = v_out
        if audio:
            a_out = "[a%d]" % i if i < n - 1 else "[aout]"
            a_parts.append("%s[%d:a]acrossfade=d=%.4f%s" % (a_prev, i, s["d"], a_out))
            a_prev = a_out
    return ";".join(v_parts + a_parts)


def ffmpeg_command(exe, paths, out_path, steps, audio, fps=FPS):
    cmd = [exe, "-y", "-hide_banner", "-loglevel", "error"]
    for p in paths:
        cmd += ["-i", p]
    cmd += ["-filter_complex", filter_graph(len(paths), steps, audio), "-map", "[vout]"]
    if audio:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-r", str(int(fps)), "-c:v", "libx264", "-crf", "17", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    return cmd


# --------------------------------------------------------------- the files
def find_ffmpeg():
    """The ffmpeg binary: PATH first, then imageio-ffmpeg's bundled one
    (a plain pip dependency many packs already bring). None = not found."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def probe(path):
    """(seconds, has_audio, width, height, fps) of a video file (PyAV)."""
    import av
    with av.open(path) as c:
        vs = next((s for s in c.streams if s.type == "video"), None)
        if vs is None:
            raise ValueError("%s has no video stream" % os.path.basename(path))
        has_a = any(s.type == "audio" for s in c.streams)
        fps = float(vs.average_rate or vs.guessed_rate or FPS)
        # v1084: the VIDEO stream's length decides (the container's is the
        # longer of video and audio -- an audio tail moved the xfade offsets
        # past the picture, measured: a hard jump instead of a dissolve)
        dur = 0.0
        if vs.frames and fps > 0:
            dur = float(vs.frames) / fps
        if dur <= 0 and vs.duration and vs.time_base:
            dur = float(vs.duration * vs.time_base)
        if dur <= 0 and c.duration:
            dur = float(c.duration) / 1e6
        return dur, has_a, int(vs.codec_context.width), int(vs.codec_context.height), fps


def next_free(folder, stem, ext=".mp4"):
    os.makedirs(folder, exist_ok=True)
    n = 1
    while os.path.exists(os.path.join(folder, "%s_%05d_%s" % (stem, n, ext))):
        n += 1
    return os.path.join(folder, "%s_%05d_%s" % (stem, n, ext))


def assemble(folder, count, transitions, kinds, name="film", exe=None, run=subprocess.run, out_dir=None):
    """Join the newest clip of every scene 1..count in `folder` with the
    transitions -> (out_path, lines). Raises ValueError by name: a missing
    scene, no ffmpeg, a size mismatch, ffmpeg's own error."""
    count = int(count)
    if count < 2:
        raise ValueError("a join needs at least two scenes (the timeline has %d)" % count)
    files = scene_files(folder, count)
    missing = [k for k in range(1, count + 1) if k not in files]
    if missing:
        raise ValueError("scene %s not rendered yet -- no scene_%s_*.mp4 in %s and none in its stack list"
                         % (", ".join(str(k) for k in missing), missing[0], folder))
    exe = exe or find_ffmpeg()
    if not exe:
        raise ValueError("ffmpeg not found (PATH or imageio-ffmpeg) -- the join needs it")
    paths = [files[k]["path"] for k in range(1, count + 1)]
    probes = []
    for k, p in enumerate(paths, start=1):
        try:
            probes.append(probe(p))
        except Exception as e:
            raise ValueError("scene %d: %s cannot be read (%s)" % (k, os.path.basename(p), e))
    sizes = set((w, h) for _, _, w, h, _ in probes)
    if len(sizes) > 1:
        raise ValueError("the scene clips differ in size %s -- one project, one size" % sorted(sizes))
    audio = all(a for _, a, _, _, _ in probes)
    fps = probes[0][4] or FPS
    steps, total = xfade_plan([d for d, _, _, _, _ in probes], transitions, kinds)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (name or "film")) or "film"
    out = next_free(out_dir or os.path.join(folder, "assembled"), safe)   # v1152: the caller names the day folder
    cmd = ffmpeg_command(exe, paths, out, steps, audio, fps=fps)
    r = run(cmd, capture_output=True, text=True)
    if getattr(r, "returncode", 1) != 0 or not os.path.isfile(out):
        raise ValueError("ffmpeg failed: %s" % ((getattr(r, "stderr", "") or "").strip()[-600:] or "no output file"))
    lines = ["Assemble: %d scene(s) -> %s (%.2f s%s)" % (count, out, total, "" if audio else ", video only -- a clip had no audio")]
    for k, p, s in zip(range(1, count + 1), paths, [None] + steps):
        if s is not None:
            lines.append("      -- %s%s --" % (s["name"], "" if s["name"] == "cut" else " %.2f s" % s["d"]))
        lines.append("  %2d  %s (%.2f s)" % (k, os.path.basename(p), probes[k - 1][0]))
    return out, lines
