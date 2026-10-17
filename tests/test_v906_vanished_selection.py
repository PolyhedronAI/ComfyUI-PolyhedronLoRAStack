# -*- coding: ascii -*-
"""Guard -- a listing is also news about what is GONE (v906).

FRANK'S FIELD REPORT (29.08.): "Ich habe gerade aus Versehen zwei Bilder aus
dem Input verschoben, sie waren also weg ... wurden aber im Media-Loader noch
angezeigt in der Preview."

MEASURED CAUSE. Both paths that replace `_files` -- refreshGrid and the silent
focus probe -- rebuilt the GRID from the fresh listing and left the SELECTION
alone. The tiles therefore corrected themselves while the preview did not: its
<img> was already in the DOM and already decoded, so nothing on screen had any
reason to change. The comment sitting on the focus probe said it out loud:

    this.renderGrid();   // page + name-based selection survive on their own

True, and precisely the bug. The selection survived the file.

The backend was never fooled -- ph_media_loader.py raises "the selected file
vanished from disk" -- so no run ever loaded a ghost. Only the picture lied,
which is its own kind of bad: it invites a run you have every reason to trust.

THE PROMISE THIS GUARD PINS, and the one that is easy to get wrong: absence is
evidence ONLY where we looked. A listing of folder A says nothing about a file
pinned in folder B, and the audio slot is routinely cross-folder (there is a
jump-to-folder button for exactly that case). A prune that ignores this would
silently wipe a perfectly good cross-folder audio pick on every focus probe --
a worse bug than the one being fixed, and invisible until someone loses work.

Driven, not read (the v903 lesson): the method is lifted out of the shipped
source and run in plain node against real state objects.
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
JS = os.path.join(ROOT, "web", "js", "ph_media_loader.js")

_FAILED = []
_PASSED = []


def check(label, cond, detail=""):
    if cond:
        _PASSED.append(label)
    else:
        _FAILED.append(label + ((" -- " + detail) if detail else ""))


def _lift(src, sig):
    """Text-lift one method out of the class so it can be driven in plain node."""
    if sig not in src:
        return None
    i = src.index(sig)
    depth, j, started = 0, i, False
    while j < len(src):
        if src[j] == "{":
            depth += 1
            started = True
        elif src[j] == "}":
            depth -= 1
            if started and depth == 0:
                return src[i:j + 1]
        j += 1
    return None


def _run_node(script):
    fd, path = tempfile.mkstemp(suffix=".mjs")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(script)
    try:
        p = subprocess.run(["node", path], capture_output=True, text=True, timeout=40)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    finally:
        os.unlink(path)


src = open(JS, encoding="utf-8").read()

body = _lift(src, "_pruneVanished() {")
if body is None:
    print("[test_v906_vanished_selection] FAIL: cannot lift _pruneVanished "
          "from ph_media_loader.js")
    sys.exit(1)
same = _lift(src, "_samePath(a, b) {")
if same is None:
    print("[test_v906_vanished_selection] FAIL: cannot lift _samePath")
    sys.exit(1)

HARNESS = """
class Loader {
  constructor(st) { Object.assign(this, st); this.synced = 0; this.stopped = 0; }
  _syncAudioToggle() { this.synced++; }
  _stopAudioHover() { this.stopped++; }
  %s
  %s
}
const F = (n) => ({ name: n, mtime: 1 });
const out = [];
const run = (label, st, files, folder) => {
  const L = new Loader(Object.assign({ _files: files, _filesFolder: folder }, st));
  const gone = L._pruneVanished();
  out.push({ label, gone, state: L.state, audioSel: L.audioSel,
             audioOn: L.audioOn, pendingSplit: L._pendingSplit });
};

// 1. the file Frank moved: same folder, gone from the listing
run("same-folder-gone",
    { state: { folder: "C:\\\\in", file: "a.png", kind: "image", mtime: 7 },
      audioSel: null, audioOn: false, _pendingSplit: { file: "a.png" } },
    [F("b.png"), F("c.png")], "C:\\\\in");

// 2. still there -> nothing may move
run("same-folder-present",
    { state: { folder: "C:\\\\in", file: "b.png", kind: "image", mtime: 7 },
      audioSel: null, audioOn: false, _pendingSplit: null },
    [F("b.png"), F("c.png")], "C:\\\\in");

// 3. THE ONE THAT MATTERS: selection lives in ANOTHER folder. This listing
//    cannot testify about it and must leave it completely alone.
run("cross-folder-untouched",
    { state: { folder: "D:\\\\other", file: "a.png", kind: "image", mtime: 7 },
      audioSel: null, audioOn: false, _pendingSplit: null },
    [F("b.png")], "C:\\\\in");

// 4. audio slot, same folder, gone
run("audio-same-folder-gone",
    { state: { folder: "C:\\\\in", file: "b.png", kind: "image", mtime: 7 },
      audioSel: { folder: "C:\\\\in", file: "s.wav", mtime: 3 }, audioOn: true,
      _pendingSplit: null },
    [F("b.png")], "C:\\\\in");

// 5. audio slot pinned in another folder -- the cross-folder pick that the
//    jump-to-folder button exists for. Must survive every probe.
run("audio-cross-folder-untouched",
    { state: { folder: "C:\\\\in", file: "b.png", kind: "image", mtime: 7 },
      audioSel: { folder: "E:\\\\sfx", file: "s.wav", mtime: 3 }, audioOn: true,
      _pendingSplit: null },
    [F("b.png")], "C:\\\\in");

// 6. path spelling must not decide it: same folder, different separators/case
run("same-folder-messy-path",
    { state: { folder: "c:/IN/", file: "a.png", kind: "image", mtime: 7 },
      audioSel: null, audioOn: false, _pendingSplit: null },
    [F("b.png")], "C:\\\\in");

// 7. no listing at all -> no verdict
const L7 = new Loader({ _files: null, _filesFolder: "C:\\\\in",
                        state: { folder: "C:\\\\in", file: "a.png" } });
out.push({ label: "no-listing", gone: L7._pruneVanished(), state: L7.state });

console.log(JSON.stringify(out));
""" % (body, same)

rc, so, se = _run_node(HARNESS)
if rc != 0:
    print("[test_v906_vanished_selection] FAIL: harness did not run: %s" % (se or so))
    sys.exit(1)

import json  # noqa: E402
res = {r["label"]: r for r in json.loads(so)}

r = res["same-folder-gone"]
check("P1 the vanished file is reported", r["gone"] == ["a.png"],
      "got %r" % (r["gone"],))
check("P1 the selection is cleared", not r["state"].get("file"),
      "preview would still show %r" % (r["state"].get("file"),))
check("P1 the folder is kept", r["state"].get("folder") == "C:\\in",
      "the user is still standing there; got %r" % (r["state"].get("folder"),))
check("P1 the pending split goes with it", not r["pendingSplit"],
      "a split armed for a file that no longer exists")

r = res["same-folder-present"]
check("P2 a present file is left alone", r["gone"] == [] and
      r["state"].get("file") == "b.png", "got %r" % (r,))

r = res["cross-folder-untouched"]
check("P3 a selection in ANOTHER folder is NOT dropped",
      r["gone"] == [] and r["state"].get("file") == "a.png",
      "this listing cannot testify about D:\\other; got %r" % (r,))

r = res["audio-same-folder-gone"]
check("P4 a vanished audio pick is cleared",
      r["gone"] == ["s.wav"] and not r["audioSel"], "got %r" % (r,))
check("P4 the audio pane disarms", r["audioOn"] is False,
      "an armed pane with no file is a second kind of ghost")
check("P4 the visual selection is untouched by it",
      r["state"].get("file") == "b.png", "got %r" % (r["state"],))

r = res["audio-cross-folder-untouched"]
check("P5 a cross-folder audio pick SURVIVES",
      r["gone"] == [] and r["audioSel"] and r["audioSel"]["file"] == "s.wav",
      "wiping this would be worse than the bug being fixed; got %r" % (r,))

r = res["same-folder-messy-path"]
check("P6 path spelling does not decide it", r["gone"] == ["a.png"],
      "c:/IN/ and C:\\in are the same folder; got %r" % (r["gone"],))

r = res["no-listing"]
check("P7 no listing yields no verdict",
      r["gone"] == [] and r["state"].get("file") == "a.png",
      "absence of a listing is not absence of a file; got %r" % (r,))


# --- both readers must actually call it -------------------------------------
def _method(sig):
    return _lift(src, sig) or ""


refresh = _method("async refreshGrid(keepPage = false) {")
focus = _method("async _focusReread() {")
check("P8 refreshGrid prunes", "_pruneVanished()" in refresh,
      "the fresh listing is the verdict; without this the tiles update and "
      "the preview does not")
check("P8 the focus probe prunes", "_pruneVanished()" in focus,
      "this is the path Frank hit -- the tab was in the background")
check("P8 refreshGrid repaints when something went",
      "_renderPreview()" in refresh,
      "clearing state without repainting leaves the old <img> on screen -- "
      "the exact symptom")
check("P8 the focus probe repaints when something went",
      "_renderPreview()" in focus)
# ORIGINALLY this pinned the absence of the old comment ("page + name-based
# selection survive on their own"). It went red on its own first run -- and it
# was RIGHT to, just not about the tree: the string now lives in the new
# explanation, which quotes the old promise in order to explain it. A substring
# is not a proof, and prose is not a behaviour. Replaced with the ordering that
# actually matters: the prune has to run BEFORE renderGrid, or the grid draws
# its selection marks from an entry that is already gone.
for _name, _body in (("refreshGrid", refresh), ("the focus probe", focus)):
    _p = _body.find("_pruneVanished()")
    _g = _body.find("renderGrid()")
    check("P8 %s prunes BEFORE it redraws the grid" % _name,
          _p != -1 and _g != -1 and _p < _g,
          "the marks would be drawn from a selection that no longer exists")

# the announcement exists and is reachable from both
ann = _method("_announceVanished(gone) {")
check("P9 there is an announcement", bool(ann))
check("P9 both readers announce",
      "_announceVanished(" in refresh and "_announceVanished(" in focus,
      "a selection that empties itself in silence is the same class of lie")


if _FAILED:
    print("[test_v906_vanished_selection] FAIL (%d of %d):"
          % (len(_FAILED), len(_FAILED) + len(_PASSED)))
    for f in _FAILED:
        print("   - " + f)
    sys.exit(1)
print("[test_v906_vanished_selection] PASS: %d promises -- a vanished file "
      "takes its selection with it, and a listing never passes judgement on a "
      "folder it did not read." % len(_PASSED))
