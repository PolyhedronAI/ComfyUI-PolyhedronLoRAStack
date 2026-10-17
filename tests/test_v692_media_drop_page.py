"""Guard #112 -- the drop actually selects the dropped file (v692).

FRANK'S FIELD REPORT: "wenn ich eine neue Grafik direkt reinziehe, dann
wechselt er auch brav den Ordner im Hintergrund; wenn ich die Tiles oeffne ist
alles da ... aber er wechselt NICHT automatisch das Bild."

MEASURED CAUSE: _dropSelect looked for the TILE in the DOM and clicked it. Two
perfectly normal states make that impossible:

  * renderGrid builds ONE PAGE of GRID_PAGE (20) tiles and a drop resets the
    view to page 1. Frank's folder holds 204 files across 11 pages, so the
    dropped file was only findable when it happened to sort into the first
    twenty -- roughly one drop in ten.
  * in Solo the grid is not built at all (v683 L1). There is never a tile.

Both ended in twenty silent retries over three seconds and then nothing at
all -- no error, no note, just a node that appeared to ignore the drop.

THE FIX, pinned here: the LIST is the truth, not the DOM. Find the file in
_files, page the grid to where it lives so the selection is visible, and go
through select() -- the same call the tile's own click makes, so no logic is
duplicated. And when the file really is not there, SAY so.

DRIVEN in node against a fabricated UI (204 files, page size 20):
  * a file on page 7 pages the grid to 6 (zero-based) and selects that file;
  * a file already on the current page does not reshuffle the pager;
  * in Solo the page is remembered but no render is attempted;
  * an unknown name reports instead of failing silently.

STATIC: the tile carries the waiting ring while its thumbnail is pending, and
drops it on BOTH load and error; the ring rule exists and animates.

MUTATIONS (wound in a COPY, catch proven):
  M1 page arithmetic dropped        -> back to "only page 1 works".
  M2 select() never called          -> the folder changes, the image does not.
  M3 the silent-failure note removed-> the original bug, only quieter.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

NAME = "v692"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JS = os.path.join(ROOT, "web", "js", "ph_media_loader.js")


def _fail(msg):
    print("[%s] FAIL -- %s" % (NAME, msg))
    sys.exit(1)


def _need(cond, msg):
    if not cond:
        _fail(msg)


def _lift(src, sig):
    """Text-lift one method out of the class so it can be driven in plain node."""
    i = src.index(sig)
    depth = 0
    j = i
    started = False
    while j < len(src):
        if src[j] == "{":
            depth += 1
            started = True
        elif src[j] == "}":
            depth -= 1
            if started and depth == 0:
                return src[i:j + 1]
        j += 1
    _fail("could not lift %r" % sig)


def _run_node(script):
    fd, path = tempfile.mkstemp(suffix=".js")
    os.write(fd, script.encode("utf-8"))
    os.close(fd)
    try:
        p = subprocess.run(["node", path], capture_output=True, text=True,
                           timeout=60)
        lines = p.stdout.strip().splitlines()
        if p.returncode != 0 or not lines:
            _fail("node produced no verdict: %s"
                  % ((p.stderr.strip() or "no output")[:400]))
        return json.loads(lines[-1])
    finally:
        os.unlink(path)


DRIVER = r"""
const GRID_PAGE = 20;

function makeUI(solo) {
    const files = [];
    for (let i = 0; i < 204; i++) files.push({ name: "f" + String(i).padStart(3, "0") + ".png" });
    return Object.assign({
        _files: files,
        _page: 0,
        view: { solo: !!solo },
        renders: 0,
        picked: null,
        notes: [],
        renderGrid() { this.renders++; },
        select(f) { this.picked = f; },
        _dropNote(msg) { this.notes.push(msg); },
    }, PROTO);
}

const out = {};

// a file deep in the folder: index 137 -> page 6 (zero-based)
let ui = makeUI(false);
ui._dropSelect("f137.png");
out.deepPage = ui._page;
out.deepPicked = ui.picked && ui.picked.name;
out.deepRenders = ui.renders;

// a file already on the current page must not reshuffle anything
ui = makeUI(false);
ui._dropSelect("f005.png");
out.nearPage = ui._page;
out.nearPicked = ui.picked && ui.picked.name;
out.nearRenders = ui.renders;

// Solo: remember the page, do not try to draw a grid that does not exist
ui = makeUI(true);
ui._dropSelect("f137.png");
out.soloPage = ui._page;
out.soloPicked = ui.picked && ui.picked.name;
out.soloRenders = ui.renders;

// a name that is not in the listing must be REPORTED, never swallowed
ui = makeUI(false);
ui._dropSelect("ghost.png", 20);
out.ghostPicked = ui.picked;
out.ghostNotes = ui.notes.length;
out.ghostText = ui.notes.join(" ");

console.log(JSON.stringify(out));
"""


def run_driven(src, tag=""):
    body = _lift(src, "_dropSelect(name, tries = 0) {")
    script = "const PROTO = {\n" + body + "\n};\n" + DRIVER
    r = _run_node(script)

    _need(r["deepPage"] == 6,
          "%sa file at index 137 lives on page 6 of 20-per-page; the grid must "
          "turn there, got page %r -- otherwise only the first twenty files in "
          "a folder can ever be selected by a drop" % (tag, r["deepPage"]))
    _need(r["deepPicked"] == "f137.png",
          "%sthe dropped file itself must be selected, got %r"
          % (tag, r["deepPicked"]))
    _need(r["deepRenders"] == 1,
          "%sturning the page must redraw the grid exactly once, got %r"
          % (tag, r["deepRenders"]))

    _need(r["nearPage"] == 0 and r["nearRenders"] == 0,
          "%sa file already on the current page must not turn the pager "
          "(page %r, %r redraws)" % (tag, r["nearPage"], r["nearRenders"]))
    _need(r["nearPicked"] == "f005.png",
          "%sthe near file must still be selected, got %r" % (tag, r["nearPicked"]))

    _need(r["soloPage"] == 6,
          "%sin Solo the page must still be remembered for the swap back, got %r"
          % (tag, r["soloPage"]))
    _need(r["soloRenders"] == 0,
          "%sin Solo there is no grid to redraw -- renderGrid must not be "
          "called, got %r" % (tag, r["soloRenders"]))
    _need(r["soloPicked"] == "f137.png",
          "%sSolo must select the file all the same (this is exactly the case "
          "that used to do nothing at all), got %r" % (tag, r["soloPicked"]))

    _need(r["ghostPicked"] is None,
          "%san unknown name must not select anything" % tag)
    _need(r["ghostNotes"] == 1 and "ghost.png" in r["ghostText"],
          "%sa file that is not in the listing must be REPORTED by name -- the "
          "original bug was not the miss, it was the silence" % tag)


def run_static(src, tag=""):
    _need(re.search(r"tile\.classList\.add\(\s*[\"']ph-thumb-wait[\"']\s*\)", src),
          "%sa tile whose thumbnail is still pending must show that work is "
          "happening -- since v684 the URL is only assigned near view and the "
          "tile sat empty until the server answered" % tag)
    _need(re.search(r"img\.onload\s*=\s*\(\)\s*=>\s*tile\.classList\.remove\("
                    r"\s*[\"']ph-thumb-wait[\"']\s*\)", src),
          "%sthe ring must stop when the thumbnail arrives" % tag)
    # HOUSE TRAP (guards #104/#108/#109/#111): anchor on the block that is
    # actually meant. There are TWO `img.onerror` handlers in this file -- the
    # preview's and the tile's -- and a bare index() finds the preview one,
    # which would have passed while the tile spun forever.
    i = src.index('tile.classList.add("ph-thumb-wait")')
    blk = src[i:i + 700]
    _need("img.onerror" in blk, "%sthe tile's error handler must follow the ring" % tag)
    errblk = blk[blk.index("img.onerror"):]
    _need(re.search(r"tile\.classList\.remove\(\s*[\"']ph-thumb-wait[\"']\s*\)",
                    errblk),
          "%sthe ring must ALSO stop on a dead thumbnail, or a broken file "
          "spins forever" % tag)
    _need(re.search(r"\.ph-media-tile\.ph-thumb-wait::after[^}]*"
                    r"animation:\s*ph-busy-spin", src, flags=re.S),
          "%sthe waiting ring must exist as a pure CSS rule (no timer, nothing "
          "to leak) and reuse the folder spinner's animation" % tag)
    _need(re.search(r"const\s+idx\s*=\s*files\.findIndex", src),
          "%sthe selection must be driven from the FILE LIST, not from the DOM "
          "-- the DOM only ever holds one page" % tag)


def main():
    src = open(JS, encoding="utf-8").read()
    run_static(src)
    run_driven(src)

    caught = 0
    muts = [
        ("M1", src.replace("const page = Math.floor(idx / GRID_PAGE);",
                           "const page = 0;")),
        ("M2", src.replace("            this.select(files[idx]);\n", "")),
        ("M3", src.replace('        this._dropNote("\\u26a0 " + name',
                           '        if (false) this._dropNote("\\u26a0 " + name')),
    ]
    for name, mutated in muts:
        _need(mutated != src, "%s did not apply" % name)
        try:
            run_driven(mutated, tag="[%s] " % name)
        except SystemExit:
            caught += 1
        else:
            print("[%s] NOTE -- mutation %s survived" % (NAME, name))

    _need(caught == len(muts),
          "only %d/%d mutations were caught" % (caught, len(muts)))
    print("[%s] PASS -- a drop pages the grid to the file and selects it "
          "wherever it sorts (204 files, 11 pages), Solo selects without a "
          "grid, a miss is reported by name instead of silently retried away, "
          "the pending tile shows the amber ring and drops it on load and on "
          "error, %d/%d mutations caught" % (NAME, caught, len(muts)))


if __name__ == "__main__":
    main()
