"""Guard v539 -- noise preview: the drag budget survives a reload.

DECLARED MOVE (v687): this invariant was born on the Empty Latent, whose
preview is gone -- the preview now lives on the Polyhedron Seed node and
shows the REAL field from uls_noise.make_noise. The LAW is unchanged and
is what this guard exists for:

    onResize never fires on configure(). A drag budget kept only in the
    widget therefore snaps back to its default on every reload, leaving a
    sized node with dead space in it. It must be written to
    node.properties and read back on load -- clamped on the way in, with
    the same MIN/MAX as the drag path, so a hand-edited or older workflow
    cannot inject an out-of-range height.

The guard follows the invariant to its new home rather than being deleted
with the old one.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(m):
    print("FAIL: " + m)
    sys.exit(1)


def main():
    js = open(os.path.join(ROOT, "web", "js", "ph_seed.js"),
              encoding="utf-8").read()
    if "this.properties.pls_preview_h = widget._h;" not in js:
        _fail("write path missing: onResize must persist the budget")
    if "properties.pls_preview_h" not in js.split("onConfigure")[-1]:
        _fail("read path missing: onConfigure must restore the budget")
    if js.count("Math.max(PREV_MIN_H, Math.min(PREV_MAX_H") < 2:
        _fail("restore path must clamp against MIN/MAX like the drag path")
    if 'w.type === "uls_seed_noise_preview"' not in js:
        _fail("restore must target the preview widget by type")
    # the Empty Latent must NOT quietly keep a second, dead copy of it
    old = open(os.path.join(ROOT, "web", "js", "ph_empty_latent.js"),
               encoding="utf-8").read()
    # HOUSE TRAP, fourth sighting (guards #104, #108, #109): a static must pin
    # EXECUTABLE code, never prose. A comment explaining why the old property is
    # no longer read is not a read. Match the access forms, not the word.
    if re.search(r"properties\.uls_preview_h\s*(=|\?|\)|;|,)", old) \
            or re.search(r"=\s*[^\n]*properties[^\n]*uls_preview_h", old):
        _fail("the Empty Latent's preview budget is dead code -- one home only")
    print("PASS: v539 noise preview -- budget persisted + clamped restore "
          "(now on the Seed node)")
    sys.exit(0)


if __name__ == "__main__":
    main()
