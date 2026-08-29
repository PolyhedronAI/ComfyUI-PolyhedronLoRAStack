"""Guard #110 -- the seed preview's DRAG IMPRESSION (v690).

THE FIELD BUG (v689, Frank's screen): "im Umbau der Noise Node kriegt man
hier ein 'building field'". Dragging the preview to hunt for a variant -- the
whole reason the control exists -- had stopped showing anything.

WHY IT HAPPENED: v687 moved the preview to the REAL field, fetched from
/uls/noise/preview. Correct, and worth keeping. But the fetch is debounced by
90ms in _requestField and the debounce RESTARTS on every state change, so a
continuous scrub never let the timer fire -- zero requests for the whole
gesture. At the same time _requestField sets widget._key immediately, and
draw() only shows the cached image while _imgKey === _key. So the valid old
image was discarded on the first mouse move and nothing replaced it: an empty
box labelled "building field..." until the mouse came to rest.

THE FIX, and its price: while the mouse is down we draw a LOCAL field
(ph_noise_field.js, ported from the pre-v685 Empty Latent preview). It is an
IMPRESSION of the noise character, not the tensor the run will use. That is a
deliberate reversal of the v687 direction, and it is only tolerable because
the readout SAYS SO on every frame. An unlabelled imitation would be the same
lie as drawing a confident 64x64 for a wired size -- the v619-v623 class.

Note for whoever reads guard #109 next: its "the javascript imitation must
not come back" now means "not in ph_empty_latent.js, and nowhere unlabelled".
The imitation lives in exactly one place, is reachable only during a drag,
and is pinned here.

DRIVEN in node (the renderer is pure, so it is executed, not read):
  * a field of the requested size for every type;
  * deterministic in (type, seed);
  * a different seed gives a different field -- INCLUDING seeds that differ
    by an exact multiple of 2^32, which a bare `| 0` would collapse onto the
    same picture (the v689 class, one layer down);
  * the grid is clamped, so a pathological wired size cannot turn one mouse
    move into a million-pixel loop;
  * the canvas cache returns the same object for the same key and rebuilds
    when the seed changes.

STATIC (regex against executable code -- never `"name" in src`, the house
trap that fired four times in the v689 session):
  * ph_seed.js imports the renderer instead of carrying a copy;
  * the fetch is gated on the drag flag;
  * the impression branch is labelled;
  * the widget's state still goes through the CLOSURE, never `this`.

MUTATIONS (wound in a COPY, catch proven):
  M1 fetch no longer gated on the drag  -> the scrub goes blind again.
  M2 the impression label removed       -> an imitation shown as the field.
  M3 seed folded with a bare `| 0`      -> 53-bit seeds collapse.
  M4 the grid clamp removed             -> unbounded loop on a wired size.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

NAME = "v690"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JS_DIR = os.path.join(ROOT, "web", "js")
SEED = os.path.join(JS_DIR, "ph_seed.js")
FIELD = os.path.join(JS_DIR, "ph_noise_field.js")
EMPTY = os.path.join(JS_DIR, "ph_empty_latent.js")


def _fail(msg):
    print("[%s] FAIL -- %s" % (NAME, msg))
    sys.exit(1)


def _need(cond, msg):
    if not cond:
        _fail(msg)


def _lift(src, sigs):
    """Text-lift whole functions out of an ES module so they can be driven in
    plain node (the module has an `export` keyword and no DOM)."""
    parts = []
    for sig in sigs:
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
                    j += 1
                    break
            j += 1
        parts.append(src[i:j])
    return "\n\n".join(parts)


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
                  % ((p.stderr.strip() or "no output")[:300]))
        return json.loads(lines[-1])
    finally:
        os.unlink(path)


# A canvas stub with just enough surface for impressionCanvas: the point is to
# drive the real caching and clamping code, not to rasterise anything.
_DRIVER = r"""
const document = {
    createElement: function () {
        return {
            width: 0, height: 0,
            getContext: function () {
                const self = this;
                return {
                    createImageData: function (w, h) {
                        return { width: w, height: h,
                                 data: new Uint8ClampedArray(w * h * 4) };
                    },
                    putImageData: function (img) { self._last = img; },
                };
            },
        };
    },
};

const out = {};

// 1. every type produces a field of exactly the requested size
out.sizes = {};
for (const t of ["gaussian", "blue", "brown", "pink", "fractal", "zeros"]) {
    const f = _renderField(t, 7, 12, 9);
    out.sizes[t] = f.length;
}

// 2. deterministic in (type, seed)
const a1 = Array.from(_renderField("gaussian", 12345, 8, 8));
const a2 = Array.from(_renderField("gaussian", 12345, 8, 8));
out.deterministic = a1.every((v, i) => v === a2[i]);

// 3. a different seed gives a different field
const b = Array.from(_renderField("gaussian", 12346, 8, 8));
out.seedMatters = a1.some((v, i) => v !== b[i]);

// 4. THE v689 CLASS, one layer down: seeds an exact multiple of 2^32 apart
//    must not collapse onto the same picture.
const big1 = Array.from(_renderField("gaussian", 1125899906842624, 8, 8));
const big2 = Array.from(_renderField("gaussian", 1125899906842624 + 4294967296, 8, 8));
out.highBitsMatter = big1.some((v, i) => v !== big2[i]);

// 5. zeros is flat
const z = Array.from(_renderField("zeros", 1, 8, 8));
out.zerosFlat = z.every((v) => v === z[0]);

// 6. the grid is clamped in both directions
// 1024 rather than something absurd on purpose: without the cap this must
// fail the ASSERTION below, not die in the allocator. A guard that catches
// its mutation by crashing is a guard that also crashes on a slow machine.
const wide = impressionCanvas(null, "gaussian", 1, 1024, 1024);
const tiny = impressionCanvas(null, "gaussian", 1, 1, 1);
out.clampHigh = wide.width;
out.clampLow = tiny.width;

// 7. the cache hands back the SAME object for the same key, and rebuilds
//    when the seed moves (otherwise the drag would freeze on one picture)
const store = {};
const c1 = impressionCanvas(store, "gaussian", 5, 16, 16);
const k1 = store._impKey;
const c2 = impressionCanvas(store, "gaussian", 5, 16, 16);
out.cacheHit = (c1 === c2 && store._impKey === k1);
impressionCanvas(store, "gaussian", 6, 16, 16);
out.cacheRebuilds = (store._impKey !== k1);

console.log(JSON.stringify(out));
"""


def run_driven(field_src, tag=""):
    lifted = _lift(field_src, ["function _hash2(", "function _normal(",
                               "function _valueNoise(", "function _seed32(",
                               "function _renderField(", "function _clampGrid("])
    # the export keyword cannot survive in a plain script
    body = field_src[field_src.index("export function impressionCanvas("):]
    body = body.replace("export function", "function", 1)
    consts = "\n".join(re.findall(r"^const GRID_(?:MAX|MIN) = \d+;", field_src,
                                  flags=re.M))
    res = _run_node(consts + "\n" + lifted + "\n" + body + "\n" + _DRIVER)

    for t, n in res["sizes"].items():
        _need(n == 12 * 9,
              "%s%s: the field must have exactly the requested sample count, "
              "got %s" % (tag, t, n))
    _need(res["deterministic"],
          "%sthe same (type, seed) must give the same field -- a shimmering "
          "impression during a drag is worse than none" % tag)
    _need(res["seedMatters"],
          "%sa different seed must give a different field, or scrubbing shows "
          "nothing happening" % tag)
    _need(res["highBitsMatter"],
          "%sseeds an exact multiple of 2^32 apart must not collapse onto the "
          "same field -- that is the v689 truncation, one layer down" % tag)
    _need(res["zerosFlat"], "%szeros must be flat" % tag)
    _need(res["clampHigh"] <= 256,
          "%sthe grid must be capped (got %s) -- a wired size must not turn a "
          "mouse move into a million-pixel loop" % (tag, res["clampHigh"]))
    _need(res["clampLow"] >= 8,
          "%sthe grid must have a floor (got %s)" % (tag, res["clampLow"]))
    _need(res["cacheHit"],
          "%sthe canvas must be cached per key, or every frame reallocates" % tag)
    _need(res["cacheRebuilds"],
          "%sthe cache must rebuild when the seed changes, or the drag freezes "
          "on one picture" % tag)


def run_static(seed_src, field_src, tag=""):
    _need(re.search(r"export\s+function\s+impressionCanvas\s*\(", field_src),
          "%sthe renderer must be exported from ph_noise_field.js" % tag)
    _need(re.search(r"import\s*\{[^}]*\bimpressionCanvas\b[^}]*\}\s*from\s*"
                    r"[\"']\./ph_noise_field\.js[\"']", seed_src),
          "%sph_seed.js must IMPORT the renderer, never carry a second copy "
          "of it" % tag)
    # Pin the CALL and the GATE, not the prose around them.
    _need(re.search(r"if\s*\(\s*!st\.pending\s*&&\s*!dragging\s*\)\s*"
                    r"_requestField\(", seed_src),
          "%sthe fetch must be gated on the drag flag -- an ungated debounce "
          "restarts on every mouse move and never fires" % tag)
    _need(re.search(r"impressionCanvas\(\s*widget\s*,", seed_src),
          "%sthe draw path must actually call the renderer, with the widget as "
          "its cache store (closure, not `this`)" % tag)
    _need(re.search(r"fillText\(\s*[\"']impression while", seed_src),
          "%sthe impression MUST be labelled on screen -- an unlabelled "
          "imitation presented as the real field is the v619-v623 lie" % tag)
    # The imitation is allowed in exactly one file.
    _need("function _renderField(" not in seed_src
          and "function _renderField(" not in open(EMPTY, encoding="utf-8").read(),
          "%sthe renderer must live in ph_noise_field.js alone" % tag)
    # v689 closure law, re-pinned here because this cut edited that widget.
    i = seed_src.index('const widget = {\n        type: "uls_seed_noise_preview"')
    j = seed_src.index("    node.addCustomWidget(widget);", i)
    _need("this." not in seed_src[i:j],
          "%sthe preview widget must use the closure `widget`, never `this`"
          % tag)


def main():
    seed_src = open(SEED, encoding="utf-8").read()
    field_src = open(FIELD, encoding="utf-8").read()
    run_static(seed_src, field_src)
    run_driven(field_src)

    caught = 0

    # M1 -- ungate the fetch: the scrub goes blind again.
    m1 = seed_src.replace("if (!st.pending && !dragging) _requestField(",
                          "if (!st.pending) _requestField(")
    _need(m1 != seed_src, "M1 did not apply")
    try:
        run_static(m1, field_src, tag="[M1] ")
    except SystemExit:
        caught += 1
    else:
        print("[%s] NOTE -- mutation M1 survived" % NAME)

    # M2 -- drop the label: an imitation shown as the real field.
    m2 = seed_src.replace('ctx.fillText("impression while", tx, ty); ty += 13;',
                          'ctx.fillText("the real field,", tx, ty); ty += 13;')
    _need(m2 != seed_src, "M2 did not apply")
    try:
        run_static(m2, field_src, tag="[M2] ")
    except SystemExit:
        caught += 1
    else:
        print("[%s] NOTE -- mutation M2 survived" % NAME)

    # M3 -- fold the seed with a bare `| 0`: 53-bit seeds collapse.
    m3 = field_src.replace("return (lo ^ Math.imul(hi, 2654435761)) | 0;",
                           "return lo | 0;")
    _need(m3 != field_src, "M3 did not apply")
    try:
        run_driven(m3, tag="[M3] ")
    except SystemExit:
        caught += 1
    else:
        print("[%s] NOTE -- mutation M3 survived" % NAME)

    # M4 -- remove the grid cap: unbounded work per mouse move.
    m4 = field_src.replace(
        "return Math.max(GRID_MIN, Math.min(GRID_MAX, n));",
        "return Math.max(GRID_MIN, n);")
    _need(m4 != field_src, "M4 did not apply")
    try:
        run_driven(m4, tag="[M4] ")
    except SystemExit:
        caught += 1
    else:
        print("[%s] NOTE -- mutation M4 survived" % NAME)

    _need(caught == 4, "only %d/4 mutations were caught" % caught)
    print("[%s] PASS -- the scrub draws a local field instead of going blind, "
          "it is labelled as an impression on every frame, the fetch is gated "
          "on the drag, the renderer is deterministic, bounded, cached, "
          "survives 53-bit seeds and lives in one place, 4/4 mutations caught"
          % NAME)


if __name__ == "__main__":
    main()
