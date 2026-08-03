"""Guard #125 -- v715: the CLIP Text Encode footer band.

TWO FIELD FINDINGS.

(1) THE BAND COULD BE PUSHED UNDER THE neg_1 FIELD. Its height was reserved
    ONLY inside _refit, while nodeType.prototype.computeSize was not overridden
    -- so LiteGraph itself did not know the band existed. Any height change that
    does not run through _refit, above all dragging the node's bottom edge, set
    size[1] freely, and size[1] - BAR_H then landed inside the last field. neg_1
    is a real DOM textarea above the canvas, so it covered the band.
    ph_reference.js has no such bug: it adds its band height inside computeSize.

(2) THE TEXT RAN OFF THE RIGHT EDGE -- one fillText, no width measurement. It
    now wraps at the counter's own " . " separators, so a narrow node gets a
    second row instead of losing text.

THE DANGEROUS PART, and the reason this guard exists: _refit used to compute
`frame = computeSize()[1] - fields` and then add the band on top. With the band
now inside computeSize, `frame` already carries it -- adding it again would grow
the node by one band height on EVERY refit, and refits are frequent. So the
guard DRIVES _refit repeatedly and proves the height is stable.

Everything is driven against the real ph_clip_encode.js in node.

MUTATIONS (source rewrites, each re-driven):
  M1 _refit adds the band on top of computeSize again -> the node creeps;
  M2 computeSize stops reserving the band          -> a drag buries the footer;
  M3 the text stops wrapping                       -> it runs off the edge;
  M4 the paint uses its own height, not _barHeight -> the two disagree.
"""

import json
import os
import subprocess
import sys
import tempfile

NAME = "v715"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JS = os.path.join(ROOT, "web", "js", "ph_clip_encode.js")

WIDE = 900          # everything fits on one row
NARROW = 260        # forces a wrap
CHAR_W = 7.2        # the fake measureText: monospace-ish, deterministic


def _fail(msg):
    print("[%s] FAIL -- %s" % (NAME, msg))
    sys.exit(1)


def _need(c, msg):
    if not c:
        _fail(msg)


def source():
    with open(JS, encoding="utf-8") as f:
        return f.read()


HARNESS_HEAD = r"""
const CHAR_W = %s;
function mkCtx() {
  return {
    // deterministic monospace metrics -- the wrap must depend on the text and
    // the width, never on a real font engine
    measureText: (t) => ({ width: String(t).length * CHAR_W }),
    save() {}, restore() {}, beginPath() {}, closePath() {}, moveTo() {},
    lineTo() {}, arcTo() {}, arc() {}, stroke() {}, fill() {}, clip() {},
    rect() {}, fillRect() {}, clearRect() {}, drawImage() {}, setTransform() {},
    fillText(t, x, y) { (globalThis._painted ||= []).push([String(t), x, y]); },
    fillStyle: "", strokeStyle: "", lineWidth: 1, font: "", textAlign: "",
    textBaseline: "", globalAlpha: 1, globalCompositeOperation: "",
  };
}
function mkEl(tag) {
  const el = {
    tagName: String(tag || "").toUpperCase(),
    className: "", style: {}, dataset: {}, children: [], value: "",
    classList: { add() {}, remove() {}, contains: () => false },
    width: 0, height: 0, readOnly: false, placeholder: "",
    appendChild(c) { this.children.push(c); return c; },
    removeChild(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); return c; },
    addEventListener() {}, removeEventListener() {},
    setAttribute(k, v) { this[k] = v; },
    getBoundingClientRect: () => ({ width: 0, height: 0, top: 0, left: 0 }),
    focus() {}, remove() {},
    getContext: () => mkCtx(),
  };
  Object.defineProperty(el, "offsetParent", { get() { return null; } });
  Object.defineProperty(el, "clientWidth", { get() { return 0; } });
  Object.defineProperty(el, "scrollHeight", { get() { return 0; } });
  Object.defineProperty(el, "offsetHeight", { get() { return 0; } });
  return el;
}
globalThis.document = {
  getElementById: () => null, createElement: (t) => mkEl(t),
  head: mkEl("head"), body: mkEl("body"),
  addEventListener() {}, removeEventListener() {},
};
globalThis.LiteGraph = { NODE_WIDGET_HEIGHT: 20, NODE_SLOT_HEIGHT: 20 };
globalThis.ResizeObserver = class { constructor(cb) { this.cb = cb; } observe() {} unobserve() {} disconnect() {} };
globalThis.requestAnimationFrame = (f) => { f(); return 1; };  // drive the refit
globalThis.devicePixelRatio = 1;
let _ext = null;
globalThis.app = {
  registerExtension(e) { _ext = e; },
  canvas: { ds: { scale: 1 }, node_over: null, selected_nodes: {} },
  graph: {},
};
globalThis.window = globalThis;
""" % (CHAR_W,)

HARNESS_TAIL = r"""
(async () => {
  const nodeType = function () {};
  // A plain LiteGraph-ish base: computeSize returns the chrome plus whatever
  // the widgets reserve, exactly as LiteGraph does.
  const CHROME = 120;
  nodeType.prototype = {
    computeSize() {
      let h = CHROME;
      for (const w of (this.widgets || [])) {
        if (w.hidden) continue;
        if (typeof w.computeSize === "function") h += w.computeSize(this.size[0])[1];
        else h += 20;
      }
      return [this.size[0], h];
    },
    setSize(s) { this.size = [s[0], s[1]]; },
    setDirtyCanvas() {},
    addDOMWidget(name, type, el, opts) {
      const w = { name, type, element: el, options: opts, value: "" };
      this.widgets.push(w);
      return w;
    },
    onDrawForeground() {},
  };
  await _ext.beforeRegisterNodeDef(nodeType, { name: "ULSCLIPTextEncode" });

  function mkNode(width) {
    const n = Object.create(nodeType.prototype);
    n.size = [width, 400];
    n.widgets = [];
    // the real widget set: the prompt fields are DOM widgets with a textarea,
    // exactly what _visibleFields looks for. Without them _refit returns on its
    // first line and nothing about the band is exercised.
    const mkField = (name, val) => {
      const el = mkEl("textarea");
      el.value = val;
      const w = { name, type: "customtext", element: el, value: val };
      n.widgets.push(w);
      return w;
    };
    mkField("pos_1", "a long positive prompt with several words in it");
    mkField("neg_1", "");
    n.widgets.push({ name: "segments", value: 1, callback: null });
    n.widgets.push({ name: "use_negative", value: true, callback: null });
    n.widgets.push({ name: "strip_comments", value: true, callback: null });
    n.widgets.push({ name: "strip_newlines", value: true, callback: null });
    n.widgets.push({ name: "comment_markers", value: "//", callback: null });
    n.widgets.push({ name: "separator", value: "comma", callback: null });
    n.widgets.push({ name: "external_mode", value: "append", callback: null });
    n.inputs = [{ name: "clip" }, { name: "pos_external", link: null },
                { name: "neg_external", link: null }];
    n.outputs = [];
    n.flags = {};
    // a realistic last-run readout, so the counter text is the long one Frank
    // actually sees on screen
    n._cteTokens = { pos: 14, neg: 0, method: "exact" };
    if (typeof nodeType.prototype.onNodeCreated === "function")
      nodeType.prototype.onNodeCreated.call(n);
    return n;
  }

  const wide = mkNode(%(WIDE)d);
  const narrow = mkNode(%(NARROW)d);

  // --- the band's own numbers
  // The module-level helpers are not exported, so reach them through the
  // behaviour instead: computeSize must EXCEED the plain widget sum by the band.
  function plainSize(n) {
    let h = 120;
    for (const w of (n.widgets || [])) {
      if (w.hidden) continue;
      if (typeof w.computeSize === "function") h += w.computeSize(n.size[0])[1];
      else h += 20;
    }
    return h;
  }
  const bandWide = wide.computeSize()[1] - plainSize(wide);
  const bandNarrow = narrow.computeSize()[1] - plainSize(narrow);

  // --- what actually gets painted, and where
  globalThis._painted = [];
  nodeType.prototype.onDrawForeground.call(narrow, mkCtx());
  const paintedNarrow = globalThis._painted.slice();
  globalThis._painted = [];
  nodeType.prototype.onDrawForeground.call(wide, mkCtx());
  const paintedWide = globalThis._painted.slice();

  // every painted row must fit the available width
  const avail = %(NARROW)d - 20;
  let widestNarrow = 0;
  for (const p of paintedNarrow) widestNarrow = Math.max(widestNarrow, p[0].length * CHAR_W);

  // the band top used by the paint must equal size[1] - band height
  const topPainted = paintedNarrow.length ? paintedNarrow[0][2] : null;
  let bottomPainted = null;
  for (const p of paintedNarrow) bottomPainted = Math.max(bottomPainted ?? -1e9, p[2]);

  // --- THE ANTI-CREEP PROPERTY: refit repeatedly, height must settle.
  // _refit is module-private; reach it the way the node does -- through the
  // callback the extension wired onto a widget, or by forcing the public path.
  // The refit is module-private; the node reaches it through the callback the
  // extension wired onto `use_negative`. Fire that five times and watch the
  // height: it must SETTLE. If the band is counted twice the node grows by one
  // band on every single call.
  const heights = [];
  const cb = wide.widgets.find((w) => w.name === "use_negative").callback;
  for (let i = 0; i < 5; i++) {
    if (typeof cb === "function") cb.call(wide);
    heights.push(wide.size[1]);
  }
  const creep = heights[heights.length - 1] - heights[0];
  // The sharper property: after a refit the node must be EXACTLY as tall as
  // computeSize asks -- no more. Counting the band twice does not make the node
  // run away (each refit recomputes from computeSize, so it settles), it parks
  // it one whole band too tall: a permanent dead strip between the last field
  // and the footer. Measured, that is 442 against a wanted 415.
  const settledDelta = wide.size[1] - wide.computeSize()[1];

  console.log(JSON.stringify({
    ok: true,
    bandWide: bandWide,
    bandNarrow: bandNarrow,
    rowsWide: paintedWide.length,
    rowsNarrow: paintedNarrow.length,
    widestNarrow: widestNarrow,
    avail: avail,
    topPainted: topPainted,
    bottomPainted: bottomPainted,
    narrowH: narrow.size[1],
    creep: creep,
    heights: heights,
    settledDelta: settledDelta,
  }));
})().catch((e) => {
  console.log(JSON.stringify({ ok: false, err: String(e && e.stack || e) }));
});
""" % {"WIDE": WIDE, "NARROW": NARROW}


def _build(src):
    body = src.replace('import { app } from "../../scripts/app.js";', "")
    return HARNESS_HEAD + "\n" + body + "\n" + HARNESS_TAIL


def _run(src):
    fd, path = tempfile.mkstemp(suffix=".mjs")
    os.write(fd, _build(src).encode("utf-8"))
    os.close(fd)
    try:
        p = subprocess.run(["node", path], capture_output=True, text=True,
                           timeout=60)
        lines = p.stdout.strip().splitlines()
        if not lines:
            _fail("node gave no verdict: %s"
                  % ((p.stderr.strip() or "no output")[:400]))
        return json.loads(lines[-1])
    finally:
        os.unlink(path)


def checks(v):
    _need(v.get("ok"), "harness threw: %s" % str(v.get("err", ""))[:300])
    # (1) computeSize MUST reserve the band -- this is what stops a manual drag
    # from burying the footer under the neg_1 textarea.
    _need(v["bandWide"] > 0,
          "computeSize must reserve the footer band (got +%s) -- without it "
          "LiteGraph does not know the band exists and dragging the node's "
          "bottom edge puts it under the neg_1 field" % v["bandWide"])
    # (2) a narrow node wraps: more rows, and a TALLER reservation.
    _need(v["rowsNarrow"] > v["rowsWide"],
          "a narrow node must wrap the counter onto more rows (wide %s, "
          "narrow %s)" % (v["rowsWide"], v["rowsNarrow"]))
    _need(v["bandNarrow"] > v["bandWide"],
          "the reserved band must GROW with the wrap (wide %s, narrow %s) -- "
          "otherwise the extra row is painted outside the reservation"
          % (v["bandWide"], v["bandNarrow"]))
    # (3) nothing runs off the right edge any more.
    _need(v["widestNarrow"] <= v["avail"],
          "every painted row must fit the band (widest %.1f > available %s)"
          % (v["widestNarrow"], v["avail"]))
    # (4) the paint starts inside the reservation, not somewhere else.
    _need(v["topPainted"] is not None
          and v["topPainted"] >= v["narrowH"] - v["bandNarrow"] - 1
          and v["topPainted"] <= v["narrowH"],
          "the first painted row must sit inside the reserved band (row at %s, "
          "band from %s to %s) -- a paint that uses its own height is exactly "
          "the two-numbers-that-must-agree bug"
          % (v["topPainted"], v["narrowH"] - v["bandNarrow"], v["narrowH"]))
    # ...and the LAST row must still be on the node. A paint that assumes a
    # shorter band starts too low and pushes its final row off the bottom edge --
    # invisible, which is how a footer quietly loses half its text.
    _need(v["bottomPainted"] is not None
          and v["bottomPainted"] <= v["narrowH"] - 2,
          "the last painted row must stay inside the node (row at %s, node "
          "bottom %s) -- painting from a band height that is not the reserved "
          "one drops the wrapped row off the edge"
          % (v["bottomPainted"], v["narrowH"]))
    # (5) THE HEIGHT MUST BE EXACTLY WHAT computeSize ASKS FOR.
    _need(v["creep"] == 0,
          "the node height must settle, not drift: repeated refits gave %s"
          % (v["heights"],))
    _need(abs(v["settledDelta"]) <= 1,
          "after a refit the node must be exactly as tall as computeSize asks "
          "(off by %s) -- counting the band in BOTH computeSize and _refit does "
          "not make the node run away, it parks it one whole band too tall and "
          "leaves a permanent dead strip above the footer"
          % v["settledDelta"])


def statics(src):
    _need("function _barHeight(" in src and "function _barLines(" in src,
          "the band height and its wrap must come from named functions")
    _need("nodeType.prototype.computeSize" in src,
          "computeSize must be overridden to reserve the band (v715)")
    _need("const BAR_H" not in src,
          "the fixed band height must be gone -- it wraps now")
    # the paint must not invent its own height
    i = src.index("nodeType.prototype.onDrawForeground")
    draw = src[i:]
    _need("_barHeight(this)" in draw and "_barLines(this)" in draw,
          "the paint must take its height AND its rows from the same functions "
          "that made the reservation")


def prove_mutations(src):
    caught = 0
    total = 0

    def bite(msrc, label):
        nonlocal caught, total
        total += 1
        v = _run(msrc)
        try:
            checks(v)
        except SystemExit:
            caught += 1
            return
        _fail("mutation %s SURVIVED" % label)

    def mut(old, new, label):
        _need(src.count(old) == 1,
              "anchor %s not unique (%d)" % (label, src.count(old)))
        return src.replace(old, new)

    # M1: the band is added on top of a computeSize that already carries it.
    bite(mut("        let total = frame;",
             "        let total = frame + _barHeight(node);", "M1"), "M1")
    # M2: computeSize stops reserving -> a drag buries the footer.
    bite(mut("            try { size[1] += _barHeight(this); } catch (e) { /* ignore */ }",
             "            try { size[1] += 0; } catch (e) { /* ignore */ }",
             "M2"), "M2")
    # M3: the text stops wrapping -> it runs off the right edge.
    bite(mut("        if (ctx.measureText(txt).width <= avail) {",
             "        if (true) {", "M3"), "M3")
    # M4: the paint invents its own height instead of asking _barHeight.
    bite(mut("                const barH = _barHeight(this);",
             "                const barH = 26;", "M4"), "M4")

    print("[%s] mutations %d/%d caught" % (NAME, caught, total))
    _need(caught == total, "not every mutation was caught")


def main():
    src = source()
    checks(_run(src))
    statics(src)
    prove_mutations(src)
    print("[%s] PASS -- footer band driven in node: computeSize reserves it "
          "(the ph_reference pattern), a narrow node wraps onto more rows and "
          "the reservation grows with them, no painted row exceeds the width, "
          "the paint sits inside the reservation, and repeated refits do NOT "
          "creep; 4/4 mutations caught" % NAME)


if __name__ == "__main__":
    main()
