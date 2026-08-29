"""
test_v889_note.py -- promises of the \u2b21 Polyhedron Note.

  N1  it NEVER RUNS. No outputs, no OUTPUT_NODE, so it is in no execution
      path and queueing costs nothing.
  N2  the colour is not a widget. A canon widget would cost a positional
      index in every saved workflow (guard #577) for a value Python never
      reads -- the v750 decision.
  N8  the control row cannot hide under the DOM textarea (the v891 wound):
      its height is declared INSIDE computeSize.
  N10 the swatch IS the outcome -- it is painted with the very bgcolor the
      node will take, so the control cannot misrepresent itself.
  N11 the note keeps NO colour state of its own: no palette, no derivation,
      no property, and above all NO serialize() patch.
  N12 a pick is visible immediately, without a graph run.
  N13 the row folds away and can always be brought back, and the node gets
      back exactly the height the row stops using.
  N14 a note coloured by v892 is translated once, by the measured mapping.

RETIRED, and named so nobody hunts for them:

  N3/N4/N6, N5, N9  belonged to the note-to-switch LINK, removed in v892. The
      click was hooked onto LiteGraph's `processNodeSelected`, which carries
      @deprecated and is called from nowhere in 0.12.0/0.14.0/0.16.0/0.17.2,
      so the link could never be made; and the promise it advertised ("a
      legend can never quietly disagree with its switch") was false anyway --
      it matched NUMBERS, never meaning.
  N7  pinned that the note took its colours from ph_tint.js. v893 removed the
      note's colour system entirely, so there is nothing left to source. The
      stronger promise is now N11: it has no colour source at all.

WHY N11 IS PHRASED AS A PROHIBITION. v892 kept a colour property, derived a
body tint from it, and deleted `color`/`bgcolor` in serialize() so the derived
value could not be written twice. Driven, that also deleted every colour set
through ComfyUI's right-click menu: pick green, save, reload, amber again --
silent, and only visible after a reload. A guard that pins "no second copy" is
worthless if the way it keeps that promise is to throw away the FIRST copy. So
the serialize patch is now forbidden outright.

The JS is DRIVEN, not read: a fake tree lets the real ph_note.js resolve a stub
app.js, `beforeRegisterNodeDef` runs against a stand-in nodeType, and
LiteGraph's node_colors table and setColorOption/getColorOption semantics are
reproduced verbatim from the published bundle, so onNodeCreated, onConfigure
and the widget's own draw/mouse handlers actually execute.
"""

import io
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILED = []


def check(cond, label):
    if cond:
        print("  PASS  " + label)
    else:
        print("  FAIL  " + label)
        FAILED.append(label)


def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _code_only(js):
    """JS with comments stripped.

    WHY THIS EXISTS: a plain substring search reads the module's own prose as
    a violation -- the docstring that WARNS about a wound names the wound, and
    the guard then fails on the warning. That happened three times in one
    session (OUTPUT_NODE in v889, and twice here). Rules that forbid a
    construct must look at CODE, never at the sentence explaining why the
    construct is forbidden.
    """
    out = []
    i = 0
    n = len(js)
    while i < n:
        c = js[i]
        if c == "/" and i + 1 < n and js[i + 1] == "*":
            j = js.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c == "/" and i + 1 < n and js[i + 1] == "/":
            j = js.find("\n", i)
            i = n if j < 0 else j
            continue
        if c in "\"'":
            q = c
            out.append(c)
            i += 1
            while i < n and js[i] != q:
                if js[i] == "\\":
                    out.append(js[i])
                    i += 1
                if i < n:
                    out.append(js[i])
                    i += 1
            if i < n:
                out.append(js[i])
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _lift_js(js, names):
    """Pull named exported functions out of ph_note.js so they can be RUN.

    The module imports app.js and ph_tint.js, which do not exist outside a
    browser. Rather than stubbing the whole ComfyUI frontend, the two PURE
    functions are lifted; they touch nothing but their arguments.
    """
    out = []
    for name in names:
        m = re.search(
            r"export function %s\([^)]*\)\s*\{.*?\n\}" % re.escape(name), js, re.S)
        if not m:
            return None
        out.append(m.group(0).replace("export function", "function"))
    return "\n\n".join(out)


def _run_node(script):
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(script)
        path = fh.name
    try:
        res = subprocess.run([("node"), path], capture_output=True, text=True,
                             timeout=60)
        return res.returncode, res.stdout.strip(), res.stderr.strip()
    finally:
        os.unlink(path)


def main():
    py = _read("nodes", "ph_note.py")
    js = _read("web", "js", "ph_note.js")
    init = _read("__init__.py")

    print("N1 - the note is never executed")
    check(re.search(r"^\s*RETURN_TYPES = \(\)", py, re.M) is not None,
          "RETURN_TYPES is empty")
    # Look for the ASSIGNMENT, not the word: the module docstring explains
    # that the flag is deliberately absent, and a substring search read its own
    # prose as a violation. The house lesson, paid for again (v755).
    check(re.search(r"^\s*OUTPUT_NODE\s*=", py, re.M) is None,
          "it does not declare itself an output node")
    check(re.search(r'"ULSNote"\s*\]\s*=\s*ULSNote', init) is not None,
          "it is registered")
    check('"\u2b21 Polyhedron Note"' in init or "Polyhedron Note" in init,
          "it has a display name")

    print("N2 - the colour is not a widget")
    types_block = re.search(r"def INPUT_TYPES.*?\n        \}", py, re.S)
    check(types_block is not None, "INPUT_TYPES is present")
    block = types_block.group(0) if types_block else ""
    check("colour" not in block and "color" not in block,
          "the colour is not declared as a widget")
    for w in ("title", "text"):
        check('"%s"' % w in block, "the %s widget exists" % w)
    check("properties" in js, "the frontend uses node.properties")

    print("N8 - the control row cannot hide under the DOM textarea (v891)")
    # THE WOUND, from Frank's field probe (28.08.): v890 painted the swatches
    # and the link button in onDrawForeground at `size[1] - PAD`. `text` is a
    # real DOM textarea ABOVE the canvas, so it covered them -- the exact
    # failure ph_clip_encode.js documents at lines 35ff, where ph_reference is
    # named as the pattern that does it right (height declared INSIDE
    # computeSize, so LiteGraph accounts for it on every path).
    check("addCustomWidget" in js,
          "the row is registered as a custom widget")
    check(re.search(r"computeSize\s*\([^)]*\)\s*\{[^}]*BAR_H", js) is not None,
          "its height is declared in computeSize - LiteGraph knows it exists")
    check("serialize: false" in js,
          "serialize:false - the canon and both baselines stay untouched")
    # The regression itself: nothing may be positioned off the node's own
    # bottom edge any more.
    code = _code_only(js)
    check(re.search(r"size\[1\]\s*-", code) is None,
          "nothing is placed from size[1] downward any more (that IS the wound)")
    check(re.search(r"onDrawForeground", code) is None,
          "the row no longer paints in onDrawForeground at all")
    check(re.search(r"draw\s*\(ctx, node, width, y\)", js) is not None,
          "it draws at the y LiteGraph hands it, not a computed one")
    check("mouse(event, pos, node)" in js,
          "clicks arrive through the widget's own mouse handler")
    check(re.search(r"onMouseDown", code) is None,
          "and no longer through a node-level onMouseDown")



    print("N10..N14 - driven against LiteGraph's own colour semantics")
    # No serialize() patch may exist at all -- v892's ate the right-click
    # colour. Checked in CODE, so the docstring explaining the ban is not
    # itself read as a violation (the v755/v889 lesson).
    codej = _code_only(js)
    check("serialize = function" not in codej
          and "prototype.serialize" not in codej,
          "N11: no serialize() patch - the right-click colour survives saving")
    check("ph_tint" not in codej,
          "N11: no palette or derivation of its own any more")
    check("setColorOption" in codej,
          "N11: colours are applied through LiteGraph's own setColorOption")

    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "web", "js"))
    os.makedirs(os.path.join(tmp, "scripts"))
    with io.open(os.path.join(tmp, "web", "js", "ph_note.js"), "w",
                 encoding="utf-8") as fh:
        fh.write(js)
    with io.open(os.path.join(tmp, "scripts", "app.js"), "w",
                 encoding="utf-8") as fh:
        fh.write("export const app = {"
                 " registerExtension(e) { globalThis.__ext = e; } };\n")

    harness = r"""
// LGraphCanvas.node_colors, verbatim from @comfyorg/litegraph 0.17.2.
globalThis.LGraphCanvas = { node_colors: {
  red:{color:"#322",bgcolor:"#533",groupcolor:"#A88"},
  brown:{color:"#332922",bgcolor:"#593930",groupcolor:"#b06634"},
  green:{color:"#232",bgcolor:"#353",groupcolor:"#8A8"},
  blue:{color:"#223",bgcolor:"#335",groupcolor:"#88A"},
  pale_blue:{color:"#2a363b",bgcolor:"#3f5159",groupcolor:"#3f789e"},
  cyan:{color:"#233",bgcolor:"#355",groupcolor:"#8AA"},
  purple:{color:"#323",bgcolor:"#535",groupcolor:"#a1309b"},
  yellow:{color:"#432",bgcolor:"#653",groupcolor:"#b58b2a"},
  black:{color:"#222",bgcolor:"#000",groupcolor:"#444"} } };

const mod = await import("%s/web/js/ph_note.js");
const ext = globalThis.__ext;
const NC = globalThis.LGraphCanvas.node_colors;
const out = {};
out.registered = !!ext;

const nodeType = { prototype: {} };
// What LiteGraph itself writes out.
nodeType.prototype.serialize = function () {
  const o = { widgets_values: ["t", "b"] };
  if (this.color !== undefined) o.color = this.color;
  if (this.bgcolor !== undefined) o.bgcolor = this.bgcolor;
  return o;
};
const baseSerialize = nodeType.prototype.serialize;
await ext.beforeRegisterNodeDef(nodeType, { name: "ULSNote" });
out.serializeUntouched = (nodeType.prototype.serialize === baseSerialize);

// setColorOption / getColorOption, verbatim from the same bundle.
function mk(props) {
  return { properties: props || {}, widgets: [], size: [300, 210], flags: {},
    addCustomWidget(w) { this.widgets.push(w); return w; },
    setDirtyCanvas() { this._dirty = (this._dirty || 0) + 1; },
    setColorOption(o) {
      if (o == null) { delete this.color; delete this.bgcolor; }
      else { this.color = o.color; this.bgcolor = o.bgcolor; }
    },
    getColorOption() {
      return Object.values(NC).find(
        (o) => o.color === this.color && o.bgcolor === this.bgcolor) ?? null;
    } };
}

const n = mk();
n.color = "#333"; n.bgcolor = "#1a1a2a";      // as ph_palette.js leaves it
nodeType.prototype.onNodeCreated.call(n);
const bar = n.widgets.find(w => w && w.name === "$ph_note_bar");
out.barPresent  = !!bar;
out.barNoSerial = bar ? bar.serialize === false : null;
out.openHeight  = bar ? bar.computeSize(300)[1] : null;

// --- draw, capturing what each swatch is actually filled with -------------
function probeCtx(rec) {
  let fill = null, stroke = null;
  return { save(){}, restore(){}, beginPath(){}, closePath(){}, fill(){},
    stroke(){}, moveTo(){}, lineTo(){},
    set fillStyle(v){ fill = v; }, get fillStyle(){ return fill; },
    set strokeStyle(v){ stroke = v; }, get strokeStyle(){ return stroke; },
    set lineWidth(v){}, 
    fillRect(x,y,w,h){ rec.push({kind:"fill", x, fill}); },
    strokeRect(x,y,w,h){ rec.push({kind:"stroke", x, stroke}); } };
}
let rec = [];
bar.draw(probeCtx(rec), n, 300, 0);
const fills = rec.filter(r => r.kind === "fill");
out.swatchFills = fills.map(f => f.fill);
out.expectedFills = Object.keys(NC).map(k => NC[k].bgcolor);
out.hitCount = (bar._hits || []).length;
out.firstHitIsNone = (bar._hits || [])[0]?.name === null;

// --- a pick goes through setColorOption ----------------------------------
// EVERY probe target is looked up defensively. A mutation this guard exists
// to catch will REMOVE one of them, and a harness that throws on the missing
// object reports a traceback instead of a named broken promise -- the v892
// lesson, in its sharper form: not a missing value, a missing thing.
function hit(name) { return (bar._hits || []).find(h => h.name === name) || null; }
function clickAt(box) {
  if (!box) return false;
  bar.mouse({type:"pointerdown"}, [box.x + 2, box.y + 2], n);
  return true;
}

const dirty0 = n._dirty || 0;
out.purpleFound = !!hit("purple");
clickAt(hit("purple"));
out.picked = [n.color ?? null, n.bgcolor ?? null];
out.pickedExpect = [NC.purple.color, NC.purple.bgcolor];
out.pickedReadBack = n.getColorOption() === NC.purple;
out.repaintedNow = (n._dirty || 0) > dirty0;

// --- and it SURVIVES a save --------------------------------------------
out.saved = nodeType.prototype.serialize.call(n);

// --- "none" clears both --------------------------------------------------
out.noneFound = !!hit(null);
clickAt(hit(null));
out.clearedColor = out.noneFound
    && n.color === undefined && n.bgcolor === undefined;

// --- N13 collapse --------------------------------------------------------
const h0 = n.size[1];
out.chevronWhenOpen = !!bar._chev;
if (bar._chev) {
  bar.mouse({type:"pointerdown"}, [bar._chev.x + 2, bar._chev.y + 2], n);
}
out.collapsedFlag  = !!n.properties["ph_note_bar_collapsed"];
out.collapsedH     = bar.computeSize(300)[1];
out.heightGivenBack = h0 - n.size[1];
rec = [];
bar.draw(probeCtx(rec), n, 300, 0);
out.noSwatchesWhenFolded = rec.filter(r => r.kind === "fill").length === 0;
out.chevronStillThere = !!bar._chev;
// and back again -- only if there IS a way back; otherwise the promise is
// simply false and says so, instead of throwing.
if (bar._chev) {
  bar.mouse({type:"pointerdown"}, [bar._chev.x + 2, bar._chev.y + 2], n);
}
out.reopened   = !n.properties["ph_note_bar_collapsed"];
out.heightBack = n.size[1] === h0;

// --- N14 migration of a v892 note ---------------------------------------
// v892 saved the property and, because its serialize deleted them, NO
// color/bgcolor at all.
const m = mk({ ph_note_colour: "amber" });
m.color = "#333"; m.bgcolor = "#1a1a2a";
nodeType.prototype.onConfigure.call(m);
out.migratedTo   = [m.color, m.bgcolor];
out.migrateExpect = [NC.yellow.color, NC.yellow.bgcolor];
out.propGone     = m.properties["ph_note_colour"] === undefined;
// every one of the five maps somewhere real
out.allMap = ["magenta","green","orange","amber","blue"].map(k => {
  const q = mk({ ph_note_colour: k });
  nodeType.prototype.onConfigure.call(q);
  return q.bgcolor;
});
out.allMapExpect = [NC.purple.bgcolor, NC.green.bgcolor, NC.brown.bgcolor,
                    NC.yellow.bgcolor, NC.blue.bgcolor];

console.log(JSON.stringify(out));
""" % (tmp,)

    rc, stdout, stderr = _run_node(harness)
    check(rc == 0, "the real module runs in node"
          + ("" if rc == 0 else ": " + stderr[:200]))
    if rc != 0:
        return 1
    g = json.loads(stdout)

    check(g["registered"], "the extension registers itself")
    check(g["serializeUntouched"],
          "N11: serialize is left EXACTLY as LiteGraph made it")
    check(g["barPresent"] and g["barNoSerial"] is True,
          "the row is a custom widget with serialize:false")

    # N10 -- the swatch IS the outcome.
    check(g["swatchFills"] == g["expectedFills"],
          "N10: every swatch is filled with the bgcolor the node will take")
    check(g["hitCount"] == 10 and g["firstHitIsNone"] and g["noneFound"],
          "N10: ten targets, 'none' first as in ComfyUI's own menu")
    check(g["purpleFound"], "N10: each standard colour has its own target")

    # N11/N12 -- one system, applied and read through LiteGraph.
    check(g["picked"] == g["pickedExpect"],
          "N11: a pick sets exactly LiteGraph's colour pair")
    check(g["pickedReadBack"],
          "N11: and getColorOption finds it - nothing of ours is stored")
    check(g["saved"].get("color") == g["pickedExpect"][0]
          and g["saved"].get("bgcolor") == g["pickedExpect"][1],
          "N11: the colour SURVIVES a save (the v892 regression, pinned)")
    check(g["clearedColor"], "N11: 'none' clears both fields")
    check(g["repaintedNow"], "N12: the pick repaints without a graph run")

    # N13 -- folding.
    check(g["collapsedFlag"] and g["collapsedH"] < g["openHeight"],
          "N13: folding really shrinks what computeSize reports")
    check(g["heightGivenBack"] == g["openHeight"] - g["collapsedH"],
          "N13: the node gets back exactly the height the row stops using")
    check(g["noSwatchesWhenFolded"], "N13: no swatches are painted when folded")
    check(g["chevronWhenOpen"] and g["chevronStillThere"],
          "N13: the chevron is there open AND folded - there is a way back")
    check(g["reopened"] and g["heightBack"],
          "N13: and unfolding restores the original height exactly")

    # N14 -- migration.
    check(g["migratedTo"] == g["migrateExpect"],
          "N14: a v892 amber note becomes LiteGraph yellow")
    check(g["propGone"], "N14: and the old property is removed, so once only")
    check(g["allMap"] == g["allMapExpect"],
          "N14: all five v892 colours land on their measured nearest")


    print("")
    if FAILED:
        print("FAIL -- %d broken promise(s)" % len(FAILED))
        for f in FAILED:
            print("   * " + f)
        return 1
    print("PASS -- v889 note holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
