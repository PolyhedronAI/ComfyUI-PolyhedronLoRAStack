#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v895 -- "No color" is a CHOICE and survives a reload.

THE WOUND, open since v600 and driven again on 28.08: right-click > Colors >
"No color" on any Suite node greys it as asked, and the next reload quietly
paints it Pack-blue again. Not a repaint bug -- an information loss:

    menu       inner_clicked -> item.setColorOption(v.value ? nine[v] : null)
    node       setColorOption(null)  =>  delete this.color; delete this.bgcolor
    serialize  `if (this.color) o.color = this.color`  -- deleted keys vanish

so in the FILE "chose none" and "never touched" are the same state: absence.
ph_palette.js then repaints its default in onNodeCreated and configure() has
nothing to override it with.

All three lines above are lifted from @comfyorg/litegraph 0.17.2 (read in the
real bundle, not remembered), and the stub below implements exactly them. That
is the point of the stub: it is LiteGraph's rule, restated once, so this guard
can fail when OUR file stops obeying it -- while a stub that echoed our own
behaviour would prove nothing.

WHAT IS PINNED

  P1  a fresh node takes the Suite default.
  P2  an untouched node writes NO colour to the file (the v600 promise: the
      default must stay repaintable from one line, never frozen into saves).
  P3  "No color" -> grey now, AND grey after save+reload. This is the fix.
  P4  the choice is recorded in node.properties, so it rides the workflow and
      the backend never sees it -- and it is ONE flag, not a colour copy.
  P5  a real colour still wins, is written, survives, and CLEARS the flag --
      the flag can only ever describe the current state.
  P6  the flag is honoured on the LOAD path specifically (onConfigure), not
      only at pick time -- onNodeCreated runs BEFORE configure(), so a fix that
      only guarded creation would still repaint every loaded node.
  P7  foreign nodes (category outside Polyhedron/) are not touched at all.

Script-style: exit 0 = pass.
"""

import json
import os
import subprocess
import sys
import tempfile

NAME = "v895"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PALETTE = os.path.join(ROOT, "web", "js", "ph_palette.js")

_fails = []
_checks = 0


def _need(cond, msg):
    global _checks
    _checks += 1
    if not cond:
        _fails.append(msg)
        print("[%s] FAIL -- %s" % (NAME, msg))


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# The harness. LiteGraph's three rules, restated from 0.17.2:
#   setColorOption(null) deletes both keys; a real option assigns both.
#   serialize() writes color/bgcolor ONLY when truthy, and clones properties.
#   configure() applies the saved fields and calls onConfigure LAST.
# ---------------------------------------------------------------------------
HARNESS = r"""
const results = [];
const PH_BG = "#1a1a2a", PH_TITLE = "#333";
const NINE = { green: { color: "#232", bgcolor: "#353" } };

// --- app stub: capture the extension ---------------------------------------
let EXT = null;
globalThis.__app = { registerExtension(e) { EXT = e; } };

%(import_line)s

// --- LiteGraph node, per 0.17.2 --------------------------------------------
function makeType() {
    class T {
        constructor() { this.properties = {}; }
        setColorOption(o) {
            if (o == null) { delete this.color; delete this.bgcolor; }
            else { this.color = o.color; this.bgcolor = o.bgcolor; }
        }
        getColorOption() {
            return Object.values(NINE).find(
                (c) => c.color === this.color && c.bgcolor === this.bgcolor) ?? null;
        }
        serialize() {
            const o = { type: "ULSFake" };
            if (this.properties) o.properties = JSON.parse(JSON.stringify(this.properties));
            if (this.color) o.color = this.color;
            if (this.bgcolor) o.bgcolor = this.bgcolor;
            return o;
        }
        configure(info) {
            if (info.properties) this.properties = JSON.parse(JSON.stringify(info.properties));
            if (info.color) this.color = info.color;
            if (info.bgcolor) this.bgcolor = info.bgcolor;
            this.onConfigure?.(info);          // LAST, as in configure()
        }
    }
    return T;
}

function register(category) {
    const T = makeType();
    EXT.beforeRegisterNodeDef(T, { category });
    return T;
}

// birth = what ComfyUI does: construct, then onNodeCreated
function birth(T) {
    const n = new T();
    if (n.onNodeCreated) n.onNodeCreated();
    return n;
}

// load = birth, then configure with the saved object
function load(T, saved) {
    const n = birth(T);
    n.configure(JSON.parse(JSON.stringify(saved)));
    return n;
}

const Suite = register("Polyhedron/Loaders");

// P1: fresh node takes the default
const fresh = birth(Suite);
results.push(["P1", fresh.color === PH_TITLE && fresh.bgcolor === PH_BG,
              JSON.stringify([fresh.color, fresh.bgcolor])]);

// P2: untouched node writes no colour
const s_fresh = fresh.serialize();
results.push(["P2", !("color" in s_fresh) && !("bgcolor" in s_fresh),
              JSON.stringify(s_fresh)]);

// P3: "No color" now, and after a round trip
const none = birth(Suite);
none.setColorOption(null);
const greyNow = none.color === undefined && none.bgcolor === undefined;
const s_none = none.serialize();
const reloaded = load(Suite, s_none);
const greyAfter = reloaded.color === undefined && reloaded.bgcolor === undefined;
results.push(["P3a", greyNow, JSON.stringify([none.color, none.bgcolor])]);
results.push(["P3b", greyAfter, JSON.stringify([reloaded.color, reloaded.bgcolor,
                                                s_none])]);

// P4: recorded as ONE flag in properties, and it is not a colour
const props = s_none.properties || {};
const keys = Object.keys(props);
results.push(["P4a", keys.length === 1, JSON.stringify(props)]);
results.push(["P4b", keys.length === 1 && props[keys[0]] === true,
              JSON.stringify(props)]);
results.push(["P4c", !("color" in props) && !("bgcolor" in props),
              JSON.stringify(props)]);

// P5: a real colour wins, survives, clears the flag
const green = birth(Suite);
green.setColorOption(null);                 // first none...
green.setColorOption(NINE.green);           // ...then a real pick
const s_green = green.serialize();
const green2 = load(Suite, s_green);
results.push(["P5a", s_green.color === NINE.green.color
                  && s_green.bgcolor === NINE.green.bgcolor,
              JSON.stringify(s_green)]);
results.push(["P5b", green2.color === NINE.green.color
                  && green2.bgcolor === NINE.green.bgcolor,
              JSON.stringify([green2.color, green2.bgcolor])]);
results.push(["P5c", Object.keys(s_green.properties || {}).length === 0,
              JSON.stringify(s_green.properties)]);

// P6: the LOAD path is what honours it. Prove creation alone is not enough:
// birth() paints the default, and only configure() may take it back off.
const midway = birth(Suite);
const paintedAtBirth = midway.color === PH_TITLE;
midway.configure(JSON.parse(JSON.stringify(s_none)));
results.push(["P6", paintedAtBirth && midway.color === undefined,
              JSON.stringify([paintedAtBirth, midway.color])]);

// P7: a foreign node is untouched
const Foreign = register("Comfy/loaders");
const f = birth(Foreign);
results.push(["P7", f.color === undefined && f.bgcolor === undefined,
              JSON.stringify([f.color, f.bgcolor])]);

process.stdout.write(JSON.stringify(results));
"""


def main():
    src = _read(PALETTE)
    # the module imports app from ComfyUI; point that at our stub
    stub_dir = tempfile.mkdtemp()
    with open(os.path.join(stub_dir, "app.js"), "w", encoding="utf-8") as fh:
        fh.write("export const app = globalThis.__app;\n")
    mod = os.path.join(stub_dir, "ph_palette.mjs")
    with open(mod, "w", encoding="utf-8") as fh:
        fh.write(src.replace('from "../../scripts/app.js"', 'from "./app.js"'))

    js = HARNESS % {"import_line": 'await import("%s");' % mod.replace("\\", "/")}
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(js)
        path = fh.name
    try:
        p = subprocess.run(["node", path], capture_output=True, text=True,
                           timeout=90)
    finally:
        os.unlink(path)

    if p.returncode != 0 or not p.stdout.strip():
        print("[%s] FAIL -- harness did not run: %s"
              % (NAME, (p.stderr or p.stdout or "")[-600:]))
        return 1

    EXPLAIN = {
        "P1": "a fresh Suite node must take the Suite default",
        "P2": "an untouched node must write NO colour to the file (v600)",
        "P3a": "'No color' must clear both keys immediately",
        "P3b": "'No color' must STILL be colourless after save + reload -- "
               "this is the v895 wound",
        "P4a": "the choice must be exactly ONE property",
        "P4b": "the flag must be a boolean marker, not a stored colour",
        "P4c": "no colour copy may be smuggled into properties",
        "P5a": "a real pick must be written to the file",
        "P5b": "a real pick must survive the reload",
        "P5c": "a real pick must CLEAR the no-colour flag",
        "P6": "the flag must be honoured on the LOAD path (onConfigure), "
              "because onNodeCreated has already painted by then",
        "P7": "a node outside Polyhedron/ must not be touched",
    }
    for tag, ok, detail in json.loads(p.stdout):
        _need(ok, "%s: %s [%s]" % (tag, EXPLAIN.get(tag, tag), detail))

    # --- structural: the flag rides properties, nothing else -----------------
    _need("PROP_NO_COLOUR" in src,
          "the flag must have ONE named constant, not a literal at each site")
    _need("onConfigure" in src,
          "the load path must be hooked (onConfigure)")
    _need("setColorOption" in src,
          "the pick must be captured at LiteGraph's own method")

    if _fails:
        print("\n[%s] %d/%d FAILED" % (NAME, len(_fails), _checks))
        return 1
    print("[%s] OK -- 'No color' is a choice and survives a reload (%d checks)"
          % (NAME, _checks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
