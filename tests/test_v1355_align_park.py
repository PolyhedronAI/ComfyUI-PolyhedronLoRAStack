#!/usr/bin/env python3
"""
test_v1355_align_park -- the Polyhedron Align bar can be parked in the top bar
(web/js/ph_align.js).

Frank, 21.11.: "die Align-Node auch oben in den ComfyUI-Kopf parken ... dort
verbleibt sie auch fest, auch nach Neustart" -- and flush in the row, where
his old (foreign) align bar sat, left of the user button.

The tab row as measured in 1.49.6 and 1.53.10 is rebuilt here as a small DOM:
.workflow-tabs-container.pointer-events-auto > div > .workflow-tabs-container.flex
> [tabs, +, .ml-auto group > user button].

  P1  HOST -- parkHost finds the right-hand .ml-auto group (first place in it);
      without it the end of the tab row; without a tab row null; a class that
      only contains "ml-auto" (xml-auto) does not count.
  P2  KEPT -- writeParked writes the ComfyUI setting (the server keeps it: the
      state survives a restart) and mirrors it in localStorage; readParked
      prefers the setting, falls back to localStorage without one.
  P3  MOUNT -- parked: first child of the right-hand group, before the user
      button, flat (position static, no shadow, no frame), buttons 26 px;
      idempotent; dropped by a redraw of the row it is put back on the next
      tick; unparked: back in the body, floating (fixed, 28 px buttons);
      parked without a tab row: floating.
  P4  WIRING -- the setting is declared (boolean, default off); the tick
      remounts; reset unparks; place() and the drag leave a parked bar alone;
      the park button is not an align op (no data-op, BAR_GROUPS unchanged);
      the node says "on (top bar)" when parked.
  M   MUTATIONS -- seven wounds, each must go red.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _groups_js as GJ  # noqa: E402

TAG = "[test_v1355_align_park]"
CORE = GJ.read("ph_groups_core.js")
TITLE = GJ.read("ph_group_title.js")
SWITCH = GJ.read("ph_group_switch.js")
ALIGN = GJ.read("ph_align.js")
FAILS = []

DRIVER = r"""
class El {
    constructor(tag, cls) { this.tagName = tag; this.className = cls || ""; this.children = []; this.parentNode = null;
        this.style = {}; this.dataset = {}; this.innerHTML = ""; this.title = ""; }
    get firstElementChild() { return this.children[0] || null; }
    get lastElementChild() { return this.children[this.children.length - 1] || null; }
    _detach(ch) { if (ch.parentNode) { const k = ch.parentNode.children; k.splice(k.indexOf(ch), 1); ch.parentNode = null; } }
    appendChild(ch) { this._detach(ch); this.children.push(ch); ch.parentNode = this; return ch; }
    insertBefore(ch, ref) { this._detach(ch); const i = ref ? this.children.indexOf(ref) : -1;
        if (i < 0) this.children.push(ch); else this.children.splice(i, 0, ch); ch.parentNode = this; return ch; }
    _match(sel) { if (/^[a-z]+$/.test(sel)) return this.tagName === sel;
        const cl = String(this.className).split(/\s+/); return sel.split(".").filter(Boolean).every((c) => cl.includes(c)); }
    _all(sel, out) { for (const c of this.children) { if (c._match(sel)) out.push(c); c._all(sel, out); } return out; }
    querySelectorAll(sel) { return this._all(sel, []); }
    querySelector(sel) { return this._all(sel, [])[0] || null; }
}
globalThis.document = { createElement: () => ({ getContext: () => ({ font: "", measureText: (s) => ({ width: String(s).length * 10 }) }) }), body: { appendChild() {} } };
const store = {};
globalThis.localStorage = { getItem: (k) => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); }, removeItem: (k) => { delete store[k]; } };
globalThis.LiteGraph = { NODE_TITLE_HEIGHT: 30 };
const { app } = await import("../../scripts/app.js");
const al = await import("./ph_align.js");
const out = {};

function row(withRight) {
    const doc = new El("div", "root");
    const outer = doc.appendChild(new El("div", "workflow-tabs-container pointer-events-auto relative w-full"));
    const wrap = outer.appendChild(new El("div", "flex h-full items-center"));
    const inner = wrap.appendChild(new El("div", "workflow-tabs-container flex h-full max-w-full flex-auto flex-row"));
    inner.appendChild(new El("div", "p-scrollpanel"));
    inner.appendChild(new El("button", "plus"));
    const right = withRight ? inner.appendChild(new El("div", withRight)) : null;
    const user = right ? right.appendChild(new El("button", "user")) : null;
    return { doc, inner, right, user };
}
// P1
const a = row("ml-auto flex shrink-0 items-center gap-2 px-2");
const h1 = al.parkHost(a.doc);
const b = row(null);
const h2 = al.parkHost(b.doc);
const c = row("xml-auto flex");
const h3 = al.parkHost(c.doc);
out.p1 = [h1 && h1.parent === a.right, h1 && h1.first, h2 && h2.parent === b.inner, h2 && h2.first,
          al.parkHost(new El("div", "root")), h3 && h3.parent === c.inner];
// P2
const calls = [];
let sv;
app.extensionManager = { setting: { get: (k) => (k === al.PARK_SETTING ? sv : undefined),
                                     set: (k, v) => { calls.push([k, v]); sv = v; return Promise.resolve(); } } };
al.writeParked(true);
const p2a = [calls.length, calls[0] && calls[0][0] === al.PARK_SETTING, calls[0] && calls[0][1], store.ph_align_parked, al.readParked()];
sv = false; store.ph_align_parked = "1";
const p2b = al.readParked();                       // the setting wins
sv = undefined;
const p2c = al.readParked();                       // no setting value: the mirror
app.extensionManager = null;
al.writeParked(false);
const p2d = ["ph_align_parked" in store, al.readParked()];
out.p2 = [p2a, p2b, p2c, p2d];
// P3
function makeBar() {
    const bar = new El("div", "ph-align-bar");
    bar.style.display = "flex";
    bar.appendChild(new El("div", "ph-align-handle"));
    const op = bar.appendChild(new El("button", "")); op.dataset.op = "left";
    bar.appendChild(new El("div", "ph-align-sep"));
    bar.appendChild(new El("button", "ph-align-park"));
    return bar;
}
const body = new El("body", "");
const r = row("ml-auto flex shrink-0 items-center gap-2 px-2");
const bar = makeBar();
body.appendChild(bar);
al.writeParked(true);
const m1 = al.mountBar(bar, r.doc, body);
const s1 = [m1.mode, m1.moved, r.right.firstElementChild === bar, r.right.children[1] === r.user,
            /position:static/.test(bar.style.cssText), /box-shadow:none/.test(bar.style.cssText), /border:none/.test(bar.style.cssText),
            bar.querySelectorAll("button").every((x) => x.style.width === "26px" && x.style.height === "26px"),
            bar.dataset.parked, bar.style.display];
const m2 = al.mountBar(bar, r.doc, body);
r.right._detach(bar);                                // Vue redrew the row
const m3 = al.mountBar(bar, r.doc, body);
const s2 = [m2.moved, m3.moved, r.right.firstElementChild === bar];
al.writeParked(false);
const m4 = al.mountBar(bar, r.doc, body);
const s3 = [m4.mode, bar.parentNode === body, r.right.children.length, /position:fixed/.test(bar.style.cssText),
            bar.querySelectorAll("button").every((x) => x.style.width === "28px"), bar.dataset.parked];
al.writeParked(true);
const m5 = al.mountBar(bar, new El("div", "root"), body);
out.p3 = [s1, s2, s3, [m5.mode, bar.parentNode === body]];
al.writeParked(false);
console.log(JSON.stringify(out));
"""


def evaluate(align):
    errs = []
    o, err = GJ.run({"ph_groups_core.js": CORE, "ph_group_title.js": TITLE,
                     "ph_group_switch.js": SWITCH, "ph_align.js": align}, DRIVER)
    if o is None:
        return ["node driver failed: " + str(err)]
    if o["p1"] != [True, True, True, False, None, True]:
        errs.append("P1 host %r" % o["p1"])
    if o["p2"] != [[1, True, True, "1", True], False, True, [False, False]]:
        errs.append("P2 kept %r" % o["p2"])
    want3 = [["parked", True, True, True, True, True, True, True, "1", "flex"],
             [False, True, True],
             ["floating", True, 1, True, True, "0"],
             ["floating", True]]
    if o["p3"] != want3:
        errs.append("P3 mount %r" % o["p3"])
    # P4 wiring, read off the source
    m = re.search(r"settings:\s*\[\{(.*?)\}\]", align, re.S)
    if not m or "id: PARK_SETTING" not in m.group(1) or 'type: "boolean"' not in m.group(1) \
            or "defaultValue: false" not in m.group(1):
        errs.append("P4 the setting is declared (boolean, default off)")
    if "try { remount(bar); } catch (e) { /* keep where it is */ }" not in align:
        errs.append("P4 the tick puts a dropped bar back")
    r = re.search(r"export function resetBarPosition\(\) \{(.*?)\n\}", align, re.S)
    if not r or "writeParked(false);" not in r.group(1):
        errs.append("P4 reset unparks")
    p = re.search(r"function place\(bar\) \{(.*?)\n\}", align, re.S)
    if not p or 'bar.dataset.parked === "1") return;' not in p.group(1):
        errs.append("P4 place() leaves a parked bar alone")
    if 'if (bar.dataset.parked === "1") return;\n        e.preventDefault();' not in align:
        errs.append("P4 the drag leaves a parked bar alone")
    if re.search(r"park\.dataset\.op", align) or "ph-align-park" not in align:
        errs.append("P4 the park button is not an align op")
    g = re.search(r"export const BAR_GROUPS = (\[.*?\]);", align, re.S)
    if not g or "park" in g.group(1):
        errs.append("P4 BAR_GROUPS unchanged")
    if '"Align bar: on (top bar)"' not in align:
        errs.append("P4 the node says when the bar is parked")
    return errs


FAILS.extend(evaluate(ALIGN))
if re.search(r"[^\x00-\x7F]", ALIGN):
    FAILS.append("the file must be plain ASCII")

MUT = [
    ("host ignores the right-hand group", "return right ? { parent: right, first: true } :",
     "return false ? { parent: right, first: true } :"),
    ("setting not written", "const p = s.set(PARK_SETTING, on);", "const p = null;"),
    ("setting not read", "const v = s.get(PARK_SETTING); if (v === true || v === false) return v;",
     "const v = s.get(PARK_SETTING);"),
    ("parked bar keeps the floating frame", "background:transparent;border:none;border-radius:0;box-shadow:none;",
     "background:#202024;border:1px solid #3c3c44;border-radius:7px;box-shadow:0 3px 12px rgba(0,0,0,.45);"),
    ("parked bar goes last", "if (host.first) p.insertBefore(bar, p.firstElementChild || null); else p.appendChild(bar);",
     "p.appendChild(bar);"),
    ("tick does not remount", "try { remount(bar); } catch (e) { /* keep where it is */ }", ""),
    ("reset keeps it parked", "    writeParked(false);\n    if (BAR)", "    if (BAR)"),
]
caught = 0
for name, a, b in MUT:
    if ALIGN.count(a) != 1:
        FAILS.append("M mutation anchor not unique: " + name)
        continue
    if evaluate(ALIGN.replace(a, b)):
        caught += 1
    else:
        FAILS.append("M mutation survived: " + name)

if FAILS:
    for f in FAILS:
        print(TAG + " FAIL: " + f)
    sys.exit(1)
print(TAG + " OK -- 4 promises, host + kept + mount + wiring, %d/%d mutations caught" % (caught, len(MUT)))
