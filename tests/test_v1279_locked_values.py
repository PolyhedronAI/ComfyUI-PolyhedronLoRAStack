#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v1279 -- a preset-locked field shows the preset's value.

THE WOUND (found 04.11. while validating FX CRT in the frontend 1.53.10)
The frontend draws a disabled widget WITHOUT its value: BaseWidget's getter
_displayValue returns "" when computedDisabled (read in its bundle). Every
preset node of the pack locks its fields with w.disabled -- so a picked preset
showed "scale - preset" and no number. Frank's rule is that the fields jump
VISIBLY to the preset's values.

THE CURE: plsShowLockedValue(w, locked) gives a locked widget its own
_displayValue (greyed, no arrows, no input, value readable) and removes it on
unlock -- inline in every preset node (their guards copy single files into a
sandbox, an import would break them), byte-equal across the copies.

  L1  the helper exists byte-equal in FX (ph_fx.js), Pixelize / Palette
      (ph_pixel.js), Music Master, Sigma List and FX CRT, and each calls it
      where it locks.
  L2  driven in node against a widget class that behaves like the frontend's:
      FX Look / FX Screen presets, a Pixelize preset, a Music Master preset and
      an FX CRT preset -> every locked numeric / combo field reads its value
      (floats with the widget's precision); 'custom' -> the override is gone and
      the widget behaves like the frontend's again.

Script-style: exit 0 = pass.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

NAME = "v1279"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JS = os.path.join(ROOT, "web", "js")
_fails = []


def _need(cond, msg):
    if not cond:
        _fails.append(msg)
    print(("  ok    " if cond else "  FAIL  ") + msg)


FILES = {"ph_fx.js": "plsShowLockedValue(w, !!locked);", "ph_pixel.js": "plsShowLockedValue(w, !!locked);",
         "ph_music_master.js": "plsShowLockedValue(w, !!locked);", "ph_fx_crt.js": "plsShowLockedValue(w, !!locked);",
         "ph_sigma_list.js": "plsShowLockedValue(w, !!off);"}
# the public cut carries only some of these nodes: the guard checks what the tree has (at least one)
FILES = {f: c for f, c in FILES.items() if os.path.isfile(os.path.join(JS, f))}
_need(len(FILES) >= 1, "L1 the tree carries preset nodes (%s)" % ", ".join(sorted(FILES)))
# ------------------------------------------------------------------ L1
bodies = {}
for f, call in FILES.items():
    s = open(os.path.join(JS, f), encoding="ascii").read()
    m = re.search(r"function plsShowLockedValue\(w, locked\) \{.*?\n\}\n", s, re.S)
    bodies[f] = m.group(0) if m else None
    _need(m is not None and call in s, "L1 %s carries the helper and calls it where it locks" % f)
_need(len(set(bodies.values())) == 1 and None not in bodies.values(), "L1 the copies are byte-equal")
if "ph_sigma_list.js" in FILES:
    _need("w.disabled = !!off;" in open(os.path.join(JS, "ph_sigma_list.js"), encoding="ascii").read(),
          "L1 sigma list still greys (the helper adds the value, it does not unlock)")

# ------------------------------------------------------------------ L2
NODE = shutil.which("node")
if not NODE:
    print("  skip  L2 (no node binary)")
else:
    tmp = tempfile.mkdtemp(prefix="pls1279_")
    os.makedirs(os.path.join(tmp, "scripts"))
    shutil.copytree(JS, os.path.join(tmp, "ext", "js"))
    open(os.path.join(tmp, "scripts", "app.js"), "w").write(
        "export const app = { registerExtension() {}, graph: { _nodes: [] }, canvas: { ds: { scale: 1 } } };\n")
    open(os.path.join(tmp, "scripts", "api.js"), "w").write(
        "export const api = { fetchApi: async () => ({ json: async () => ({ ok: false }) }), addEventListener() {} };\n")
    drv = r"""
// a widget that draws like the frontend 1.53: disabled -> no value
class FrontWidget {
  constructor(name, value, options) { this.name = name; this.value = value; this.options = options || {}; this.disabled = false; }
  get computedDisabled() { return !!this.disabled; }
  get _displayValue() { return this.computedDisabled ? "" : String(this.value); }
}
const mkNode = (type, fields) => ({ type, _fxType: type, setDirtyCanvas() {}, size: [300, 200], properties: {},
  widgets: [new FrontWidget("preset", "custom (your settings)", { values: [] })].concat(fields.map((f) => new FrontWidget(f, 0, { precision: 2 }))) });
const W = (n, f) => n.widgets.find((w) => w.name === f);
const out = {};
async function probe(label, mod, type, fields, preset) {
  const M = await import(mod);
  const n = mkNode(type, fields);
  W(n, "preset").value = preset; M.applyPreset(n);
  const locked = n.widgets.filter((w) => w.name !== "preset" && w.disabled);
  const shown = locked.map((w) => [w.name, w._displayValue, w.value]);
  const blank = shown.filter(([k, d, v]) => d === "" && v !== "" && v != null);
  W(n, "preset").value = "custom (your settings)"; M.applyPreset(n);
  const after = n.widgets.filter((w) => w.name !== "preset");
  const own = after.filter((w) => Object.prototype.hasOwnProperty.call(w, "_displayValue")).length;
  after[0].disabled = true;
  out[label] = { locked: locked.length, blank: blank.length, sample: shown, own, front: after[0]._displayValue };
}
const HAS = __HAS__;
if (HAS.includes("ph_fx.js")) {
  const C = await import("./ext/js/ph_fx_core.js");
  await probe("fx look", "./ext/js/ph_fx.js", "ULSFXLook", C.L_FIELDS, "crt monitor");
  await probe("fx screen", "./ext/js/ph_fx.js", "ULSFXScreen", C.S_FIELDS, "night vision");
}
if (HAS.includes("ph_pixel.js")) { const PX = await import("./ext/js/ph_pixel.js");
  await probe("pixelize", "./ext/js/ph_pixel.js", "ULSPixelize", PX.FIELDS, Object.keys(PX.PRESETS || {})[0]); }
if (HAS.includes("ph_music_master.js")) { const MM = await import("./ext/js/ph_music_master.js");
  await probe("music master", "./ext/js/ph_music_master.js", "ULSMusicMaster", MM.FIELDS, Object.keys(MM.PRESETS)[0]); }
if (HAS.includes("ph_fx_crt.js")) { const CR = await import("./ext/js/ph_fx_crt.js");
  await probe("fx crt", "./ext/js/ph_fx_crt.js", "ULSFXCRT", CR.FIELDS, "pixel tv (slot mask)"); }
console.log(JSON.stringify(out));
"""
    drv = drv.replace("__HAS__", json.dumps(sorted(FILES)))
    open(os.path.join(tmp, "drive.mjs"), "w", encoding="ascii").write(drv)
    r = subprocess.run([NODE, "drive.mjs"], capture_output=True, text=True, timeout=180, cwd=tmp)
    o = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else None
    _need(o is not None, "L2 the preset modules run in node (%s)" % (r.stderr or "")[-400:])
    if o:
        for k, v in o.items():
            _need(v["locked"] > 0 and v["blank"] == 0,
                  "L2 %s: %d locked fields, none blank (e.g. %s)" % (k, v["locked"], v["sample"][1:3]))
            _need(v["own"] == 0 and v["front"] == "", "L2 %s: 'custom' removes the override (frontend behaviour back)" % k)
        fl = [s for v in o.values() for s in v["sample"] if isinstance(s[2], float) and not float(s[2]).is_integer()]
        _need((len(fl) >= 5 or not o) and all(s[1] == ("%.2f" % s[2]) for s in fl), "L2 floats keep the widget's precision (%s)" % fl[:2])
    shutil.rmtree(tmp, ignore_errors=True)

print("[%s] %d failed" % (NAME, len(_fails)))
print("[%s] %s" % (NAME, "PASS" if not _fails else "FAIL"))
sys.exit(1 if _fails else 0)
