# -*- coding: ascii -*-
"""Guard v1346 -- what the Power Upscale and the blank picture SHOW is what they DO.

(Frank 19.11.: "Warum haben wir hier nur einen Step und keine 6 oder mehr?" -- the ESRGAN stage ran no step at all,
refine_order 'off', and still showed steps 1 / denoise 0.16. And the blank picture 0-0.25 said "steps 1-2 of 6" while
it covered steps 1-4 of 8 on the HyperFlow curve shifted 12. "Ja, dann bitte erst mal alles fixen ... go.")

  C1 Power Upscale JS: REFINE_ONLY lists the dials the backend says reach nothing with refine_order 'off', they hide
     through the one _setDisabled path, refine_order re-applies on change, one OR-verdict per row (no second loop
     can show a row the first hid), no curve label without a refine, banner v1346
  C2 the backend's own console line names the same dials (the list is not invented in JS)
  C3 Sampler: _cond_windows on Core's MiniMaxH3ImageModelSampling semantics -- window 0-0.25 on the PDD grid
     [1, .9675, .8875, .75, .5328, .232, 0] -> steps 1-2 of the own 6-step grid, 1-4 of 8 on the HyperFlow curve
     shifted 12; a full 0-1 cond is not reported; the readout is called on the wired-curve path
  C4 Image Reference: the info tail says the count is on the model's own grid and that a curve moves it
Script-style: exit 0 = pass.
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


JS = (ROOT / "web" / "js" / "ph_power_upscale.js").read_text(encoding="utf-8")
m = re.search(r"const REFINE_ONLY = \[([^\]]*)\]", JS)
names = re.findall(r'"([^"]+)"', m.group(1)) if m else []
_need(set(names) == {"upscale_by", "denoise", "steps", "cfg", "sampler_name", "scheduler", "tile_size",
                     "tile_overlap", "sigma_shift"}, "C1 REFINE_ONLY = %s" % names)
body = JS[JS.index("function _applyModeState(node) {"):JS.index("// \u2500\u2500 v1212: a wired curve IS the schedule")]
_need("for (const name of REFINE_ONLY) add(name, refOff);" in body and "for (const [name, off] of hide)" in body
      and body.count("_setDisabled(") == 1, "C1 one OR-verdict per row, one _setDisabled call")
_need('String(ro.value == null ? "" : ro.value).trim() === "off"' in body, "C1 refOff read from refine_order")
_need("_curveLabels(node, wired && !refOff, dual)" in body, "C1 no curve label without a refine")
_need('const row = _findWidget(self, "refine_order");' in JS and "row.callback = function" in JS,
      "C1 refine_order re-applies on change")
_need('ph_power_upscale.js v1346 loaded' in JS, "C1 banner v1346")
vd = JS[JS.index("function _applyVerdict("):JS.index("// \u2550\u2550\u2550 v549")] if "function _applyVerdict(" in JS else ""
_need("joint || (refOff && REFINE_ONLY.includes(name))" in vd,
      "C1 the verdict (JOINT_UNREAD) does not show the refine-off dials again (field 19.11.: tile rows stayed)")

PY = (ROOT / "nodes" / "ph_power_upscale.py").read_text(encoding="utf-8")
_need("upscale_by, denoise, steps, \"\n                  \"cfg and the tile dials reach nothing" in PY
      or "upscale_by, denoise, steps, cfg and the tile dials reach nothing" in PY.replace('"\n                  "', ""),
      "C2 the backend names the unread dials itself")

S = (ROOT / "nodes" / "uls_sampler.py").read_text(encoding="utf-8")
a, b = S.index("def _cond_windows"), S.index("# \u2500\u2500 v544: per-expert sampler")
ns = {}
exec(S[a:b], ns)


class _MS:
    """Core's MiniMaxH3ImageModelSampling.percent_to_sigma (comfy/ldm/minimax/sampling.py), pure python."""
    grid = [0.0, 0.2320, 0.5328, 0.75, 0.8875, 0.9675, 1.0]

    def percent_to_sigma(self, p):
        if p <= 0.0:
            return 1.0
        if p >= 1.0:
            return 0.0
        n = len(self.grid) - 1
        pos = min(max(1.0 - p, 0.0), 1.0) * n
        i = min(int(pos), n - 1)
        return self.grid[i] + (self.grid[i + 1] - self.grid[i]) * (pos - i)


class _M:
    def get_model_object(self, name):
        return _MS()


class _T(list):
    def flatten(self):
        return self

    def tolist(self):
        return list(self)

    def numel(self):
        return len(self)


raw = [1, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0]
s12 = _T([12.0 * v / (1.0 + 11.0 * v) for v in raw])
grid = _T([1.0, 0.9675, 0.8875, 0.75, 0.5328, 0.2320, 0.0])
cond = [[None, {"start_percent": 0.0, "end_percent": 0.25}], [None, {}]]
w12 = ns["_cond_windows"](_M(), s12, cond)
wg = ns["_cond_windows"](_M(), grid, cond)
_need(len(w12) == 1 and w12[0][4] == [1, 2, 3, 4], "C3 S12 curve: window 0-0.25 -> steps %s of 8" % (w12[0][4] if w12 else None))
_need(len(wg) == 1 and wg[0][4] == [1, 2], "C3 own grid: window 0-0.25 -> steps %s of 6" % (wg[0][4] if wg else None))
_need(abs(w12[0][3] - 0.9275) < 1e-4 if w12 else False, "C3 the window ends at sigma 0.9275")
_need("_report_cond_windows(model, sigmas, positive)" in S and S.index("_report_cond_windows(model, sigmas, positive)")
      < S.index("        if dual_moe:\n            # High + Low: the boundary drives"), "C3 readout on the wired-curve path")

R = (ROOT / "nodes" / "ph_image_ref.py").read_text(encoding="utf-8")
_need("of %d on the model's own grid" in R and "the Sampler names the real steps" in R, "C4 info says own grid + curve")
ast.parse(S); ast.parse(R)

print("FAIL" if FAILS else "PASS", "test_v1346_refine_off_cond_window")
sys.exit(1 if FAILS else 0)
