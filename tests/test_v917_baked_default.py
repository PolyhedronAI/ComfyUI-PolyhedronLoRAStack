"""v917 guard -- the field decided: baked by default, and the console says so.

Frank's A/B of 05.09. (MiniMax H3 int8-convrot, same noise seed): baked
visibly better and ~10 % faster per step than bypass. Four cuts, one guard:

  P1  _apply_decision: auto -> BAKED on quantized AND plain targets;
      "bypass" and "patch" by hand still do what they say
  P2  the decision is never silent: auto/patch say "BAKED", auto on a
      quantized target additionally names the pill's BYPASS hand switch
  P3  the mixed-convention fallback in uls_stack_node lists every LoRA of
      the group with its convention label (05.09.: eight LoRAs, no name)
  P4  Engine Apply pill sits LEFT after the S|C|D buttons (after the DARE
      variant pill when shown), never at W - PAD (the output-pin column)
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "nodes"))

import uls_merge_policy as MP  # noqa: E402


def _fail(msg):
    print("[test_v917_baked_default] FAIL -- " + msg)
    sys.exit(1)


# P1
for q in (True, False):
    for a in ("auto", None, "nonsense"):
        if MP._apply_decision(a, q, "SEQ")[0] is not False:
            _fail("P1 apply=%r quantized=%r must be BAKED" % (a, q))
if MP._apply_decision("bypass", False, "SEQ")[0] is not True:
    _fail("P1 explicit bypass must stay bypass")
if MP._apply_decision("patch", True, "SEQ")[0] is not False:
    _fail("P1 explicit patch must stay baked")

# P2
for a, q in (("auto", True), ("auto", False), ("patch", True), ("patch", False),
             ("bypass", True), ("bypass", False)):
    ub, mode, notes = MP._apply_decision(a, q, "SEQ")
    if not notes:
        _fail("P2 apply=%r quantized=%r printed nothing" % (a, q))
    head = notes[0]
    if ub and not head.startswith("BYPASS"):
        _fail("P2 bypass note must start with BYPASS: " + head)
    if not ub and not head.startswith("BAKED"):
        _fail("P2 baked note must start with BAKED: " + head)
ub, mode, notes = MP._apply_decision("auto", True, "SEQ")
if "BYPASS" not in notes[0] or "pill" not in notes[0]:
    _fail("P2 auto on quantized must name the pill's BYPASS hand switch: " + notes[0])
ub, mode, notes = MP._apply_decision("auto", False, "SEQ")
if "BYPASS" in notes[0]:
    _fail("P2 auto on a plain target must not advertise bypass: " + notes[0])
if mode != "SEQ":
    _fail("P2 baked SEQ must stay SEQ")

# P3
src = open(os.path.join(ROOT, "nodes", "uls_stack_node.py"), encoding="utf-8").read()
m = re.search(r"group mixes LoRA naming conventions(.*?)return _apply_seq", src, re.S)
if not m:
    _fail("P3 mixed-convention fallback not found")
blk = m.group(1)
if "zip(valid_names, convs)" not in blk or "_convention_label(c)" not in blk:
    _fail("P3 the fallback must list every LoRA with its convention")
if "def _convention_label(conv)" not in src:
    _fail("P3 _convention_label helper missing")
ns = {}
exec(compile(re.search(r"def _convention_label\(conv\):.*?\n\n\n", src, re.S).group(0),
             "<lifted>", "exec"), ns)
lab = ns["_convention_label"]
if "kohya" not in lab((".lora_up.weight", ".lora_down.weight")):
    _fail("P3 kohya label")
if "WAN/FLUX" not in lab((".lora_B.weight", ".lora_A.weight")):
    _fail("P3 WAN/FLUX label")
if lab(None) != "unknown":
    _fail("P3 None label")

# P4
js = open(os.path.join(ROOT, "web", "js", "uls_node.js"), encoding="utf-8").read()
m = re.search(r"// v913: Apply pill -- cycles Auto / Bypass / Baked\.(.*?)uls\._applyRect = ", js, re.S)
if not m:
    _fail("P4 Engine Apply pill block not found")
blk = m.group(1)
code = "\n".join(l for l in blk.splitlines() if not l.strip().startswith("//"))
if "W - PAD - pillW" in code:
    _fail("P4 Engine pill still anchored at the right edge (output-pin column)")
if "afterX + 8" not in code or "uls._dareVariantRect" not in code or ": mbx" not in code:
    _fail("P4 Engine pill must sit after S|C|D (or after the DARE variant pill)")

print("[test_v917_baked_default] OK - auto=BAKED on every target, the decision "
      "always speaks (BAKED / BYPASS), quantized auto names the pill switch, "
      "mixed-convention fallback lists the LoRAs, Engine pill left of the pins "
      "(4 pins)")
