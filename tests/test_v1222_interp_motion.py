# -*- coding: ascii -*-
"""Guard v1222 -- Interpolate: fast motion is not a cut (Frank 24.10., the reef clip: 57 of 123 in-between frames
came out as COPIES -- the cut_guard threshold 0.1 held every pair of the fast camera move as a 'cut', although the
Cine Timeline's cuts pin was wired and the shot had no cut at all).

  M1  the real series of that clip (123 pair differences, median 0.098, max 0.146): without the pin, cut_guard 0.1
      holds 56 pairs (the old behaviour, kept for unwired graphs); with the pin wired and NO cut asked -> 0 held,
      every pair interpolated, and the console says the threshold is off.
  M2  with the pin and one asked cut: exactly the cut's pair is held, the motion around it is interpolated -- also
      when the motion is above the threshold.
  M3  without the pin: the threshold works as before (0 disables, a clear jump is held); static_skip unchanged.
  M4  the node uses _verdicts; the tooltip no longer claims 0.1 "leaves fast motion alone".
  S   ASCII.
Script-style: exit 0 = pass. Pure -- the gate functions are lifted from the source (no torch)."""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "nodes" / "ph_interpolate.py"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


txt = SRC.read_text(encoding="utf-8")
tree = ast.parse(txt)
want = {"_gate_verdict", "_cut_pairs", "_verdicts", "CUT_WINDOW_S", "CUT_RATIO", "CUT_FLOOR", "CUT_GUARD_DEFAULT"}
mod = ast.Module(body=[n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in want) or
                       (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in want for t in n.targets))],
                 type_ignores=[])
ns = {}
exec(compile(mod, str(SRC), "exec"), ns)
V = ns["_verdicts"]
REEF = [0.091, 0.0898, 0.0931, 0.093, 0.0975, 0.0966, 0.099, 0.0992, 0.1052, 0.104, 0.107, 0.1071, 0.1125, 0.1089, 0.1132, 0.1128, 0.1157, 0.1175, 0.1148, 0.1157, 0.1133, 0.1169, 0.1128, 0.1128, 0.1117, 0.1161, 0.1155, 0.1276, 0.1287, 0.1404, 0.1373, 0.137, 0.1278, 0.1448, 0.1456, 0.1333, 0.134, 0.1272, 0.1325, 0.1258, 0.1257, 0.1239, 0.1261, 0.1168, 0.116, 0.1152, 0.1155, 0.1074, 0.1051, 0.1046, 0.104, 0.1016, 0.0976, 0.0956, 0.0946, 0.0968, 0.0944, 0.0973, 0.0985, 0.1043, 0.1058, 0.1083, 0.1074, 0.1087, 0.104, 0.1024, 0.1029, 0.1051, 0.1074, 0.1023, 0.0969, 0.092, 0.0981, 0.0914, 0.094, 0.0861, 0.0994, 0.0938, 0.0982, 0.0857, 0.1079, 0.0906, 0.0919, 0.0877, 0.091, 0.0871, 0.0736, 0.0796, 0.0566, 0.0781, 0.062, 0.0515, 0.0539, 0.0592, 0.0522, 0.059, 0.0456, 0.0544, 0.0472, 0.0455, 0.0415, 0.0464, 0.0462, 0.0421, 0.041, 0.0364, 0.0442, 0.0403, 0.0392, 0.0391, 0.0449, 0.0417, 0.041, 0.0412, 0.0463, 0.0432, 0.0424, 0.0402, 0.0448, 0.0424, 0.0382, 0.0381, 0.0357]

# ---- M1
v0, n0, on0 = V(REEF, 0.0, 0.1, None, 24.0)
_need(v0.count("hold") == 56 and on0, "M1 without the pin: cut_guard 0.1 holds %d of 123 pairs of fast motion (the old way)" % v0.count("hold"))
v1, n1, on1 = V(REEF, 0.0, 0.1, {"seconds": [], "fps": 24.0, "frames": 124}, 24.0)
_need(v1.count("hold") == 0 and v1.count("interp") == 123 and not on1,
      "M1 pin wired, no cut asked: 0 held, all 123 interpolated")
_need(n1 and n1[0].startswith("cuts pin wired: the cut_guard threshold (0.1) is OFF") and "every pair is interpolated" in n1[0],
      "M1 the console says so: %r" % (n1[:1],))

# ---- M2
fast = [0.12] * 60
fast[29] = 0.55                                      # a real cut between frames 29|30 inside fast motion
v2, n2, _ = V(fast, 0.0, 0.1, {"seconds": [30 / 24.0], "fps": 24.0, "frames": 61}, 24.0)
_need([i for i, x in enumerate(v2) if x == "hold"] == [29] and v2.count("interp") == 59,
      "M2 one asked cut inside fast motion: exactly its pair held, the 59 motion pairs interpolated")
_need("only the 1 cut(s) the timeline asked for are held (1 found)" in n2[0], "M2 the console counts asked and found: %r" % n2[0])
v2b, n2b, _ = V(fast, 0.0, 0.1, {"seconds": [30 / 24.0], "frames": 70}, 24.0)
_need(any("the timeline rendered 70 frames, 61 came in" in x for x in n2b), "M2 a frame-count mismatch is still named")

# ---- M3
v3, _, on3 = V(REEF, 0.0, 0.0, None, 24.0)
_need(v3.count("hold") == 0 and not on3, "M3 without the pin, cut_guard 0 disables the threshold")
jump = [0.02] * 20
jump[10] = 0.6
v4, _, _ = V(jump, 0.0, 0.1, None, 24.0)
_need([i for i, x in enumerate(v4) if x == "hold"] == [10], "M3 without the pin a clear jump is still held")
v5, _, _ = V([0.0001, 0.05, 0.0001], 0.001, 0.1, {"seconds": []}, 24.0)
_need(v5 == ["dup", "interp", "dup"], "M3 static_skip works with the pin as before")
_need(ns["CUT_GUARD_DEFAULT"] == 0.1, "M3 the default for graphs without the pin is unchanged (0.1)")

# ---- M4
_need("verdicts, cut_notes, _theta_on = _verdicts(maes, static_skip, cut_guard, cuts, float(source_fps))" in txt
      and "verdicts = [_gate_verdict(m, static_skip, cut_guard) for m in maes]" not in txt,
      "M4 the node decides through _verdicts only")
_need("leaves fast motion" not in txt and "this threshold is OFF" in txt and "measured 0.10-0.15 per frame" in txt,
      "M4 the tooltip tells the truth: fast motion can reach 0.1; with the pin the threshold is off")

for f in (SRC, Path(__file__)):
    _need(all(b < 128 for b in f.read_bytes()), "S  ASCII %s" % f.name)
if FAILS:
    print("test_v1222_interp_motion: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1222_interp_motion: all checks passed")
