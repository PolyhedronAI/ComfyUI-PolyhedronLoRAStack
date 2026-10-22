# -*- coding: ascii -*-
"""Guard v1202 -- the model door's line: a check mark ONLY for what is really there.

(Frank 21.10.: "Wieso kriege ich dann einen gruenen Haken, obwohl nichts installiert ist ... kompletter Unsinn!")

AMENDED v1204: v1202's lineVerdict (grey lines without a mark) was replaced by the need-driven line (v1204, Frank:
"Das sollte dann erscheinen, sobald man es wirklich braucht"): a line appears only for files the node NEEDS now, so
the grey "nothing on disk" states are gone -- no line instead. The v1202 promise is pinned on the new code:

  V1 the green check sits ONLY in the branch where every NEEDED file is installed (need non-empty, miss empty)
  V2 no line ever says "ready" without a file of its own: the v1095 phrase and the status field 'covered' are gone
  V3 no line is grey WITH a Download (the optional / recommended offer is gone; a Download only beside orange)
  Z  ASCII
Script-style: exit 0 = pass.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


WJ = (ROOT / "web" / "js" / "ph_world_weights.js").read_text(encoding="utf-8")
WP = (ROOT / "nodes" / "ph_world_weights.py").read_text(encoding="utf-8")
us = WJ[WJ.find("async function updateSet(node)"):WJ.find("function setup(node, kind)")]
i_need = us.find("if (!need.length) { line.hide(); return; }")
i_miss = us.find("if (!miss.length) {")
i_ok = us.find('line.show("\\u2713 " + txt.title')
_need(0 < i_need < i_miss < i_ok and us.count("\\u2713 ") >= 1 and us[i_miss:i_ok].count("return") == 0,
      "V1 the green check only where every needed file is installed")
_need("ready (a model on disk does the job)" not in WJ and '"covered"' not in WP, "V2 no 'ready' without a file; 'covered' gone")
_need("optional " not in us.replace("optional &&", "") and "recommended" not in us
      and us.count("line.show(") == 4 and '"#e0a030", total ? "Download "' in us, "V3 a Download only beside orange, never grey")
for p in ("web/js/ph_world_weights.js", "tests/test_v1202_line_verdict.py"):
    _need(all(c < 128 for c in (ROOT / p).read_bytes()), "Z ASCII %s" % p)
print("%s: %d check(s) failed" % ("test_v1202_line_verdict", len(FAILS)) if FAILS else "test_v1202_line_verdict: all checks passed")
sys.exit(1 if FAILS else 0)
