# -*- coding: ascii -*-
"""Guard v1160 -- the Cine Timeline reads only the clips it may read.

Found 17.10. while cutting the public build: the timeline's routes (/pls/cine/probe, /edit, /export -- and the run)
read a layer's clip through layer_path(). A clip "read where it lies" is an ABSOLUTE path the client names; under
--listen every client on the network could probe any video on this machine and pull it out through an export. And a
"copied" clip (relative to input/) was joined to input/ without a check -- "../.." climbed out of it.

  L1  an absolute path is refused while the pack's LAN lock holds (bound beyond localhost, no ULS_ALLOW_LAN_BROWSE);
      with the lock open (localhost, or the override) it is read as before
  L2  a relative path must stay inside input/ -- "../" out of it is refused, also on localhost
  L3  a render layer stays inside output/ (unchanged)
  L4  _lan_locked() asks the pack's ONE lock (uls_routes / ph_media_routes._lan_browse_locked), fail-open
  L5  every way a layer's file is read goes through layer_path (probe_layer, stack_of)
  S   ASCII
"""
import os
import re
import shutil
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1160_cine_paths"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


SRC = (ROOT / "nodes" / "ph_cine_timeline.py").read_text(encoding="ascii")
TMP = Path(tempfile.mkdtemp(prefix="v1160_"))
out, inp, far = TMP / "output", TMP / "input", TMP / "elsewhere"
for d in (out / "day", inp / "cine_media", far):
    d.mkdir(parents=True)
for f in (out / "day" / "r.mp4", inp / "cine_media" / "c.mp4", far / "x.mp4"):
    f.write_bytes(b"v")
fp = types.ModuleType("folder_paths")
fp.get_output_directory = lambda: str(out)
fp.get_input_directory = lambda: str(inp)
sys.modules["folder_paths"] = fp

LOCK = {"on": False}
ns = {"os": os, "re": re, "__package__": None, "VIDEO_EXT": re.compile(r"\.(mp4|webm|mov|mkv|m4v|avi)$", re.I)}
exec("def _output_dir():\n    return %r\n" % str(out), ns)
exec(SRC[SRC.index("def _lan_locked():"):SRC.index("def _layer_clip(")], ns)
ns["_lan_locked_real"] = ns["_lan_locked"]
ns["_lan_locked"] = lambda: LOCK["on"]
lp = ns["layer_path"]


def refused(src):
    try:
        lp(src)
        return False
    except ValueError:
        return True


absf = str(far / "x.mp4")
LOCK["on"] = False
_need(lp({"kind": "file", "path": absf}) == os.path.realpath(absf), "L1 lock open: an absolute clip is read where it lies")
LOCK["on"] = True
_need(refused({"kind": "file", "path": absf}), "L1 lock on (--listen): an absolute clip is refused")
try:
    lp({"kind": "file", "path": absf})
except ValueError as e:
    _need("ULS_ALLOW_LAN_BROWSE" in str(e) and "copy" in str(e), "L1 the refusal names the override and the copy way")
_need(lp({"kind": "file", "path": "cine_media/c.mp4"}) == os.path.realpath(str(inp / "cine_media" / "c.mp4")),
      "L1 lock on: a copied clip (in input/) still works")
LOCK["on"] = False
_need(refused({"kind": "file", "path": "../elsewhere/x.mp4"}), "L2 '../' out of input/ is refused (also on localhost)")
_need(refused({"kind": "file", "path": "cine_media/../../elsewhere/x.mp4"}), "L2 a climb hidden inside the path is refused")
_need(lp({"kind": "render", "path": "day/r.mp4"}) == os.path.realpath(str(out / "day" / "r.mp4")) and refused({"kind": "render", "path": "../elsewhere/x.mp4"}),
      "L3 a render layer: inside output/ read, outside refused")

# L4: the real _lan_locked asks the pack's lock, fail-open
lk = SRC[SRC.index("def _lan_locked():"):SRC.index("def layer_path(")]
_need("_lan_browse_locked()" in lk and '"uls_routes", "ph_media_routes"' in lk and lk.rstrip().endswith("return False"),
      "L4 _lan_locked asks uls_routes / ph_media_routes._lan_browse_locked and fails open")
ns2 = {"__package__": None}
exec(lk, ns2)
sys.modules.pop("uls_routes", None)
sys.modules.pop("ph_media_routes", None)
fake = types.ModuleType("ph_media_routes")
fake._lan_browse_locked = lambda: True
sys.modules["ph_media_routes"] = fake
_need(ns2["_lan_locked"]() is True, "L4 the lock found in ph_media_routes (a build without the shared route file) is used")
del sys.modules["ph_media_routes"]
_need(ns2["_lan_locked"]() is False, "L4 no lock module importable -> unlocked (fail-open, local behaviour)")

# L5: every read goes through layer_path
_need(len(re.findall(r"layer_path\(", SRC)) == 4 and "full = layer_path(src)" in SRC and "pth = layer_path(sx)" in SRC
      and "path = layer_path(sx)" in SRC,   # AMENDED v1198: every file of a track; AMENDED v1221: + snap_edit (v1219)
      "L5 probe_layer, stack_of and snap_edit read through layer_path (the definition + three callers, nothing else)")

shutil.rmtree(TMP, ignore_errors=True)
_need(all(b < 128 for b in (ROOT / "nodes" / "ph_cine_timeline.py").read_bytes()) and all(b < 128 for b in Path(__file__).read_bytes()), "S  ASCII")
print("%s: %d check(s) failed" % (NAME, len(FAILS)) if FAILS else "%s: all checks passed" % NAME)
sys.exit(1 if FAILS else 0)
