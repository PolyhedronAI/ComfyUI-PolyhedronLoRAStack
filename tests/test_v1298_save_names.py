# -*- coding: ascii -*-
"""Guard v1298 -- Save: one save, one time in every file name; the keypoints JSON is named after its media.

(Frank 06.11.: "bitte alle Fixes mit rein ... go" -- the find of the v1295 test run: PH_112605_..._keypoints.json
beside PH_112606_..._00001_.mp4. The date token was resolved once for the JSON and once more for the clip, a second
boundary in between split the names; and the JSON took a counter of its own.)

  N1 freeze_date_tokens: every %date:...% at ONE moment, %width% / %height% and other tokens left for later
  N2 a clip + keypoints under a clock that ticks a second on every call: the JSON is <clip name>_keypoints.json --
     the same time, the same counter, the same folder (ComfyUI's own get_save_image_path, real files); the save
     resolves its date tokens once
  N3 stills + keypoints: the JSON beside the last still, the stills share one time
  N4 keypoints alone: as v1218 (<name>_<counter>_keypoints.json); a second save counts on
  N5 ASCII
Script-style: exit 0 = pass. Needs ComfyUI core on PYTHONPATH (folder_paths) and the video writer.
"""
import datetime as dt
import importlib
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1298_save_names"
FAILS = []
TMP = Path(tempfile.mkdtemp(prefix="pls1298_"))


def _need(c, m):
    print(("  ok    " if c else "  FAIL  ") + m)
    if not c:
        FAILS.append(m)


sys.path.insert(0, os.environ.get("PLS_CORE_ROOT") or "/home/claude/core_pin")   # v1302: the core, also in the cold runner
import folder_paths  # noqa: E402  (ComfyUI core)
folder_paths.set_output_directory(str(TMP / "out"))
folder_paths.set_temp_directory(str(TMP / "temp"))
pkg = types.ModuleType("plsn1298")
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules["plsn1298"] = pkg
SV = importlib.import_module("plsn1298.ph_save")
U = importlib.import_module("plsn1298.ph_save_util")

# ---- N1
now = dt.datetime(2026, 11, 6, 11, 26, 5)
f = U.freeze_date_tokens("%date:yyyy-MM-dd%/PH_%date:hhmmss%_%width%x%height%_%Node.seed%", now)
_need(f == "2026-11-06/PH_112605_%width%x%height%_%Node.seed%", "N1 the dates at one moment, the rest left (%s)" % f)


class Tick(dt.datetime):
    """A clock that is one second later on every call -- every second boundary at once."""
    n = 0

    @classmethod
    def now(cls, tz=None):
        Tick.n += 1
        return dt.datetime(2026, 11, 6, 11, 26, 5) + dt.timedelta(seconds=Tick.n)


U._dt.datetime = Tick
_calls = []
_freeze = U.freeze_date_tokens
U.freeze_date_tokens = lambda prefix, now=None: (_calls.append(prefix), _freeze(prefix, now))[1]
kp = [{"canvas_width": 64, "canvas_height": 48, "people": [{"pose_keypoints_2d": [1.0, 2.0, 0.9] * 18}]} for _ in range(5)]
W = list(SV.ULSSave.INPUT_TYPES()["required"])
vals = {"filename_prefix": "%date:yyyy-MM-dd%/PH_%date:hhmmss%_STAGE", "media_kind": "auto", "image_format": "png",
        "image_quality": 100, "video_preset": "H.264 MP4 (delivery)", "quality": 17, "frame_rate": 24, "autoplay": False,
        "pingpong": False, "loop_count": 0, "trim_to_audio": False, "save_metadata": False, "save_output": True}
args = {k: vals[k] for k in W if k in vals}

# ---- N2 clip + keypoints
img = torch.rand(5, 48, 64, 3)
try:
    r = SV.ULSSave().save(**args, image=img, keypoints=kp)
    clip = r["result"][0]
    folder = Path(clip).parent
    js = sorted(x for x in os.listdir(folder) if x.endswith("_keypoints.json"))
    want = os.path.splitext(os.path.basename(clip))[0].rstrip("_") + "_keypoints.json"
    _need(clip.endswith(".mp4") and js == [want], "N2 the JSON is named after the clip (%s beside %s)" % (js, os.path.basename(clip)))
    d = json.load(open(folder / want))
    _need(d["format"] == "openpose" and len(d["frames"]) == 5, "N2 the JSON holds the 5 frames")
    _need(len(_calls) == 1, "N2 the save resolved its date tokens once (%d)" % len(_calls))
except Exception as ex:
    _need(False, "N2 clip + keypoints: %s" % ex)

# ---- N3 stills + keypoints
args3 = dict(args, filename_prefix="%date:yyyy-MM-dd%/stills/PH_%date:hhmmss%_S", media_kind="image")
r = SV.ULSSave().save(**args3, image=torch.rand(2, 48, 64, 3), keypoints=kp[:2])
files = sorted(os.listdir(Path(r["result"][0]).parent))
pngs = [x for x in files if x.endswith(".png")]
stems = {x.split("_")[1] for x in files}
_need(len(pngs) == 2 and any(x.endswith("_keypoints.json") for x in files) and len(stems) == 1,
      "N3 stills and the JSON carry one time (%s)" % files)

# ---- N4 keypoints alone
args4 = dict(args, filename_prefix="pose/PH_kp")
p1 = SV.ULSSave().save(**args4, keypoints=kp)["result"][0]
p2 = SV.ULSSave().save(**args4, keypoints=kp)["result"][0]
_need(os.path.basename(p1) == "PH_kp_00001_keypoints.json" and os.path.basename(p2) == "PH_kp_00002_keypoints.json",
      "N4 keypoints alone: <name>_<counter>_keypoints.json, counting on (%s, %s)" % (os.path.basename(p1), os.path.basename(p2)))

for p in ("nodes/ph_save_util.py", "tests/" + NAME + ".py"):
    _need(all(c < 128 for c in (ROOT / p).read_bytes()), "N5 ASCII %s" % p)
src = (ROOT / "nodes" / "ph_save.py").read_text(encoding="utf-8")       # (it carries older non-ASCII UI text)
_need(all(all(ord(c) < 128 for c in ln) for ln in src.splitlines() if "v1298" in ln), "N5 ASCII: the v1298 lines of ph_save.py")
print("%s: %d check(s) failed" % (NAME, len(FAILS)) if FAILS else "%s: all checks passed" % NAME)
sys.exit(1 if FAILS else 0)
