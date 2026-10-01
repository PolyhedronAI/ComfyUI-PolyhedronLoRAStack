# -*- coding: ascii -*-
"""Guard v1019 -- ph_reference_board.js: the board's pure helpers and parity.

  R1  ROLES / RETENTION / LIMIT are the backend's (h3_prompt.ROLES /
      RETENTION, ph_reference_board N_*): a role the UI offers is a role the
      backend accepts.
  R2  sanitizeTag(): from a file name to a valid, unique, non-slot tag; a
      video soundtrack name (<tag>_sound) is taken too.
  R3  kindOf / defaultEntry / counts / canAdd / moveEntry / parseBoard behave.
  R4  The board widget is hidden, the view is a fixed-height DOM field that
      never serialises, files go through Core's /upload/image into pls_board.

Driven in node.js; `app`, `api` stubbed, the real ph_widget_vis.js beside it.
Script-style: exit 0 = pass.
"""
import ast
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "web" / "js" / "ph_reference_board.js"
FAILS = []


def _fail(m):
    FAILS.append(m)
    print("  FAIL  " + m)


def _ok(m):
    print("  ok    " + m)


if shutil.which("node") is None:
    print("  note  node.js not found -- SKIPPED")
    sys.exit(0)

hp = ast.parse((ROOT / "nodes" / "h3_prompt.py").read_text(encoding="utf-8"))
py = {}
for n in hp.body:
    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("ROLES", "RETENTION"):
        py[n.targets[0].id] = ast.literal_eval(n.value)
rb = (ROOT / "nodes" / "ph_reference_board.py").read_text(encoding="utf-8")

src = JS.read_text(encoding="utf-8")
tmp = Path(tempfile.mkdtemp(prefix="v1019js_"))
(tmp / "m.mjs").write_text(src.replace('import { app } from "../../scripts/app.js";', "const app = { registerExtension() {} };")
                           .replace('import { api } from "../../scripts/api.js";', "const api = { apiURL: (p) => p };"), encoding="utf-8")
shutil.copy(ROOT / "web" / "js" / "ph_widget_vis.js", tmp / "ph_widget_vis.js")
(tmp / "h.mjs").write_text(r"""
import * as B from "./m.mjs";
const f = (n) => ({ filename: n, subfolder: "pls_board", type: "input" });
const e1 = B.defaultEntry("image", f("My Fox!.PNG"), []);
const e2 = B.defaultEntry("image", f("my_fox.png"), [e1.tag]);
const v = Object.assign(B.defaultEntry("video", f("walk.mp4"), []), { sound: "own" });
const out = {
  roles: B.ROLES, ret: B.RETENTION, limit: B.LIMIT,
  tags: [B.sanitizeTag("image_2.png", []), B.sanitizeTag("12 monkeys.jpg", []), B.sanitizeTag("x.png", ["x"]),
         B.sanitizeTag("walk_sound.wav", B.takenTags([v]))],
  e1, e2, kinds: [B.kindOf("a.WEBP"), B.kindOf("b.mov"), B.kindOf("c.flac"), B.kindOf("d.txt")],
  counts: B.counts([e1, e2, v]), can: [B.canAdd(Array(9).fill({ kind: "image" }), "image"), B.canAdd([v, v], "video")],
  moved: B.moveEntry([1, 2, 3], 0, 1), edge: B.moveEntry([1, 2, 3], 2, 1),
  parse: [B.parseBoard("nonsense"), B.parseBoard("{}"), B.parseBoard('[{"a":1}]')],
};
console.log(JSON.stringify(out));
""", encoding="utf-8")
r = subprocess.run(["node", str(tmp / "h.mjs")], capture_output=True, text=True, timeout=60)
if r.returncode:
    print(r.stderr[-1500:])
    _fail("harness crashed")
    sys.exit(1)
d = json.loads(r.stdout.strip().splitlines()[-1])

if d["roles"] == py["ROLES"] and d["ret"] == py["RETENTION"]:
    _ok("R1 ROLES and RETENTION are the backend's, value for value")
else:
    _fail("R1 js %r / %r vs py %r" % (d["roles"], d["ret"], py))
if d["limit"] == {"image": 9, "video": 3, "audio": 3} and "N_IMAGES, N_VIDEOS, N_AUDIOS = 9, 3, 3" in rb:
    _ok("R1 limits 9 / 3 / 3 on both sides")
else:
    _fail("R1 limits %r" % d["limit"])

t = d["tags"]
if t[0] == "ref_image_2" and t[1] == "ref_12_monkeys" and t[2] == "x_2" and t[3] == "walk_sound_2":
    _ok("R2 tags: slot names and leading digits prefixed, duplicates numbered, soundtrack names taken")
else:
    _fail("R2 tags %r" % t)
if d["e1"]["tag"] == "my_fox" and d["e2"]["tag"] == "my_fox_2" and d["e1"]["role"] == "subject" \
        and d["e1"]["retention"] == "fully_preserved":
    _ok("R2 defaultEntry: clean tag from the file name, unique, first role, visual retention")
else:
    _fail("R2 entries %r %r" % (d["e1"], d["e2"]))

if d["kinds"] == ["image", "video", "audio", None] and d["counts"] == {"image": 2, "video": 1, "audio": 0} \
        and d["can"] == [False, True] and d["moved"] == [2, 1, 3] and d["edge"] == [1, 2, 3] \
        and d["parse"] == [[], [], [{"a": 1}]]:
    _ok("R3 kindOf / counts / canAdd / moveEntry / parseBoard behave")
else:
    _fail("R3 %r" % {k: d[k] for k in ("kinds", "counts", "can", "moved", "edge", "parse")})

if 'setHidden(_w(this, "board"), true)' in src and "w.computeSize = (width) => [width, HEIGHT]" in src \
        and "serialize: false" in src and '"/upload/image"' in src and 'const SUB = "pls_board"' in src:
    _ok("R4 hidden board widget, fixed-height non-serialising view, Core upload into pls_board")
else:
    _fail("R4 view contract not pinned")

shutil.rmtree(tmp, ignore_errors=True)
print()
if FAILS:
    print("test_v1019_reference_board_js: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1019_reference_board_js: PASS (R1-R4)")
