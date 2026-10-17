"""GATE v580 - the wire that was THERE and still WRONG, plus the control frame.

PART A -- THE MIS-WIRED SOCKET.

Frank's HIGH token counter read TRIGGERS: 1710 / 512 (334%). His LOW counter,
same eleven LoRAs, same prompt, read 38. That gap was treated as EVIDENCE for
three rounds: a "tag soup" was diagnosed from it, a fix (v579) was built for it,
a guard was written for the fix, and a Trigger Law was written into the handover.

All of it was fiction. The workflow JSON settled it in one read:

    #155 Token Counter . trigger_words  <-  #256 Stack HIGH . uls_config_out
    #156 Token Counter . trigger_words  <-  #257 Stack LOW  . trigger_words

The HIGH counter was tokenising the serialised Stack config -- 15 rows of JSON,
about 6000 characters. STRING fits into STRING, so LiteGraph passed it in
silence. The eleven real triggers measure 129 characters.

The v568 wire law guards "unwired". NOTHING guarded "wired, to the wrong socket"
-- and a socket NAME is a label, not a contract. So the counter now looks at what
it was handed, and says so ON the number it distrusts.

PART B -- THE CONTROL FRAME.

media_kind='auto' on a real clip now also writes ONE still beside the video:
the MIDDLE frame, carrying the workflow (an .mp4 cannot). 'video' and 'image'
are unchanged. A single frame under 'auto' is not a clip and stays one image.
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STACK = os.path.join(HERE, "..", "nodes", "uls_stack_node.py")
SAVE = os.path.join(HERE, "..", "nodes", "ph_save.py")


def _fail(msg):
    print(f"[test_v580_wire_and_control] FAIL: {msg}")
    sys.exit(1)


# =============================================================================
# PART A - execute the real detector
# =============================================================================
src = open(STACK, encoding="utf-8").read()
tree = ast.parse(src)
ns = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("_mis_wired_trigger_input", "_wrap_note"):
        exec(compile(ast.Module([node], []), STACK, "exec"), ns)
    elif isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "_TRIG_SANE_CHARS":
                exec(compile(ast.Module([node], []), STACK, "exec"), ns)
if "_mis_wired_trigger_input" not in ns:
    _fail("_mis_wired_trigger_input is not at module level in uls_stack_node.py")
detect = ns["_mis_wired_trigger_input"]

# ---- A1: THE ACTUAL PAYLOAD Frank's counter was handed -----------------------
# A faithful miniature of uls_config_out: the config dict with lora_info glued on.
CONFIG_OUT = (
    '{"rows": [{"name": "polyhedron_wan2.2_polyhedron_all_high_noise_high_noise.safetensors",'
    ' "weight": 0.25, "group": "subject", "enabled": true, "order": 0},'
    ' {"name": "wan2.2_glamour_high_noise.safetensors", "weight": 0.15,'
    ' "group": "subject", "enabled": true, "order": 1}],'
    ' "lora_info": [{"name": "polyhedron_wan2.2_polyhedron_all", "weight": 0.25,'
    ' "group": "subject", "trigger_words": "all", "trigger_src": "name"}]}'
)
if not detect(CONFIG_OUT, 1710):
    _fail("the config object that Frank's HIGH counter actually tokenised is NOT "
          "flagged. That string read 1710 tokens against a 512 budget and was "
          "believed for three rounds.")

# ---- A2: the REAL trigger string must pass, silently -------------------------
REAL_TRIGGERS = ("noise, glamour, du8ne, oxyge8n, r8ings, vi8ctorian, lac8e, "
                 "rai8n, hour, rays, apple tree")            # 87 chars, ~35 tokens
if detect(REAL_TRIGGERS, 35):
    _fail(f"the REAL trigger string of Frank's 11-LoRA stack is being flagged: "
          f"{detect(REAL_TRIGGERS, 35)!r}. A false alarm on the healthy wire is "
          f"worse than no alarm at all -- it teaches you to ignore the line.")
for ok in ("", "   ", "golden hour", "lac8e", "du8ne, oxyge8n",
           "a, b, c, d, e, f, g, h, i, j, k, l"):
    if detect(ok, 12):
        _fail(f"false alarm on a legitimate trigger string: {ok!r}")

# ---- A3: every shape of "that is not a trigger list" -------------------------
if not detect('{"a": 1}', 5):
    _fail("a string starting with '{' must be flagged (a config object announces "
          "itself in the first character)")
if not detect('[1, 2, 3]', 5):
    _fail("a string starting with '[' must be flagged")
if not detect('  {"rows": []}  ', 5):
    _fail("leading whitespace must not hide the brace")
if not detect("x" * 700, 800):
    _fail(f"a {700}-char 'trigger list' must be flagged -- the real one measures 129")
if detect("x" * 400, 400):
    _fail("a 400-char string must NOT be flagged: a hand-curated list is INTENT, and "
          "the trigger law says intent is not capped. Only the absurd is called out.")

# ---- A4: the alarm must reach the REPORT, not just the console ---------------
if "THIS NUMBER IS NOT A TRIGGER COUNT" not in src:
    _fail("the alarm must be printed IN the report, on the line of the number it "
          "distrusts. Frank's 1710 was believed for three rounds precisely because "
          "nothing standing next to it ever asked whether it could be a trigger count.")
if "_mis_wired_trigger_input(trigger_words" not in src:
    _fail("count() must run the detector on its trigger_words input")

# =============================================================================
# PART B - the control frame
# =============================================================================
save_src = open(SAVE, encoding="utf-8").read()
if "_write_control_frame" not in save_src:
    _fail("_write_control_frame is missing from ph_save.py")

body = save_src.split("def _write_control_frame(")[1]
body = body[:body.find("\n    def ")] if "\n    def " in body else body

# B1: 'video' and 'image' must be untouched -- only 'auto' grows a second file
if 'if str(media_kind) != "auto":' not in body:
    _fail("the control frame must be written ONLY under media_kind='auto'. Forcing "
          "'video' means the clip and nothing else; that contract cannot change.")
if "return" not in body.split('if str(media_kind) != "auto":')[1][:40]:
    _fail("a non-'auto' media_kind must return immediately")

# B2: a single frame is not a clip
if "n < 2" not in body:
    _fail("a single-frame batch under 'auto' is not a clip -- it must stay one image, "
          "which is what resolve_media_kind already decided. Writing a 'control frame' "
          "for a still would duplicate the still.")

# B3: the MIDDLE frame, on purpose
if "n // 2" not in body:
    _fail("the control frame must be the MIDDLE frame. Frame 1 of a video-diffusion "
          "run is the cleanest the model produces -- drift has not started yet -- so "
          "it is exactly the frame that hides what you opened the file to check.")

# B4: it must carry the workflow (the whole point: an .mp4 cannot)
if "_png_info" not in body:
    _fail("the PNG must carry the workflow (_png_info) -- an .mp4 cannot, and that "
          "round-trip IS the reason this file earns its place on disk")

# B5: it must share the clip's stem
if "os.path.splitext(video_path)[0]" not in body:
    _fail("the still must share the clip's stem and differ only in extension, so the "
          "pair stays together in the folder and in the sort order")

# B6: it must never take the video down with it, and never fail silently
if "except Exception" not in body:
    _fail("the control frame must never be fatal -- the video is already on disk")
if "FAILED" not in body:
    _fail("a swallowed exception is how 'sometimes the workflow just isn't in the "
          "file' becomes a ghost story (audit B3). If the still fails, SAY so.")

# B7: the caller must pass media_kind through
if "media_kind=media_kind" not in save_src:
    _fail("save() must pass media_kind into _save_video, or 'auto' can never be seen")

print("[test_v580_wire_and_control] PASS: the mis-wired socket is detected on the "
      "REAL config payload (1710 tokens) and stays silent on the REAL trigger string "
      "(129 chars); the control frame is middle-frame, workflow-carrying, "
      "'auto'-only, and can never take the video down with it.")
sys.exit(0)
