"""GATE v580 - stage 4 of the trigger ladder, executed against REAL filenames.

WHY THIS GUARD EXISTS, AND WHY THE v579 ONE DID NOT CATCH ANY OF IT:

test_v579_triggers.py is green. It always was. It checks the DISCRIMINATOR
(`_looks_like_trigger_list("golden hour") is True`) and the ORDER of the ladder
(stage 4 sits below stage 3). It never once asked what stage 4 RETURNS.

It even asserts, in its own failure message, that the filename fallback "for
'polyhedron_wan2.2_golden_hour_low_noise' yields exactly 'golden hour'".

It yielded "hour".

Then a filesystem scan of Frank's 459 LoRAs found the second wound: the stamp
is REPEATED in half the WAN library --

    polyhedron_wan2.2_golden_hour_low_noise_low_noise
    polyhedron_wan2.2_rings_low_noise__low_noise_low_noise_r8ing

-- and the old regex was `$`-anchored and ran once, so it peeled one stamp and
handed back the last word of what was left: "noise".

Every case below is a REAL file in Frank's loras folder. The expectations are
what the name plainly says. This guard executes the shipped function; it does
not re-implement it.
"""
import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "nodes", "uls_stack_node.py")


def _fail(msg):
    print(f"[test_v580_filename_trigger] FAIL: {msg}")
    sys.exit(1)


# ---- lift the pure helpers out of the module (no torch, no comfy) ------------
src = open(SRC, encoding="utf-8").read()
tree = ast.parse(src)

WANT_FUNCS = {"_trigger_from_filename"}
WANT_NAMES = {"_STAMP_RE", "_SIDE_RE", "_COINED_RE"}

ns = {"re": re, "os": os}
found_funcs, found_names = set(), set()
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in WANT_FUNCS:
        exec(compile(ast.Module([node], []), SRC, "exec"), ns)
        found_funcs.add(node.name)
    elif isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in WANT_NAMES:
                exec(compile(ast.Module([node], []), SRC, "exec"), ns)
                found_names.add(t.id)

missing = (WANT_FUNCS - found_funcs) | (WANT_NAMES - found_names)
if missing:
    _fail(f"cannot find {sorted(missing)} at module level in uls_stack_node.py")

stage4 = ns["_trigger_from_filename"]

# ---- 1: the REAL library. Every name below exists on Frank's disk. -----------
REAL = {
    # -- the double stamp. v579 said "noise" for every one of these. ----------
    "polyhedron_wan2.2_golden_hour_low_noise_low_noise":            "golden hour",
    "polyhedron_wan2.2_apple_tree_low_noise_low_noise":             "apple tree",
    "polyhedron_wan2.2_glamour_low_noise_low_noise":                "glamour",
    "polyhedron_wan2.2_god_rays_low_noise__low_noise":              "god rays",
    "polyhedron_wan2.2_victorian_dress_low_noise__low_noise":       "victorian dress",
    "polyhedron_wan2.2_polyhedron_all_high_noise_high_noise":       "all",
    "polyhedron_wan2.2_polyhedron_all_low_noise_low_noise":         "all",
    # -- the single stamp. v579 returned the LAST WORD, a fragment. -----------
    "polyhedron_wan2.2_golden_hour_high_noise":                     "golden hour",
    "polyhedron_wan2.2_god_rays_high_noise":                        "god rays",
    "polyhedron_wan2.2_victorian_dress_high_noise":                 "victorian dress",
    "wan2.2_glamour_high_noise":                                    "glamour",
    "polyhedron_wan2.2_lace_low_noise":                             "lace",
    # -- coined tokens: letters AND digits. These ARE the trigger. ------------
    "polyhedron_wan2.2_lace_lac8e_high_noise":                      "lac8e",
    "polyhedron_wan2.2_rain_rai8n_high_noise":                      "rai8n",
    "polyhedron_wan2.2_rain_rai8n_low_noise":                       "rai8n",
    "polyhedron_wan2.2_oxygen_oxyge8n_low_noise":                   "oxyge8n",
    "polyhedron_wan2.2_film_poster_2_du8ne_high_noise":             "du8ne",
    "polyhedron_wan2.2_film_poster_low_noise_low_noise_du8ne":      "du8ne",
    # -- a coined token behind a leading index. v579 returned "2". ------------
    "polyhedron_wan2.2_1_oxygen_oxyge8n_high_noise":                "oxyge8n",
    "polyhedron_wan2.2_1_fern_fer8n_2_high_noise":                  "fer8n",
    # -- the triple stamp, coined token last. -------------------------------
    "polyhedron_wan2.2_rings_low_noise__low_noise_low_noise_r8ing": "r8ing",
}
for name, want in REAL.items():
    got = stage4(name)
    if got != want:
        _fail(f"stage 4 on the REAL file '{name}' returned '{got}', expected '{want}'")
    # ... and with the extension, which is how the caller passes it
    if stage4(name + ".safetensors") != want:
        _fail(f"stage 4 differs with/without the .safetensors extension on '{name}'")

# ---- 2: THE TWO WOUNDS, named, so a regression cannot hide -------------------
if stage4("polyhedron_wan2.2_golden_hour_high_noise") == "hour":
    _fail("stage 4 is back to returning the LAST WORD ('hour' for golden_hour). "
          "The whole phrase IS the trigger; the last word is a fragment of it.")
if stage4("polyhedron_wan2.2_golden_hour_low_noise_low_noise") == "noise":
    _fail("stage 4 is back to peeling only ONE '_low_noise' stamp. Half of Frank's "
          "WAN library carries it twice; one file carries it three times.")

# ---- 3: a coined token OUTRANKS the phrase ----------------------------------
if stage4("polyhedron_wan2.2_lace_lac8e_high_noise") != "lac8e":
    _fail("a coined token (letters AND digits) must win over the phrase: when a "
          "trigger is spelled 'lac8e' it is 'lac8e', not 'lace lac8e'.")

# ---- 4: never crash, never return junk on hostile input ---------------------
for junk in ("", "_", "___", "low_noise", "_high_noise", "wan2.2_",
             "polyhedron_", "42", "polyhedron_wan2.2_low_noise",
             "a" * 300, "no_extension_at_all"):
    out = stage4(junk)
    if not isinstance(out, str):
        _fail(f"stage 4 returned a non-str on '{junk[:30]}': {type(out)}")
    if "_" in out:
        _fail(f"stage 4 leaked an underscore on '{junk[:30]}': '{out}' -- the phrase "
              f"is joined with spaces, so an underscore means a path escaped cleanup")
    if out.strip() != out:
        _fail(f"stage 4 returned padded output on '{junk[:30]}': '{out}'")

# a name that is ONLY stamps must not invent a trigger out of them
if stage4("polyhedron_wan2.2_low_noise") in ("noise", "low", "low noise"):
    _fail("a name made of nothing but stamps must not yield a stamp as its trigger")

# ---- 5: the source tag must exist and must mark a guess ----------------------
if "_get_trigger" not in src:
    _fail("_get_trigger is gone")
gt = src.split("def _get_trigger(")[1]
gt = gt[:gt.find("\ndef ")] if "\ndef " in gt else gt
for rung, tag in (('_read_uls_meta_trigger', '"meta"'),
                  ('_read_txt_trigger', '"txt"'),
                  ('_looks_like_trigger_list', '"header"')):
    if rung not in gt or tag not in gt:
        _fail(f"_get_trigger no longer returns the source tag {tag} for rung {rung} -- "
              f"the source must travel WITH the value, or a guess arrives looking "
              f"exactly like a fact (it did, and the Inspector printed it as one)")
if '"name"' not in gt:
    _fail("_get_trigger must tag the filename fallback 'name' -- it is a GUESS and "
          "the Inspector prints 'name?' for it")

print("[test_v580_filename_trigger] PASS: stage 4 executed against "
      f"{len(REAL)} REAL filenames from Frank's library; repeated stamps peel, the "
      f"whole phrase survives, coined tokens win, and the guess is tagged as one.")
sys.exit(0)
