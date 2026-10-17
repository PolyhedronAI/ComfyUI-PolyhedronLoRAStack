"""Guard v579 -- ss_tag_frequency is not a trigger list.

AMENDED IN v580 (1st amendment): THE PREMISE BELOW IS FALSE. IT WAS INVENTED.

Everything from "THE FIELD MEASUREMENT" down to the end of this header was
reasoned backwards from two numbers, and both numbers were misread. Read it as
history, not as fact. What actually happened:

  * A filesystem scan of all 459 LoRAs found ZERO `.uls-meta.json` files and
    ZERO tag soup in ANY WAN header. Frank's WAN triggers measure 5 to 11
    characters each -- `du8ne`, `lac8e`, `golden hour`. There was never a soup.
    (The 11 soup headers in his library are all FLUX LoRAs, none of them in the
    run this guard claims to be built on.)

  * The 1710 was a MIS-WIRED SOCKET. The workflow JSON:
        #155 Token Counter . trigger_words <- #256 Stack HIGH . uls_config_out
        #156 Token Counter . trigger_words <- #257 Stack LOW  . trigger_words
    The HIGH counter was tokenising the serialised Stack config -- ~6000 chars of
    JSON. The eleven real triggers measure 129. STRING fits into STRING, so
    LiteGraph passed it without a word.

  * The "golden hour" / "hour" split was the FILENAME FALLBACK, not the header:
    stage 4 returned parts[-1] -- the last word -- and peeled only one `_noise`
    stamp off names that carry two. See test_v580_filename_trigger.py, which
    executes the function against the real filenames instead of theorising
    about them.

WHY THIS GUARD WAS GREEN THROUGH ALL OF IT -- and this is the lesson, restated
because it was already written down once and repeated anyway (v576 lesson 4,
"green guards only prove what they measure"):

    It checks the DISCRIMINATOR. It checks the LADDER ORDER. It never once asks
    what stage 4 RETURNS. It even asserts, in its own failure text below, that
    the filename fallback "yields exactly 'golden hour'" -- an assertion it never
    executes. It yielded "hour".

The checks below are still worth keeping: a caption table genuinely must not be
mistaken for a trigger list, and stage 3 must genuinely validate before it
returns. The mechanism is sound. The story that motivated it was not.

Original header follows, uncorrected, so the mistake stays legible:

THE FIELD MEASUREMENT that produced this guard (Frank's console, 11 LoRAs, one
prompt, two wires):

    LOW wire  (headers carry NO tag table -> the filename fallback speaks)
        TRIGGERS :   38 / 512  (  7.4%)   "golden hour", "god rays", "apple tree"

    HIGH wire (headers DO carry tag tables)
        TRIGGERS : 1710 / 512  (334.0%)   20 caption fragments x 11 LoRAs

Same LoRAs, same concepts, same prompt. The difference was entirely in what the
safetensors header happened to contain -- and the tag soup BEAT the correct
answer, because step 3 of the _get_trigger ladder short-circuited step 4.

`ss_tag_frequency` is kohya's DATASET CAPTION STATISTICS: every tag that ever
appeared in a training caption, with its count. It is not a trigger list, and it
never was. And this is not cosmetic: `trigger_words` is an output STRING meant
to be wired into a prompt. Wire it and 1710 tokens land in a 512-token budget --
silent truncation, or kijai's 'RuntimeError: Trying to create tensor with
negative dimension'. The Token Counter was the ONLY node that told the truth;
the Inspector printed triggers[0] truncated to its column, so the soup showed up
as the innocent word "hour".

The discriminator: a trigger list is SHORT. Anything wider is caption statistics
wearing a trigger's coat.

Pure functions, exec'd whole - no torch, no ComfyUI, no folder_paths.
Script-style: exit 0 = pass.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "nodes", "uls_stack_node.py")


def _fail(msg):
    print("[test_v579_triggers] FAIL: " + msg)
    sys.exit(1)


def main():
    src = open(SRC, encoding="utf-8").read()

    # ---- carve the pure trigger helpers and RUN them ---------------------------
    m = re.search(r"_TRIG_MAX_TAGS\s*=.*?\n(?=\ndef _flatten_tag_frequency)",
                  src, re.S)
    if not m:
        _fail("cannot carve the trigger laws (anchor _TRIG_MAX_TAGS .. "
              "def _flatten_tag_frequency)")
    ns = {"os": os}
    exec(compile(m.group(0), "<trig>", "exec"), ns)          # noqa: S102
    cap = ns["_cap_tags"]
    looks = ns["_looks_like_trigger_list"]

    # ---- 1: a real trigger list passes ----------------------------------------
    for good in ("golden hour", "god rays", "apple tree", "du8ne",
                 "vi8ctorian, lac8e", "oxyge8n"):
        ok, n = looks(good)
        if not ok:
            _fail(f"a real trigger list must pass: {good!r} -> ok={ok} n={n}")

    # ---- 2: Frank's actual soup FAILS -----------------------------------------
    # A kohya caption table: 20 wordy fragments. This is what produced 1710
    # tokens across 11 LoRAs.
    soup = ", ".join([
        "hour", "a woman standing at a window", "golden hour lighting",
        "warm rim light", "long red hair", "victorian lace dress",
        "rain streaked glass", "shallow depth of field", "looking at viewer",
        "soft focus background", "green foliage", "indoor", "portrait",
        "cinematic color grading", "35mm film grain", "backlit",
        "hand pressed against glass", "melancholic mood", "overcast sky",
        "muted palette"])
    ok, n = looks(soup)
    if ok:
        _fail(f"a {n}-tag caption table must NOT pass as a trigger list - that "
              f"is the 1710-token bug walking back in")
    if n != 20:
        _fail(f"the probe soup should carry 20 tags, counted {n}")

    # ---- 3: nothing may escape the flattener unbounded ------------------------
    # The two paths that used to `return s` verbatim.
    monster = ", ".join(f"tag_{i}" for i in range(500))
    capped = cap(monster)
    n_capped = len([t for t in capped.split(",") if t.strip()])
    if n_capped > ns["_TRIG_HARD_TAGS"]:
        _fail(f"_cap_tags let {n_capped} tags through - the hard cap is "
              f"{ns['_TRIG_HARD_TAGS']}. An unbounded path is how a raw header "
              f"blob became 'trigger words' in full.")
    if not capped:
        _fail("_cap_tags must not swallow the string entirely")

    # ---- 4: the flattener's OWN return paths are all capped -------------------
    fl = re.search(r"def _flatten_tag_frequency.*?(?=\ndef _extract_lora_info)",
                   src, re.S)
    if not fl:
        _fail("cannot carve _flatten_tag_frequency")
    for bad in ("return s\n", "return s  ", "return str(tw)"):
        if bad in fl.group(0):
            _fail(f"an UNCAPPED return path is back in the flattener ({bad!r}). "
                  f"Every way out must go through _cap_tags - a raw "
                  f"ss_tag_frequency blob is thousands of characters.")

    # ---- 5: THE LAW - tag frequency may not short-circuit the filename --------
    gt = re.search(r"def _get_trigger.*?(?=\ndef |\nclass )", src, re.S)
    if not gt:
        _fail("cannot carve _get_trigger")
    gt = gt.group(0)
    if "_looks_like_trigger_list" not in gt:
        _fail("step 3 of the ladder (safetensors header) must VALIDATE what it "
              "found before returning it. Without that, a dataset caption table "
              "beats the filename fallback - which for "
              "'polyhedron_wan2.2_golden_hour_low_noise' yields exactly "
              "'golden hour'. The garbage was winning.")
    # the filename fallback must still be REACHABLE after a rejected soup
    i_check = gt.index("_looks_like_trigger_list")
    i_fname = gt.find("Last resort")
    if i_fname == -1 or i_fname < i_check:
        _fail("the filename fallback must sit AFTER the header check and stay "
              "reachable when the header is rejected")
    if "_warn_tag_soup_once" not in gt:
        _fail("a rejected tag soup must SAY so on the console. Silence is how a "
              "1710-token string walked into a 512-token budget unnoticed.")

    # ---- 6: the Inspector must stop hiding the soup ---------------------------
    if "triggers[0][:col_trig]" in src:
        _fail("the Inspector is printing triggers[0] truncated again - that is "
              "the line that showed a 20-tag caption soup as the word 'hour'. "
              "It must say how many tags are behind it.")

    print("[test_v579_triggers] PASS: a caption table can no longer pass as a "
          "trigger list, no path out of the flattener is unbounded, the filename "
          "fallback is reachable again, and the Inspector says how many tags it "
          "is not showing.")
    sys.exit(0)


if __name__ == "__main__":
    main()
