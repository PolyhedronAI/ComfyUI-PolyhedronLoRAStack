"""Guard v542 -- dialog UX overhaul, zoom fix, CLIP auto-resolve from MODEL."""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _fail(m): print("FAIL: " + m); sys.exit(1)
def _read(*p): return open(os.path.join(ROOT, *p), encoding="utf-8").read()

def main():
    ml = _read("web", "js", "ph_media_loader.js")
    sv = _read("web", "js", "ph_save.js")
    pb = _read("nodes", "ph_basics.py")

    # -- zoom fix (both DOM widgets opt out, like the 3D nodes already do) ----
    if 'hideOnZoom: false' not in ml: _fail("media loader DOM widget still hides on zoom")
    if 'hideOnZoom: false' not in sv: _fail("save preview DOM widget still hides on zoom")

    # -- Bug-B evidence: DIAG markers must survive the overhaul ---------------
    if "[PLS v536 DIAG] ph_media_loader.js" not in ml: _fail("loader banner lost")
    if "MediaLoader restore(): batch_config=" not in ml: _fail("restore() dump lost")
    if "[PLS v536 DIAG] ph_save.js" not in sv: _fail("save banner lost")
    if "v531 loaded" in ml or "v531 loaded" in sv: _fail("stale v531 banner payload still present")

    # -- label != value: the WIRE values must be untouched --------------------
    for v in ('"name (natural)"', '"name (literal)"', '"mtime (oldest first)"', '"created"',
              '"none (strict)"', '"resize to first"', '"pad to first"', '"center crop to first"'):
        if v not in ml: _fail(f"stored value {v} vanished -- JS/Py parity would break")
    # ...while the jargon is gone from the UI text
    for jargon in ("Every Nth</label>", "Checks := matches", "<label>Wrap</label>",
                   "<label>Source folder</label>", "<label>Sort</label>"):
        if jargon in ml: _fail(f"old jargon still in the dialog: {jargon!r}")
    # v543 evolved the vocabulary (purpose names; labels = functions, tooltips
    # explain). The PRINCIPLE this guard protects is unchanged -- the needles track it.
    for plain in ("Video frames", "Separate files", "Name contains", "Select the matches",
                  "Start over after the last file", ">Number<", ">Stop<",
                  "every 2nd file", "Saved sequences"):
        if plain not in ml: _fail(f"plain-language label missing: {plain!r}")

    # -- structure: mode switches the form, live line, examples, popovers -----
    if "applyModeSwitch" not in ml or 'row.style.display' not in ml:
        _fail("mode must SWITCH the form (display), not grey it out")
    if "ph-batch-live" not in ml or "updateLive" not in ml: _fail("live outcome line missing")
    if "checks beat the filter" not in ml: _fail("running-set rule must be encoded, not just prose")
    if "ph-adv-body" not in ml or "data-ex=" not in ml: _fail("clickable filter examples missing")
    if 'class="ph-q"' not in ml: _fail("per-row info popovers missing")
    if "getNth" not in ml or "ph-batch-nth-custom" not in ml: _fail("plain-language nth dropdown missing")

    # -- CLIP auto-resolve ----------------------------------------------------
    if "_MODEL_CLASS_CLIP_TYPE" not in pb: _fail("model-class -> clip-type table missing")
    if "__mro__" not in pb: _fail("MRO walk missing (WAN22 must resolve via WAN21)")
    if '"WAN21": "wan"' not in pb: _fail("WAN mapping missing")
    if "_DualEncoderNeeded" not in pb or "clip_l + t5xxl" not in pb:
        _fail("dual-encoder detection missing (flux is not in the single-CLIP list)")
    if '"default": "auto"' not in pb: _fail("type default must be auto")
    # v716 REHUNG: the validation now runs against _clip_type_choices(), the
    # union of the single- and dual-encoder lists, because the node serves both.
    # Same invariant -- a resolved type is never used unless the RUNNING ComfyUI
    # actually offers it.
    if "not in _clip_type_choices()" not in pb: _fail("resolved type must be validated against the live list")
    if pb.count("set the type explicitly") < 1: _fail("must fail loudly instead of guessing")
    if len(re.findall(r'"[a-z0-9_]+",', pb.split("_CLIP_TYPE_FALLBACK = [")[1].split("]")[0])) < 20:
        _fail("frozen fallback list not refreshed to the measured core list")
    print("PASS: v542 -- zoom fix, plain-language dialog (wire values intact), CLIP auto from MODEL")
    sys.exit(0)

if __name__ == "__main__":
    main()
