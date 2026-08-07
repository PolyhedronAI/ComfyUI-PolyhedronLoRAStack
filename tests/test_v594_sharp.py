"""v594 guard: THE SHARP PREVIEW DECODES ONE FRAME, PAYS ITS OWN WAY, AND SAYS
WHAT IT IS SHOWING.

Frank, 2026-07-14, after v592 made the process pane big: "Jetzt ist die
Voransicht zu gross skaliert und deshalb wohl unscharf."

He was right, and the cause was mine. The refine probe does not send a picture
of his tile - it sends `latent2rgb`, which reads the LATENT's own grid. His 768
tile is 96x96 there (the WAN vae is /8), and `PIL.thumbnail()` only ever scales
DOWN, so a 96px jpeg went out and the v592 pane blew it up 5.2x. It looked
exactly like what it was.

And test_v591_preview was GREEN over it, because it pinned
`PV_TARGET_W <= _PROBE_MAX_EDGE` - a CEILING, which I had read as a promise
about the resolution. Third time in this sprint that a number was believed
instead of measured: a name out of scope (v590), int() where the code rounds
(v593), and now a cap mistaken for a size.

So v594 adds a real decode - `process_preview = 'vae (sharp)'` - and this guard
holds it to three things it could easily lie about:

  1. IT DECODES ONE FRAME. Not seventeen. The x0 latent is [B,C,T,H,W] with T=17
     on Frank's runs; handing that whole thing to the vae is a 65-frame decode -
     ~16 seconds, per preview event, inside the sampler loop. That is not a
     preview, it is a catastrophe with a progress bar. The temporal slice is the
     single most important line in the feature, so it is pinned by the SHAPE the
     vae actually receives, through a fake that records it.
  2. IT AUDITS ITSELF. The probe already carries the stage clock (v567). It
     compares what it has spent against that clock and disarms sharp mode past
     _SHARP_MAX_SHARE - loudly. A preview that costs a measurable slice of the
     pass is a second pass, and no tooltip promising "small overhead" is worth
     as much as code that checks.
  3. THE COARSE PATH SAYS IT IS COARSE. latent2rgb is /8 by construction and no
     kernel puts back what was never sampled. It may stay soft; it may not stay
     silent about why - or the next person reads the softness as a broken render
     and goes looking for a bug that is not there.
"""
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PU = ROOT / "nodes" / "ph_power_upscale.py"


def _fail(msg):
    print(f"[test_v594_sharp] FAIL: {msg}")
    sys.exit(1)


def _pure(src, name, ns):
    m = re.search(r"^def " + name + r"\(.*?(?=\n(?:def |class )|\Z)",
                  src, re.S | re.M)
    if not m:
        _fail(f"{name}() is gone")
    exec(m.group(0), ns)  # noqa: S102 - our own source, measured not believed
    return ns[name]


class _FakeTensor:
    """Ducks just enough torch for _sharp_frame: dim(), shape, slicing. Every
    slice is recorded, so the guard can see exactly what reached the vae."""

    def __init__(self, shape, log):
        self.shape = tuple(shape)
        self.log = log

    def dim(self):
        return len(self.shape)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            new = list(self.shape)
            for i, k in enumerate(key):
                if isinstance(k, slice):
                    start = k.start or 0
                    stop = k.stop if k.stop is not None else new[i]
                    new[i] = max(0, stop - start)
            return _FakeTensor(new, self.log)
        if isinstance(key, int):
            return _FakeTensor(self.shape[1:], self.log)
        return self


class _FakeVAE:
    def __init__(self, out_shape, log):
        self.out_shape = out_shape
        self.log = log

    def decode(self, lat):
        self.log.append(tuple(lat.shape))       # what did it ACTUALLY get?
        return _FakeTensor(self.out_shape, self.log)


def _coarse_announces(src):
    """Does the latent2rgb branch PRINT its own resolution?

    Pinned by the AST, in the RIGHT branch. The first draft looked for the flag
    `state["coarse"]` anywhere in the probe - and that string also lives in the
    state dict's initialiser, so deleting the announcement left the guard green.
    A pin that matches an initialiser instead of the statement is the v590
    disease with a new coat.

    This walks to the `if state["sharp"]` and reads its ELSE branch - which IS
    latent2rgb - and demands a print there whose arguments name the array's own
    shape. A flag without a print is a flag; a print without the shape is noise.
    """
    for fn in ast.walk(ast.parse(src)):
        if not isinstance(fn, ast.FunctionDef) or fn.name != "probe":
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.If) or "sharp" not in ast.dump(node.test):
                continue
            for sub in node.orelse:                     # the latent2rgb path
                for c in ast.walk(sub):
                    if (isinstance(c, ast.Call)
                            and isinstance(c.func, ast.Name)
                            and c.func.id == "print"):
                        d = ast.dump(c)
                        if "arr" in d and "shape" in d:
                            return True
    return False


def main():
    src = PU.read_text(encoding="utf-8")
    sharp_frame = _pure(src, "_sharp_frame", {})

    # ---- 1: ONE frame reaches the vae. Not seventeen. ----------------------
    # Frank's latent: [B=1, C=16, T=17, H=96, W=96]. The vae must see T=1.
    log = []
    x0 = _FakeTensor((1, 16, 17, 96, 96), log)
    vae = _FakeVAE((1, 1, 768, 768, 3), log)    # video vae: [B,T,H,W,C]
    out = sharp_frame(vae, x0)
    if not log:
        _fail("_sharp_frame never called vae.decode - there is no sharp path")
    got = log[0]
    if len(got) != 5 or got[2] != 1:
        _fail(f"the vae received a latent of shape {got}. The temporal "
              f"dimension must be 1 - handing it all {got[2] if len(got) > 2 else '?'} "
              f"frames is a full 65-frame decode (~16s in Frank's log) inside "
              f"the sampler loop, on EVERY preview event. That is not a "
              f"preview, it is a second render.")
    if tuple(out.shape) != (768, 768, 3):
        _fail(f"_sharp_frame returned {tuple(out.shape)}, expected (H, W, C) - "
              f"the pane needs one image, not a batch")

    # ---- 1b: shape tolerance. Image vaes hand back [B,H,W,C]. ---------------
    log2 = []
    out2 = sharp_frame(_FakeVAE((1, 768, 768, 3), log2),
                       _FakeTensor((1, 16, 17, 96, 96), log2))
    if tuple(out2.shape) != (768, 768, 3):
        _fail(f"_sharp_frame breaks on a 4-D vae return ({tuple(out2.shape)}) - "
              f"a preview must not care which family the vae belongs to")

    # ---- 2: the mode exists, reaches the probe, and carries the vae ---------
    if '"vae (sharp)"' not in src:
        _fail("process_preview has no 'vae (sharp)' option")
    m = re.search(r'"process_preview":\s*\(\[([^\]]+)\]', src)
    if not m or "latent2rgb" not in m.group(1) or "Off" not in m.group(1):
        _fail("the process_preview combo lost 'Off' or 'latent2rgb' - the free "
              "paths are not optional")
    tree = ast.parse(src)
    call_ok = False
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_make_tile_probe"):
            kws = {k.arg for k in node.keywords}
            if "vae" in kws and "sharp" in kws:
                call_ok = True
    if not call_ok:
        _fail("_make_tile_probe is not handed the vae and the sharp flag - the "
              "mode cannot work, however the widget is set")

    # ---- 3: it audits itself, and the fallback is LOUD ----------------------
    if not re.search(r"^_SHARP_MAX_SHARE\s*=", src, re.M):
        _fail("_SHARP_MAX_SHARE is gone - the preview has no budget and nothing "
              "stops it from eating the pass")
    share = float(re.search(r"^_SHARP_MAX_SHARE\s*=\s*([\d.]+)", src, re.M).group(1))
    if not (0.0 < share <= 0.10):
        _fail(f"_SHARP_MAX_SHARE = {share}: a preview allowed more than 10% of "
              f"the pass is a second pass wearing a preview's coat")
    probe = re.search(r"def _make_tile_probe\(.*?(?=\ndef )", src, re.S)
    body = probe.group(0) if probe else ""
    if 'state["sharp"] = False' not in body:
        _fail("sharp mode never disarms - it must drop back to latent2rgb when "
              "it passes its budget, on its own, without asking")
    if "elapsed" not in body or "_SHARP_MAX_SHARE" not in body:
        _fail("the budget is not checked against the stage clock the probe "
              "already carries (v567) - an unmeasured budget is a wish")
    # the drop must SAY it happened, or the user silently gets a coarser
    # picture and never learns why
    if "Falling back to latent2rgb" not in body:
        _fail("the fallback is silent - the picture changes under the user and "
              "nothing explains it")

    # ---- 4: the coarse path admits what it is ------------------------------
    # Pinned by the two things that MAKE the statement, not by a sentence: the
    # once-flag that fires it and the shape it reports. (The first draft searched
    # for the literal "latent2rgb source is" - which does not exist in the
    # source, because the print is split across two f-string lines. A text pin
    # that cannot survive a line break is not measuring the code, it is
    # measuring the formatter. Fourth time this sprint.)
    if 'state["coarse"] = True' not in body:
        _fail("the latent2rgb notice has no once-flag - it would fire on every "
              "probe event and bury the log it is trying to inform")
    if not _coarse_announces(src):
        _fail("latent2rgb never states its resolution. It is /8 by construction "
              "(96px on a 768 tile) and the pane scales it up - unsaid, that "
              "softness reads as a broken render and sends the next person "
              "hunting a bug that does not exist")

    print(f"PASS: v594 -- sharp decodes exactly ONE latent frame (T=1, not 17), "
          f"tolerates 4-D and 5-D vae returns, is wired with vae+sharp, audits "
          f"itself against the stage clock at {share:.0%} and drops back loudly; "
          f"latent2rgb states its /8 resolution")


if __name__ == "__main__":
    main()
