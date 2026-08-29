"""Guard #108 -- external NOISE source (v685).

The field question this answers: "I pick fractal and nothing changes."
It did not change because a latent's CONTENTS reach the model through
noise_scaling, which at a full schedule is sigma*noise + (1-sigma)*latent
with sigma[0] = 1.0 -- the latent is multiplied by zero. The noise that
IS denoised is the sampler's own. v685 therefore lets a NOISE source
replace that one, and this guard drives the whole path.

DRIVEN with real torch (this is tensor work; reading the source would
prove nothing):

  * ULSNoiseSource takes its GEOMETRY from the latent handed to it at
    generate time -- which is the entire reason it is an object and not
    a tensor: the seed is picked where the size is unknown.
  * Every type is deterministic in (type, shape, seed, strength): two
    calls give the identical tensor. The MoE low phase re-generates the
    noise, so a non-deterministic source would silently break WAN 2.2.
  * Unit scale: strength is expressed in standard-latent-noise units.
  * is_default is EXACT -- gaussian at 1.0 delegates to prepare_noise
    (bit identity with the pre-v685 sampler), gaussian at 2.0 does not.

  * The sampler seam _initial_noise, lifted out of uls_sampler.py and run
    against a stubbed comfy: add_noise off -> zeros; no source -> the
    untouched prepare_noise call; a source -> its field, generated
    against the CORRECTED latent geometry.
  * The run context restores the previous value -- including when the
    body raises. A leaked source would apply to the NEXT render, which
    is exactly the kind of ghost this house does not ship.

STATIC, because they are canon and canon is forever:
  * the Seed node's outputs and widgets are APPENDED, never inserted;
  * the Empty Latent's noise widgets are still in the Python canon at
    their old positions (hidden in the UI is not removed from the array);
  * the frontend hides them by type, and does not splice node.widgets.

MUTATIONS (wound in a COPY, catch proven):
  M1 _initial_noise ignores the parked source -> the override does nothing.
  M2 the context does not restore     -> a source leaks into the next run.
  M3 is_default ignores strength      -> gaussian at 2.0 silently becomes 1.0.
  M4 the Empty Latent canon is short  -> every saved graph renumbers.
"""

import os
import re
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)          # v896: tests/_lift.py, the shared closed lift
NOISE_PY = os.path.join(ROOT, "nodes", "uls_noise.py")
SAMPLER_PY = os.path.join(ROOT, "nodes", "uls_sampler.py")
BASICS_PY = os.path.join(ROOT, "nodes", "ph_basics.py")
EMPTY_PY = os.path.join(ROOT, "nodes", "ph_empty_latent.py")
EMPTY_JS = os.path.join(ROOT, "web", "js", "ph_empty_latent.js")

NAME = "test_v685_noise_source"


def _fail(msg):
    print("[%s] FAIL -- %s" % (NAME, msg))
    sys.exit(1)


def _need(ok, msg):
    if not ok:
        _fail(msg)


def _stub_comfy_sample():
    """The default path delegates to comfy.sample.prepare_noise. Stub it with
    plain unit-scale randn -- that is what Core's does -- so a mutation that
    wrongly delegates produces a WRONG NUMBER here instead of an ImportError.
    A guard must catch the wound, not the scaffolding falling over."""
    import torch
    if "comfy.sample" in sys.modules:
        return
    comfy = sys.modules.setdefault("comfy", types.ModuleType("comfy"))
    sample = types.ModuleType("comfy.sample")
    sample.prepare_noise = (lambda latent_image, seed, batch_inds=None:
                            torch.randn(latent_image.shape,
                                        generator=torch.Generator().manual_seed(int(seed))))
    sys.modules["comfy.sample"] = sample
    comfy.sample = sample


def _load_noise(src=None):
    import importlib.util
    path = NOISE_PY
    if src is not None:
        f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                        encoding="utf-8")
        f.write(src)
        f.close()
        path = f.name
    spec = importlib.util.spec_from_file_location("_uls_noise_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lift_seam(src=None):
    """Lift the noise seam out of uls_sampler.py. Importing the module would
    need the whole comfy stack; the seam is self-contained apart from torch and
    comfy.sample, both of which are provided here.

    v896: the piece list used to be written by hand -- three names, correct on
    the day this guard was written. v870 gave _initial_noise a call to
    _latent_parts, the list was not updated, and this guard stood red for many
    versions with `NameError: _latent_parts` -- a healthy tree looking broken.
    The list is now CLOSED transitively by tests/_lift.py: ask for the seam, get
    whatever the seam needs, and hear about anything genuinely missing BY NAME.
    """
    import torch
    import _lift
    text = src if src is not None else open(SAMPLER_PY, encoding="utf-8").read()
    calls = []
    comfy = types.ModuleType("comfy")
    sample = types.ModuleType("comfy.sample")

    def prepare_noise(latent_image, seed, batch_inds=None):
        calls.append(("prepare_noise", tuple(latent_image.shape), seed,
                      batch_inds))
        return torch.full(latent_image.shape, 0.5)

    sample.prepare_noise = prepare_noise
    comfy.sample = sample
    provided = {"torch", "comfy"}
    seam, missing = _lift.close_over(
        text, ["_ACTIVE_NOISE", "_NoiseContext", "_initial_noise"], provided)
    _need(not missing,
          "seam lift is SHORT of %s -- the harness must provide these or the "
          "names have moved; this is a GUARD fault, not a tree fault"
          % ", ".join(missing))
    for sig in ("_ACTIVE_NOISE = None", "class _NoiseContext:",
                "def _initial_noise("):
        _need(sig in seam, "seam piece %r not found in uls_sampler.py" % sig)
    ns = {"torch": torch, "comfy": comfy}
    exec(seam, ns)
    ns["_calls"] = calls
    return ns


def run_noise(mod, tag=""):
    import torch
    lat = {"samples": torch.zeros(1, 16, 40, 24)}

    # geometry comes from the latent, not from the source
    for t in ("fractal", "brown", "pink", "blue", "gaussian"):
        s = mod.ULSNoiseSource(1234, t, 1.0)
        a = s.generate_noise(lat) if t != "gaussian" else None
        if a is None:
            continue
        _need(tuple(a.shape) == (1, 16, 40, 24),
              "%s%s must take its shape from the latent, got %s"
              % (tag, t, tuple(a.shape)))
        b = s.generate_noise(lat)
        _need(bool(torch.equal(a, b)),
              "%s%s must be deterministic -- the MoE low phase regenerates it"
              % (tag, t))
        _need(abs(float(a.std()) - 1.0) < 0.05,
              "%s%s must be unit scale at strength 1.0, got %.3f"
              % (tag, t, float(a.std())))

    # a different seed is a different field
    x = mod.ULSNoiseSource(1, "fractal", 1.0).generate_noise(lat)
    y = mod.ULSNoiseSource(2, "fractal", 1.0).generate_noise(lat)
    _need(not bool(torch.equal(x, y)), "%sthe seed must change the field" % tag)

    # strength is a scale in units of standard latent noise
    s2 = mod.ULSNoiseSource(1, "gaussian", 2.0).generate_noise(lat)
    _need(abs(float(s2.std()) - 2.0) < 0.1,
          "%sstrength 2.0 must double the scale, got %.3f"
          % (tag, float(s2.std())))

    # zeros = no noise at all (the DisableNoise equivalent)
    z = mod.ULSNoiseSource(1, "zeros", 1.0).generate_noise(lat)
    _need(float(z.abs().sum()) == 0.0, "%szeros must be exactly zero" % tag)

    # is_default is what buys bit identity -- it must be exact
    _need(mod.ULSNoiseSource(7, "gaussian", 1.0).is_default(),
          "%sgaussian at 1.0 must delegate to prepare_noise" % tag)
    _need(not mod.ULSNoiseSource(7, "gaussian", 2.0).is_default(),
          "%sgaussian at 2.0 must NOT delegate -- the strength would be lost"
          % tag)
    _need(not mod.ULSNoiseSource(7, "fractal", 1.0).is_default(),
          "%sfractal must never delegate" % tag)


def run_seam(ns, mod, tag=""):
    import torch
    img = torch.zeros(1, 16, 8, 8)
    lat = {"samples": img}
    calls = ns["_calls"]

    # add_noise off -> zeros, whatever is parked
    with ns["_NoiseContext"](mod.ULSNoiseSource(3, "fractal", 1.0)):
        out = ns["_initial_noise"](lat, img, 3, False)
    _need(float(out.abs().sum()) == 0.0,
          "%sadd_noise off must stay zeros" % tag)

    # nothing parked -> the untouched original call
    del calls[:]
    out = ns["_initial_noise"](lat, img, 99, True)
    _need(len(calls) == 1 and calls[0][2] == 99,
          "%swithout a source the sampler must call prepare_noise as before"
          % tag)

    # a parked source -> its own field, at the latent's geometry
    del calls[:]
    src = mod.ULSNoiseSource(5, "fractal", 1.0)
    with ns["_NoiseContext"](src):
        out = ns["_initial_noise"](lat, img, 99, True)
    _need(not calls,
          "%sa wired source must REPLACE prepare_noise, not run beside it"
          % tag)
    _need(tuple(out.shape) == (1, 16, 8, 8),
          "%sthe generated field must match the latent" % tag)
    _need(bool(torch.equal(out, src.generate_noise({"samples": img}))),
          "%sthe seam must hand the CORRECTED latent to the source" % tag)

    # the context restores -- including through an exception
    _need(ns["_ACTIVE_NOISE"] is None,
          "%sthe context must be clear between runs" % tag)
    try:
        with ns["_NoiseContext"](src):
            raise RuntimeError("sampler blew up")
    except RuntimeError:
        pass
    # re-read the module global (exec namespace holds the live value)
    del calls[:]
    ns["_initial_noise"](lat, img, 77, True)
    _need(len(calls) == 1,
          "%sa source must NOT survive a failed run -- it would silently "
          "apply to the next render" % tag)


def run_static(tag="", empty_py=None):
    basics = open(BASICS_PY, encoding="utf-8").read()
    _need('RETURN_TYPES = ("INT", "STRING", "NOISE")' in basics
          and 'RETURN_NAMES = ("seed", "seed_string", "noise")' in basics,
          "%sthe Seed node's noise output must be APPENDED as slot 2" % tag)
    i_seed = basics.find('"seed": ("INT"')
    i_type = basics.find('"noise_type"')
    i_str = basics.find('"noise_strength"')
    _need(0 < i_seed < i_type < i_str,
          "%sthe Seed node's new widgets must be APPENDED after seed" % tag)

    empty = open(empty_py or EMPTY_PY, encoding="utf-8").read()
    for w in ("noise_type", "noise_seed", "noise_strength"):
        _need('"%s"' % w in empty,
              "%sthe Empty Latent CANON must keep %s -- widgets_values is "
              "positional, dropping it renumbers every saved graph"
              % (tag, w))
    m = re.search(r'"batch_size".*?"noise_type".*?"noise_seed".*?'
                  r'"noise_strength"', empty, flags=re.S)
    _need(m is not None,
          "%sthe Empty Latent canon order must be unchanged" % tag)

    js = open(EMPTY_JS, encoding="utf-8").read()
    _need('w.type = "hidden"' in js and "_hideLegacyNoise" in js,
          "%sthe frontend must HIDE the legacy rows" % tag)
    _need("splice" not in js.split("function _hideLegacyNoise")[1][:600],
          "%sthe frontend must never splice node.widgets -- a hidden widget "
          "is still serialised, a removed one is not" % tag)
    _need("_addNoisePreview(self);" not in js,
          "%sthe preview must no longer be attached to the Empty Latent"
          % tag)

    sampler = open(SAMPLER_PY, encoding="utf-8").read()
    _need('"noise": ("NOISE"' in sampler,
          "%sthe sampler must expose an optional NOISE pin" % tag)
    # HOUSE TRAP, third time (see guard #104): a static must pin the CALL, not
    # prose. uls_sampler.py mentions comfy.sample.prepare_noise in a docstring,
    # so a substring count matches twice and fails on a correct tree. Pin the
    # executable form instead.
    real_calls = len(re.findall(r"^\s*(?:return|noise\s*=)\s*comfy\.sample\.prepare_noise\(",
                                sampler, flags=re.M))
    _need(real_calls == 1,
          "%sprepare_noise must be CALLED in exactly one place (the seam), "
          "found %d -- a second call site is a path the override would "
          "silently miss" % (tag, real_calls))
    _need("with _NoiseContext(noise):" in sampler,
          "%sthe run must be wrapped in the noise context" % tag)


def main():
    try:
        import torch  # noqa: F401
    except Exception:
        print("[%s] note: torch absent -- driven half skipped as pass; "
              "statics still ran." % NAME)
        run_static()
        print("[%s] PASS (statics only)" % NAME)
        return

    _stub_comfy_sample()
    mod = _load_noise()
    run_noise(mod)
    run_seam(_lift_seam(), mod)
    run_static()

    noise_src = open(NOISE_PY, encoding="utf-8").read()
    sampler_src = open(SAMPLER_PY, encoding="utf-8").read()
    caught = 0

    # M1 -- the seam ignores the parked source
    m1 = sampler_src.replace("    src = _ACTIVE_NOISE\n", "    src = None\n")
    _need(m1 != sampler_src, "M1 could not be injected")
    try:
        run_seam(_lift_seam(m1), mod, tag="[M1] ")
    except SystemExit:
        caught += 1
    else:
        print("[%s] NOTE -- mutation M1 survived" % NAME)

    # M2 -- the context never restores
    m2 = sampler_src.replace("        _ACTIVE_NOISE = self.prev\n",
                             "        pass\n")
    _need(m2 != sampler_src, "M2 could not be injected")
    try:
        run_seam(_lift_seam(m2), mod, tag="[M2] ")
    except SystemExit:
        caught += 1
    else:
        print("[%s] NOTE -- mutation M2 survived" % NAME)

    # M3 -- is_default forgets the strength
    m3 = noise_src.replace(
        'return self.noise_type == "gaussian" and abs(self.strength - 1.0) < 1e-9',
        'return self.noise_type == "gaussian"')
    _need(m3 != noise_src, "M3 could not be injected")
    try:
        run_noise(_load_noise(m3), tag="[M3] ")
    except SystemExit:
        caught += 1
    else:
        print("[%s] NOTE -- mutation M3 survived" % NAME)

    # M4 -- the Empty Latent canon loses a widget (static, on a copy)
    empty = open(EMPTY_PY, encoding="utf-8").read()
    m4 = re.sub(r'\n *"noise_strength": \("FLOAT".*?\}\),', "", empty,
                count=1, flags=re.S)
    _need(m4 != empty, "M4 could not be injected")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                      encoding="utf-8")
    tmp.write(m4)
    tmp.close()
    try:
        run_static(tag="[M4] ", empty_py=tmp.name)
    except SystemExit:
        caught += 1
    else:
        print("[%s] NOTE -- mutation M4 survived" % NAME)
    finally:
        os.unlink(tmp.name)

    _need(caught == 4, "only %d/4 mutations were caught" % caught)
    print("[%s] PASS -- the source takes its geometry from the latent, is "
          "deterministic and unit-scaled per type, gaussian at 1.0 keeps bit "
          "identity with prepare_noise, the seam replaces the sampler's noise "
          "on every path, the context survives an exception without leaking, "
          "and both canons are intact, 4/4 mutations caught" % NAME)


if __name__ == "__main__":
    main()
