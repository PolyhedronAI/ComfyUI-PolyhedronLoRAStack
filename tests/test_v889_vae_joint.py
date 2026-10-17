"""
test_v889_vae_joint.py -- Load VAE must not refuse a joint model's VAE.

THE WOUND (measured 28.08. against Frank's MiniMax H3 workflow): ULSLoadVAE
raises ValueError when the connected model's latent_channels differ from the
VAE's. That is right for SD/Wan, where one model means one latent space. An H3
model carries TWO streams and `latent_format.latent_channels` reports only the
MAXIMUM over both (32), while the correct VIDEO VAE has 24. Wiring the model
pin -- the normal thing to do -- would therefore kill a legitimate run with a
message pointing at the wrong thing. It had not fired in the field only because
both Load VAE nodes in that workflow leave `model` unconnected (link=None).

WHAT IS PINNED, in the shape the house asks for: the promise, not the wording.
  P1  the family table NAMES both H3 channel counts
  P2  an unknown count still falls back to the generic line (no silent claim)
  P3  the joint answer comes from ONE source -- ph_basics delegates to
      ph_joint_probe and does not re-implement a probe of its own
  P4  the DECISION BLOCK IS RUN, both ways: a joint model is let through, an
      ordinary mismatch STILL RAISES. Re-grounding a promise never means
      softening it -- the check that protects SD/Wan users must survive intact
  P5  ph_power_upscale still re-exports both probe names, so every existing
      caller and test_v880 keep working after the move
  P6  ph_basics does NOT import ph_power_upscale (a base module importing a
      heavy leaf is the inversion this move exists to avoid)
"""

import os
import re
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nodes"))

FAILED = []


def check(cond, label):
    if cond:
        print("  PASS  " + label)
    else:
        print("  FAIL  " + label)
        FAILED.append(label)


def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


class _Nested(object):
    """Stands in for Core's NestedTensor: what matters is `.tensors`."""

    def __init__(self, *t):
        self.tensors = list(t)


class _Fmt(object):
    def __init__(self, channels, streams):
        self.latent_channels = channels
        self._streams = streams

    def fix_empty_latent(self, x):
        return _Nested(x, x) if self._streams > 1 else x


class _Model(object):
    def __init__(self, channels, streams):
        self.model = types.SimpleNamespace(latent_format=_Fmt(channels, streams))


def _lift_decision_block(src):
    """Lift the channel-check block out of ULSLoadVAE so it can be RUN.

    Deliberately NOT a substring test: the block is executed against real
    models below. The v685/v839 lesson applies -- everything the block needs
    goes into the namespace explicitly, so this guard cannot rot into a
    NameError the way those two did.
    """
    start = src.index("        model_ch = _model_latent_channels(model)")
    end = src.index("        shown_dtype = precision", start)
    block = src[start:end]
    # de-indent from method body to module level
    return "".join(
        (ln[8:] if ln.startswith("        ") else ln) + "\n"
        for ln in block.splitlines()
    )


def _run_block(block, helpers, model, channels, vae_name="test.safetensors"):
    ns = dict(helpers)
    ns.update({"model": model, "channels": channels, "vae_name": vae_name})
    exec(compile(block, "<loadvae-block>", "exec"), ns)
    return ns


def main():
    py = _read("nodes", "ph_basics.py")
    pu = _read("nodes", "ph_power_upscale.py")

    print("P1 - the family table names both H3 counts")
    import ph_joint_probe as probe

    ns = {}
    tbl = re.search(r"_VAE_FAMILIES = \{.*?\n\}", py, re.S)
    check(tbl is not None, "the table is present")
    exec(compile(tbl.group(0), "<table>", "exec"), ns)
    fam = ns["_VAE_FAMILIES"]
    check(24 in fam, "24ch (H3 video) is in the table")
    check(32 in fam, "32ch (H3 audio) is in the table")
    check("24" in fam.get(32, ("", ""))[1] or "max" in fam.get(32, ("", ""))[1].lower(),
          "the 32ch note warns that 32 is the MAX over both streams")
    for known in (4, 16, 48):
        check(known in fam, "pre-existing entry %dch survived" % known)

    print("P2 - an unknown count still gets the generic fallback")
    fn = re.search(r"def _vae_family\(channels\):.*?\n\n", py, re.S)
    exec(compile(fn.group(0), "<famfn>", "exec"), ns)
    check(ns["_vae_family"](24)[0].find("H3") >= 0, "24ch is named, not generic")
    check("not in the family table" in ns["_vae_family"](99)[1],
          "99ch still falls back to the generic line")

    print("P3 - the joint answer comes from ONE source")
    check("ph_joint_probe" in py, "ph_basics reaches for ph_joint_probe")
    helper = re.search(r"def _is_joint_model\(model\):.*?\n\n\ndef ", py, re.S)
    check(helper is not None, "_is_joint_model exists")
    body = helper.group(0)
    check("_joint_latent_parts" in body, "it calls the shared probe")
    check("FLOW_AV" not in body and "audio_shift" not in body,
          "it does NOT re-implement the witnesses (no second source)")

    print("P4 - the decision block is RUN, both ways")
    block = _lift_decision_block(py)
    said = []
    helpers = {
        "_model_latent_channels": lambda m: (
            int(m.model.latent_format.latent_channels) if m is not None else None),
        "_is_joint_model": lambda m: probe._joint_latent_parts(m) > 1,
        "print": lambda *a, **k: said.append(" ".join(str(x) for x in a)),
        "ValueError": ValueError,
    }

    # (a) joint model, 24ch video VAE against a model reporting 32 -> NO raise
    said[:] = []
    raised = None
    try:
        _run_block(block, helpers, _Model(32, 2), 24)
    except ValueError as exc:
        raised = str(exc)
    check(raised is None, "joint model + 24ch VAE is LET THROUGH")
    check(any("JOINT" in s for s in said), "and it says so out loud")

    # (b) ordinary model, real mismatch -> STILL raises (promise not softened)
    raised = None
    try:
        _run_block(block, helpers, _Model(16, 1), 48)
    except ValueError as exc:
        raised = str(exc)
    check(raised is not None, "ordinary mismatch STILL raises")
    check(raised is not None and "48" in raised and "16" in raised,
          "and the message still names both counts")

    # (c) ordinary model, matching counts -> no raise, no joint noise
    said[:] = []
    raised = None
    try:
        _run_block(block, helpers, _Model(16, 1), 16)
    except ValueError as exc:
        raised = str(exc)
    check(raised is None, "ordinary match passes")
    check(not any("JOINT" in s for s in said),
          "and stays quiet about joint models")

    # (d) no model connected -> no raise (the field's actual configuration)
    raised = None
    try:
        _run_block(block, helpers, None, 24)
    except ValueError as exc:
        raised = str(exc)
    check(raised is None, "unconnected model pin still passes")

    print("P5 - the old home still re-exports both names")
    # RE-GROUNDED during this build's own mutation round: the first version
    # asked `"ph_joint_probe" in pu`, which a mutation to `ph_joint_probe_GONE`
    # satisfied as a SUBSTRING -- the guard stayed green while the re-export was
    # broken. A name is not a promise; the module it points at has to EXIST and
    # carry both symbols. Blindgaenger caught, pinned by effect instead.
    imported = re.findall(r"from \.?(\w+) import \(?_joint_latent_parts", pu)
    check(len(imported) >= 1, "ph_power_upscale imports the probe from somewhere")
    for mod in set(imported):
        path = os.path.join(ROOT, "nodes", mod + ".py")
        check(os.path.exists(path),
              "the module it names (%s) actually exists" % mod)
        if os.path.exists(path):
            text = _read("nodes", mod + ".py")
            for name in ("_joint_latent_parts", "_joint_streams"):
                check(re.search(r"^def %s\(" % name, text, re.M) is not None,
                      "%s is defined in %s" % (name, mod))
    check(hasattr(probe, "_joint_streams") and hasattr(probe, "_joint_latent_parts"),
          "both live in the new module")
    check(probe._joint_latent_parts(_Model(32, 2)) == 2,
          "the probe itself still measures a joint model as 2 streams")
    check(probe._joint_latent_parts(_Model(16, 1)) == 1,
          "and an ordinary model as 1")

    print("P6 - no inversion: base module must not pull in the heavy leaf")
    check(not re.search(r"^\s*from \.ph_power_upscale import", py, re.M),
          "ph_basics does not import ph_power_upscale")

    print("")
    if FAILED:
        print("FAIL -- %d broken promise(s)" % len(FAILED))
        for f in FAILED:
            print("   * " + f)
        return 1
    print("PASS -- v889 VAE joint exception holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
