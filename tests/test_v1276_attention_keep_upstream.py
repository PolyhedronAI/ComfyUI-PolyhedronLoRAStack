"""Guard v1276 -- Polyhedron Attention: a path that patches nothing keeps what came in.

Fix C from the Singularity audit (02.11.): up to v1275, 'default (leave the
model alone)' and every give-up path (mode not offered here, launch probe
failed, sparse dropped, unknown mode) POPPED transformer_options
["optimized_attention_override"]. A backend set UPSTREAM -- Core's
ModelAttentionBackend, a Veda layer, any node writing that key -- was deleted
without a word, while the collision report told the user to pick 'default'
"if you meant to keep it".

  K1  default + foreign upstream callable -> the very same object is still in
      the slot, and the console says it stays in place.
  K2  default + our own upstream override -> kept, with the Polyhedron wording.
  K3  default + parked None -> the key is gone (Trap A unchanged).
  K4  default + nothing upstream -> no key, old wording "own backend".
  K5  mode NOT offered here + foreign upstream -> kept, said.
  K6  launch probe fails + foreign upstream -> kept, said.
  K7  unknown mode (build_override None) + foreign upstream -> kept, said.
  K8  the caller's dict is never mutated (copy-on-write).
  K9  a mode that DOES patch still replaces the upstream one (v903 unchanged).

Script-style: exit 0 = pass.
"""

import importlib.util
import io
import os
import sys
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY = "optimized_attention_override"
FAILS = []


def _fail(msg):
    FAILS.append(msg)
    print("  FAIL  " + msg)


def _ok(msg):
    print("  ok    " + msg)


def _load():
    path = os.path.join(ROOT, "nodes", "ph_attention.py")
    spec = importlib.util.spec_from_file_location("ph_attention_v1276", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PH = _load()


class FakeModel:
    def __init__(self, options=None):
        self.model_options = dict(options or {})

    def clone(self):
        return FakeModel(dict(self.model_options))


def _foreign(*a, **k):
    return None


def _run(model, attention):
    buf = io.StringIO()
    with redirect_stdout(buf):
        node = PH.ULSAttention()
        # the internal tree patches the dense backend in _patch_dense (a sparse layer goes on top in patch());
        # the public cut has no sparse layer and patches in patch() itself -- same arguments, same order
        dense = getattr(node, "_patch_dense", None) or node.patch
        (out,) = dense(model, attention, True, PH.SAME_AS_MAIN, 0, PH.SAME_AS_MAIN, 0,
                       3, True, False, None, None)
    return out, buf.getvalue()


def _with(value):
    return FakeModel({"transformer_options": {KEY: value}})


def _slot(m):
    return m.model_options.get("transformer_options", {}).get(KEY, "ABSENT")


# K1
src = _with(_foreign)
out, said = _run(src, PH.DEFAULT_MODE)
if _slot(out) is not _foreign:
    _fail("K1: default removed/replaced the upstream override (%r)" % (_slot(out),))
elif "stays in place" not in said or "ModelAttentionBackend" not in said:
    _fail("K1: keeping it was not said, said:\n%s" % said)
else:
    _ok("K1 default keeps a foreign upstream backend and says so")
# K8
if src.model_options["transformer_options"].get(KEY) is not _foreign:
    _fail("K8: the caller's transformer_options were mutated")
else:
    _ok("K8 caller's dict untouched")

# K2
def _ours(*a, **k):
    return None
_ours._pls_owner = "ULSAttention"
out, said = _run(_with(_ours), PH.DEFAULT_MODE)
if _slot(out) is not _ours or "earlier Polyhedron Attention" not in said:
    _fail("K2: own upstream override not kept/said: %r\n%s" % (_slot(out), said))
else:
    _ok("K2 default keeps an earlier Polyhedron override, own wording")

# K3
out, said = _run(_with(None), PH.DEFAULT_MODE)
if _slot(out) != "ABSENT":
    _fail("K3: a parked None stayed in the slot")
else:
    _ok("K3 parked None removed (Trap A)")

# K4
out, said = _run(FakeModel(), PH.DEFAULT_MODE)
if _slot(out) != "ABSENT" or "own backend" not in said or "stays in place" in said:
    _fail("K4: nothing upstream -> wrong slot/wording: %r\n%s" % (_slot(out), said))
else:
    _ok("K4 nothing upstream: no key, old wording")

# K5 -- a mode this installation does not offer
out, said = _run(_with(_foreign), "xformers (not here v1276)")
if _slot(out) is not _foreign or "does not offer" not in said or "stays in place" not in said:
    _fail("K5: unoffered mode did not keep/say: %r\n%s" % (_slot(out), said))
else:
    _ok("K5 unoffered mode keeps the upstream backend")

# K6 / K7 need an offered mode: fake the gates
offered = list(PH.available_modes())
real = [m for m in offered if m != PH.DEFAULT_MODE and m != PH.SPARSE_LOCAL]
if not real:
    real = ["pytorch sdpa"]
mode = real[0]
_av, _pr, _bo = PH.available_modes, PH.probe, PH.build_override
try:
    PH.available_modes = lambda: offered + ([mode] if mode not in offered else [])
    PH.probe = lambda m: (False, "faked launch failure v1276")
    out, said = _run(_with(_foreign), mode)
    if _slot(out) is not _foreign or "did NOT run" not in said or "stays in place" not in said:
        _fail("K6: failed probe did not keep/say: %r\n%s" % (_slot(out), said))
    else:
        _ok("K6 failed launch keeps the upstream backend")
    PH.probe = lambda m: (True, "faked ok")
    PH.build_override = lambda *a, **k: None
    out, said = _run(_with(_foreign), mode)
    if _slot(out) is not _foreign or "unknown mode" not in said or "stays in place" not in said:
        _fail("K7: unknown mode did not keep/say: %r\n%s" % (_slot(out), said))
    else:
        _ok("K7 unknown mode keeps the upstream backend")
    # K9 -- a real patch still replaces it
    def _new(*a, **k):
        return None
    PH.build_override = lambda *a, **k: _new
    out, said = _run(_with(_foreign), mode)
    if _slot(out) is not _new or "REPLACES" not in said:
        _fail("K9: a patching mode no longer replaces the upstream one: %r\n%s"
              % (_slot(out), said))
    else:
        _ok("K9 a patching mode still replaces it, reported (v903)")
finally:
    PH.available_modes, PH.probe, PH.build_override = _av, _pr, _bo

if FAILS:
    print("\n[test_v1276_attention_keep_upstream] %d FAILED" % len(FAILS))
    sys.exit(1)
print("\n[test_v1276_attention_keep_upstream] all passed")
