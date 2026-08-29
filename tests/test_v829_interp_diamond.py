"""Guard v829 -- the diamond on the Interpolate checkpoints (sweep stage 2).

WHAT THIS PINS (driven where it can be, and the one text pin named):

  D1  every ckpt_name combo entry carries the diamond (ALL entries are
      house definitions here -- the pack pins, defaults and fetches
      them), and a size where one is known: from the disk when the file
      lies in a vfi folder, else from the measured byte pin, else
      nothing. The DEFAULT is the decorated entry, so a fresh node never
      shows a value its own list does not offer.
  D2  the strip is exact and complete: diamond off, one size suffix off,
      dashes and unit-less middots survive (the v828 rule).
  D3  REGEX PARITY: the size-suffix pattern here equals
      ph_basics._SIZE_SUFFIX character for character -- one truth, held
      by measurement instead of an import that would drag comfy.sd into
      this module.
  D4  ROUND TRIP: interpolate() strips before anything reads the name --
      driven by calling it with a decorated value and catching where it
      fails: _arch_for must receive the BARE name (a decorated one is not
      in CKPT_ARCH and the append-only map must not learn junk).
  D5  a pre-v829 workflow (bare filename) passes validation:
      VALIDATE_INPUTS exists and is permissive.
  D6  MUTATION: with the strip removed, the decorated value must reach
      _arch_for undecorated no longer -- the run must fail differently,
      proving D4 is driven by the strip.
  D7  the JS half: ph_interpolate.js exists, mirrors the suffix regex,
      strips the diamond, and is wired post-pour (text pin).

Script-style: exit 0 = pass.
"""
import importlib.util
import os
import re
import sys
import types

NAME = "v829"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MOD = os.path.join(ROOT, "nodes", "ph_interpolate.py")
BAS = os.path.join(ROOT, "nodes", "ph_basics.py")
JS = os.path.join(ROOT, "web", "js", "ph_interpolate.js")


def _fail(msg):
    print("[%s] FAIL -- %s" % (NAME, msg))
    sys.exit(1)


def _need(cond, msg):
    if not cond:
        _fail(msg)


# folder_paths stub -- models_dir without any vfi files, so sizes come
# from the byte pins (the middle step of the three-step rule).
import tempfile
tmp = tempfile.mkdtemp()
fp = types.ModuleType("folder_paths")
fp.models_dir = tmp
sys.modules["folder_paths"] = fp

comfy = types.ModuleType("comfy")
comfy_mm = types.ModuleType("comfy.model_management")
comfy_mm.get_torch_device = lambda: "cpu"
comfy_utils = types.ModuleType("comfy.utils")
comfy.model_management = comfy_mm
comfy.utils = comfy_utils
sys.modules["comfy"] = comfy
sys.modules["comfy.model_management"] = comfy_mm
sys.modules["comfy.utils"] = comfy_utils


def _load(src=None):
    pkg = types.ModuleType("plspack")
    pkg.__path__ = [os.path.join(ROOT, "nodes")]
    sys.modules["plspack"] = pkg
    if src is None:
        spec = importlib.util.spec_from_file_location(
            "plspack.ph_interpolate", MOD)
        m = importlib.util.module_from_spec(spec)
        sys.modules["plspack.ph_interpolate"] = m
        spec.loader.exec_module(m)
        return m
    m = types.ModuleType("plspack.ph_interpolate_mut")
    m.__package__ = "plspack"
    m.__file__ = MOD
    exec(compile(src, MOD, "exec"), m.__dict__)
    return m


mod = _load()

# ---- D1: decorated list + decorated default ------------------------------
choices = mod._ckpt_choices()
_need(all(c.startswith("\u25c8 ") for c in choices),
      "D1: not every entry carries the diamond: %r" % (choices,))
_need("\u25c8 rife426.pth \u00b7 23 MB" in choices,
      "D1: the pinned-size entry is wrong, got %r" % (choices,))
# RE-GROUNDED v836: ALL FOUR checkpoints are byte-pinned now (audit B2
# measured them from the release assets), so no combo entry is naturally
# unpinned any more. The PROMISE is unchanged -- disk > pin > nothing --
# and the "nothing" branch is driven directly on the entry builder with a
# name the pin table does not know.
_need(all("\u00b7" in c for c in choices),
      "D1: every pinned, absent file must carry its pin size, got %r"
      % (choices,))
_need("\u00b7" not in mod._ckpt_entry("rife999.pth"),
      "D1: an unpinned name must still build a size-less entry -- the "
      "nothing branch died")
it = mod.ULSInterpolate.INPUT_TYPES()["required"]["ckpt_name"]
_need(it[0] == choices, "D1: the widget list is not _ckpt_choices()")
_need(it[1]["default"] == mod._ckpt_entry("rife426.pth"),
      "D1: the default is not the decorated entry: %r" % it[1]["default"])
_need(it[1]["default"] in choices,
      "D1: the default is not IN the list it defaults for")

# ---- D1b: disk beats pin -------------------------------------------------
vfi = os.path.join(tmp, "vfi")
os.makedirs(vfi, exist_ok=True)
with open(os.path.join(vfi, "rife426.pth"), "wb") as fh:
    fh.truncate(50 * 1024 * 1024)
_need(mod._ckpt_entry("rife426.pth") == "\u25c8 rife426.pth \u00b7 50 MB",
      "D1b: a file on disk must beat the byte pin")
os.remove(os.path.join(vfi, "rife426.pth"))

# ---- D2: the strip -------------------------------------------------------
_need(mod._strip_deco("\u25c8 rife426.pth \u00b7 23 MB") == "rife426.pth",
      "D2: strip failed on the house entry")
_need(mod._strip_deco("rife426.pth") == "rife426.pth",
      "D2: strip changed a bare value")
_need(mod._strip_deco("a - b \u00b7 c.pth") == "a - b \u00b7 c.pth",
      "D2: strip ate a dash or unit-less middot")

# ---- D3: regex parity with ph_basics -------------------------------------
bas_src = open(BAS, encoding="utf-8").read()
m = re.search(r'_SIZE_SUFFIX = re\.compile\(r"([^"]+)"\)', bas_src)
_need(m is not None, "D3: ph_basics._SIZE_SUFFIX not found")
_need(mod._DECO_SIZE.pattern == m.group(1),
      "D3: the size-suffix regex drifted from ph_basics "
      "(%r vs %r)" % (mod._DECO_SIZE.pattern, m.group(1)))

# ---- D4 + D6: the strip is load-bearing in interpolate() -----------------
SEEN = {}
real_arch_for = mod._arch_for


def _spy(name):
    SEEN["name"] = name
    raise RuntimeError("spy stop")  # stop before torch/model machinery


mod._arch_for = _spy
node = mod.ULSInterpolate()
try:
    node.interpolate(frames=None, ckpt_name="\u25c8 rife426.pth \u00b7 23 MB")
except RuntimeError:
    pass
mod._arch_for = real_arch_for
_need(SEEN.get("name") == "rife426.pth",
      "D4: interpolate() did not strip before _arch_for "
      "(got %r)" % SEEN.get("name"))

# ---- D5: validation ------------------------------------------------------
_need(hasattr(mod.ULSInterpolate, "VALIDATE_INPUTS"),
      "D5: no VALIDATE_INPUTS -- a pre-v829 graph dies at validation")
_need(mod.ULSInterpolate.VALIDATE_INPUTS(ckpt_name="rife426.pth") is True,
      "D5: VALIDATE_INPUTS is not permissive")

# ---- D6: mutation --------------------------------------------------------
SRC = open(MOD, encoding="utf-8").read()
MUT = SRC.replace(
    "        ckpt_name = _strip_deco(ckpt_name)"
    "  # v829: diamond+size decoration\n",
    "        pass\n", 1)
_need(MUT != SRC, "D6: mutation anchor not found")
mmod = _load(MUT)
SEEN.clear()
mmod._arch_for = _spy
mnode = mmod.ULSInterpolate()
try:
    mnode.interpolate(frames=None,
                      ckpt_name="\u25c8 rife426.pth \u00b7 23 MB")
except RuntimeError:
    pass
_need(SEEN.get("name") != "rife426.pth",
      "D6: with the strip removed the name STILL arrived bare -- D4 is "
      "not driven by the strip")

# ---- D7: the JS half -----------------------------------------------------
js = open(JS, encoding="utf-8").read()
_need("\\s\\u00b7\\s[\\d.,]+\\s?(KB|MB|GB|TB)$" in js,
      "D7: the JS regex no longer mirrors the backend")
_need("\\u25c8 " in js and "slice(2)" in js,
      "D7: the JS strip no longer removes the diamond")
_need("ULSInterpolate" in js and "onNodeCreated" in js,
      "D7: the normalisation is not wired to the node")

print("[%s] PASS -- diamond on the interpolate combo: list+default "
      "decorated, disk beats pin, strip exact and load-bearing, regex "
      "parity held, mutation caught, JS half present" % NAME)
