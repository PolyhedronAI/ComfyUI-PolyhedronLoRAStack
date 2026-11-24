#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v1354 -- blank picture: strength, any colour, the blank as a real picture (Image Reference).

Frank, 21.11.: WAS' "H3 Image Blank Guidance" (WAS Node Suite, MIT) mixes plain + s x (blank - plain) on the
first steps ("subtraction", s = 2), takes any colour and gives the blank to Core as a real picture (1024 px,
vision rows AND a reference latent). Built into OUR node, no guider of our own: Core's cond strengths do the mix.

WHAT IS PINNED

  S1  weigh_window: s = 1 returns the conditioning UNCHANGED (the v1338 result bit for bit); s != 1 gives the
      window's blank entries strength s and adds the plain conditioning in the window at strength 1 - s.
  S2  The mix through REAL Core: comfy.samplers.calculate_start_end_timesteps + get_area_and_mult on the converted
      entries, summed as calc_cond_batch sums them (output * mult / sum mult): inside the window
      plain + s x (blank - plain) for s in 2, 0.5, -1; outside the plain prediction alone.
  S3  parse_colour: #rrggbb, #rgb, without "#"; a wrong colour raises by name. blank_image("colour") fills the
      colour, colour_name writes it back as #rrggbb.
  S4  Image Reference end to end with a fake encoder: defaults (strength 1, mid grey, anchor off) give exactly the
      v1338 conditioning; strength 2 adds the plain entry; strength 0 refuses in the info; colour reaches the
      encoder's picture; the info names strength and colour.
  S5  Anchor: Core's own class is asked for prompt + a 1024 px blank with the VAE (stand-in of Core master's
      MiniMaxH3ImageConditioning); its rows and minimax_refs stand in the window, the plain ones outside; anchor
      without VAE and anchor on p-h3-image raise by name.
  S6  The three widgets appended LAST (after v1338's three), defaults that keep old graphs; ASCII-clean.

Script-style: exit 0 = pass.
"""

import contextlib
import importlib.util
import io
import os
import re
import sys
import types

NAME = "v1354"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NODES = os.path.join(ROOT, "nodes")
_fails = []


def _need(cond, msg):
    if not cond:
        _fails.append(msg)
        print("  FAIL " + msg)
    else:
        print("  ok   " + msg)


import torch  # noqa: E402

# the pack as a package (ph_image_ref imports `from . import h3_image`)
PKG = "pls_v1354"
pkg = types.ModuleType(PKG)
pkg.__path__ = [ROOT]
sys.modules[PKG] = pkg
npkg = types.ModuleType(PKG + ".nodes")
npkg.__path__ = [NODES]
sys.modules[PKG + ".nodes"] = npkg


def _load(name):
    full = PKG + ".nodes." + name
    spec = importlib.util.spec_from_file_location(full, os.path.join(NODES, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[full] = m
    spec.loader.exec_module(m)
    return m


HI = _load("h3_image")

# ---- S1 -----------------------------------------------------------------------------------------------------
print("S1 weigh_window")
txt = torch.arange(6, dtype=torch.float32).view(1, 3, 2)
plain = [[txt, {"minimax_token_tags": torch.zeros(3)}]]
pic = torch.full((1, 2, 2), 9.0)
primed = HI.prime_blank_picture(plain, [pic, {"minimax_token_tags": torch.ones(2)}], 0.0, 0.25)
same = HI.weigh_window(primed, 0.0, 0.25, 1.0, plain)
_need(same is primed, "s = 1: the conditioning untouched (v1338 bit for bit)")
w2 = HI.weigh_window(primed, 0.0, 0.25, 2.0, plain)
_need(len(w2) == len(primed) + 1, "s = 2: one plain entry joins the window")
_need(w2[0][1]["strength"] == 2.0 and torch.equal(w2[0][0][:, :2], pic), "blank entry at strength 2")
_need(w2[-1][1]["strength"] == -1.0 and torch.equal(w2[-1][0], txt)
      and (w2[-1][1]["start_percent"], w2[-1][1]["end_percent"]) == (0.0, 0.25), "plain entry in the window at 1 - s")
_need("strength" not in w2[1][1], "the plain tail outside the window keeps strength 1")

# ---- S2 -----------------------------------------------------------------------------------------------------
print("S2 the mix through real Core")
CORE = os.environ.get("PLS_CORE_ROOT") or "/home/claude/core_pin"
if CORE not in sys.path:
    sys.path.insert(0, CORE)
try:
    import comfy.cli_args
    comfy.cli_args.args.cpu = True          # the sandbox has no card; Core's sampler module wants a device
    import comfy.samplers as CS
    have_core = True
except Exception as e:  # pragma: no cover - the cold set always has Core
    have_core = False
    _need(False, "Core importable from %s (%s)" % (CORE, e))

if have_core:
    class _MS:
        @staticmethod
        def percent_to_sigma(p):
            return 1.0 - p          # linear: step k of 6 at sigma 1 - k/6 (the mixing is what is pinned here)

    class _Model:
        model_sampling = _MS()

    VALUE = {"blank": 10.0, "plain": 4.0}

    def _mixed(entries, sigma):
        conds = []
        for c, d in entries:
            x = dict(d)
            x["model_conds"] = {}
            x["uuid"] = len(conds)          # convert_cond sets one; get_area_and_mult hands it on
            x["tag"] = "blank" if c.shape[1] != txt.shape[1] else "plain"
            conds.append(x)
        CS.calculate_start_end_timesteps(_Model(), conds)
        xin = torch.zeros(1, 1, 2, 2)
        tin = torch.tensor([sigma])
        num = torch.zeros_like(xin)
        den = torch.ones_like(xin) * 1e-37
        for x in conds:
            r = CS.get_area_and_mult(x, xin, tin)
            if r is None:
                continue
            mult = r.mult if hasattr(r, "mult") else r[1]
            num += VALUE[x["tag"]] * mult
            den += mult
        return float((num / den).mean())

    for s in (2.0, 0.5, -1.0):
        ent = HI.weigh_window(primed, 0.0, 0.25, s, plain)
        want = VALUE["plain"] + s * (VALUE["blank"] - VALUE["plain"])
        got1 = _mixed(ent, 1.0)                 # step 1 (percent 0)
        got2 = _mixed(ent, 1.0 - 1.0 / 6.0)     # step 2 (percent 1/6)
        got4 = _mixed(ent, 0.5)                 # step 4 (percent 1/2), outside
        _need(abs(got1 - want) < 1e-4 and abs(got2 - want) < 1e-4,
              "s = %g: steps 1-2 give plain + s x (blank - plain) = %g (got %.4f, %.4f)" % (s, want, got1, got2))
        _need(abs(got4 - VALUE["plain"]) < 1e-4, "s = %g: step 4 the plain prediction alone (got %.4f)" % (s, got4))
    _need(abs(_mixed(primed, 1.0) - VALUE["blank"]) < 1e-4, "s = 1: step 1 the blank prediction (v1338)")

# ---- S3 -----------------------------------------------------------------------------------------------------
print("S3 colour")
_need(HI.parse_colour("#808080") == (128 / 255.0,) * 3, "#808080")
_need(HI.parse_colour("1a2440") == (0x1a / 255.0, 0x24 / 255.0, 0x40 / 255.0), "without #")
_need(HI.parse_colour("#fff") == (1.0, 1.0, 1.0), "#rgb")
for bad in ("", "#12345", "grey", "#gg0000"):
    try:
        HI.parse_colour(bad)
        _need(False, "bad colour %r refused" % bad)
    except ValueError as e:
        _need("blank_colour" in str(e), "bad colour %r refused by name" % bad)
im = HI.blank_image("colour", HI.parse_colour("#1a2440"), 64)
_need(tuple(im.shape) == (1, 64, 64, 3) and abs(float(im[0, 5, 7, 2]) - 0x40 / 255.0) < 1e-6, "colour image filled")
_need(HI.colour_name("colour", HI.parse_colour("#1A2440")) == "#1a2440" and HI.colour_name("black") == "black",
      "colour named back")
_need(tuple(HI.blank_image("mid grey").shape) == (1, 512, 512, 3), "tones keep 512 px")

# ---- S4 / S5 Image Reference ---------------------------------------------------------------------------------
print("S4 Image Reference")
IR = _load("ph_image_ref")


class _Tok:
    pass


class _Clip:
    """H3 tokenizer stand-in: text rows = 3, a picture adds 2 rows in front."""

    def __init__(self):
        self.tokenizer = _Tok()
        self.calls = []

    def tokenize(self, t, **kw):
        self.calls.append((t, kw))
        return {"t": t, "kw": kw}

    def encode_from_tokens_scheduled(self, tok):
        if tok["kw"].get("images"):
            return [[torch.full((1, 2, 2), 9.0), {"minimax_token_tags": torch.ones(2)}]]
        return [[txt.clone(), {"minimax_token_tags": torch.zeros(3)}]]


core_calls = []


class _CoreH3Image:
    """Stand-in of Core master's MiniMaxH3ImageConditioning.execute (read 21.11.): one encode of prompt + pictures,
    with a VAE the pictures as minimax_refs, latent 2048 or from image_1."""

    @classmethod
    def execute(cls, clip, prompt, width, height, vae=None, images=None):
        core_calls.append({"prompt": prompt, "width": width, "height": height, "vae": vae, "images": images})
        if images:
            img = images["image_1"]
            rows = torch.cat([torch.full((1, 5, 2), 7.0), txt], dim=1)
            d = {"minimax_token_tags": torch.ones(8)}
            if vae is not None:
                d["minimax_refs"] = [{"kind": "image", "latent": torch.zeros(1, 24, 1, 64, 64),
                                      "side": int(img.shape[1])}]
            return ([[rows, d]], {"samples": torch.zeros(1, 24, 1, 64, 64)})
        return ([[txt.clone(), {"minimax_token_tags": torch.zeros(3)}]],
                {"samples": torch.zeros(1, 24, 1, 128, 128)})


class _Vae:
    pass


HI.core_class = lambda variant: _CoreH3Image          # ph_image_ref uses this very module object
fam_h3 = "minimax_h3_image"


def _run(**kw):
    clip = _Clip()
    with contextlib.redirect_stdout(io.StringIO()):
        out = IR.ULSImageReference()._minimax_image(clip, kw.pop("vae", _Vae()), None, "a portrait", [], "test",
                                                     "MiniMaxH3ImageVAE", **kw)
    return clip, out


clip0, base = _run(blank="mid grey", blank_start=0.0, blank_end=0.25)
clip1, dflt = _run(blank="mid grey", blank_start=0.0, blank_end=0.25, blank_strength=1.0,
                   blank_colour="#808080", blank_anchor=False)
_need(len(base[0]) == len(dflt[0]) == 2 and all(torch.equal(a[0], b[0]) and a[1].keys() == b[1].keys()
                                                for a, b in zip(base[0], dflt[0])),
      "defaults: the v1338 conditioning")
_need("strength" not in dflt[0][0][1], "defaults: no strength key")
_, s2 = _run(blank="mid grey", blank_start=0.0, blank_end=0.25, blank_strength=2.0)
_need(len(s2[0]) == 3 and s2[0][0][1]["strength"] == 2.0 and s2[0][2][1]["strength"] == -1.0,
      "strength 2: blank at 2, plain in the window at -1")
_need("blank_strength 2: plain + 2 x (blank - plain)" in s2[3], "info names the strength")
_, s0 = _run(blank="mid grey", blank_start=0.0, blank_end=0.25, blank_strength=0.0)
_need(len(s0[0]) == 1 and "NOT used: blank_strength 0" in s0[3], "strength 0 refused in the info")
cc, col = _run(blank="colour", blank_colour="#1a2440", blank_start=0.0, blank_end=0.25)
pics = [kw["images"][0] for t, kw in cc.calls if kw.get("images")]
_need(len(pics) == 1 and abs(float(pics[0][0, 0, 0, 1]) - 0x24 / 255.0) < 1e-6, "colour reaches the encoder's picture")
_need("blank_picture #1a2440 0.000-0.250" in col[3], "info names the colour")
try:
    _run(blank="colour", blank_colour="night", blank_start=0.0, blank_end=0.25)
    _need(False, "bad colour raises")
except ValueError as e:
    _need("blank_colour" in str(e), "bad colour raises by name")

print("S5 anchor")
core_calls.clear()
_, an = _run(blank="mid grey", blank_start=0.0, blank_end=0.25, blank_anchor=True, blank_strength=2.0)
blank_call = [c for c in core_calls if c["images"]]
_need(len(blank_call) == 1 and blank_call[0]["prompt"] == "a portrait" and blank_call[0]["vae"] is not None,
      "Core's own class asked for prompt + blank with the VAE")
bimg = blank_call[0]["images"]["image_1"] if blank_call else None
_need(bimg is not None and tuple(bimg.shape) == (1, 1024, 1024, 3) and abs(float(bimg.mean()) - 0.5) < 1e-6,
      "the blank is 1024 px, mid grey")
win = [e for e in an[0] if e[1].get("start_percent") == 0.0 and e[1].get("end_percent") == 0.25]
_need(len(win) == 2 and any("minimax_refs" in e[1] and e[1].get("strength") == 2.0 for e in win)
      and any("minimax_refs" not in e[1] and e[1].get("strength") == -1.0 for e in win),
      "window: Core's rows + minimax_refs at 2, plain at -1")
tail = [e for e in an[0] if e[1].get("start_percent") == 0.25]
_need(len(tail) == 1 and "minimax_refs" not in tail[0][1] and torch.equal(tail[0][0], txt),
      "outside the window the plain text, no reference")
_need(tuple(an[2]["samples"].shape[-2:]) == (128, 128), "canvas stays the plain run's (not the blank's 1024)")
_need("anchored" in an[3], "info says anchored")
try:
    _run(blank="mid grey", blank_start=0.0, blank_end=0.25, blank_anchor=True, vae=None)
    _need(False, "anchor without VAE raises")
except ValueError as e:
    _need("needs the VAE" in str(e), "anchor without VAE raises by name")
except Exception as e:
    # without a VAE Core's plain encode already runs; our refusal must come first or name the VAE
    _need("VAE" in str(e), "anchor without VAE raises naming the VAE (%s)" % e)
try:
    clip = _Clip()
    with contextlib.redirect_stdout(io.StringIO()):
        IR.ULSImageReference()._minimax_image(clip, _Vae(), None, "x", [], "test", "PH3ImageVAE",
                                              blank="mid grey", blank_start=0.0, blank_end=0.25, blank_anchor=True)
    _need(False, "anchor on p-h3-image raises")
except ValueError as e:
    _need("MiniMax H3 Image" in str(e), "anchor on p-h3-image raises by name")

# ---- S6 -----------------------------------------------------------------------------------------------------
print("S6 widgets, ASCII")
src = open(os.path.join(NODES, "ph_image_ref.py")).read()
opt = src[src.index('"optional": {'):src.index("RETURN_TYPES")]
keys = [k for k in re.findall(r'^\s{16}"(\w+)": \(', opt, re.M) if k != "prompt_in"]
_need(keys[-6:] == ["blank_picture", "blank_start", "blank_end", "blank_strength", "blank_colour", "blank_anchor"],
      "v1354 widgets appended LAST after v1338's: %s" % keys[-6:])
_need('"blank_strength": ("FLOAT", {"default": 1.0' in opt, "blank_strength defaults to 1 (old graphs unchanged)")
_need('"blank_colour": ("STRING", {"default": "#808080"' in opt, "blank_colour defaults to mid grey")
_need('"blank_anchor": ("BOOLEAN", {"default": False' in opt, "blank_anchor defaults to off")
_need('"white", "colour"]' in opt, "blank_picture offers colour last")
for f in ("h3_image.py", "ph_image_ref.py", os.path.join("..", "tests", os.path.basename(__file__))):
    _need(all(ord(ch) < 128 for ch in open(os.path.join(NODES, f)).read()), f + " ASCII-clean")

print("%s: %s" % (NAME, "PASS" if not _fails else "FAIL %d" % len(_fails)))
sys.exit(0 if not _fails else 1)
