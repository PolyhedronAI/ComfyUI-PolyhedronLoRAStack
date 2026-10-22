# -*- coding: ascii -*-
"""Guard v1201 -- MiniMax Reference: ref_noise, the noise the PICTURES carry into the model.

(Frank 21.10.: "ob zusaetzlicher Noise beim Einbringen der Input-Bilder den gewuenschten Upgrade mit sich bringt oder
das nur Verzerrungen produziert" -- the test series needs a dial Core's node does not offer.)

  N1 the widget: ref_noise APPENDED as the LAST widget (behind tags, the #577 law), FLOAT 0..1, default 0.001
  N2 ref_noise_cond: Core's default (0.001) sets NOTHING (the run stays bit for bit); 0.3 sets the conditioning key
     to a = 0.7 on every entry and says "looser"; 0 sets a = 1.0 and says "TIGHTER"; with no picture wired nothing is
     set and that is said; outside 0..1 refused by name; the input conditioning is never mutated
  N3 the contract with Core, read from Core's source: model_base hands minimax_visual_cond_noise_aug to the payload as
     visual_cond_noise_aug; the model mixes  a * picture + (1 - a) * noise  only when a < 1 and presents the picture
     rows at max(t, a); the default constant is 0.999 (= 1 - 0.001); Core passes conditioning keys to extra_conds
  N4 Core's mixing MEASURED: _cond_video_rows at a = 0.7 equals 0.7 * rows + 0.3 * the seeded noise; at a = 1.0 the
     rows pass untouched
  N5 the node wires it: build() calls ref_noise_cond AFTER the references and the keyframes are set, with
     has_visual = references or a pinned first frame, and its line joins the node's info
  Z  ASCII
Script-style: exit 0 = pass.
"""
import importlib
import inspect
import os
import re
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


CORE = os.environ.get("PLS_CORE_ROOT") or "/home/claude/core_pin"
sys.path.insert(0, CORE)
sys.path.insert(0, str(ROOT / "nodes"))
import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True
fp = types.ModuleType("folder_paths")
for _n in ("get_temp_directory", "get_output_directory", "get_input_directory", "get_user_directory"):
    setattr(fp, _n, lambda: "/tmp")
fp.get_filename_list = lambda *a, **k: []
fp.get_full_path = lambda *a, **k: None
fp.models_dir = "/tmp/models"
fp.add_model_folder_path = lambda *a, **k: None
fp.get_folder_paths = lambda f: ["/tmp/models/" + f]
sys.modules.setdefault("folder_paths", fp)
pkg = types.ModuleType("plsn1201")
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules["plsn1201"] = pkg
MR = importlib.import_module("plsn1201.ph_minimax_ref")

# ---- N1
it = MR.ULSMiniMaxReference.INPUT_TYPES()
req, opt = it["required"], it["optional"]
widgets = [k for k in req if req[k][0] in ("FLOAT", "STRING", "INT") or isinstance(req[k][0], list)] \
    + [k for k in opt if opt[k][0] in ("FLOAT", "STRING", "INT")]
_need(widgets[-2:] == ["tags", "ref_noise"], "N1 ref_noise is the LAST widget, behind tags (%s)" % widgets[-3:])
spec = opt["ref_noise"]
_need(spec[0] == "FLOAT" and spec[1]["default"] == 0.001 and spec[1]["min"] == 0.0 and spec[1]["max"] == 1.0,
      "N1 FLOAT 0..1, default 0.001")

# ---- N2
base = [[torch.zeros(1, 2, 4), {"minimax_refs": ["x"]}], [torch.zeros(1, 2, 4), {"pooled_output": None}]]
c0, l0 = MR.ref_noise_cond(base, 0.001, True)
_need(c0 is base and l0 is None, "N2 Core's default sets nothing (the run stays bit for bit)")
c1, l1 = MR.ref_noise_cond(base, 0.3, True)
_need(all(abs(e[1][MR.REF_NOISE_KEY] - 0.7) < 1e-12 for e in c1) and "looser" in l1 and "0.700 x picture + 0.300 x noise" in l1,
      "N2 0.3 -> a = 0.7 on every entry, said 'looser' (%s)" % l1)
_need(all(MR.REF_NOISE_KEY not in e[1] for e in base), "N2 the input conditioning is not mutated")
c2, l2 = MR.ref_noise_cond(base, 0.0, True)
_need(all(e[1][MR.REF_NOISE_KEY] == 1.0 for e in c2) and "TIGHTER" in l2, "N2 0 -> a = 1.0, said 'TIGHTER'")
c3, l3 = MR.ref_noise_cond(base, 0.5, False)
_need(c3 is base and "no picture is wired" in l3, "N2 no picture wired: nothing set, said")
try:
    MR.ref_noise_cond(base, 1.5, True)
    _need(False, "N2 1.5 must be refused")
except ValueError as e:
    _need("0..1" in str(e), "N2 outside 0..1 refused by name")

# ---- N3 the contract, read from Core's source
mb = (Path(CORE) / "comfy" / "model_base.py").read_text(encoding="utf-8")
md = (Path(CORE) / "comfy" / "ldm" / "minimax" / "model.py").read_text(encoding="utf-8")
sm = (Path(CORE) / "comfy" / "samplers.py").read_text(encoding="utf-8")
_need('payload["visual_cond_noise_aug"] = kwargs["minimax_visual_cond_noise_aug"]' in mb and MR.REF_NOISE_KEY == "minimax_visual_cond_noise_aug",
      "N3 model_base hands minimax_visual_cond_noise_aug to the payload")
_need("r = aug * r + (1.0 - aug) * noise.to(r.device)" in md and "if aug < 1.0:" in md, "N3 the model mixes a * picture + (1 - a) * noise, only when a < 1")
_need('"cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),' in md, "N3 the picture rows are presented at max(t, a)")
_need(re.search(r"^VISUAL_COND_TIMESTEP = 0\.999$", md, re.M) is not None and abs((1 - MR.REF_NOISE_DEFAULT) - 0.999) < 1e-12,
      "N3 Core's default a = 0.999 = 1 - our default")
_need("params = x.copy()" in sm and "out = model_function(**params)" in sm, "N3 Core passes the conditioning keys to extra_conds")

# ---- N4 Core's mixing measured
import comfy.ldm.minimax.model as MM  # noqa: E402


class _Self:
    patch_size = getattr(MM, "PATCH_SIZE", None) or (1, 2, 2)


z = torch.randn(1, 4, 2, 4, 4)
try:
    ps = _Self.patch_size
    r_clean = MM.patchify_video(z.to(torch.float32), ps)
    out7 = MM.MiniMaxH3Model._cond_video_rows(_Self(), {"visual_cond_noise_aug": 0.7, "seed": 5, "cond_video_latents": [z]}, "cpu") \
        if hasattr(MM, "MiniMaxH3Model") else None
    if out7 is None:
        cls = [v for v in vars(MM).values() if inspect.isclass(v) and hasattr(v, "_cond_video_rows")][0]
        out7 = cls._cond_video_rows(_Self(), {"visual_cond_noise_aug": 0.7, "seed": 5, "cond_video_latents": [z]}, "cpu")
        out1 = cls._cond_video_rows(_Self(), {"visual_cond_noise_aug": 1.0, "seed": 5, "cond_video_latents": [z]}, "cpu")
    else:
        out1 = MM.MiniMaxH3Model._cond_video_rows(_Self(), {"visual_cond_noise_aug": 1.0, "seed": 5, "cond_video_latents": [z]}, "cpu")
    gen = torch.Generator("cpu").manual_seed(5)
    noise = torch.randn(r_clean.shape, generator=gen, dtype=torch.float32)
    _need(torch.allclose(out7, 0.7 * r_clean + 0.3 * noise, atol=1e-6), "N4 a = 0.7: 0.7 x rows + 0.3 x the seeded noise (measured)")
    _need(torch.equal(out1, r_clean), "N4 a = 1.0: the rows pass untouched")
except Exception as e:
    _need(False, "N4 Core's _cond_video_rows could not be driven: %s" % e)

# ---- N5 the node wires it
src = (ROOT / "nodes" / "ph_minimax_ref.py").read_text(encoding="utf-8")
b = src[src.find("    def build(self"):]
i_ref = b.find('cond, {"minimax_refs": ref_blocks})')
i_kf = b.find('cond, {"minimax_keyframes": kf_list, "minimax_frame_count": frame_count})')
i_rn = b.find('cond, rn_line = ref_noise_cond(cond, opt.get("ref_noise", REF_NOISE_DEFAULT), bool(ref_blocks or kf_list))')
i_hd = b.find("header = (")
_need(0 < i_ref < i_kf < i_rn < i_hd and "lines.append(rn_line)" in b[i_rn:i_hd],
      "N5 build() applies it after the references and keyframes, has_visual = refs or a first frame, the line joins the info")
for p in ("nodes/ph_minimax_ref.py", "tests/test_v1201_ref_noise.py"):
    _need(all(c < 128 for c in (ROOT / p).read_bytes()), "Z ASCII %s" % p)
print("%s: %d check(s) failed" % ("test_v1201_ref_noise", len(FAILS)) if FAILS else "test_v1201_ref_noise: all checks passed")
sys.exit(1 if FAILS else 0)
