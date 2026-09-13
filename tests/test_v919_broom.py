"""v919 guard -- the broom cut (four measured leftovers, no field needed).

  P1  NAG: a model with NEITHER `text_embedding` NOR `cross_attn` (MiniMax
      H3 as ComfyUI builds it) is refused with the CROSS-ATTENTION reason
      and told that cfg > 1 is the way to the negative -- not with the
      "has no text_embedding" line, which reads like a missing detail
      (field, 29.08.). The text_embedding refusal still exists for a Wan
      fork that kept its blocks. No person's name in a shipped line.
  P2  test_v665 pins the locate route whitespace-tolerantly (the public
      build had the better form since v362; pulled back).
  P3  ph_weights prints the unit it computes: `>> 20` is MiB, so the
      lines say MiB (the "661 MB" for a 694-MB file, v650).
  P4  ph_empty_latent.js carries no `_fitBox` and no PREVIEW_* constants
      any more -- dead since v686 (the preview moved to the Seed node).
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NODES = os.path.join(ROOT, "nodes")
sys.path.insert(0, NODES)


def _fail(msg):
    print("[test_v919_broom] FAIL -- " + msg)
    sys.exit(1)


# P1 -- lift describe_model from the source (ph_nag imports comfy at top)
src = open(os.path.join(NODES, "ph_nag.py"), encoding="utf-8").read()
m = re.search(r"\ndef describe_model\(model\):.*?\n    return dm, name, None\n", src, re.S)
if not m:
    _fail("P1 describe_model not found")
ns = {}
exec(compile(m.group(0), "<lifted ph_nag.describe_model>", "exec"), ns)
describe_model = ns["describe_model"]


class _Block:
    def __init__(self):
        self.attn = object()          # self-attention only, like H3's DiTBlock


class _H3:
    """MiniMax H3 as Core builds it: blocks, no cross_attn, no text_embedding."""
    def __init__(self):
        self.blocks = [_Block()]

    def get_model_object(self, name):
        return self


dm, kind, why = describe_model(_H3())
if why is None:
    _fail("P1 H3-shaped model must be refused")
if "cross-attention" not in why or "MiniMax H3" not in why:
    _fail("P1 refusal must give the cross-attention reason: %r" % why)
if "text_embedding" in why:
    _fail("P1 the text_embedding line must not speak first for H3: %r" % why)
if "cfg" not in why:
    _fail("P1 refusal must point to cfg > 1 as the way to the negative: %r" % why)
if "Frank" in src.split("class ULSNag")[0]:
    _fail("P1 a shipped refusal line names a person")

te_idx = src.index('if not hasattr(dm, "text_embedding")')
ca_idx = src.index('if not blocks or not hasattr(blocks[0], "cross_attn")')
if te_idx < ca_idx:
    _fail("P1 text_embedding check still runs before the cross-attention check")


class _WanFork:
    """blocks with cross_attn, but the text_embedding was dropped."""
    class _B:
        cross_attn = object()

    def __init__(self):
        self.blocks = [self._B()]

    def get_model_object(self, name):
        return self


dm, kind, why = describe_model(_WanFork())
if why is None or "text_embedding" not in why:
    _fail("P1 a Wan fork without text_embedding must still be refused for that: %r" % why)

# P2
t = open(os.path.join(HERE, "test_v665_drop_locate.py"), encoding="utf-8").read()
if "re.search(r'\"/uls/media/locate\"\\s*,\\s*handle_media_locate'" not in t:
    _fail("P2 test_v665 does not pin the locate route whitespace-tolerantly")
if "'\"/uls/media/locate\",  handle_media_locate' not in PY" in t:
    _fail("P2 the exact-spacing pin is still there")

# P3
w = open(os.path.join(NODES, "ph_weights.py"), encoding="utf-8").read()
for frag in ('" ~%d MiB" % (total >> 20)', '%d%s MiB"', 'download done (%d MiB)'):
    if frag not in w:
        _fail("P3 ph_weights lacks the MiB label: " + frag)
if re.search(r'%d MB"|%d%s MB"|~%d MB"', w):
    _fail("P3 ph_weights still labels a >>20 number as MB")

# P4
js = open(os.path.join(ROOT, "web", "js", "ph_empty_latent.js"), encoding="utf-8").read()
for dead in ("_fitBox", "PREVIEW_MIN_H", "PREVIEW_MAX_H", "PREVIEW_DEF_H"):
    if dead in js:
        _fail("P4 dead symbol back in ph_empty_latent.js: " + dead)

print("[test_v919_broom] OK - H3 refused for the right reason (cfg > 1 named), "
      "Wan fork still refused on text_embedding, v665 regex, MiB honest, dead "
      "preview code gone (4 pins)")
