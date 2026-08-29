"""v872 -- the joint AV latent leaves the hood: noise, latent, reference.

Three cuts, one guard:

  A  uls_noise.shaped_noise_joint  -- shaped noise on a joint latent shapes the
     VIDEO half and gives the audio half plain gaussian. Before v872 the shaped
     branch read `.shape`, which delegates to tensors[0], and returned a FLAT
     video-shaped tensor against a nested latent.
  B  uls_latent_math.minimax_*     -- the 17k+5 arithmetic, and its EXACT
     inverse, which is what lets a downstream node recover the size from a
     latent it did not build. Plus: duration_seconds is APPENDED, never
     inserted.
  C  ph_minimax_ref                -- pure scale rule, the latent-derived size,
     and the two Core coupling points, pinned against Core's own source when it
     can be found.

Runs without torch and without comfy: the pure helpers are lifted with ast.
"""
import ast
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FAILED = []


def _fail(m):
    FAILED.append(m)
    print("FAIL: {}".format(m))


def _ok(m):
    print("ok  : {}".format(m))


def _lift(relpath, names, extra_assigns=()):
    """exec the named module-level functions (and constants) with no imports.

    `math` is seeded because the lifted bodies use it; seeding a stdlib module
    is not the same as importing the node's own dependency tree, which is the
    whole point of lifting."""
    import math as _math
    src = (ROOT / relpath).read_text(encoding="utf-8")
    tree = ast.parse(src)
    ns, got = {"math": _math}, {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            tgt = node.targets[0].id if isinstance(node.targets[0], ast.Name) else ""
            if extra_assigns and tgt.startswith(extra_assigns):
                exec(compile(ast.Module([node], []), "<v872>", "exec"), ns)
        if isinstance(node, ast.FunctionDef) and node.name in names:
            got[node.name] = node
    for name in names:
        if name in got:
            exec(compile(ast.Module([got[name]], []), "<v872>", "exec"), ns)
    return ns, src, set(got)


# =====================================================================  B
MATH = ("minimax_align_frames", "minimax_video_latent_t",
        "minimax_frames_from_latent_t", "minimax_temporal_shape",
        "minimax_frames_from_seconds", "minimax_size_from_latent")
ns, lm_src, got = _lift("nodes/uls_latent_math.py", MATH, ("MINIMAX_",))
missing = set(MATH) - got
if missing:
    _fail("uls_latent_math is missing {}".format(sorted(missing)))
else:
    _ok("six minimax_* helpers lifted without torch")

if not missing:
    align = ns["minimax_align_frames"]
    tshape = ns["minimax_temporal_shape"]
    inv = ns["minimax_frames_from_latent_t"]
    from_s = ns["minimax_frames_from_seconds"]

    bad = [n for n in (5, 6, 81, 120, 124, 125, 362, 600) if align(n) % 17 != 5]
    if bad:
        _fail("align_frames left {} off the 17k+5 grid".format(bad))
    else:
        _ok("align_frames lands every probe on the 17k+5 grid")

    if align(3) != 5:
        _fail("align_frames must floor at 5 (Core: max(5, length))")
    else:
        _ok("align_frames floors at 5")

    broken = [n for n in (5, 81, 120, 124, 125, 362, 600, 1000)
              if inv(tshape(n)[1]) != tshape(n)[0]]
    if broken:
        _fail("the latent_t inverse is not exact for {} -- a node that recovers "
              "the frame count from a latent would lie".format(broken))
    else:
        _ok("frames <-> latent_t round-trips exactly on every probe")

    # Pinned against Core's own template default and against Frank's real run.
    if from_s(5.0) != 124:
        _fail("5.0 s must give 124 frames (Core's template default), got {}"
              .format(from_s(5.0)))
    elif tshape(124) != (124, 37, 207):
        _fail("temporal_shape(124) must be (124, 37, 207) -- the field run's "
              "latent was (1, 24, 37, 30, 54) with a 207-long audio half; got {}"
              .format(tshape(124)))
    else:
        _ok("5.0 s -> 124 frames -> latent_t 37 / audio_t 207 (matches the field run)")

    class _T(object):
        def __init__(self, shape):
            self.shape = shape
            self.ndim = len(shape)
            self.is_nested = False

    class _J(object):
        is_nested = True

        def __init__(self, parts):
            self._p = parts

        def unbind(self):
            return self._p

    size = ns["minimax_size_from_latent"]
    real = _J([_T((1, 24, 37, 30, 54)), _T((1, 32, 2, 207))])
    if size(real) != (864, 480, 124):
        _fail("size recovery from the FIELD latent (1,24,37,30,54) must give "
              "(864, 480, 124), got {}".format(size(real)))
    elif size(_T((1, 24, 37, 30, 54))) is not None:
        _fail("size recovery must return None for a plain tensor -- guessing "
              "would be worse than refusing")
    elif size(_J([_T((1, 16, 30, 54))])) is not None:
        _fail("size recovery must return None when the video half is not 5D")
    else:
        _ok("size recovery: field latent -> (864, 480, 124), non-joint -> None")

# duration_seconds must be APPENDED, or saved workflows shift.
base = sorted(ROOT.glob("WIDGET_ORDER_baseline_v*.txt"))
if len(base) != 1:
    _fail("expected exactly ONE widget baseline, found {}".format(
        [b.name for b in base]))
else:
    line = [l for l in base[0].read_text(encoding="utf-8").splitlines()
            if l.startswith("ULSEmptyLatent\t")]
    if not line:
        _fail("ULSEmptyLatent is missing from the widget baseline")
    elif not line[0].split("\t")[1].endswith(",duration_seconds"):
        _fail("duration_seconds must be the LAST widget of ULSEmptyLatent "
              "(v585: widgets_values is positional) -- baseline says: {}"
              .format(line[0].split("\t")[1]))
    else:
        _ok("duration_seconds is appended LAST (v585 honoured)")

el_src = (ROOT / "nodes" / "ph_empty_latent.py").read_text(encoding="utf-8")
if 'key == "minimax_h3"' not in el_src:
    _fail("ph_empty_latent has no minimax_h3 branch")
elif "_minimax_av" not in el_src:
    _fail("the minimax builder _minimax_av is missing")
elif "NestedTensor" not in el_src:
    _fail("the minimax builder must emit a NestedTensor, not a plain latent")
else:
    _ok("ph_empty_latent branches to a NestedTensor builder for minimax_h3")

# =====================================================================  A
nz_src = (ROOT / "nodes" / "uls_noise.py").read_text(encoding="utf-8")
tree = ast.parse(nz_src)
joint = [n for n in tree.body
         if isinstance(n, ast.FunctionDef) and n.name == "shaped_noise_joint"]
if not joint:
    _fail("uls_noise.shaped_noise_joint is missing -- shaped noise on a joint "
          "latent still returns a flat video-shaped tensor")
else:
    body = ast.get_source_segment(nz_src, joint[0]) or ""
    if "parts[0]" not in body or "make_noise" not in body:
        _fail("shaped noise must be applied to parts[0], the VIDEO half")
    elif "prepare_noise" not in body:
        _fail("the remaining halves must get Core's plain gaussian")
    elif "NestedTensor" not in body:
        _fail("the per-part noises must be re-wrapped as a NestedTensor")
    else:
        _ok("shaped_noise_joint: video half shaped, rest gaussian, re-wrapped")

    if "tuple(samples.shape)" in body:
        _fail("shaped_noise_joint still reads samples.shape -- that delegates "
              "to tensors[0] and is the v872 wound itself")
    else:
        _ok("shaped_noise_joint never reads .shape off the joint latent")

    gen = [n for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef) and n.name == "generate_noise"]
    gsrc = ast.get_source_segment(nz_src, gen[0]) if gen else ""
    if "is_nested" not in (gsrc or ""):
        _fail("generate_noise does not check for a joint latent before taking "
              "the shaped branch")
    else:
        _ok("generate_noise routes a joint latent to the joint rule")

# =====================================================================  C
mr = ROOT / "nodes" / "ph_minimax_ref.py"
if not mr.exists():
    _fail("nodes/ph_minimax_ref.py is missing")
else:
    mr_src = mr.read_text(encoding="utf-8")
    ns2, _, got2 = _lift("nodes/ph_minimax_ref.py", ("ref_scale",), ("REF_IMAGE_",))
    if "ref_scale" not in got2:
        _fail("ref_scale is not a pure module-level function -- the guard "
              "cannot drive it")
    else:
        rs = ns2["ref_scale"]
        # Core's 'match': sqrt(gen_area / src_area), DOWN ONLY.
        if abs(rs(2000, 2000, 1000, 1000, "match", 0.0) - 0.5) > 1e-9:
            _fail("'match' must scale by sqrt(gen_area/src_area)")
        elif rs(100, 100, 1000, 1000, "match", 0.0) != 1.0:
            _fail("'match' must NEVER upscale a reference")
        elif abs(rs(4096, 4096, 100, 100, "max", 0.0) - 0.5) > 1e-9:
            _fail("'max' must scale to a 2048 short edge")
        elif rs(500, 500, 100, 100, "max", 0.0) != 1.0:
            _fail("'max' must never upscale either")
        else:
            _ok("ref_scale mirrors Core's two rules and never upscales")

        # our addition: a per-image megapixel target wins over the global rule
        # 2000x2000 = 4 MP; a 1 MP target is a LINEAR x0.5 (0.5 per edge is a
        # quarter of the area). Asserting 0.25 here would be confusing the
        # area ratio with the scale factor -- which is exactly what this probe
        # caught while it was being written.
        got_mp = rs(2000, 2000, 4000, 4000, "match", 1.0)
        if abs(got_mp - 0.5) > 1e-9:
            _fail("a megapixel target must override the global rule "
                  "(2000x2000 -> 1 MP is a linear x0.5), got {}".format(got_mp))
        elif abs(rs(2000, 2000, 4000, 4000, "match", 0.0) - 1.0) > 1e-9:
            _fail("without a megapixel target the same call must follow the "
                  "global rule instead (here: no downscale at all)")
        elif rs(500, 500, 100, 100, "match", 9.0) != 1.0:
            _fail("a megapixel target must not upscale either")
        else:
            _ok("per-image megapixel target overrides the rule, still down-only")

    if "minimax_size_from_latent" not in mr_src:
        _fail("the reference node must READ the size from the latent, not from "
              "its own widgets")
    elif '"width"' in mr_src or '"height"' in mr_src:
        _fail("the reference node must not carry width/height widgets -- a "
              "second source of truth can drift out of step with the latent")
    else:
        _ok("size comes from the latent only (one source of truth)")

    if "raise ValueError" not in mr_src:
        _fail("a non-joint latent must be REFUSED by name, not guessed at")
    else:
        _ok("a non-joint latent is refused by name")

    # --- THE COUPLING, pinned AT THE AST ------------------------------
    # NOT a text search. The module docstring quotes these very names, and a
    # reconstructed-source search still matched after the call itself had been
    # renamed -- measured twice while this guard was written. So we find the
    # actual CALL and read its actual keyword, which no comment can fake.
    tok_kw, cond_keys, enc_calls = set(), set(), set()
    for node in ast.walk(ast.parse(mr_src)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name == "tokenize":
            tok_kw |= {k.arg for k in node.keywords if k.arg}
        if name in ("encode_from_tokens_scheduled", "encode_from_tokens"):
            enc_calls.add(name)
        if name == "conditioning_set_values":
            for a in node.args:
                if isinstance(a, ast.Dict):
                    cond_keys |= {k.value for k in a.keys
                                  if isinstance(k, ast.Constant)}
    if "minimax_ref_items" not in tok_kw:
        _fail("clip.tokenize() is not called with minimax_ref_items={} -- the "
              "<Picture n> tags would resolve to nothing and the run would "
              "carry NO references, silently".format(sorted(tok_kw) or "nothing"))
    else:
        _ok("clip.tokenize() carries the minimax_ref_items keyword (AST)")
    if "minimax_refs" not in cond_keys:
        _fail("conditioning_set_values() does not set the 'minimax_refs' key "
              "(found {})".format(sorted(cond_keys) or "nothing"))
    else:
        _ok("conditioning_set_values() sets minimax_refs (AST)")
    # ...and the call must be REACHABLE. `if False:` around it would leave
    # every AST needle above satisfied while shipping a run with no references.
    dead = False
    for node in ast.walk(ast.parse(mr_src)):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Constant):
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call)
                        and getattr(sub.func, "attr", "") == "conditioning_set_values"):
                    dead = True
    if dead:
        _fail("the conditioning_set_values call sits behind a CONSTANT test -- "
              "it is dead code and the references never reach the model")
    else:
        _ok("the conditioning_set_values call is reachable, not dead code")

    if "encode_from_tokens_scheduled" not in enc_calls:
        _fail("the scheduled encode is gone (found {}) -- Core's own node uses "
              "encode_from_tokens_scheduled".format(sorted(enc_calls) or "nothing"))
    else:
        _ok("the encode call matches Core's (AST)")

    # If Core is reachable, the names must still exist THERE.
    core = None
    for up in ROOT.parents:
        cand = up / "comfy_extras" / "nodes_minimax_h3.py"
        if cand.exists():
            core = cand
            break
    if core is None:
        print("note: Core's nodes_minimax_h3.py not found from here -- the "
              "upstream half of the coupling pin is SKIPPED (it fires on "
              "Frank's machine, where the pack sits inside ComfyUI).")
    else:
        core_src = core.read_text(encoding="utf-8", errors="replace")
        drifted = [n for n in ("minimax_ref_items", "minimax_refs")
                   if n not in core_src]
        if drifted:
            _fail("Core no longer uses {} -- our reference node would build a "
                  "run with NO references and say nothing. Re-read "
                  "comfy_extras/nodes_minimax_h3.py.".format(drifted))
        else:
            _ok("both coupling names still exist in Core's own source")

# node id baseline
nb = sorted(ROOT.glob("NODE_IDS_baseline_v*.txt"))
if len(nb) != 1:
    _fail("expected exactly ONE node-id baseline, found {}".format(
        [b.name for b in nb]))
elif "ULSMiniMaxReference" not in nb[0].read_text(encoding="utf-8"):
    _fail("ULSMiniMaxReference is not in the node-id baseline")
else:
    _ok("ULSMiniMaxReference is carried by the node-id baseline")

init_src = (ROOT / "__init__.py").read_text(encoding="utf-8")
if "_MMREF_OK" not in init_src or 'NODE_CLASS_MAPPINGS["ULSMiniMaxReference"]' not in init_src:
    _fail("ULSMiniMaxReference is not registered the way this pack registers")
else:
    _ok("ULSMiniMaxReference registered with the _XXX_OK pattern")

# =====================================================================  v873
# The chain must be WIRABLE. positive/negative are born DOWNSTREAM of this
# latent (in the reference stage), so as required sockets they made the graph a
# ring. This pin is the whole reason v873 exists.
import re as _re
_it = ""
for node in ast.walk(ast.parse(el_src)):
    if isinstance(node, ast.ClassDef) and node.name == "ULSEmptyLatent":
        for m in node.body:
            if isinstance(m, ast.FunctionDef) and m.name == "INPUT_TYPES":
                _it = ast.get_source_segment(el_src, m) or ""
_sec = {}
for name in ("required", "optional"):
    mm = _re.search(r'"%s"\s*:\s*\{(.*?)\n\s{12}\}' % name, _it, _re.S)
    _sec[name] = set(_re.findall(r'"([a-z_0-9]+)"\s*:\s*\(', mm.group(1))) if mm else set()
if {"positive", "negative"} & _sec["required"]:
    _fail("positive/negative are REQUIRED on ULSEmptyLatent -- the MiniMax "
          "chain cannot be wired at all, because the conditioning is born "
          "downstream in the reference stage")
elif not {"positive", "negative"} <= _sec["optional"]:
    _fail("positive/negative must still EXIST as optional sockets")
else:
    _ok("positive/negative are optional -- the MiniMax chain is wirable")

# AT THE AST, and REACHABLE: `if False:` around the raise would leave the
# message in the file and satisfy any text search (measured -- it did). PARSE
# ONCE: the first version of this check parsed el_src twice and compared nodes
# from two different trees, so the identity test could never match and it
# always passed. Measured too.
_tree = ast.parse(el_src)
_dead = set()
for _n in ast.walk(_tree):
    if isinstance(_n, ast.If) and isinstance(_n.test, ast.Constant):
        for _sub in ast.walk(_n):
            if isinstance(_sub, ast.Raise):
                _dead.add(id(_sub))
_wan_raise = False
for _n in ast.walk(_tree):
    if not isinstance(_n, ast.Raise):
        continue
    _seg = ast.get_source_segment(el_src, _n) or ""
    if "latent_type=WAN needs positive AND negative" in _seg and id(_n) not in _dead:
        _wan_raise = True
if not _wan_raise:
    _fail("the WAN lane must REFUSE by name when conditioning is missing, and "
          "the raise must be REACHABLE -- it consumes the conditioning (core's "
          "I2V surgery) instead of passing it through")
else:
    _ok("the WAN lane refuses missing conditioning by name, reachably")

_gen = [m for m in ast.walk(ast.parse(el_src))
        if isinstance(m, ast.FunctionDef) and m.name == "generate"]
if _gen:
    _args = [a.arg for a in _gen[0].args.args]
    _defaults = _gen[0].args.defaults
    _named = dict(zip(_args[len(_args) - len(_defaults):], _defaults))
    _bad = [k for k in ("positive", "negative")
            if not (isinstance(_named.get(k), ast.Constant)
                    and _named[k].value is None)]
    if _bad:
        _fail("generate() must default {} to None specifically -- any other "
              "default would make an unwired socket look wired".format(_bad))
    elif _args[1:3] != ["positive", "negative"]:
        _fail("positive/negative must KEEP their positions in generate() -- "
              "callers in this tree pass them positionally (test_v679)")
    else:
        _ok("generate() defaults positive/negative to None, positions kept")

# =====================================================================  v875
# The reference-weight report. 'match' scales DOWN only, so two sources of very
# different size get very different token counts -- and reference tokens ride
# through every sampling step. Until v875 that spread showed up only in the
# picture, where it reads as a model problem instead of a budget problem.
_ns3, _mr_src3, _got3 = _lift("nodes/ph_minimax_ref.py", ("balance_note",),
                              ("BALANCE_WARN",))
if "balance_note" not in _got3:
    _fail("balance_note is missing -- an unbalanced reference budget would "
          "again be invisible in the log")
else:
    _bn = _ns3["balance_note"]
    if _bn([]) is not None or _bn([(1, 768)]) is not None:
        _fail("balance_note must stay silent below two references -- there is "
              "no ratio to report")
    else:
        _ok("balance_note is silent below two references")

    _eq = _bn([(1, 768), (2, 768)])
    if not _eq or "1 : 1.0" not in _eq or "balanced" not in _eq:
        _fail("equal references must report 1 : 1.0 and read as balanced")
    elif "<Picture 1> 768 cells vs <Picture 1>" in _eq:
        _fail("the equal case must not name the same picture twice")
    else:
        _ok("equal references: one clean 1 : 1.0 line")

    # The field case: 385x500 and 1638x2048 into a 1344x768 generation.
    _field = _bn([(1, 768), (2, 4032)])
    if not _field or "1 : 5.2" not in _field:
        _fail("the measured field case (768 vs 4032 cells) must report 1 : 5.2, "
              "got: {}".format(_field))
    elif "UNBALANCED" not in _field:
        _fail("a 1 : 5.2 spread must be called out, not merely printed")
    elif "never upscaled" not in _field:
        _fail("the line must say WHICH WAY the control works -- megapixels_n "
              "only scales down, and a reader who does not know that will try "
              "to raise the small one")
    else:
        _ok("the field case reports 1 : 5.2 and names it UNBALANCED")

    # ROUNDED ONCE: printing a rounded ratio while judging the raw one puts
    # "1 : 2.0 (balanced)" next to "1 : 2.0 -- UNBALANCED".
    _edge = _bn([(1, 768), (2, 1500)])          # raw 1.953 -> shown 2.0
    if _edge and "1 : 2.0" in _edge and "(balanced)" in _edge:
        _fail("the verdict is taken on the RAW ratio while a rounded one is "
              "printed -- the same printed number would appear with both "
              "verdicts")
    else:
        _ok("verdict and printed ratio come from the same rounded number")

    # AT THE AST OF build(), not by text search. "latent cells" also appears in
    # balance_note's own message, and "balance_note(cells)" is literally the
    # DEFINITION line -- both let a mutation through when this was a grep.
    # Fifth time in this tree; the rule is now: a needle that means "this code
    # runs" is checked where it would run.
    _build = None
    for _n in ast.walk(ast.parse(_mr_src3)):
        if isinstance(_n, ast.ClassDef) and _n.name == "ULSMiniMaxReference":
            for _m in _n.body:
                if isinstance(_m, ast.FunctionDef) and _m.name == "build":
                    _build = _m
    if _build is None:
        _fail("could not lift build() -- nothing below can be checked")
    else:
        # There are SEVERAL lines.append calls carrying "<Picture %d>" (the
        # report line and the renumbering NOTE). Collect them all and look for
        # one that carries the cell count -- taking "the last match" picked the
        # NOTE and turned this guard red on a healthy file.
        _appends = []
        for _n in ast.walk(_build):
            if (isinstance(_n, ast.Call)
                    and getattr(_n.func, "attr", "") == "append"
                    and getattr(_n.func.value, "id", "") == "lines"):
                _appends.append(ast.get_source_segment(_mr_src3, _n) or "")
        _pic_line = next((a for a in _appends if "<Picture %d>" in a
                          and "n_cells" in a), "")
        if not any("<Picture %d>" in a for a in _appends):
            _fail("build() no longer appends a per-picture report line")
        elif not _pic_line or "latent cells" not in _pic_line:
            _fail("the per-picture line must carry its latent cell count -- "
                  "the ratio alone does not say how big either reference "
                  "actually is")
        else:
            _ok("each reference line carries its own latent cell count")

        # ...and the list it reads must actually be FILLED. Dropping the
        # cells.append leaves every needle above satisfied and reports nothing,
        # because balance_note([]) is silent by design.
        _fills = any(isinstance(_n, ast.Call)
                     and getattr(_n.func, "attr", "") == "append"
                     and getattr(_n.func.value, "id", "") == "cells"
                     for _n in ast.walk(_build))
        if not _fills:
            _fail("build() never appends to `cells` -- balance_note would be "
                  "handed an empty list and stay silent, which looks exactly "
                  "like a balanced run")
        else:
            _ok("build() fills the cells list it reports from")

        _calls_bn = any(isinstance(_n, ast.Call)
                        and getattr(_n.func, "id", "") == "balance_note"
                        for _n in ast.walk(_build))
        if not _calls_bn:
            _fail("build() never CALLS balance_note -- the report line would "
                  "silently disappear")
        else:
            _ok("build() calls balance_note")

print("\n{}: {} failure(s)".format(pathlib.Path(__file__).name, len(FAILED)))
sys.exit(1 if FAILED else 0)
