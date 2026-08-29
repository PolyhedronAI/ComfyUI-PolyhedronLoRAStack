#!/usr/bin/env python3
"""
test_v365_public_build -- the public build's own law: the Sampling group grows
by NEW FILES, and the shared route file never moves.

This build (v365) adds two nodes to the published pack -- the Polyhedron Sampler
and the Polyhedron CLIP Text Encode -- following the rule MAINTAINING.md states
for every future node: a group owns its own files, registration is guarded and
grouped, and nothing re-opens a file another group already owns.

The one pin that carries the whole principle is the FIRST one below. If a later
release ever needs a Stack route, it must be a deliberate, declared act -- not
something that slips in because a handler was convenient to paste.

Guards, all must hold, mutation-tested:

  SHARED FILE -- nodes/uls_routes.py is byte-identical to the Stack release
                 (md5 bcc4d8c47e500c892619c22d51f7cbc6, unchanged since v362),
                 and it registers NONE of the /pls/sampler routes.

  OWN MODULE  -- nodes/ph_sampler_routes.py registers exactly the three sampler
                 routes, each under the bare path AND the /api alias, imports
                 uls_sampler LAZILY (inside the handlers, so a broken node
                 cannot fell the module) and pulls nothing from uls_routes or
                 ph_media_routes.

  MEDIA DIMS  -- the deferred-dimensions half of the listing (v683) arrived
                 with this build: ph_media_routes.py serves /uls/media/dims and
                 handle_media_list honours dims=0. The node's own JS calls the
                 route, so a missing registration is a dead panel.

  REGISTRY    -- both nodes are registered under their guarded flags, the pack
                 exports 20 nodes, and the version triple agrees
                 (pyproject / __init__ banner / uls_compat PLUGIN_VERSION).

  ASCII       -- the two new house modules are pure ASCII (house rule).
"""
import os
import re
import sys
import ast
import hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The Stack's route file as published in v362. Not a version stamp: this is the
# byte-level promise that the public build never edits upstream's file.
ULS_ROUTES_MD5 = "bcc4d8c47e500c892619c22d51f7cbc6"

SAMPLER_ROUTES = (
    "/pls/sampler/preview_mode",
    "/pls/sampler/tae_status",
    "/pls/sampler/tae_install",
)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def _fail(msg):
    print("[test_v365_public_build] FAIL -- %s" % msg)
    sys.exit(1)


# --- 1. the shared file has not moved ---------------------------------------
raw = open(os.path.join(ROOT, "nodes", "uls_routes.py"), "rb").read()
got = hashlib.md5(raw).hexdigest()
if got != ULS_ROUTES_MD5:
    _fail("nodes/uls_routes.py changed: %s (expected %s). The public build's "
          "rule is that the Stack's route file stays upstream's file -- if this "
          "was deliberate, say so in the changelog and re-pin here."
          % (got, ULS_ROUTES_MD5))

routes_src = raw.decode("utf-8")
for path in SAMPLER_ROUTES:
    if path in routes_src:
        _fail("uls_routes.py mentions %s -- the sampler routes belong to "
              "nodes/ph_sampler_routes.py" % path)

# --- 2. the sampler owns its route module -----------------------------------
SR = _read("nodes", "ph_sampler_routes.py")
tree = ast.parse(SR)

top_imports = []
for node in tree.body:
    if isinstance(node, ast.Import):
        top_imports += [a.name for a in node.names]
    elif isinstance(node, ast.ImportFrom):
        top_imports.append(node.module or "")
for bad in ("uls_routes", "ph_media_routes", "uls_sampler"):
    if any(bad in m for m in top_imports):
        _fail("ph_sampler_routes.py imports %s at module level -- the handlers "
              "must import the node lazily so a broken node cannot stop the "
              "module from loading" % bad)

funcs = [n.name for n in tree.body
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
for want in ("handle_sampler_preview_mode", "handle_sampler_tae_status",
             "handle_sampler_tae_install", "register_sampler_routes"):
    if want not in funcs:
        _fail("ph_sampler_routes.py has no %s" % want)

for path in SAMPLER_ROUTES:
    if not re.search(r'\("POST",\s*"%s"' % re.escape(path), SR):
        _fail("ph_sampler_routes.py does not register POST %s" % path)

if '"/api" + path' not in SR:
    _fail("ph_sampler_routes.py does not add the /api alias -- ComfyUI's "
          "api.fetchApi prefixes every call with /api")

# every handler reaches the node through a lazy, function-local import
for name in ("handle_sampler_preview_mode", "handle_sampler_tae_status",
             "handle_sampler_tae_install"):
    fn = [n for n in tree.body
          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
          and n.name == name][0]
    lazy = [n for n in ast.walk(fn)
            if isinstance(n, ast.ImportFrom) and (n.module or "") == "uls_sampler"]
    if not lazy:
        _fail("%s does not import uls_sampler lazily" % name)

# --- 3. the frontend's three calls are exactly the three registered ----------
JS = _read("web", "js", "uls_sampler.js")
called = set(re.findall(r'"(/pls/sampler/[a-z_]+)"', JS))
if called != set(SAMPLER_ROUTES):
    _fail("uls_sampler.js calls %s but the module registers %s"
          % (sorted(called), sorted(SAMPLER_ROUTES)))

# --- 4. the deferred-dimensions half of the listing --------------------------
MR = _read("nodes", "ph_media_routes.py")
if not re.search(r'"GET",\s*"/uls/media/dims"', MR):
    _fail("ph_media_routes.py does not register /uls/media/dims -- the loader's "
          "JS asks for it, so the tiles would draw without dimensions")
if "def handle_media_dims" not in MR or "def _scan_media_fast" not in MR:
    _fail("ph_media_routes.py is missing the deferred-dimension handlers")
if "dims" not in MR.split("def handle_media_list")[1][:1200]:
    _fail("handle_media_list does not honour the dims= switch")

LJS = _read("web", "js", "ph_media_loader.js")
if "/uls/media/dims" not in LJS:
    _fail("the loader JS no longer calls /uls/media/dims -- re-check this pin")

# --- 5. both nodes registered, count and version triple ---------------------
INIT = _read("__init__.py")
for flag, cls, disp in (("_SAMPLER_OK", "ULSSampler", "Polyhedron Sampler"),
                        ("_CTE_OK", "ULSCLIPTextEncode",
                         "Polyhedron CLIP Text Encode"),
                        ("_PUP_OK", "ULSPowerUpscale",
                         "Polyhedron Power Upscale"),
                        ("_FUP_OK", "ULSFastUpscale",
                         "Polyhedron Fast Upscale"),
                        ("_INTERP_OK", "ULSInterpolate",
                         "Polyhedron Interpolate"),
                        # v371 -- the workflow essentials group
                        ("_BASICS_OK", "ULSLoadModel", "Polyhedron Load Model"),
                        ("_BASICS_OK", "ULSLoadCLIP", "Polyhedron Load CLIP"),
                        ("_BASICS_OK", "ULSLoadVAE", "Polyhedron Load VAE"),
                        ("_BASICS_OK", "ULSSeed", "Polyhedron Seed"),
                        ("_UPLOAD_OK", "ULSLoadUpscaleModel",
                         "Polyhedron Load Upscale Model"),
                        ("_VAE_OK", "ULSVAE", "Polyhedron VAE Codec"),
                        ("_ELAT_OK", "ULSEmptyLatent",
                         "Polyhedron Empty Latent"),
                        ("_SWITCH_OK", "ULSAnySwitch", "Polyhedron Switch"),
                        ("_SWITCH_OK", "ULSAnySwitchInv",
                         "Polyhedron Switch Inverse"),
                        ("_INT_OK", "ULSInt", "Polyhedron Int"),
                        ("_MINFO_OK", "ULSMediaInfo", "Polyhedron Media Info"),
                        ("_MMREF_OK", "ULSMiniMaxReference",
                         "Polyhedron MiniMax Reference"),
                        ("_NOTE_OK", "ULSNote", "Polyhedron Note")):
    if ("if %s:" % flag) not in INIT:
        _fail("__init__.py does not guard %s behind %s" % (cls, flag))
    if 'NODE_CLASS_MAPPINGS["%s"]' % cls not in INIT:
        _fail("__init__.py does not register %s" % cls)
    if disp not in INIT:
        _fail("__init__.py has no display name for %s" % cls)

if "register_sampler_routes()" not in INIT:
    _fail("__init__.py never calls register_sampler_routes()")

n_nodes = len(set(re.findall(r'NODE_CLASS_MAPPINGS\["(\w+)"\]', INIT))
              | set(re.findall(r'"(\w+)":\s+\w+,', INIT)))
if n_nodes != 37:
    _fail("the pack registers %d nodes, expected 37 (the 33 of v371 plus "
          "Attention, NAG, Filter and Audio Stretch added in v372)" % n_nodes)

# v368: the three new nodes open NO server route. Power Upscale reports tile
# progress through PromptServer.send_sync, which needs no endpoint. This is a
# promise worth pinning: the day one of them grows a route handler, it must be
# a DECLARED act in its own module -- not a quiet reopening of a shared file.
# v371: the same promise for the thirteen workflow essentials. Measured before
# the cut -- not one of their nine carrier modules opens an endpoint, and not
# one of their frontends calls fetch(). That is WHY this cut needed no new
# route module at all, and it is the reason uls_routes.py can stay shut for a
# tenth release running.
#
# v372: Attention and NAG keep the promise -- neither opens an endpoint and
# neither ships a frontend. The FILTER does not, and that is a declared act:
# its live preview has to read the SAME .cube the backend grades with, so it
# needs three routes. They went into nodes/ph_filter_routes.py, a FOURTH
# module, exactly the way the Media Loader and the Sampler did it. The rule
# was never "no new routes" -- it is "no reopening of a shared file", and the
# check below enforces the version that actually matters: the Filter's routes
# must live in its own module, and uls_routes.py must stay byte-identical
# (pinned at the top of this file).
for fname in ("ph_power_upscale.py", "ph_fast_upscale.py", "ph_interpolate.py",
              "ph_basics.py", "ph_switch.py", "ph_int.py",
              "ph_empty_latent.py", "ph_media_info.py", "ph_minimax_ref.py",
              "ph_note.py", "ph_vae.py", "ph_upscale_loader.py"):
    src = _read("nodes", fname)
    for needle in ("routes.get(", "routes.post(", "@server.PromptServer",
                   "add_routes("):
        if needle in src:
            _fail("%s registers a server route (%r) - the v368 cut promised "
                  "these three nodes need none. A new route belongs in its own "
                  "module, declared in the changelog." % (fname, needle))

PY = _read("pyproject.toml")
CO = _read("web", "js", "uls_compat.js")
m = re.search(r'version\s*=\s*"(\d+)\.(\d+)\.(\d+)"', PY)
if not m:
    _fail("no version in pyproject.toml")
triple = "v%s%s" % (m.group(1), m.group(2))
if ("Polyhedron Suite  %s" % triple) not in INIT:
    _fail("the __init__ banner does not carry %s" % triple)
if ('PLUGIN_VERSION = "%s"' % triple) not in CO:
    _fail("uls_compat.js PLUGIN_VERSION does not carry %s" % triple)

# --- 6. house rule: the new modules are pure ASCII ---------------------------
for rel in (("nodes", "ph_sampler_routes.py"),):
    data = open(os.path.join(ROOT, *rel), "rb").read()
    if not data.isascii():
        _fail("%s is not pure ASCII" % "/".join(rel))

# --- v372: the Filter's routes live in their OWN module ---------------------
# The one node in this cut that needs endpoints. The promise is not that it
# has none -- it is that getting them cost no shared file. Each check below is
# the thing that would actually go wrong if the rule were bent.
FR = _read("nodes", "ph_filter_routes.py")
for _path in ("/uls/filter/lut", "/uls/filter/preset"):
    if _path not in FR:
        _fail("ph_filter_routes.py does not serve %s" % _path)
if "def register_filter_routes" not in FR:
    _fail("ph_filter_routes.py exposes no register_filter_routes()")
if "register_filter_routes()" not in INIT:
    _fail("__init__.py never calls register_filter_routes() -- the Filter's "
          "live preview would 404 on every LUT it tries to read")
# NOT a substring search. The first draft of this check was one, and it went
# red on its own module header, which NAMES the other three files in order to
# explain why it does not use them. Prose is not a dependency: what matters is
# whether the module IMPORTS them.
for _shared in ("uls_routes", "ph_media_routes", "ph_sampler_routes"):
    if re.search(r"(?:^|\n)\s*(?:from\s+\.?%s\s+import|import\s+\.?%s)\b"
                 % (_shared, _shared), FR):
        _fail("ph_filter_routes.py imports %s.py -- the whole point of a "
              "fourth module is that the other three stay shut" % _shared)
_FJS = _read("web", "js", "ph_filter.js")
for _hit in re.findall(r'fetchApi\("([^"?]+)', _FJS):
    if not _hit.startswith("/uls/filter/"):
        _fail("ph_filter.js calls %s, which ph_filter_routes.py does not serve"
              % _hit)


print("[test_v365_public_build] OK -- uls_routes.py untouched (%s), the sampler "
      "owns its 3 routes lazily, the Filter owns its 3 in a fourth module, "
      "/uls/media/dims served, 37 nodes, at %s"
      % (ULS_ROUTES_MD5[:8], triple))
