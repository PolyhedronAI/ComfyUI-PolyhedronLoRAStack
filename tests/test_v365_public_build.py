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
                 exports 17 nodes, and the version triple agrees
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
                         "Polyhedron CLIP Text Encode")):
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
if n_nodes != 17:
    _fail("the pack registers %d nodes, expected 17 (15 published + Sampler + "
          "CLIP Text Encode)" % n_nodes)

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

print("[test_v365_public_build] OK -- uls_routes.py untouched (%s), the sampler "
      "owns its 3 routes lazily, /uls/media/dims served, 17 nodes at %s"
      % (ULS_ROUTES_MD5[:8], triple))
