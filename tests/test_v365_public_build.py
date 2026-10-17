# -*- coding: ascii -*-
"""Guard: the public build is whole -- every registered node, every route a frontend calls, one version.

Rewritten in v388 (the cut without legacy rules). What it holds:

  N  the node set: 41 node ids, each registered behind its own import flag, each with a display name;
     the V3 registry imports only modules this build ships and maps only registered nodes
  R  the routes: every /uls/... or /pls/... path any web/js file calls is registered by some module of
     this build (the check that would have caught the two 404s of v371-v385: the Seed's noise preview,
     the Empty Latent's presets); each route module is registered from __init__.py, under the bare
     path and the /api alias
  W  the model door serves only the sets of this build's nodes (rife)
  V  the version triple agrees: pyproject x.NNN.0, the banner "Polyhedron Suite  vNNN", PLUGIN_VERSION vNNN
  F  nothing of the removed legacy comes back: no install.py, no runtime pip, no standalone CLI
  S  the route modules are pure ASCII
"""
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = "test_v365_public_build"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


INIT = _read("__init__.py")

# --- N: the node set ------------------------------------------------------------------------------
ids = sorted(set(re.findall(r'NODE_CLASS_MAPPINGS\["(\w+)"\]', INIT)) | set(re.findall(r'^\s+"(\w+)":\s+\w+,', INIT, re.M)))
_need(len(ids) == 41, "N  41 node ids registered (%d)" % len(ids))
base = [l.strip() for l in open(os.path.join(ROOT, [f for f in os.listdir(ROOT) if f.startswith("NODE_IDS_baseline_v")][0]),
                                encoding="utf-8") if l.strip() and not l.startswith("#")]
_need(sorted(base) == ids, "N  the registered ids are exactly the NODE_IDS baseline")
names = set(re.findall(r'NODE_DISPLAY_NAME_MAPPINGS\["(\w+)"\]', INIT)) | set(re.findall(r'^\s+"(\w+)":\s+"', INIT, re.M))
_need(not [i for i in ids if i not in names], "N  every node has a display name")
V3X = _read("nodes", "uls_v3_extension.py")
_need(all(os.path.isfile(os.path.join(ROOT, "nodes", m + ".py")) for m in re.findall(r"^from \.(\w+) import", V3X, re.M)),
      "N  the V3 registry imports only modules this build ships")
_need(all(('NODE_CLASS_MAPPINGS["%s"]' % k) in INIT for k in re.findall(r'^\s+"(\w+)":\s+\w+V3,', V3X, re.M)),
      "N  the V3 registry maps only registered nodes")

# --- R: every call a frontend makes is served -------------------------------------------------------
py_all = "".join(_read("nodes", f) for f in sorted(os.listdir(os.path.join(ROOT, "nodes"))) if f.endswith(".py")) + INIT
served = {x.rstrip("/") for x in re.findall(r'["\'](/(?:uls|pls)/[a-z0-9_/\-]+)["\']', py_all)}
missing = []
for js in sorted(os.listdir(os.path.join(ROOT, "web", "js"))):
    if js.endswith(".js"):
        for hit in set(re.findall(r'["\'`](/(?:uls|pls)/[a-z0-9_/\-]*)', _read("web", "js", js))):
            if hit.rstrip("/") not in served:
                missing.append("%s -> %s" % (js, hit))
_need(not missing, "R  every /uls|/pls path a frontend calls is registered %r" % missing[:5])
ROUTE_MODULES = {"uls_routes": "register_routes", "ph_media_routes": "register_media_routes",
                 "ph_sampler_routes": "register_sampler_routes", "ph_filter_routes": "register_filter_routes",
                 "ph_sigma_routes": "register_sigma_routes", "ph_basics_routes": "register_basics_routes",
                 "ph_world_weights": "register_weights_routes", "ph_cine_timeline": "register_timeline_routes"}
for mod, fn in ROUTE_MODULES.items():
    src = _read("nodes", mod + ".py")
    _need(("def %s(" % fn) in src and ("%s()" % fn) in INIT, "R  %s.%s() exists and __init__.py calls it" % (mod, fn))
    _need('"/api" + p' in src or '"/api" + path' in src or "'/api' +" in src or '"/api"' in src,
          "R  %s registers the /api alias too" % mod)

# --- W: the model door ------------------------------------------------------------------------------
WW = _read("nodes", "ph_world_weights.py")
_need('SETS = ("rife",)' in WW and "REGISTRY = {}" in WW, "W  the model door serves the rife set only")

# --- V: the version triple --------------------------------------------------------------------------
m = re.search(r'^version\s*=\s*"(\d+)\.(\d+)\.(\d+)"', _read("pyproject.toml"), re.M)
tag = "v%s%s" % (m.group(1), m.group(2)) if m else "?"
_need(bool(m) and ("Polyhedron Suite  %s" % tag) in INIT, "V  the banner carries %s" % tag)
_need(('PLUGIN_VERSION = "%s"' % tag) in _read("web", "js", "uls_compat.js"), "V  uls_compat.js PLUGIN_VERSION carries %s" % tag)
_need(os.path.isfile(os.path.join(ROOT, "CHANGELOG_%s.md" % tag)), "V  CHANGELOG_%s.md at the root" % tag)

# --- F: no legacy back ------------------------------------------------------------------------------
for f in ("install.py", "uls_preview_gen.py", os.path.join("nodes", "ph_pip_provision.py")):
    _need(not os.path.exists(os.path.join(ROOT, f)), "F  %s is not part of the build" % f)
pip_hits = []
for d, _, fs in os.walk(os.path.join(ROOT, "nodes")):
    for f in fs:
        if f.endswith(".py"):
            t = ast.parse(open(os.path.join(d, f), encoding="utf-8").read())
            for n in ast.walk(t):
                if isinstance(n, ast.Call) and any(isinstance(a, ast.Constant) and a.value == "pip" for a in ast.walk(n)) \
                        and "subprocess" in ast.dump(n.func) + "".join(ast.dump(a) for a in n.args):
                    pip_hits.append(f)
_need(not pip_hits, "F  no module runs pip %r" % pip_hits)

# --- S ----------------------------------------------------------------------------------------------
for mod in ROUTE_MODULES:
    _need(open(os.path.join(ROOT, "nodes", mod + ".py"), "rb").read().isascii(), "S  nodes/%s.py is ASCII" % mod)

print("%s: %d check(s) failed" % (NAME, len(FAILS)) if FAILS else "%s: all checks passed" % NAME)
sys.exit(1 if FAILS else 0)
