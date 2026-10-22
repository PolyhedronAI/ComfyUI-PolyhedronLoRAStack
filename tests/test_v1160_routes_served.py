# -*- coding: ascii -*-
"""Guard v1160 -- every endpoint a frontend calls is registered by a module of the pack.

The public build answered 404 for two routes its frontends called since v371 (the Seed's noise preview, the Empty
Latent's presets): their handlers lived only in a route file that build did not carry, and no check looked at the
frontends' calls. This one does, on every tree: each /uls/... or /pls/... path a web/js file names must appear as a
route path in some Python module (path level; methods are the modules' own guards' job).
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


py = "".join(_read("nodes", f) for f in sorted(os.listdir(os.path.join(ROOT, "nodes"))) if f.endswith(".py")) + _read("__init__.py")
served = {x.rstrip("/") for x in re.findall(r'["\'](/(?:uls|pls)/[a-z0-9_/\-]+)["\']', py)}
missing = []
for js in sorted(os.listdir(os.path.join(ROOT, "web", "js"))):
    if js.endswith(".js"):
        for hit in sorted(set(re.findall(r'["\'`](/(?:uls|pls)/[a-z0-9_/\-]*)', _read("web", "js", js)))):
            if hit.rstrip("/") not in served:
                missing.append("%s -> %s" % (js, hit))
if missing or len(served) < 10:
    print("[test_v1160_routes_served] FAIL -- frontend calls without a registered route: %r" % missing[:8])
    sys.exit(1)
print("[test_v1160_routes_served] PASS -- every frontend call is served (%d route paths)" % len(served))
