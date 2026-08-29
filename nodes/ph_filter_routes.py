"""
Polyhedron Filter -- server-side routes (self-contained).

WHY ITS OWN MODULE (maintenance rule, please keep it that way):
`nodes/uls_routes.py` carries the LoRA-Stack routes and is upstream's file;
`nodes/ph_media_routes.py` carries the Media Loader's, `ph_sampler_routes.py`
the Sampler's. Everything the Filter needs lives HERE instead, so all three of
those stay untouched and the next node can be added the same way -- one new
module, one registration call in __init__.py -- without ever re-opening a
shared file. Nothing in this module is imported by the Stack, the Media Loader
or the Sampler.

WHAT IT SERVES
The node UI (web/js/ph_filter.js) drives these endpoints:
  * filter/lut        -- hand one .cube LUT to the live preview as plain text,
    so the browser grades with the SAME file the backend does. Without it the
    preview would have to guess at a table it cannot read.
  * filter/preset GET -- load one saved look, sanitized.
  * filter/preset POST-- save the current look.

WHY THESE THREE ARE NOT BEHIND THE LAN LOCK
The lock added in v367 guards the routes that take a path FROM THE CLIENT and
resolve it on the host -- those turn a --listen instance into a file browser.
These three take no such path. Both folders are FIXED, computed from this
file's own location (`<pack>/luts`, `<pack>/presets`), and the only thing the
caller supplies is a bare name that is immediately reduced with
`os.path.basename` and forced to the one permitted extension. Nothing outside
the pack is reachable even in principle, which is the same reasoning that left
seq_list / seq_delete / proc_count open in the Media Loader module: a managed
project directory is not a file browser.

Both handlers that accept parameters run them through
`ph_filter._sanitize_preset` -- the SAME whitelist the node itself uses, not a
second copy of it. A hand-edited or downloaded preset can therefore never push
unknown keys into the frontend, and the load and save paths cannot drift apart.
The import is LAZY, so this module stays importable even when the heavy node is
unavailable and a duplicate path can never abort plugin startup.

Routes (3), each registered under the bare path AND the /api alias:
    GET  /uls/filter/lut
    GET  /uls/filter/preset
    POST /uls/filter/preset
"""

import json
import os
import re

from aiohttp import web
from server import PromptServer


def _pack_dir(name: str) -> str:
    """<pack>/<name> -- derived from this file, never from the caller."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), name)


async def handle_filter_lut(request: web.Request) -> web.Response:
    """Serve one .cube LUT from the pack's luts/ folder as plain text, so the
    Polyhedron Filter's live preview can parse and apply the SAME file the
    backend grades with. Name is basename-sanitized and must end in .cube --
    nothing outside luts/ is reachable."""
    name = request.query.get("name", "")
    if not name:
        return web.Response(status=400, text="Missing 'name' parameter")
    name = os.path.basename(name)
    if not name.lower().endswith(".cube"):
        return web.Response(status=400, text="Only .cube files are served")
    path = os.path.join(_pack_dir("luts"), name)
    if not os.path.isfile(path):
        return web.Response(status=404, text=f"LUT not found: {name}")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return web.Response(text=f.read(), content_type="text/plain")
    except OSError as e:
        return web.Response(status=500, text=f"Read error: {e}")


async def handle_filter_preset_get(request: web.Request) -> web.Response:
    """Serve one preset from the pack's presets/ folder, SANITIZED through
    ph_filter._sanitize_preset -- a hand-edited or downloaded file can never
    push unknown keys or paths into the frontend."""
    name = request.query.get("name", "")
    if not name:
        return web.Response(status=400, text="Missing 'name' parameter")
    name = os.path.basename(name)
    if not name.lower().endswith(".json"):
        return web.Response(status=400, text="Only .json presets are served")
    path = os.path.join(_pack_dir("presets"), name)
    if not os.path.isfile(path):
        return web.Response(status=404, text=f"Preset not found: {name}")
    try:
        from .ph_filter import _sanitize_preset
        raw = json.loads(open(path, encoding="utf-8", errors="replace").read())
        params = _sanitize_preset(raw.get("params", raw))
        return web.json_response({"file": name, "params": params})
    except Exception as e:
        return web.Response(status=500, text=f"Preset unreadable: {e}")


async def handle_filter_preset_post(request: web.Request) -> web.Response:
    """Save the current look as a preset JSON into the pack's presets/
    folder. The name is reduced to a safe stem ([A-Za-z0-9_-], spaces to
    underscores); the params run through the same _sanitize_preset whitelist
    as the load path."""
    try:
        body = await request.json()
    except Exception:
        return web.Response(status=400, text="Body must be JSON")
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(body.get("name", "")).strip()).strip("_")
    if not stem:
        return web.Response(status=400, text="Missing or empty preset name")
    from .ph_filter import _sanitize_preset
    params = _sanitize_preset(body.get("params", {}))
    if not params:
        return web.Response(status=400, text="No storable parameters in body")
    preset_dir = _pack_dir("presets")
    fname = stem + ".json"
    try:
        os.makedirs(preset_dir, exist_ok=True)
        with open(os.path.join(preset_dir, fname), "w", encoding="utf-8") as f:
            f.write(json.dumps({"name": stem, "params": params}, indent=2) + "\n")
        return web.json_response({"ok": True, "file": fname})
    except OSError as e:
        return web.Response(status=500, text=f"Write error: {e}")


_FILTER_ROUTES = (
    ("GET",  "/uls/filter/lut",     handle_filter_lut),
    ("GET",  "/uls/filter/preset",  handle_filter_preset_get),
    ("POST", "/uls/filter/preset",  handle_filter_preset_post),
)


def register_filter_routes() -> int:
    """Register the three routes under the bare path AND the /api alias.

    Duplicate registrations are swallowed per route: a reload must never abort
    plugin startup over a path that is already there. Returns the number of
    paths actually added, which __init__.py prints alongside the other route
    counts so a missing module is visible in the console rather than only in a
    node that quietly stops working.
    """
    added = 0
    routes = PromptServer.instance.routes
    for method, path, handler in _FILTER_ROUTES:
        for full in (path, "/api" + path):
            try:
                if method == "GET":
                    routes.get(full)(handler)
                else:
                    routes.post(full)(handler)
                added += 1
            except Exception:
                pass
    return added
