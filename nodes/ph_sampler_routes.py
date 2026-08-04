"""
Polyhedron Sampler -- server-side routes (self-contained).

WHY ITS OWN MODULE (maintenance rule, please keep it that way):
`nodes/uls_routes.py` carries the LoRA-Stack routes and is upstream's file;
`nodes/ph_media_routes.py` carries the Media Loader's. Everything the Sampler
needs lives HERE instead, so both of those stay untouched and the next node
can be added the same way -- one new module, one registration call in
__init__.py -- without ever re-opening a shared file. Nothing in this module
is imported by the Stack or by the Media Loader.

WHAT IT SERVES
The node UI (web/js/uls_sampler.js) drives these endpoints:
  * preview_mode -- switch the live preview decoder WHILE a render runs; the
    running sampling callback picks it up on the next step. Preview only; it
    never affects the output latent.
  * tae_status   -- is the TAE preview decoder for a mode on disk? Read-only.
  * tae_install  -- fetch the pinned TAE through ph_weights.ensure_weights
    (the one download door: .part file, sha256 verify, atomic rename).

Every handler imports uls_sampler LAZILY, so this module stays importable even
when the heavy node is unavailable, and a duplicate path can never abort
plugin startup.

Routes (3), each registered under the bare path AND the /api alias:
    POST /pls/sampler/preview_mode
    POST /pls/sampler/tae_status
    POST /pls/sampler/tae_install
"""

from aiohttp import web
from server import PromptServer


async def handle_sampler_preview_mode(request: web.Request) -> web.Response:
    """v415: live (mid-render) preview-mode switch for the Polyhedron Sampler. The
    node POSTs {node_id, mode} when preview_mode is changed WHILE a render runs; the
    running sampling callback picks it up on the next step. Preview only -- it never
    affects the output latent. Unknown modes are rejected; the validation + storage
    live in nodes/uls_sampler.set_live_preview_mode."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "bad json"}, status=400)
    node_id = data.get("node_id")
    mode = data.get("mode")
    if node_id is None or not mode:
        return web.json_response({"ok": False, "error": "node_id and mode required"}, status=400)
    try:
        from .uls_sampler import set_live_preview_mode
        ok = set_live_preview_mode(node_id, mode)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)[:200]}, status=500)
    if not ok:
        return web.json_response({"ok": False, "error": f"unknown mode: {mode}"}, status=400)
    return web.json_response({"ok": True, "node_id": str(node_id), "mode": mode})


async def handle_sampler_tae_status(request: web.Request) -> web.Response:
    """v830: is the TAE preview decoder for a mode actually on disk? The
    frontend asks this when a TAE preview mode is picked (and once on load)
    and raises the amber bubble on a miss. Read-only, never touches disk."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "bad json"}, status=400)
    mode = data.get("mode") or data.get("name")
    if not mode:
        return web.json_response({"ok": False, "error": "mode required"},
                                 status=400)
    try:
        from .uls_sampler import tae_status
        st = tae_status(mode)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)[:200]},
                                 status=500)
    st["ok"] = True
    return web.json_response(st)


async def handle_sampler_tae_install(request: web.Request) -> web.Response:
    """v830: fetch the pinned TAE through ph_weights.ensure_weights (the one
    download door: .part file, sha256 verify, primary vae_approx folder).
    Blocks until done -- 22 MB, seconds; the console prints the MB marks.
    The unpinned decoder (lighttaew2_1) returns its honest error text."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "bad json"}, status=400)
    name = data.get("name") or data.get("mode")
    if not name:
        return web.json_response({"ok": False, "error": "name required"},
                                 status=400)
    try:
        import asyncio
        from .uls_sampler import tae_install
        loop = asyncio.get_running_loop()   # v367: house rule (get_event_loop deprecated 3.12+)
        path = await loop.run_in_executor(None, tae_install, name)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)[:300]},
                                 status=500)
    return web.json_response({"ok": True, "path": str(path)})


def register_sampler_routes():
    """Register the Polyhedron Sampler endpoints on the ComfyUI PromptServer.

    Called from __init__.py after the Stack's register_routes() and the Media
    Loader's register_media_routes(). Each route is added under BOTH the bare
    path and the "/api"-prefixed path: ComfyUI's frontend helpers prefix every
    call with "/api", and routes added directly through app.router.add_route do
    not get that alias automatically. Every add is guarded, so a duplicate path
    can never abort plugin startup.
    """
    try:
        app = PromptServer.instance.app
    except Exception as e:
        print(f"[PLS] \u26a0 Sampler routes: PromptServer not available: {e}")
        return

    routes = [
        ("POST", "/pls/sampler/preview_mode", handle_sampler_preview_mode),
        ("POST", "/pls/sampler/tae_status",   handle_sampler_tae_status),
        ("POST", "/pls/sampler/tae_install",  handle_sampler_tae_install),
    ]
    registered = 0
    for method, path, handler in routes:
        for p in (path, "/api" + path):
            try:
                app.router.add_route(method, p, handler)
                registered += 1
            except Exception as e:
                print(f"[PLS] \u26a0 could not add sampler route {method} {p}: {e}")
    print(f"[PLS] \u2713 Polyhedron Sampler routes registered "
          f"(root + /api alias, {registered} paths)")
