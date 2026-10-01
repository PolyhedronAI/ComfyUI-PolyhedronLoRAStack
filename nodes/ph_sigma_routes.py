"""
Polyhedron Sigma List / Sigma Curve / Dual Sigma Curve -- server-side routes
(self-contained, v384).

WHY ITS OWN MODULE: the same rule as ph_filter_routes.py. `uls_routes.py`,
`ph_media_routes.py`, `ph_sampler_routes.py` and `ph_filter_routes.py` stay
untouched; the sigma nodes' three endpoints live here, registered by one call
in __init__.py. Nothing in this module is imported by any other route module.

WHAT IT SERVES (the node frontends ask, they never calculate -- the drawn curve
comes from the SAME function the run uses, so the picture cannot drift):
  * GET  /pls/sigma_presets       -- the Sigma List's preset table
                                     (web/js/ph_sigma_list.js)
  * POST /pls/sigma_preview       -- a Sigma List's settings -> the grid the run
                                     would use, or the run's refusal text
  * POST /pls/sigma_curve_preview -- Sigma Curve / Dual Sigma Curve -> their
                                     drawn curve (web/js/ph_sigma_curves.js)

NO LAN LOCK NEEDED: none of the three takes a path, reads a file or writes
one. They take numbers and preset names and answer with numbers.
"""

from aiohttp import web
from server import PromptServer


async def handle_sigma_preview_post(request: web.Request) -> web.Response:
    """Resolve a Sigma List's settings into the grid the RUN would use (v976).

    The node draws this curve. It must therefore be computed by the SAME code
    the run uses -- mirroring the shift formula into JavaScript would be a
    second place for one arithmetic, and two places drift. So the frontend asks
    and never calculates.

    Body: {sigmas_text, shift, enforce_terminal_zero, preset}
    Reply: {ok:true, sigmas:[...], steps:N, shift, source, note}
        or {ok:false, error:"..."} -- the refusal text the run would raise,
    which is how a typo becomes visible in the node instead of at run time."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid JSON"}, status=400)
    try:
        from .wan_sigma_schedule import resolve_sigma_request
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)

    try:
        vals, eff_shift, source, note = resolve_sigma_request(
            data.get("sigmas_text") or "",
            data.get("shift") or 1.0,
            data.get("enforce_terminal_zero"),
            data.get("preset") or "custom",
        )
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)})
    return web.json_response({"ok": True, "sigmas": vals, "steps": len(vals) - 1,
                              "shift": eff_shift, "source": source, "note": note})


async def handle_sigma_curve_preview_post(request: web.Request) -> web.Response:
    """The Sigma Curve / Dual Sigma Curve nodes' drawn curve (v984).

    Same rule as the Sigma List's preview: the frontend asks, the backend
    answers from the function the node's compute() runs (curve_preview ->
    universal_curve / split_curves), so the picture cannot drift from the run.
    Body: {node: "curve"|"dual", <widget name>: value, ...}"""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid JSON"}, status=400)
    try:
        from .wan_sigma_schedule import curve_preview
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)
    return web.json_response(curve_preview(data))


async def handle_sigma_presets_get(request: web.Request) -> web.Response:
    """The sigma preset table, served from its ONE definition (v974).

    The Sigma List's frontend needs the values to show them in the widgets. A
    copy in the JS would be a second place for one truth, and two places that
    hold the same numbers drift -- a lesson this tree paid for more than once.
    So the table is served, never mirrored."""
    try:
        from .wan_sigma_schedule import SIGMA_PRESETS
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)
    out = {}
    for name, entry in SIGMA_PRESETS.items():
        grid, shift, zero, note = entry
        out[name] = {
            "grid": list(grid) if grid is not None else None,
            "shift": shift,
            "enforce_terminal_zero": zero,
            "note": note,
        }
    return web.json_response({"ok": True, "presets": out})


_SIGMA_ROUTES = (
    ("GET",  "/pls/sigma_presets",       handle_sigma_presets_get),
    ("POST", "/pls/sigma_preview",       handle_sigma_preview_post),
    ("POST", "/pls/sigma_curve_preview", handle_sigma_curve_preview_post),
)


def register_sigma_routes() -> int:
    """Register the three routes under the bare path AND the /api alias.
    Duplicate registrations are swallowed per route (a reload must never abort
    startup). Returns the number of paths added."""
    added = 0
    routes = PromptServer.instance.routes
    for method, path, handler in _SIGMA_ROUTES:
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
