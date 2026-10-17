# -*- coding: ascii -*-
"""Polyhedron basics routes (public build v388) -- the Seed's noise preview and the
Empty Latent's presets.

Both frontends called these routes since v687 / the Empty Latent rebuild, but the
handlers lived only in the internal route file, so the public release answered 404:
the Seed drew no real noise field and the Empty Latent's preset list stayed empty.
They get their own module and their own registration call here, so the Stack's route
file (nodes/uls_routes.py) stays untouched. The handlers are the internal ones (two
docstring characters made ASCII).

  GET  /uls/noise/preview   the REAL noise field (uls_noise.make_noise) as a PNG
  GET  /pls/latent_presets  built-in + user presets (user dir, polyhedron_latent_presets.json)
  POST /pls/latent_presets  save / delete a user preset (whitelisted keys only)
"""
import json
import os

from aiohttp import web


_LATENT_PRESETS_FILE = None

# Frank's dauerbrenner are built in (1280x1024 / 1024x1280), plus common WAN/Flux
# sizes. Built-ins are read-only defaults; the UI adds/removes USER presets only.
_LATENT_BUILTIN_PRESETS = [
    {"name": "Square 1024",        "width": 1024, "height": 1024, "aspect": "1:1"},
    {"name": "Portrait 1024x1280", "width": 1024, "height": 1280, "aspect": "4:5"},
    {"name": "Landscape 1280x1024","width": 1280, "height": 1024, "aspect": "5:4"},
    {"name": "Portrait 832x1216",  "width": 832,  "height": 1216, "aspect": "2:3"},
    {"name": "Landscape 1216x832", "width": 1216, "height": 832,  "aspect": "3:2"},
    {"name": "WAN 16:9 832x480",   "width": 832,  "height": 480,  "length": 81, "family": "wan", "aspect": "16:9"},
    {"name": "WAN 9:16 480x832",   "width": 480,  "height": 832,  "length": 81, "family": "wan", "aspect": "9:16"},
    {"name": "WAN 720p 1280x720",  "width": 1280, "height": 720,  "length": 81, "family": "wan", "aspect": "16:9"},
]


def _get_latent_presets_file():
    """Path to the user presets JSON, preferring the ComfyUI user directory so it
    survives plugin updates. Falls back to base_path if the user dir is unknown."""
    global _LATENT_PRESETS_FILE
    if _LATENT_PRESETS_FILE is None:
        import folder_paths as fp
        base = None
        for getter in ("get_user_directory",):
            fn = getattr(fp, getter, None)
            if callable(fn):
                try:
                    base = fn()
                except Exception:
                    base = None
            if base:
                break
        if not base:
            base = os.path.join(getattr(fp, "base_path", "."), "user")
        try:
            os.makedirs(base, exist_ok=True)
        except Exception:
            base = getattr(fp, "base_path", ".")
        _LATENT_PRESETS_FILE = os.path.join(base, "polyhedron_latent_presets.json")
    return _LATENT_PRESETS_FILE


def _aspect_label(w, h):
    """Reduced W:H label (e.g. 1280x1024 -> 5:4). Defensive on zero/invalid."""
    try:
        import math
        w, h = int(w), int(h)
        if w <= 0 or h <= 0:
            return ""
        g = math.gcd(w, h)
        return f"{w // g}:{h // g}"
    except Exception:
        return ""


def _sanitize_preset(p):
    """Coerce one incoming preset dict to the stored schema, or None if invalid.
    Only whitelisted keys survive -- arbitrary strings never reach the JSON."""
    if not isinstance(p, dict):
        return None
    name = str(p.get("name", "")).strip()[:80]
    if not name:
        return None
    try:
        w = int(p.get("width"))
        h = int(p.get("height"))
    except (TypeError, ValueError):
        return None
    if not (16 <= w <= 16384 and 16 <= h <= 16384):
        return None
    out = {"name": name, "width": w, "height": h,
           "aspect": _aspect_label(w, h)}
    if p.get("length") is not None:
        try:
            L = int(p.get("length"))
            if 1 <= L <= 16384:
                out["length"] = L
        except (TypeError, ValueError):
            pass
    fam = p.get("family")
    if isinstance(fam, str) and fam[:16]:
        out["family"] = fam[:16]
    return out


def _load_user_latent_presets():
    try:
        f = _get_latent_presets_file()
        if os.path.isfile(f):
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return [q for q in (_sanitize_preset(p) for p in data) if q]
    except Exception:
        pass
    return []


def _save_user_latent_presets(presets):
    try:
        f = _get_latent_presets_file()
        with open(f, "w", encoding="utf-8") as fh:
            json.dump(presets, fh, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[PLS] \u26a0 Failed to save latent presets: {e}")
        return False


def _merged_latent_presets():
    """Built-ins first, then user presets; a user preset with the same name (case-
    insensitive) overrides the built-in. Each carries builtin:true/false for the UI."""
    out = []
    seen = {}
    for p in _LATENT_BUILTIN_PRESETS:
        q = dict(p)
        q.setdefault("aspect", _aspect_label(q["width"], q["height"]))
        q["builtin"] = True
        seen[q["name"].lower()] = len(out)
        out.append(q)
    for p in _load_user_latent_presets():
        q = dict(p)
        q["builtin"] = False
        key = q["name"].lower()
        if key in seen:
            out[seen[key]] = q          # user overrides built-in of same name
        else:
            seen[key] = len(out)
            out.append(q)
    return out


async def handle_noise_preview(request: web.Request) -> web.Response:
    """GET ?type=&seed=&strength=&w=&h= -> a greyscale PNG of the REAL noise
    field (v687).

    Why this exists: the in-node preview used to be a javascript IMITATION of
    each noise character -- value noise in the browser, while the run builds
    torch.randn plus a frequency filter in python. Same family, different
    bytes, and no amount of tuning would have made them agree. This route runs
    the ACTUAL generator (uls_noise.make_noise), so what the node draws is what
    the sampler will denoise.

    The dimensions are LATENT dimensions, not pixels -- that is the grid the
    noise really lives on (1440px at /16 is 90x90). One channel is rendered:
    every channel is drawn from the same distribution, so one is representative
    and 128 of them would be unreadable.

    CONTRAST: the field is normalised min..max for display, exactly as the old
    javascript preview did. That makes STRUCTURE visible at any strength and
    deliberately makes strength itself invisible here -- it is a scalar, it
    belongs in the readout text, not in a picture that would just get greyer."""
    q = request.query
    ntype = (q.get("type", "gaussian") or "gaussian").strip()
    try:
        seed = int(q.get("seed", "0") or 0)
    except ValueError:
        seed = 0
    try:
        strength = float(q.get("strength", "1.0") or 1.0)
    except ValueError:
        strength = 1.0
    # v833: the character dial reaches the REAL preview too -- the point of
    # this route is that the node draws what the sampler will denoise.
    try:
        character = float(q.get("character", "1.0") or 1.0)
    except ValueError:
        character = 1.0
    def _dim(name, default):
        try:
            v = int(q.get(name, default) or default)
        except ValueError:
            v = default
        return max(8, min(512, v))
    w, h = _dim("w", 64), _dim("h", 64)
    try:
        from . import uls_noise
        import numpy as _np
        from PIL import Image as _Image
        import io as _io
        field = uls_noise.make_noise(ntype, (1, 1, h, w), seed, strength,
                                     character)
        a = field.detach().cpu().numpy().reshape(h, w).astype("float32")
        mn = float(a.min())
        mx = float(a.max())
        rng = (mx - mn) or 1.0
        a = (a - mn) / rng
        img = _Image.fromarray((_np.clip(a, 0.0, 1.0) * 255.0).astype("uint8"), mode="L")
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        return web.Response(body=buf.getvalue(), content_type="image/png",
                            headers={"Cache-Control": "no-store"})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)


async def handle_latent_presets_get(request: web.Request) -> web.Response:
    """All presets (built-in merged with user)."""
    return web.json_response({"presets": _merged_latent_presets()})


async def handle_latent_presets_set(request: web.Request) -> web.Response:
    """Add/update or delete a USER preset. Built-ins cannot be deleted (a delete
    of a built-in name just removes any user override of it).
    Body: {"action":"save","preset":{...}} | {"action":"delete","name":"..."}"""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid JSON"}, status=400)
    action = str(data.get("action", "save"))
    users = _load_user_latent_presets()
    if action == "delete":
        name = str(data.get("name", "")).strip().lower()
        if not name:
            return web.json_response({"ok": False, "error": "no name"}, status=400)
        users = [p for p in users if p["name"].lower() != name]
        _save_user_latent_presets(users)
        return web.json_response({"ok": True, "presets": _merged_latent_presets()})
    if action == "save":
        preset = _sanitize_preset(data.get("preset"))
        if preset is None:
            return web.json_response({"ok": False, "error": "invalid preset"}, status=400)
        users = [p for p in users if p["name"].lower() != preset["name"].lower()]
        users.append(preset)
        _save_user_latent_presets(users)
        return web.json_response({"ok": True, "presets": _merged_latent_presets()})
    return web.json_response({"ok": False, "error": f"unknown action: {action}"}, status=400)


def register_basics_routes():
    """Bare + /api paths, as the other route modules do."""
    try:
        from server import PromptServer
        app = PromptServer.instance.app
    except Exception as e:
        print("[PLS] basics routes not registered: %s" % e)
        return 0
    n = 0
    for method, path, fn in (("GET", "/uls/noise/preview", handle_noise_preview),
                             ("GET", "/pls/latent_presets", handle_latent_presets_get),
                             ("POST", "/pls/latent_presets", handle_latent_presets_set)):
        for p in (path, "/api" + path):
            try:
                app.router.add_route(method, p, fn)
                n += 1
            except Exception as e:
                print("[PLS] could not add route %s %s: %s" % (method, p, e))
    return n
