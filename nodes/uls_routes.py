"""
ULS API Routes
=============
Endpoints (each registered under both the bare path and the /api alias):
  GET  /uls/preview/image?lora=<name>  -> image bytes
  GET  /uls/preview/video?lora=<name>  -> video (with Range-request support)
  GET  /uls/metadata?lora=<name>       -> JSON metadata
  GET  /uls/list                       -> all LoRAs with preview flags
  GET  /uls/groups                     -> all group assignments
  POST /uls/groups                     -> set a group assignment
  POST /uls/triggers                   -> save trigger words (.uls-meta.json)
  GET  /uls/group_modes                -> all per-group merge modes
  POST /uls/group_modes                -> set the merge mode for a group
  POST /uls/civitai_fetch              -> hash-based Civitai preview + triggers
"""

import os
import asyncio
import json
import re
import mimetypes
import folder_paths

import aiohttp
from aiohttp import web
from server import PromptServer

from .uls_stack_node import (
    _extract_lora_info, _find_preview, _read_txt_trigger,
    _uls_meta_read, _uls_meta_write, _path_within_loras,
)


# Max bytes accepted when downloading a Civitai preview image (v251 hardening).
_MAX_PREVIEW_BYTES = 32 * 1024 * 1024  # 32 MB


def _pick_preview_url(images, max_nsfw_level: int = 2):
    """First image-type entry at or below the NSFW threshold, else None.
    v267 (audit A-2): NO fallback to images[0] any more -- that fallback
    returned exactly an image that had FAILED the filter, contradicting the
    documented "with NSFW filter". No SFW preview found -> no preview saved.
    Threshold unified with the CLI (N-1): Civitai nsfwLevel 1=None / 2=Soft
    / 4=Mature / 8+=X -- <=2 keeps previews SFW. Want Mature previews back?
    That is this one number.\n    Pure function -- covered by test_v267."""
    for img in images or []:
        if img.get("type", "image") == "image" and img.get("nsfwLevel", 0) <= max_nsfw_level:
            url = img.get("url")
            if url:
                return url
    return None


async def _download_capped_image(resp, max_bytes: int):
    """Read an HTTP response body as an image with two guards (v251):
      * Content-Type must be image/* (no writing arbitrary bodies as .jpg)
      * a hard byte cap, checked against Content-Length up front AND enforced
        mid-stream so a chunked/lying response can't exhaust RAM.
    Returns the bytes on success, or None (reason logged) on any rejection.
    Does not raise for the type/size checks themselves."""
    if resp.status != 200:
        return None
    ctype = (resp.headers.get("Content-Type", "") or "").split(";")[0].strip().lower()
    if not ctype.startswith("image/"):
        print(f"[ULS] \u26a0 Preview skipped: not an image (Content-Type: {ctype or 'unknown'})")
        return None
    clen = resp.headers.get("Content-Length")
    if clen and str(clen).isdigit() and int(clen) > max_bytes:
        print(f"[ULS] \u26a0 Preview skipped: declared {clen} bytes > {max_bytes} cap")
        return None
    buf = bytearray()
    async for chunk in resp.content.iter_chunked(64 * 1024):
        buf += chunk
        if len(buf) > max_bytes:
            print(f"[ULS] \u26a0 Preview skipped: exceeded {max_bytes} byte cap mid-stream")
            return None
    return bytes(buf)


def _get_lora_full_path(lora_name: str):
    p = folder_paths.get_full_path("loras", lora_name)
    # Defense-in-depth (v251): never hand back a path outside the loras dir(s).
    if p and not _path_within_loras(p):
        return None
    return p


def _load_uls_meta(full_path: str) -> dict:
    """Load companion metadata for a LoRA, tolerating both the canonical and
    legacy .uls-meta.json locations. Thin wrapper over the shared reader so
    every route resolves metadata identically to the Stack backend.
    Accepts the FULL .safetensors path (not the splitext base)."""
    return _uls_meta_read(full_path)


# --- Route Handler ---------------------------------------------------------

async def handle_preview_image(request: web.Request) -> web.Response:
    lora_name = request.rel_url.query.get("lora", "").strip()
    if not lora_name:
        return web.Response(status=400, text="Missing 'lora' parameter")

    previews = _find_preview(lora_name)
    img_path = previews.get("image")
    if not img_path or not os.path.isfile(img_path):
        return web.Response(status=404, text="No preview image found")

    mime = mimetypes.guess_type(img_path)[0] or "image/png"
    try:
        with open(img_path, "rb") as f:
            data = f.read()
        return web.Response(
            body=data,
            content_type=mime,
            headers={"Cache-Control": "public, max-age=3600"},
        )
    except OSError as e:
        return web.Response(status=500, text=f"Read error: {e}")


async def handle_preview_video(request: web.Request) -> web.Response:
    """
    Serve video with Range-Request support.
    Browsers need this for <video> seek/preload.
    """
    lora_name = request.rel_url.query.get("lora", "").strip()
    if not lora_name:
        return web.Response(status=400, text="Missing 'lora' parameter")

    previews = _find_preview(lora_name)
    vid_path = previews.get("video")
    if not vid_path or not os.path.isfile(vid_path):
        return web.Response(status=404, text="No preview video found")

    mime = mimetypes.guess_type(vid_path)[0] or "video/mp4"
    # v1154 (audit, v1155): aiohttp's FileResponse answers Range itself (suffix ranges, 416, sendfile in chunks) --
    # the hand-made reader took a browser's first "bytes=0-" as the WHOLE file into RAM, on the event loop
    return web.FileResponse(vid_path, headers={"Content-Type": mime, "Cache-Control": "public, max-age=3600"})


async def handle_metadata(request: web.Request) -> web.Response:
    lora_name = request.rel_url.query.get("lora", "").strip()
    if not lora_name:
        return web.Response(status=400, text="Missing 'lora' parameter")

    full_path = _get_lora_full_path(lora_name)
    if not full_path:
        return web.json_response({"error": f"LoRA not found: {lora_name}"}, status=404)

    info      = _extract_lora_info(full_path)
    previews  = _find_preview(lora_name)
    uls_meta  = _load_uls_meta(full_path)

    # Trigger words priority: uls-meta.json > .txt file > safetensors header
    txt_trigger = _read_txt_trigger(lora_name)
    trigger_words = (
        uls_meta.get("trigger_words")
        or txt_trigger
        or info.get("trigger_words", "")
    )
    # ss_tag_frequency als Dict sauber formatieren
    if isinstance(trigger_words, dict):
        trigger_words = ", ".join(list(trigger_words.keys())[:20])

    response = {
        "name":              lora_name,
        "has_preview_image": "image"  in previews,
        "has_preview_video": "video"  in previews,
        "trigger_words":     trigger_words,
        "base_model":        uls_meta.get("base_model")    or info.get("base_model", "?"),
        "rank":              info.get("rank", "?"),
        "algo":              info.get("algo", "lora"),
        "description":       uls_meta.get("description")   or info.get("description", ""),
        "civitai_name":      uls_meta.get("civitai_name",  ""),
        "civitai_id":        uls_meta.get("civitai_id",    ""),
        "has_uls_meta":      bool(uls_meta),
    }
    return web.json_response(response)


async def handle_list(request: web.Request) -> web.Response:
    """All LoRAs with preview flags."""
    try:
        loras = folder_paths.get_filename_list("loras")
    except Exception:
        loras = []

    def _scan():
        result = []
        for name in loras:
            try:
                previews = _find_preview(name)
                result.append({
                    "name":      name,
                    "has_image": "image" in previews,
                    "has_video": "video" in previews,
                })
            except Exception:
                result.append({"name": name, "has_image": False, "has_video": False})
        return result

    # v1155 (audit): up to 12 stats per LoRA -- 2000 LoRAs on a network drive held the event loop for seconds
    return web.json_response(await asyncio.to_thread(_scan))


# --- Route Registration ----------------------------------------------------



_GROUPS_FILE = None

def _get_groups_file():
    global _GROUPS_FILE
    if _GROUPS_FILE is None:
        import folder_paths as fp
        import os
        lora_dirs = fp.get_folder_paths("loras")
        base = lora_dirs[0] if lora_dirs else fp.base_path
        _GROUPS_FILE = os.path.join(base, "uls_groups.json")
    return _GROUPS_FILE

# v267 (audit A-6): valid group names for POST /uls/groups. The UI only ever
# posts these; the backend maps unknown strings to "custom" anyway -- this is
# purely defensive input validation so arbitrary strings never reach the JSON.
_VALID_GROUPS = {"acc", "style", "scene", "motion", "subject", "detail", "custom"}

_GROUP_MIGRATIONS = {
    # Full old names
    "character": "subject",
    "lighting":  "scene",
    "artist":    "style",
    # Short uppercase variants from older builds
    "CHAR":      "subject",
    "LIGH":      "scene",
    "SUBJ":      "subject",
    "SCEN":      "scene",
    "DETA":      "detail",
    "STYL":      "style",
    "MOTI":      "motion",
    "CUST":      "custom",
    "ACC":        "acc",
}

def _load_groups() -> dict:
    try:
        f = _get_groups_file()
        import os
        if os.path.isfile(f):
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            migrated = False
            for lora, grp in list(data.items()):
                if grp in _GROUP_MIGRATIONS:
                    data[lora] = _GROUP_MIGRATIONS[grp]
                    migrated = True
            if migrated:
                _save_groups(data)
                print("[ULS] \u2139 Migrated group names in uls_groups.json")
            return data
    except Exception:
        pass
    return {}

def _save_groups(data: dict):
    try:
        f = _get_groups_file()
        with open(f, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ULS] \u26a0 Failed to save groups: {e}")

async def handle_groups_get(request: web.Request) -> web.Response:
    """Alle gespeicherten Gruppen-Zuordnungen."""
    return web.json_response(_load_groups())

async def handle_groups_set(request: web.Request) -> web.Response:
    """Gruppen-Zuordnung setzen: POST {lora_name, group}"""
    try:
        body = await request.json()
        lora_name = body.get("lora_name", "").strip()
        group     = body.get("group", "\u2014").strip()
        if not lora_name:
            return web.Response(status=400, text="Missing lora_name")
        if group != "\u2014" and group not in _VALID_GROUPS:
            return web.Response(status=400, text=f"Unknown group: {group}")
        groups = _load_groups()
        if group == "\u2014":
            groups.pop(lora_name, None)
        else:
            groups[lora_name] = group
        _save_groups(groups)
        return web.json_response({"ok": True, "lora_name": lora_name, "group": group})
    except Exception as e:
        return web.Response(status=500, text=str(e))

async def handle_triggers_set(request: web.Request) -> web.Response:
    """Set trigger words for a LoRA (writes .uls-meta.json)."""
    try:
        body = await request.json()
        lora_name     = body.get("lora_name", "").strip()
        trigger_words = body.get("trigger_words", "").strip()
        if not lora_name:
            return web.Response(status=400, text="Missing lora_name")
        full_path = _get_lora_full_path(lora_name)
        if not full_path:
            return web.Response(status=404, text=f"LoRA not found: {lora_name}")
        # Write through the shared canonical writer so the file lands where the
        # Stack backend and /uls/metadata both read it (and any legacy file is
        # migrated + removed).
        written = _uls_meta_write(full_path, {"trigger_words": trigger_words})
        if not written:
            return web.Response(status=500, text="Failed to write .uls-meta.json")
        return web.json_response({"ok": True, "lora_name": lora_name, "trigger_words": trigger_words})
    except Exception as e:
        return web.Response(status=500, text=str(e))


# --- Group Apply Modes -----------------------------------------------------
# Persisted per-group (not per-LoRA): "character" -> "DARE", "detail" -> "CONCAT", etc.
# Stored in uls_group_modes.json next to uls_groups.json. Valid modes: SEQ/CONCAT/DARE.

_GROUP_MODES_FILE = None
_VALID_MODES = {"SEQ", "CONCAT", "DARE"}

def _get_group_modes_file():
    global _GROUP_MODES_FILE
    if _GROUP_MODES_FILE is None:
        import folder_paths as fp
        lora_dirs = fp.get_folder_paths("loras")
        base = lora_dirs[0] if lora_dirs else fp.base_path
        _GROUP_MODES_FILE = os.path.join(base, "uls_group_modes.json")
    return _GROUP_MODES_FILE

def _load_group_modes() -> dict:
    try:
        f = _get_group_modes_file()
        if os.path.isfile(f):
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    # filter invalid entries
                    return {k: v for k, v in data.items()
                            if isinstance(k, str) and isinstance(v, str) and v.upper() in _VALID_MODES}
    except Exception:
        pass
    return {}

def _save_group_modes(data: dict):
    try:
        f = _get_group_modes_file()
        with open(f, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ULS] \u26a0 Failed to save group modes: {e}")

async def handle_group_modes_get(request: web.Request) -> web.Response:
    """Return all group->mode mappings."""
    return web.json_response(_load_group_modes())

async def handle_group_modes_set(request: web.Request) -> web.Response:
    """Set the apply mode for a group: POST {group, mode}.
    Pass mode='SEQ' (or empty) to remove the entry (SEQ is the default)."""
    try:
        body = await request.json()
        group = (body.get("group") or "").strip()
        mode  = (body.get("mode")  or "").strip().upper()
        if not group or group == "\u2014":
            return web.Response(status=400, text="Group missing or '\u2014' (no mode applies)")
        if mode and mode not in _VALID_MODES:
            return web.Response(status=400, text=f"Invalid mode: {mode}")
        modes = _load_group_modes()
        if not mode or mode == "SEQ":
            modes.pop(group, None)
        else:
            modes[group] = mode
        _save_group_modes(modes)
        return web.json_response({"ok": True, "group": group, "mode": mode or "SEQ"})
    except Exception as e:
        return web.Response(status=500, text=str(e))


async def handle_civitai_fetch(request: web.Request) -> web.Response:
    """Fetch preview image + trigger words from Civitai using sshs_model_hash.
    POST { lora_name }
    Returns { ok, source, trigger_words, preview_saved }

    Fully async: the Civitai API call and preview download run through aiohttp
    so they never block ComfyUI's event loop (the old urllib.urlopen path froze
    the whole server -- websocket, queue, every client -- for up to ~25 s).
    The small local file ops (safetensors header read, image/JSON write) stay
    synchronous; they are negligible compared to the network round-trips."""
    import struct

    try:
        body      = await request.json()
        lora_name = body.get("lora_name", "").strip()
        if not lora_name:
            return web.Response(status=400, text="Missing lora_name")

        path = _get_lora_full_path(lora_name)
        if not path:
            return web.json_response({"ok": False, "error": "LoRA file not found"})

        # 1. Read sshs_model_hash from safetensors header (local, fast)
        model_hash = None
        try:
            with open(path, "rb") as f:
                header8 = f.read(8)
                if len(header8) < 8:
                    return web.json_response({"ok": False, "error": "Not a valid safetensors file (truncated header)"})
                length = struct.unpack("<Q", header8)[0]
                # v267 (audit A-1): same 50 MB header cap as _read_meta -- a corrupt
                # or malicious local file must not trigger a giant allocation.
                if length <= 0 or length > 50 * 1024 * 1024:
                    return web.json_response({"ok": False, "error": "Safetensors header invalid or oversized"})
                header = json.loads(f.read(length).decode("utf-8", errors="replace"))
            meta = header.get("__metadata__", {})
            model_hash = meta.get("sshs_model_hash") or meta.get("sshs_legacy_hash")
        except Exception as e:
            return web.json_response({"ok": False, "error": f"Could not read header: {e}"})

        if not model_hash:
            return web.json_response({"ok": False, "error": "No sshs_model_hash in safetensors header"})

        # v347: the hash is interpolated into a URL path below -- accept only hex
        # (SHA256 or the shorter AutoV2 hash) so a crafted header value cannot
        # manipulate the outgoing request URL.
        if not re.fullmatch(r"[0-9a-fA-F]{8,64}", model_hash):
            return web.json_response({"ok": False, "error": "Malformed model hash in safetensors header"})

        ua = {"User-Agent": "ComfyUI-PolyhedronLoRAStack/1.0"}
        api_url = f"https://civitai.com/api/v1/model-versions/by-hash/{model_hash}"

        async with aiohttp.ClientSession(headers=ua) as session:
            # 2. Query Civitai API (async -- no event-loop blocking)
            try:
                async with session.get(
                        api_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 404:
                        return web.json_response({"ok": False, "error": "Not found on Civitai (private or custom LoRA)"})
                    if resp.status != 200:
                        return web.json_response({"ok": False, "error": f"Civitai API error: {resp.status}"})
                    civ_data = await resp.json()
            except Exception as e:
                return web.json_response({"ok": False, "error": f"Network error: {e}"})

            # 3. Extract trigger words
            trigger_words = ""
            trained_words = civ_data.get("trainedWords", [])
            if trained_words:
                trigger_words = ", ".join(trained_words)

            # 4. Extract civitai_id and model_id
            civitai_version_id = civ_data.get("id")
            civitai_model_id   = civ_data.get("modelId")

            # 5. Download first preview image (async)
            preview_saved = False
            images = civ_data.get("images", [])
            # v267 (audit A-2): filter-strict -- no fallback past the NSFW filter.
            preview_url = _pick_preview_url(images)

            if preview_url:
                jpg_path = os.path.splitext(path)[0] + ".jpg"
                try:
                    async with session.get(
                            preview_url, timeout=aiohttp.ClientTimeout(total=15)) as resp2:
                        img_bytes = await _download_capped_image(resp2, _MAX_PREVIEW_BYTES)
                    if img_bytes is not None:
                        with open(jpg_path, "wb") as f:
                            f.write(img_bytes)
                        preview_saved = True
                except Exception as e:
                    print(f"[ULS] \u26a0 Preview download failed: {e}")

        # 6. Persist via the shared canonical writer (same file the Stack
        #    backend and /uls/metadata read; migrates any legacy file).
        updates = {}
        if trigger_words:
            updates["trigger_words"] = trigger_words
        if civitai_model_id:
            updates["civitai_id"] = civitai_model_id
        if civitai_version_id:
            updates["civitai_version_id"] = civitai_version_id
        if updates:
            _uls_meta_write(path, updates)

        model_name = civ_data.get("model", {}).get("name", "")
        return web.json_response({
            "ok":            True,
            "model_name":    model_name,
            "trigger_words": trigger_words,
            "preview_saved": preview_saved,
            "civitai_id":    civitai_model_id,
        })

    except Exception as e:
        return web.Response(status=500, text=str(e))


def register_routes():
    """Register all ULS routes with the ComfyUI PromptServer."""
    try:
        app = PromptServer.instance.app
    except Exception as e:
        print(f"[ULS] \u26a0 PromptServer not available: {e}")
        return

    routes = [
        ("GET",  "/uls/preview/image", handle_preview_image),
        ("GET",  "/uls/preview/video", handle_preview_video),
        ("GET",  "/uls/metadata",      handle_metadata),
        ("GET",  "/uls/list",          handle_list),
        ("GET",  "/uls/groups",        handle_groups_get),
        ("POST", "/uls/groups",        handle_groups_set),
        ("POST", "/uls/triggers",      handle_triggers_set),
        ("GET",  "/uls/group_modes",   handle_group_modes_get),
        ("POST", "/uls/group_modes",   handle_group_modes_set),
        ("POST", "/uls/civitai_fetch", handle_civitai_fetch),
    ]
    # -- Register each route under BOTH the bare path and the /api-prefixed path
    #
    # Why: ComfyUI's frontend helpers (api.fetchApi / api.apiURL) now prefix
    # every call with "/api" (frontend >= ~1.4x). Routes added directly via
    # app.router.add_route -- i.e. NOT through the PromptServer RouteTableDef --
    # do not receive that "/api" alias automatically, so a frontend asking for
    # "/api/uls/list" hits nothing and gets a 404 with an empty body, which the
    # UI reports as "Could not load LoRA list: unexpected end of data".
    #
    # Registering both paths fixes current frontends (via the /api alias) while
    # keeping older ones working (via the bare path). Each add is guarded so a
    # duplicate/already-registered path can never abort plugin startup.
    registered = 0
    for method, path, handler in routes:
        for p in (path, "/api" + path):
            try:
                app.router.add_route(method, p, handler)
                registered += 1
            except Exception as e:
                print(f"[ULS] \u26a0 could not add route {method} {p}: {e}")
    print(f"[ULS] \u2713 API routes registered (root + /api alias, {registered} paths)")
