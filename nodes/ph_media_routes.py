"""
Polyhedron Media Loader -- server-side routes (self-contained).

WHY ITS OWN MODULE (maintenance rule, please keep it that way):
`nodes/uls_routes.py` carries the LoRA-Stack routes and is upstream's file.
Everything the Media Loader needs lives HERE instead, so uls_routes.py stays
byte-identical to the Stack release and a future node can be added the same
way -- one new module, one registration call in __init__.py -- without ever
re-opening a shared file. Nothing in this module is imported by the Stack.

WHAT IT SERVES
The node UI (web/js/ph_media_loader.js) drives these endpoints so the user can
pin ANY local folder, see a thumbnail grid, hover-preview, drop files in and
drive the batch panel. The pinned folder is an absolute path the user navigated
to; this is a LOCAL tool, so browsing the machine's filesystem is intentional.
Every thumb/upload target is realpath-checked to resolve inside its stated
folder (`_within`), and uploads keep the streaming byte cap.

Routes (14), each registered under the bare path AND the /api alias:
    GET  /uls/media/folders      GET  /uls/media/list
    GET  /uls/media/dims
    GET  /uls/media/thumb        GET  /uls/media/file
    GET  /uls/media/native_pick  POST /uls/media/upload
    POST /uls/media/locate       GET  /uls/media/resolve
    GET  /uls/media/seq/list     POST /uls/media/seq/build
    POST /uls/media/seq/delete   POST /uls/media/open_folder
    GET  /uls/media/proc_count
"""

import os
import json
import time
import asyncio
import shutil
import subprocess
import mimetypes
import re

import folder_paths
from aiohttp import web
from server import PromptServer


# ---------------------------------------------------------------------------
# Shared helpers. Copied in deliberately: this module must not depend on
# uls_routes.py, so the Stack file can be replaced wholesale on the next
# upstream release without touching the Media Loader.
# ---------------------------------------------------------------------------
_MAX_UPLOAD_BYTES = 256 * 1024 * 1024   # generous cap; guards a runaway upload

_CV2_HINTED = [False]


def _hint_cv2_once(what):
    """v577: cv2 (opencv-python) is OPTIONAL and ComfyUI core does NOT ship it,
    so on a clean install video thumbnails and the dimension probe simply do
    not appear - and nobody is told why. Say it ONCE per process, then be
    quiet. The feature still degrades gracefully; it just stops being a
    mystery."""
    if _CV2_HINTED[0]:
        return
    _CV2_HINTED[0] = True
    print(f"[PLS] Media: {what} needs opencv-python, which ComfyUI does not "
          f"bundle. Install it (pip install opencv-python) to enable video "
          f"thumbnails and the dimension probe. Everything else works without it.")

def _within(parent: str, child: str) -> bool:
    """True iff `child` resolves inside `parent` (defense-in-depth). A realpath +
    separator-boundary check: Windows-safe (os.path.commonpath raises on paths
    that span two drives), and mirrors ph_viewport3d._resolve_mesh_out's guard so
    both 3D code paths contain the same way."""
    try:
        rp = os.path.realpath(parent)
        rc = os.path.realpath(child)
        return rc == rp or rc.startswith(rp + os.sep)
    except Exception:
        return False

async def _stream_part_to_disk(part, out_path: str) -> int:
    """Stream one multipart part to out_path with the byte cap. Returns bytes
    written; raises ValueError('cap') when the cap is exceeded (caller cleans up)."""
    size = 0
    with open(out_path, "wb") as f:
        while True:
            chunk = await part.read_chunk(256 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > _MAX_UPLOAD_BYTES:
                raise ValueError("cap")
            f.write(chunk)
    return size

def _cleanup_names(out_dir: str, names) -> None:
    """Best-effort remove a list of files from out_dir (failure-path cleanup)."""
    for nm in names:
        try:
            os.remove(os.path.join(out_dir, nm))
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────────────
# Media Loader (ULSMediaLoader) — server-side folder browser + thumbnail/list +
# upload routes. The node UI (web/js/ph_media_loader.js) drives these so the user
# can pin ANY local folder, see a thumbnail grid, and upload files. The pinned
# folder is an absolute path the user navigated to; this is a LOCAL tool, so
# browsing the machine's filesystem is intentional. A thumb/upload target is
# realpath-checked to resolve inside its stated folder (_within), and uploads
# keep the streaming byte cap.
# ──────────────────────────────────────────────────────────────────────────
_MEDIA_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")
_MEDIA_VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v")
# v457 (Stufe A): audio is a third listed/served media kind. Kept in lock-step with
# ph_media_loader._AUDIO_EXTS. Listing tags these kind="audio" (w/h=None); the file
# route serves them for browser <audio> playback (hover-preview + Selection).
_MEDIA_AUDIO_EXTS = (".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".opus",
                     ".aiff", ".aif", ".wma")
# mimetypes.guess_type misses .flac/.m4a/.opus on some Windows installs; this
# fallback keeps the Content-Type honest so the browser picks the right decoder.
_AUDIO_MIME = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
               ".flac": "audio/flac", ".m4a": "audio/mp4", ".aac": "audio/aac",
               ".opus": "audio/ogg", ".aiff": "audio/aiff", ".aif": "audio/aiff",
               ".wma": "audio/x-ms-wma"}
_MEDIA_THUMB_PX = 256


def _media_kind(name: str) -> str:
    ext = os.path.splitext(name)[1].lower()
    if ext in _MEDIA_VIDEO_EXTS:
        return "video"
    if ext in _MEDIA_IMAGE_EXTS:
        return "image"
    if ext in _MEDIA_AUDIO_EXTS:
        return "audio"
    return ""


def _drive_roots():
    """Top-level roots for the folder picker: the ComfyUI input dir as a handy
    start, then drive letters on Windows / '/' on POSIX."""
    roots = []
    try:
        inp = os.path.realpath(folder_paths.get_input_directory())
        roots.append({"name": "input/", "path": inp})
    except Exception:
        pass
    if os.name == "nt":
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            d = c + ":\\"
            if os.path.exists(d):
                roots.append({"name": d, "path": d})
    else:
        roots.append({"name": "/", "path": "/"})
    return roots


def _safe_media_name(filename: str) -> str:
    """Path-traversal-safe, separator-free name; keep an image/video/AUDIO
    extension, else fall back to .png. Mirrors _safe_companion_name's defense.

    v661 (B-03): audio was missing from the whitelist, so an uploaded/dropped
    .m4a was renamed to .png — the extension-driven kind detection then routed
    an AAC file into the image path (black preview, no audio pairing possible).
    Audio has been a first-class browsable kind since v457, so its extensions
    belong here too."""
    base = os.path.basename(str(filename or "")).replace("\\", "_").replace("/", "_")
    base = base.lstrip(".") or "upload"
    stem, ext = os.path.splitext(base)
    if ext.lower() not in (_MEDIA_IMAGE_EXTS + _MEDIA_VIDEO_EXTS + _MEDIA_AUDIO_EXTS):
        ext = ".png"
    return (stem or "upload") + ext


def _media_names_in(rp, cap=60):
    """Top-level media files of a folder for the picker's peek list: sorted
    (name, kind) tuples capped at `cap`, plus the TOTAL count. Pure listing --
    dotfiles and subfolders excluded; kind by extension (image/video/audio)."""
    names = []
    total = 0
    try:
        with os.scandir(rp) as it:
            for e in it:
                try:
                    if not e.is_file() or e.name.startswith("."):
                        continue
                except OSError:
                    continue
                ext = os.path.splitext(e.name)[1].lower()
                if ext in _MEDIA_IMAGE_EXTS:
                    kind = "image"
                elif ext in _MEDIA_VIDEO_EXTS:
                    kind = "video"
                elif ext in _MEDIA_AUDIO_EXTS:
                    kind = "audio"
                else:
                    continue
                total += 1
                names.append((e.name, kind))
    except OSError:
        return [], 0
    names.sort(key=lambda t: t[0].lower())
    return names[:cap], total


# ── v367: LAN browse lock (internal v840; audit B1 -- opt-in flag) ───────────
# These routes anchor on a CLIENT-supplied folder/path. On the default
# localhost bind that is the feature (browse anywhere); bound to the LAN
# (--listen) it would let every client on the network read arbitrary
# directories, upload into them, and pop native dialogs on this machine.
# Local behaviour stays untouched; under a non-local bind the path-anchored
# routes refuse with 403 unless ULS_ALLOW_LAN_BROWSE=1 opens them
# deliberately. Own copy on purpose -- this module imports nothing from the
# stack side (MAINTAINING.md rule 1).
_LAN_BROWSE_ENV = "ULS_ALLOW_LAN_BROWSE"


def _lan_browse_locked():
    """True when the path-anchored media routes must refuse: the server is
    bound beyond localhost AND the env override is not set. An UNKNOWN bind
    (comfy args unavailable) reads as LOCAL -- fail-open by design: this
    lock is an opt-in hardening for the --listen case and must never be able
    to break a localhost setup (stability first)."""
    if str(os.environ.get(_LAN_BROWSE_ENV, "")).strip().lower() in ("1", "true", "yes", "on"):
        return False
    try:
        from comfy.cli_args import args as _cli_args
        bind = str(getattr(_cli_args, "listen", "") or "")
    except Exception:
        return False
    _local = ("", "127.0.0.1", "localhost", "::1")
    return any(h.strip() not in _local for h in bind.split(",") if h.strip())


def _lan_deny(request):
    """The ONE refusal door: a 403 naming the override, or None when open."""
    if not _lan_browse_locked():
        return None
    return web.json_response(
        {"error": "media browsing is locked while ComfyUI is bound beyond "
                  "localhost (--listen). Set %s=1 to allow it on your "
                  "network." % _LAN_BROWSE_ENV},
        status=403)


async def handle_media_folders(request: web.Request) -> web.Response:
    """GET ?path=<abs> -> immediate SUBFOLDERS of path (for the picker), plus a
    PEEK at the folder's own media files (sorted names + total count) so the
    picker can show what a pin would load. Empty path -> top-level roots."""
    _deny = _lan_deny(request)          # v367: LAN browse lock (audit B1)
    if _deny is not None:
        return _deny
    path = (request.query.get("path", "") or "").strip()
    try:
        if not path:
            return web.json_response({"ok": True, "path": "", "parent": "",
                                      "folders": _drive_roots(), "is_root": True})
        rp = os.path.realpath(path)
        if not os.path.isdir(rp):
            return web.json_response({"ok": False, "error": "not a folder", "path": rp}, status=404)
        subs = []
        with os.scandir(rp) as it:
            for e in it:
                try:
                    if e.is_dir() and not e.name.startswith("."):
                        subs.append({"name": e.name, "path": os.path.join(rp, e.name)})
                except OSError:
                    continue
        subs.sort(key=lambda d: d["name"].lower())
        stripped = rp.rstrip(os.sep)
        parent = os.path.dirname(stripped)
        if parent == rp:
            parent = ""
        media, media_total = _media_names_in(rp)
        return web.json_response({"ok": True, "path": rp, "parent": parent,
                                  "folders": subs, "is_root": False,
                                  "media": [{"name": n, "kind": k} for n, k in media],
                                  "media_total": media_total})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e), "folders": []}, status=500)


def _media_img_dims(path):
    """(w, h) for an image via a Pillow *header* read (no full decode), corrected
    for EXIF orientation so it matches the loader's exif_transpose. (None, None)
    on any error / Pillow missing — dimensions are best-effort, never fatal."""
    try:
        from PIL import Image
        from .ph_media_util import oriented_size
        with Image.open(path) as im:
            w, h = im.size
            try:
                o = im.getexif().get(274)   # 274 = EXIF Orientation tag
            except Exception:
                o = None
        return oriented_size(w, h, o)
    except Exception:
        return (None, None)


def _media_vid_dims(path):
    """(w, h, fps) for a video via an OpenCV *header* probe — reads the frame-size
    and fps properties WITHOUT decoding a frame. Uses the FFMPEG backend so the
    reported width matches the loader's own decode (v394: Windows MSMF can pad the
    width, e.g. 720->728). v625: fps rides along for the frontend's fixed-length
    trim (frame counts are native-fps counts — the very fps _slice_frames cuts
    with). (None, None, None) if cv2 is missing or the probe fails."""
    try:
        import cv2  # opencv-python (optional, same dep as the thumb/video decode)
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        try:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        finally:
            cap.release()
        if w > 0 and h > 0:
            return (w, h, fps if fps > 0 else None)
    except Exception:
        pass
    return (None, None, None)


def _scan_media_fast(rp):
    """Blocking folder scan -> entries (newest first) with name/size/mtime/kind
    and NO pixel dimensions. This is the cheap half of the listing: os.scandir
    plus a stat per entry, no file is ever OPENED. v683 exists because the
    expensive half (a Pillow header read or a cv2 probe PER FILE) ran for every
    file in the folder before the UI could show anything -- in Solo view too,
    where nothing but the selected file is even visible."""
    entries = []
    with os.scandir(rp) as it:
        for e in it:
            kind = _media_kind(e.name)
            if not kind:
                continue
            try:
                st = e.stat()
            except OSError:
                continue
            entries.append({"name": e.name, "size": st.st_size,
                            "mtime": st.st_mtime, "kind": kind,
                            "w": None, "h": None, "fps": None})
    entries.sort(key=lambda d: d["mtime"], reverse=True)
    return entries


def _scan_media_with_dims(rp):
    """Blocking folder scan -> entries (newest first) with name/size/mtime/kind
    PLUS pixel w/h. Runs in a worker thread (see handle_media_list) so the
    per-file dimension probes never block the asyncio event loop.

    MEASURED COST (v683): the scandir half is milliseconds; the probe half is
    one file OPEN per entry and dominates everything. Callers that do not need
    every dimension up front should ask for ?dims=0 and fetch what they display
    from /uls/media/dims."""
    t0 = time.time()
    entries = _scan_media_fast(rp)
    t1 = time.time()
    for d in entries:
        path = os.path.join(rp, d["name"])
        if d["kind"] == "image":
            d["w"], d["h"] = _media_img_dims(path)
        elif d["kind"] == "video":
            d["w"], d["h"], d["fps"] = _media_vid_dims(path)
    t2 = time.time()
    print("[PLS] media list: %d files, scan %d ms, dimension probe %d ms"
          % (len(entries), int((t1 - t0) * 1000), int((t2 - t1) * 1000)))
    return entries


async def handle_media_list(request: web.Request) -> web.Response:
    """GET ?folder=<abs>[&dims=0] -> image/video files in folder, NEWEST FIRST,
    each with name/size/mtime/kind and pixel w/h. Dimensions are probed in a
    worker thread (image header read / cv2 FFMPEG header probe, NO full decode)
    so the listing stays responsive; w/h is null when a probe is unavailable
    (e.g. cv2 missing), in which case the frontend fills it from the browser
    when the media loads.

    v683: `dims=0` returns the listing WITHOUT probing any file -- milliseconds
    instead of one open per file. w/h/fps come back as null and the caller
    fetches the ones it actually displays from /uls/media/dims. The DEFAULT is
    unchanged (dims on), so every other caller keeps its old contract."""
    _deny = _lan_deny(request)          # v367: LAN browse lock (audit B1)
    if _deny is not None:
        return _deny
    folder = (request.query.get("folder", "") or "").strip()
    want_dims = (request.query.get("dims", "1") or "1").strip() not in ("0", "false", "no")
    try:
        rp = os.path.realpath(folder)
        if not os.path.isdir(rp):
            return web.json_response({"ok": False, "error": "not a folder", "files": []}, status=404)
        loop = asyncio.get_running_loop()   # v577: get_event_loop() is deprecated (3.12+)
        scan = _scan_media_with_dims if want_dims else _scan_media_fast
        entries = await loop.run_in_executor(None, scan, rp)
        return web.json_response({"ok": True, "folder": rp, "files": entries,
                                  "dims": bool(want_dims)})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e), "files": []}, status=500)


async def handle_media_thumb(request: web.Request) -> web.Response:
    """GET ?folder=&file= -> a downscaled JPEG thumbnail. Images via Pillow; a
    video's first frame via OpenCV. 404 if no thumbnail can be made (the UI then
    shows a placeholder tile)."""
    _deny = _lan_deny(request)          # v367: LAN browse lock (audit B1)
    if _deny is not None:
        return _deny
    folder = (request.query.get("folder", "") or "").strip()
    file = (request.query.get("file", "") or "").strip()
    if not folder or not file:
        return web.Response(status=400, text="folder and file required")
    path = os.path.realpath(os.path.join(folder, file))
    if not _within(folder, path) or not os.path.isfile(path):
        return web.Response(status=404, text="not found")
    kind = _media_kind(file)
    try:
        import io as _io
        from PIL import Image, ImageOps
        if kind == "image":
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        elif kind == "video":
            try:
                import cv2  # opencv-python (optional; core does not bundle it)
            except ImportError:
                _hint_cv2_once("the video thumbnail")
                return web.Response(status=404, text="no cv2")
            cap = cv2.VideoCapture(path)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                return web.Response(status=404, text="no frame")
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            return web.Response(status=404, text="unsupported")
        img.thumbnail((_MEDIA_THUMB_PX, _MEDIA_THUMB_PX))
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        return web.Response(body=buf.getvalue(), content_type="image/jpeg",
                            headers={"Cache-Control": "no-cache"})
    except Exception as e:
        # Pillow/cv2 missing or a decode error -> let the UI show its placeholder.
        return web.Response(status=404, text=f"thumb error: {e}")


async def handle_media_file(request: web.Request) -> web.Response:
    """GET ?folder=&file= -> the raw media file WITH Range-Request support, so a
    <video> can stream/seek for hover-play (the thumb route only yields a still
    frame). Mirrors handle_preview_video's Range handling; _within-guarded."""
    _deny = _lan_deny(request)          # v367: LAN browse lock (audit B1)
    if _deny is not None:
        return _deny
    folder = (request.query.get("folder", "") or "").strip()
    file = (request.query.get("file", "") or "").strip()
    if not folder or not file:
        return web.Response(status=400, text="folder and file required")
    path = os.path.realpath(os.path.join(folder, file))
    if not _within(folder, path) or not os.path.isfile(path):
        return web.Response(status=404, text="not found")

    mime = mimetypes.guess_type(path)[0]
    if not mime:
        # v457: keep audio Content-Type honest when the OS mimetypes DB is thin.
        mime = _AUDIO_MIME.get(os.path.splitext(path)[1].lower(), "application/octet-stream")
    file_size = os.path.getsize(path)

    range_header = request.headers.get("Range", "")
    if range_header.startswith("bytes="):
        try:
            start_str, end_str = range_header[6:].split("-", 1)
            if not start_str and end_str:          # RFC-7233 suffix range bytes=-N
                n_suffix = int(end_str)
                start = max(0, file_size - n_suffix) if n_suffix > 0 else file_size
                end = file_size - 1
            else:
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else file_size - 1
            end = min(end, file_size - 1)
            if start < 0 or start > end or start >= file_size:
                return web.Response(status=416,
                                    headers={"Content-Range": f"bytes */{file_size}"},
                                    text="Requested Range Not Satisfiable")
            length = end - start + 1
            with open(path, "rb") as f:
                f.seek(start)
                data = f.read(length)
            return web.Response(status=206, body=data, content_type=mime,
                                headers={"Content-Range": f"bytes {start}-{end}/{file_size}",
                                         "Accept-Ranges": "bytes",
                                         "Content-Length": str(length),
                                         "Cache-Control": "no-cache"})
        except (ValueError, OSError):
            pass  # fall back to full response

    try:
        with open(path, "rb") as f:
            data = f.read()
        return web.Response(body=data, content_type=mime,
                            headers={"Accept-Ranges": "bytes",
                                     "Content-Length": str(file_size),
                                     "Cache-Control": "no-cache"})
    except OSError as e:
        return web.Response(status=500, text=f"read error: {e}")


# v437: PowerShell for the NATIVE folder pick. Primary path = the modern Common Item
# Dialog (IFileOpenDialog) with FOS_PICKFOLDERS — the Explorer-style picker (breadcrumb
# + search box) that returns an absolute filesystem path. If the COM interop is
# unavailable for any reason the catch falls back to the original FolderBrowserDialog
# (older tree look, but always present on Windows), so there is never a regression.
# NOTE: this is LIVE-only verifiable — it needs Windows + PowerShell, which the build
# sandbox does not have. The gate tests assert its STRUCTURE (interop + fallback), not
# its on-screen behaviour.
# v448: bring an EXISTING Explorer window for a folder to the foreground (or open a
# fresh one). os.startfile / ShellExecute from this background server cannot beat
# Windows' foreground lock, so a repeat press only flashes the taskbar. This helper
# finds the open window via the Shell.Application COM list and force-foregrounds it
# (AttachThreadInput onto the current foreground thread, then SetForegroundWindow).
# The target path arrives in $env:PLS_OPEN_PATH (never on the command line).
_OPEN_FOLDER_PS = '''
$ErrorActionPreference = 'Stop'
$target = $env:PLS_OPEN_PATH
if (-not $target) { exit 1 }
try { $target = [System.IO.Path]::GetFullPath($target).TrimEnd('\\') } catch { }

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
namespace PHFg {
  public static class Win {
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool ShowWindowAsync(IntPtr h, int n);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
    [DllImport("user32.dll")] public static extern bool AttachThreadInput(uint a, uint b, bool f);
    [DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();
    public static void Force(IntPtr h) {
      uint pid;
      uint fg = GetWindowThreadProcessId(GetForegroundWindow(), out pid);
      uint me = GetCurrentThreadId();
      AttachThreadInput(me, fg, true);
      ShowWindowAsync(h, 9);            // SW_RESTORE (un-minimize if needed)
      SetForegroundWindow(h);
      AttachThreadInput(me, fg, false);
    }
  }
}
'@

$hwnd = [IntPtr]::Zero
try {
  $shell = New-Object -ComObject Shell.Application
  foreach ($w in $shell.Windows()) {
    try {
      $p = $w.Document.Folder.Self.Path
      if ($p) {
        $p = [System.IO.Path]::GetFullPath($p).TrimEnd('\\')
        if ($p -ieq $target) { $hwnd = [IntPtr]([int64]$w.HWND); break }
      }
    } catch { }
  }
} catch { }

if ($hwnd -ne [IntPtr]::Zero) {
  [PHFg.Win]::Force($hwnd)             # existing window -> force it to the front
} else {
  Invoke-Item -LiteralPath $target     # none open -> a fresh window (gets focus normally)
}
exit 0
'''


_NATIVE_FOLDER_PS = '''
$ErrorActionPreference = 'Stop'
$picked = ''
try {
  Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
namespace PHPick {
  [ComImport, ClassInterface(ClassInterfaceType.None), Guid("DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7")]
  public class FileOpenDialogRCW { }
  // FOS_PICKFOLDERS (0x20) turns the open dialog into the modern folder picker.
  [ComImport, Guid("d57c7288-d4ad-4768-be02-9d969532d960"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  public interface IFileOpenDialog {
    [PreserveSig] int Show(IntPtr parent);
    void SetFileTypes(uint c, IntPtr r);
    void SetFileTypeIndex(uint i);
    void GetFileTypeIndex(out uint i);
    void Advise(IntPtr e, out uint c);
    void Unadvise(uint c);
    void SetOptions(uint o);
    void GetOptions(out uint o);
    void SetDefaultFolder(IntPtr i);
    void SetFolder(IntPtr i);
    void GetFolder(out IntPtr i);
    void GetCurrentSelection(out IntPtr i);
    void SetFileName(string n);
    void GetFileName(out string n);
    void SetTitle(string t);
    void SetOkButtonLabel(string t);
    void SetFileNameLabel(string l);
    void GetResult(out IShellItem ppsi);
    void AddPlace(IntPtr i, int a);
    void SetDefaultExtension(string e);
    void Close(int hr);
    void SetClientGuid(ref Guid g);
    void ClearClientData();
    void SetFilter(IntPtr f);
    void GetResults(out IntPtr e);
    void GetSelectedItems(out IntPtr a);
  }
  [ComImport, Guid("43826d1e-e718-42ee-bc55-a1e261c37bfe"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  public interface IShellItem {
    void BindToHandler(IntPtr c, ref Guid h, ref Guid r, out IntPtr v);
    void GetParent(out IShellItem i);
    void GetDisplayName(uint sigdn, out IntPtr name);
    void GetAttributes(uint m, out uint a);
    void Compare(IShellItem i, uint h, out int o);
  }
  public static class FolderPick {
    public static string Run() {
      IFileOpenDialog d = (IFileOpenDialog)(new FileOpenDialogRCW());
      uint o; d.GetOptions(out o);
      // v652: NO FOS_PICKFOLDERS -- folder mode hides every file (OS behavior),
      // which made navigating blind. A normal FILE dialog shows the folder's
      // contents; the pick's PARENT FOLDER is what gets pinned (server-side).
      d.SetOptions(o | 0x40 | 0x1000);               // FOS_FORCEFILESYSTEM | FOS_FILEMUSTEXIST
      d.SetTitle("Pick any file inside your media folder - the FOLDER gets pinned");
      d.SetOkButtonLabel("Use this folder");
      if (d.Show(IntPtr.Zero) != 0) return "";       // user cancelled / error
      IShellItem it; d.GetResult(out it);
      IntPtr p; it.GetDisplayName(0x80058000, out p); // SIGDN_FILESYSPATH
      string s = Marshal.PtrToStringUni(p);
      Marshal.FreeCoTaskMem(p);
      return s;
    }
  }
}
'@
  $picked = [PHPick.FolderPick]::Run()
} catch {
  Add-Type -AssemblyName System.Windows.Forms
  $owner = New-Object System.Windows.Forms.Form; $owner.TopMost = $true
  $fb = New-Object System.Windows.Forms.FolderBrowserDialog
  $fb.Description = 'Select a folder for the Polyhedron Media Loader'
  $fb.ShowNewFolderButton = $true
  if ($fb.ShowDialog($owner) -eq 'OK') { $picked = $fb.SelectedPath }
  $owner.Dispose()
}
if ($picked) { Write-Output $picked }
'''


def _native_pick_resolve(path):
    """The native-pick contract: the dialog returns a FILE (so the user
    browses with the folder's contents visible); the FOLDER of that file is
    the pin and the FILE ITSELF loads immediately (v660) -- same behavior as
    a dropped path. A directory (the FolderBrowserDialog fallback, or a
    tkinter askdirectory) passes through as (itself, no file); anything else
    is a non-pick. Returns (folder, file_name)."""
    if not path:
        return "", ""
    if os.path.isdir(path):
        return os.path.realpath(path), ""
    if os.path.isfile(path):
        return os.path.realpath(os.path.dirname(path)), os.path.basename(path)
    return "", ""


async def handle_media_native_pick(request: web.Request) -> web.Response:
    """Open a NATIVE OS folder-picker on the machine running ComfyUI and return the
    chosen absolute path. This is a LOCAL-tool feature: the dialog appears on the
    server's desktop (= the user's own screen for a local ComfyUI). Windows opens the
    modern Common Item Dialog (IFileOpenDialog, FOS_PICKFOLDERS — Explorer look), with
    the older PowerShell FolderBrowserDialog as a no-regression fallback; other
    platforms try tkinter (needs a display). Returns {ok, path} on a pick,
    {ok:false, cancelled:true} on cancel, or {ok:false, reason:...} when no native
    dialog can be opened — the frontend then falls back to the list/recent/paste."""
    _deny = _lan_deny(request)          # v367: LAN browse lock (audit B1)
    if _deny is not None:
        return _deny
    try:
        if os.name == "nt":
            # base64 UTF-16LE -> -EncodedCommand so the multi-line C# interop crosses
            # the process boundary verbatim, with no -Command quoting/newline mangling.
            import base64
            enc = base64.b64encode(_NATIVE_FOLDER_PS.encode("utf-16-le")).decode("ascii")
            proc = await asyncio.create_subprocess_exec(
                "powershell", "-STA", "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-EncodedCommand", enc,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        else:
            import sys
            code = ("import tkinter, tkinter.filedialog as fd;"
                    "r=tkinter.Tk();r.withdraw();r.attributes('-topmost',True);"
                    "p=fd.askopenfilename(title='Pick any file inside your "
                    "media folder')or fd.askdirectory();print(p or '')")
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", code,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=300)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return web.json_response({"ok": False, "reason": "timeout"})
        path = (out or b"").decode("utf-8", "ignore").strip()
        folder, fname = _native_pick_resolve(path)
        if folder:
            return web.json_response({"ok": True, "path": folder, "file": fname})
        if proc.returncode and not path:
            return web.json_response({"ok": False, "reason": "unavailable",
                                      "error": (err or b"").decode("utf-8", "ignore")[:300]})
        return web.json_response({"ok": False, "cancelled": True})
    except FileNotFoundError:
        return web.json_response({"ok": False, "reason": "unavailable"})
    except Exception as e:
        return web.json_response({"ok": False, "reason": "unavailable", "error": str(e)[:300]})


async def handle_media_upload(request: web.Request) -> web.Response:
    """POST ?folder=<abs> multipart -> save image/video file(s) into folder.
    Returns {ok, folder, names:[...]}."""
    _deny = _lan_deny(request)          # v367: LAN browse lock (audit B1)
    if _deny is not None:
        return _deny
    folder = (request.query.get("folder", "") or "").strip()
    rp = os.path.realpath(folder) if folder else ""
    if not rp or not os.path.isdir(rp):
        return web.json_response({"ok": False, "error": "pick a folder first"}, status=400)
    try:
        reader = await request.multipart()
    except Exception as e:
        return web.json_response({"ok": False, "error": f"not a multipart upload: {e}"}, status=400)
    saved = []
    while True:
        try:
            part = await reader.next()
        except Exception as e:
            _cleanup_names(rp, saved)
            return web.json_response({"ok": False, "error": f"multipart read error: {e}"}, status=400)
        if part is None:
            break
        if part.name not in ("file", "files", "image", "video", "media"):
            continue
        raw = getattr(part, "filename", None)
        if not raw or not _media_kind(raw):
            continue  # skip non-image/video parts
        name = _safe_media_name(raw)
        out_path = os.path.join(rp, name)
        if not _within(rp, out_path):
            _cleanup_names(rp, saved)
            return web.json_response({"ok": False, "error": "unsafe target path"}, status=400)
        try:
            size = await _stream_part_to_disk(part, out_path)
        except ValueError:
            try:
                os.remove(out_path)
            except OSError:
                pass
            _cleanup_names(rp, saved)
            return web.json_response({"ok": False, "error": f"a file exceeds the {_MAX_UPLOAD_BYTES} byte cap"}, status=413)
        except Exception as e:
            _cleanup_names(rp, saved)
            return web.json_response({"ok": False, "error": f"write error: {e}"}, status=500)
        if name not in saved:
            saved.append(name)
        print(f"[PLS] MediaLoader: uploaded {name} ({size} bytes) -> {rp}")
    if not saved:
        return web.json_response({"ok": False, "error": "no image/video file in the upload"}, status=400)
    return web.json_response({"ok": True, "folder": rp, "names": saved})


# ─── Media Loader: sequence library (v443) ──────────────────────
# Build clean, ordered image sequences WITHOUT mutating originals, and keep them
# permanently. Every built sequence is a NAMED, persistent folder under one fixed
# library root, "<output>/PLS_sequences/<name>/", filled with 00001.ext, ... .
# The raw source folder is only ever READ; the library is the only thing written,
# so a build can never nest inside a source (the v424 _pls_prep bug) and two
# sequences never collide. A sequence is directly selectable as the active batch.
#
# The pure path/naming logic (safe_project_name / is_within_library /
# renumbered_targets) lives in the torch-free engine ph_media_util so the guard
# in front of Delete's recursive delete is gate-testable without ComfyUI.
# uls_routes keeps the filesystem I/O and supplies the one folder_paths touch the
# pure logic can't: the output root the library lives under. Engine imports stay
# function-local so a hypothetical engine-import problem can never take down the
# rest of the route module.
# NOTE: _SEQUENCES_DIRNAME mirrors ph_media_util.SEQUENCES_DIRNAME (asserted in tests).
_SEQUENCES_DIRNAME = "PLS_sequences"


def _library_dir() -> str:
    """<output>/PLS_sequences -- the one fixed, persistent sequence library. Not
    created here; callers that write create it on demand."""
    try:
        out = folder_paths.get_output_directory()
    except Exception:
        out = ""
    return os.path.join(out, _SEQUENCES_DIRNAME) if out else ""


def _is_within_library(project: str) -> bool:
    """is_within_library with the live library root resolved in -- refuses the
    library itself, drive roots, non-children, traversal, and any unsafe name."""
    from .ph_media_util import is_within_library
    lib = _library_dir()
    if not lib:
        return False
    return is_within_library(lib, project)


def _list_top_images(folder_rp: str):
    """Top-level image NAMES in `folder_rp`, excluding subfolders and dotfiles.
    Names only -- the engine pipeline works on names. Used by Build to read the
    immutable source originals."""
    out = []
    try:
        with os.scandir(folder_rp) as it:
            for e in it:
                try:
                    if e.name.startswith(".") or e.is_dir():
                        continue
                    if _media_kind(e.name) == "image":
                        out.append(e.name)
                except OSError:
                    continue
    except OSError:
        pass
    return out


async def handle_media_seq_list(request: web.Request) -> web.Response:
    """GET -> {ok, library, projects:[{name, count, path}]}. Lists the named
    sequence folders in <output>/PLS_sequences, each with its image-frame count,
    natural-sorted by name. A missing library is an empty list (not an error) --
    nothing is created on a read."""
    lib = _library_dir()
    projects = []
    if lib and os.path.isdir(lib):
        from .ph_media_util import natural_sort_key
        names = []
        try:
            with os.scandir(lib) as it:
                for e in it:
                    try:
                        if e.is_dir() and not e.name.startswith("."):
                            names.append(e.name)
                    except OSError:
                        continue
        except OSError:
            names = []
        for nm in sorted(names, key=natural_sort_key):
            p = os.path.join(lib, nm)
            projects.append({"name": nm, "count": len(_list_top_images(p)), "path": p})
    return web.json_response({"ok": True, "library": lib, "projects": projects})


async def handle_media_seq_build(request: web.Request) -> web.Response:
    """POST ?source=<abs>&name=&sort=&filter=&nth=&overwrite= -> build a NAMED,
    persistent sequence under <output>/PLS_sequences/<name>/ from the SAME
    ordered/filtered/strided selection the batch loader would see, copied out as
    00001.ext, ... . The source is read-only; the library is created on demand.
    If <name> already exists and overwrite is not set, returns {ok:False,
    exists:True} so the UI can confirm; overwrite=1 rebuilds it (clear + rebuild,
    guarded). skip_first/cap are load-time trims and are NOT baked in here."""
    _deny = _lan_deny(request)          # v367: LAN browse lock (audit B1)
    if _deny is not None:
        return _deny
    source = (request.query.get("source", "") or "").strip()
    rs = os.path.realpath(source) if source else ""
    if not rs or not os.path.isdir(rs):
        return web.json_response({"ok": False, "error": "pick a source folder first"}, status=400)

    from .ph_media_util import safe_project_name, sequence_dir_for
    name = safe_project_name(request.query.get("name", "") or "")
    if not name:
        return web.json_response({"ok": False, "error": "enter a valid sequence name"}, status=400)

    sort_mode = (request.query.get("sort", "") or "name (natural)").strip() or "name (natural)"
    name_filter = (request.query.get("filter", "") or "*").strip() or "*"
    try:
        every_nth = max(1, int(request.query.get("nth", "1") or "1"))
    except (TypeError, ValueError):
        every_nth = 1
    overwrite = (request.query.get("overwrite", "") or "").strip() in ("1", "true", "yes")

    names = _list_top_images(rs)
    if not names:
        return web.json_response({"ok": False, "error": "no images in the source folder"}, status=400)

    times = None
    if sort_mode in ("mtime (oldest first)", "created"):
        times = {}
        for n in names:
            try:
                times[n] = os.path.getmtime(os.path.join(rs, n))
            except OSError:
                times[n] = 0.0

    from .ph_media_util import select_frames as _select_frames
    # filter -> order -> every_nth ; no skip/cap (those are load-time trims)
    selected = _select_frames(names, sort_mode=sort_mode, name_filter=name_filter,
                              skip_first=0, every_nth=every_nth, cap=0, times=times)
    if not selected:
        return web.json_response({"ok": False, "error": "filter matched no images"}, status=400)

    lib = _library_dir()
    if not lib:
        return web.json_response({"ok": False, "error": "output directory unavailable"}, status=500)
    target = sequence_dir_for(lib, name)
    if not target or not _is_within_library(target):
        return web.json_response({"ok": False, "error": "unsafe sequence path -- refused"}, status=400)

    if os.path.isdir(target) and not overwrite:
        return web.json_response({"ok": False, "exists": True, "name": name,
                                  "error": f'sequence "{name}" already exists'})
    try:
        os.makedirs(lib, exist_ok=True)
        if os.path.isdir(target):
            shutil.rmtree(target)          # overwrite -> fresh rebuild
        os.makedirs(target, exist_ok=True)
    except OSError as e:
        return web.json_response({"ok": False, "error": f"cannot prepare folder: {e}"}, status=500)

    from .ph_media_util import renumbered_targets
    written = []
    for src_name, dst_name in renumbered_targets(selected):
        dst = os.path.join(target, dst_name)
        if not _within(target, dst):
            continue
        try:
            shutil.copy2(os.path.join(rs, src_name), dst)
            written.append(dst_name)
        except OSError:
            continue
    print(f"[PLS] MediaLoader: built sequence '{name}' ({len(written)} frame(s)) -> {target}")
    return web.json_response({"ok": True, "name": name, "path": target, "count": len(written)})


async def handle_media_seq_delete(request: web.Request) -> web.Response:
    """POST ?name=<sequence> -> delete <output>/PLS_sequences/<name> entirely.
    Guarded by _is_within_library: refuses the library itself, drive roots,
    non-children, traversal, and any unsafe name. A missing folder is a no-op
    success."""
    from .ph_media_util import safe_project_name, sequence_dir_for
    name = safe_project_name(request.query.get("name", "") or "")
    if not name:
        return web.json_response({"ok": False, "error": "no sequence name given"}, status=400)
    lib = _library_dir()
    if not lib:
        return web.json_response({"ok": False, "error": "output directory unavailable"}, status=500)
    target = sequence_dir_for(lib, name)
    if not target or not _is_within_library(target):
        return web.json_response({"ok": False, "error": "unsafe sequence path -- refused"}, status=400)
    if not os.path.isdir(target):
        return web.json_response({"ok": True, "name": name, "removed": False, "note": "nothing to delete"})
    try:
        shutil.rmtree(target)
    except OSError as e:
        return web.json_response({"ok": False, "error": f"delete failed: {e}"}, status=500)
    print(f"[PLS] MediaLoader: deleted sequence '{name}' -> {target}")
    return web.json_response({"ok": True, "name": name, "removed": True})


async def handle_media_resolve(request: web.Request) -> web.Response:
    """GET ?path=<abs file> -> {ok, folder, file, kind} for the drag&drop path
    flow (v644): a dropped PATH TEXT is resolved server-side so the frontend
    can pin the ORIGIN folder and select the file. Only real files with a
    known image/video/audio extension resolve; anything else is a 400 -- the
    drop is a convenience, never a guess."""
    _deny = _lan_deny(request)          # v367: LAN browse lock (audit B1)
    if _deny is not None:
        return _deny
    path = (request.query.get("path", "") or "").strip().strip('"')
    if not path:
        return web.json_response({"ok": False, "error": "no path"}, status=400)
    rp = os.path.realpath(os.path.expanduser(path))
    if not os.path.isfile(rp):
        return web.json_response({"ok": False, "error": "not a file"}, status=400)
    ext = os.path.splitext(rp)[1].lower()
    if ext in _MEDIA_VIDEO_EXTS:
        kind = "video"
    elif ext in _MEDIA_IMAGE_EXTS:
        kind = "image"
    elif ext in _MEDIA_AUDIO_EXTS:
        kind = "audio"
    else:
        return web.json_response({"ok": False, "error": "not a media file"}, status=400)
    return web.json_response({"ok": True, "folder": os.path.dirname(rp),
                              "file": os.path.basename(rp), "kind": kind})


async def handle_media_locate(request: web.Request) -> web.Response:
    """POST {name, size, mtime, folders:[abs,...]} -> {ok, folder, file} | {ok:False, reason}.
    v665: the drag&drop pin path for browsers that hide the origin of an OS file
    drop. The browser still hands out the file's NAME, byte SIZE and MTIME; this
    checks that triple against the CANDIDATE folders the frontend supplies (the
    node's recents + current pin + input) — never a disk-wide search. Rules:
      exactly one folder holds a file matching name AND size (mtime within 2 s,
      tolerating FAT/exFAT's 2-second stamps and timezone-less copies) -> that hit;
      several folders match -> {'reason': 'ambiguous'} (an honest refusal beats a
      coin flip); none -> {'reason': 'not_found'}. Verification against known
      ground, not a guess."""
    _deny = _lan_deny(request)          # v367: LAN browse lock (audit B1)
    if _deny is not None:
        return _deny
    try:
        data = await request.json()
    except Exception as e:
        return web.json_response({"ok": False, "error": f"bad json: {e}"}, status=400)
    name = os.path.basename(str(data.get("name", "") or ""))
    try:
        size = int(data.get("size", -1))
        mtime = int(data.get("mtime", 0))
    except (TypeError, ValueError):
        return web.json_response({"ok": False, "error": "bad size/mtime"}, status=400)
    folders = data.get("folders") or []
    if not name or size < 0 or not isinstance(folders, list):
        return web.json_response({"ok": False, "error": "need name, size, folders"}, status=400)
    if not _media_kind(name):
        return web.json_response({"ok": False, "reason": "not_media"})
    # input/ is always a candidate — a previously copied file re-locates there.
    try:
        folders = list(folders) + [os.path.realpath(folder_paths.get_input_directory())]
    except Exception:
        folders = list(folders)
    hits = []
    seen = set()
    for f in folders[:24]:                      # recents are capped at 12; stay bounded
        rp = os.path.realpath(str(f or ""))
        if not rp or rp in seen or not os.path.isdir(rp):
            continue
        seen.add(rp)
        cand = os.path.join(rp, name)
        try:
            st = os.stat(cand)
        except OSError:
            continue
        if int(st.st_size) != size:
            continue
        if mtime and abs(int(st.st_mtime) - mtime) > 2:
            continue
        hits.append(rp)
    if len(hits) == 1:
        return web.json_response({"ok": True, "folder": hits[0], "file": name})
    return web.json_response({"ok": False,
                              "reason": "ambiguous" if hits else "not_found"})



async def handle_media_open_folder(request: web.Request) -> web.Response:
    """POST ?path=<abs> -> reveal the folder in the OS file manager on the machine
    running ComfyUI AND bring an already-open window for that folder to the front. A
    LOCAL-tool feature like the native picker (the window opens on the server's own
    desktop = the user's screen for a local ComfyUI). On Windows a short PowerShell
    helper finds an Explorer window already showing the folder (Shell.Application COM
    window list) and force-foregrounds it past Windows' foreground lock
    (AttachThreadInput + SetForegroundWindow) -- so a SECOND press re-focuses the
    existing window instead of only flashing the taskbar -- otherwise it opens a fresh
    window. macOS/Linux Popen the platform file manager. The path must be an existing
    directory and travels in an environment variable, never on a command line, so it
    can never be reinterpreted as a shell command."""
    _deny = _lan_deny(request)          # v367: LAN browse lock (audit B1)
    if _deny is not None:
        return _deny
    path = (request.query.get("path", "") or "").strip()
    if not path:
        return web.json_response({"ok": False, "error": "no path given"}, status=400)
    if not os.path.isdir(path):
        return web.json_response({"ok": False, "error": "not a folder"}, status=400)
    real = os.path.realpath(path)
    import sys
    try:
        if os.name == "nt":
            import base64
            enc = base64.b64encode(_OPEN_FOLDER_PS.encode("utf-16-le")).decode("ascii")
            env = {**os.environ, "PLS_OPEN_PATH": real}   # path in env, never on argv
            proc = await asyncio.create_subprocess_exec(
                "powershell", "-STA", "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-EncodedCommand", enc,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
            try:
                _, err = await asyncio.wait_for(proc.communicate(), timeout=20)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                return web.json_response({"ok": False, "error": "timeout"}, status=500)
            if proc.returncode:
                msg = (err or b"").decode("utf-8", "ignore")[:300] or "powershell failed"
                return web.json_response({"ok": False, "error": msg}, status=500)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", real])         # arg list -> no shell, no injection
        else:
            subprocess.Popen(["xdg-open", real])     # arg list -> no shell, no injection
    except FileNotFoundError:
        return web.json_response({"ok": False, "error": "no file manager available"}, status=500)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)[:300]}, status=500)
    print(f"[PLS] MediaLoader: opened/foregrounded folder -> {real}")
    return web.json_response({"ok": True, "path": real})


async def handle_media_proc_count(request: web.Request) -> web.Response:
    """GET ?folder=&sort=&filter= -> {ok, folder, total}: how many media files the
    Batch Processing cursor will walk for this folder/sort/filter. Calls the SAME
    resolver load() uses (ph_media_loader._proc_resolve_files), so the count is the
    node's exact total -- the "Run all" button queues precisely this many runs and
    the sweep stops on its own. The scan runs in a worker thread so os.scandir never
    blocks the event loop (mirrors handle_media_list)."""
    folder = (request.query.get("folder", "") or "").strip()
    sort_mode = (request.query.get("sort", "") or "name (natural)")
    name_filter = (request.query.get("filter", "") or "*") or "*"
    try:
        rp = os.path.realpath(folder)
        if not os.path.isdir(rp):
            return web.json_response({"ok": False, "error": "not a folder", "total": 0},
                                     status=404)
        # Lazy import (mirrors the open-folder / sampler-preview routes) so the route
        # module never hard-imports the heavy media node at load time.
        from .ph_media_loader import _proc_resolve_files
        loop = asyncio.get_running_loop()   # v577: get_event_loop() is deprecated (3.12+)
        files = await loop.run_in_executor(None, _proc_resolve_files, rp, sort_mode, name_filter)
        return web.json_response({"ok": True, "folder": rp, "total": len(files)})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)[:300], "total": 0}, status=500)


def _media_dims_for(rp, names):
    """Probe pixel dimensions for the NAMED files only (the deferred half of
    the listing). Unknown or unreadable names are simply absent from the
    result -- dimensions stay best-effort, never fatal."""
    out = {}
    for name in names:
        kind = _media_kind(name)
        if not kind:
            continue
        path = os.path.realpath(os.path.join(rp, name))
        if not _within(rp, path) or not os.path.isfile(path):
            continue
        fps = None
        if kind == "image":
            w, h = _media_img_dims(path)
        elif kind == "video":
            w, h, fps = _media_vid_dims(path)
        else:
            w, h = (None, None)
        out[name] = {"w": w, "h": h, "fps": fps}
    return out


async def handle_media_dims(request: web.Request) -> web.Response:
    """GET ?folder=<abs>&files=<name>|<name>|... -> {name: {w, h, fps}} for the
    NAMED files only (v683, the deferred half of the listing). The frontend asks
    for the page it is about to draw plus the selected file, so a thousand-file
    folder costs the probes of the dozen tiles on screen instead of a thousand.
    Names are matched inside the folder and never escape it."""
    _deny = _lan_deny(request)          # v367: LAN browse lock (audit B1)
    if _deny is not None:
        return _deny
    folder = (request.query.get("folder", "") or "").strip()
    raw = (request.query.get("files", "") or "").strip()
    names = [n for n in raw.split("|") if n][:512]
    try:
        rp = os.path.realpath(folder)
        if not os.path.isdir(rp):
            return web.json_response({"ok": False, "error": "not a folder", "dims": {}}, status=404)
        if not names:
            return web.json_response({"ok": True, "folder": rp, "dims": {}})
        loop = asyncio.get_running_loop()
        dims = await loop.run_in_executor(None, _media_dims_for, rp, names)
        return web.json_response({"ok": True, "folder": rp, "dims": dims})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e), "dims": {}}, status=500)


def register_media_routes():
    """Register the Media Loader endpoints on the ComfyUI PromptServer.

    Called from __init__.py right after the Stack's own register_routes().
    Each route is added under BOTH the bare path and the "/api"-prefixed path:
    ComfyUI's frontend helpers (api.fetchApi / api.apiURL) prefix every call
    with "/api", and routes added directly through app.router.add_route do not
    get that alias automatically. Every add is guarded, so a duplicate path can
    never abort plugin startup.
    """
    try:
        app = PromptServer.instance.app
    except Exception as e:
        print(f"[PLS] \u26a0 Media routes: PromptServer not available: {e}")
        return

    routes = [
        ("GET",  "/uls/media/folders",     handle_media_folders),
        ("GET",  "/uls/media/list",        handle_media_list),
        ("GET",  "/uls/media/dims",        handle_media_dims),
        ("GET",  "/uls/media/thumb",       handle_media_thumb),
        ("GET",  "/uls/media/file",        handle_media_file),
        ("GET",  "/uls/media/native_pick", handle_media_native_pick),
        ("POST", "/uls/media/upload",      handle_media_upload),
        ("POST", "/uls/media/locate",      handle_media_locate),
        ("GET",  "/uls/media/resolve",     handle_media_resolve),
        ("GET",  "/uls/media/seq/list",    handle_media_seq_list),
        ("POST", "/uls/media/seq/build",   handle_media_seq_build),
        ("POST", "/uls/media/seq/delete",  handle_media_seq_delete),
        ("POST", "/uls/media/open_folder", handle_media_open_folder),
        ("GET",  "/uls/media/proc_count",  handle_media_proc_count),
    ]
    registered = 0
    for method, path, handler in routes:
        for p in (path, "/api" + path):
            try:
                app.router.add_route(method, p, handler)
                registered += 1
            except Exception as e:
                print(f"[PLS] \u26a0 could not add media route {method} {p}: {e}")
    print(f"[PLS] \u2713 Media Loader routes registered "
          f"(root + /api alias, {registered} paths)")
