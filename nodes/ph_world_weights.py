# -*- coding: ascii -*-
"""
Polyhedron model door -- the ONE way a model file arrives (public build).

A node run never downloads. When a file is missing, the node offers it; the
click goes to POST /pls/weights/install, which fetches the file over https
into the folder ComfyUI reports (ph_weights.ensure_weights: a .part file,
size and sha256 checked, then moved into place). GET /pls/weights/status
answers what is on disk -- no network.

In this build the door serves three sets: "rife", the checkpoints of
Polyhedron Interpolate, and (v392) "restore", the four optional SeedVR2 files
of the Power Upscale's restore path (Comfy-Org's repack, Apache-2.0), and
"h3_latent_upscale", the learned 3D upscaler of Polyhedron H3 Latent Upscale
(LBH-123-AI, MIT) -- pinned like the rest. Their pins live with the node (ph_interpolate.
RIFE_WEIGHTS) and are read lazily, so this module imports without torch or
ComfyUI. (The development tree's door also serves sets of nodes that are not
part of this release; they are not carried here.)

Registry hygiene: fixed https URLs, each pinned by sha256 + size; weight
files only, no code; no child processes, no package installation, nothing
runs at import.
"""

import os
import re

try:
    from . import ph_weights as PW
except ImportError:                      # flat import in the guards
    import ph_weights as PW

REGISTRY = {}
SETS = ("rife", "restore", "h3_latent_upscale", "control", "sdpose", "da3")   # v392: + the Power Upscale's restore path (SeedVR2, optional)
                                                  #        + the H3 Latent Upscale's learned upscaler (LBH-123-AI, MIT)
PROGRESS_EVENT = "pls.weights.progress"
_MISSING = "not installed"
_LABEL_TAIL = re.compile(r"\s\u00b7\s[\d.,]+\s?(KB|MB|GB)(\s\u00b7\s" + _MISSING + r")?$")
_BUSY = set()


def _merge_rife():
    """The RIFE checkpoints of Polyhedron Interpolate, read from the node once."""
    if any(v.get("set") == "rife" for v in REGISTRY.values()):   # v392: the restore rows are static
        return
    try:
        from . import ph_interpolate as I
    except Exception:
        return
    for name, spec in I.RIFE_WEIGHTS.items():
        REGISTRY[name] = {"set": "rife", "folder": "vfi", "label": "RIFE " + name.split(".")[0][4:],
                          "url": spec["url"], "bytes": spec["bytes"], "sha256": spec["sha256"],
                          "note": "Polyhedron Interpolate", "file": spec["file"]}


# v1193 (H3 Latent Upscale): the learned 3D latent upscaler (LBH-123-AI, MIT) -- fp16, one safetensors into Core's
# latent_upscale_models. Pin from the API listing, 20.10.2026.
REGISTRY["minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"] = {
    "set": "h3_latent_upscale", "folder": "latent_upscale_models", "label": "H3 latent upscaler 3D conv v1 (fp16)",
    "url": "https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_conv_v1/"
           "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors",
    "bytes": 690592672, "sha256": "043e5a48e161610ef6c3ea974645220354d06fa618abca15f76d084812eb55c2",
    "note": "Polyhedron H3 Latent Upscale -- the learned upscaler on the first pass's latent (LBH-123-AI, MIT)"}
# v1194 (Power Upscale, the RESTORE path): Core's SeedVR2 family -- Comfy-Org's repack (Apache-2.0, not gated), pins
# from the API listing 20.10.2026. ALL OPTIONAL: the Power Upscale works without them (ESRGAN + refine); the line under
# the node stays green and offers them. The 3B int8 is the one for a 16 GB card (3.2 GB); the VAE is shared.
_SVR = "https://huggingface.co/Comfy-Org/SeedVR2/resolve/main/"
for _name, _sub, _bytes, _sha, _label, _note in (
        ("seedvr2_3b_int8_convrot.safetensors", "diffusion_models", 3458259704,
         "c3dec8bcc5916843a8a858572970597462e1f2dc598d6dfd818f6cd40f53a157", "restoration model 3B int8 (optional)",
         "Power Upscale restore path -- SeedVR2 3B, int8 convrot (the one for a 16 GB card); into models/diffusion_models, "
         "picked in the node's restore_model"),
        ("seedvr2_ema_vae_fp16.safetensors", "vae", 501324814,
         "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1", "restoration VAE (optional, shared)",
         "Power Upscale restore path -- the family's own video VAE; into models/vae, the node pairs it with restore_model itself"),
        ("seedvr2_3b_fp16.safetensors", "diffusion_models", 6784268336,
         "98669fd2c06df5eca88baf68cd5c478775c8e61fc110e598c52b350145ea2660", "restoration model 3B fp16 (optional)",
         "Power Upscale restore path -- SeedVR2 3B fp16 (6.3 GB: fewer frames per chunk on a 16 GB card)"),
        ("seedvr2_7b_sharp_int8_convrot.safetensors", "diffusion_models", 8334897976,
         "db48be2f1cc7e36b01a2aa529810f5d9c6a971edd29be225cf1b0eb18d51c366", "restoration model 7B sharp int8 (optional)",
         "Power Upscale restore path -- SeedVR2 7B sharp, int8 (7.8 GB: stills on a 16 GB card, not clips)")):
    REGISTRY[_name] = {"set": "restore", "folder": _sub, "label": _label, "optional": True, "url": _SVR + _sub + "/" + _name,
                       "bytes": _bytes, "sha256": _sha, "note": _note,
                       # v1194: the line's optional button fetches the RECOMMENDED pair first (3B int8 + VAE, 3.7 GB), the
                       # other optional files only once those are on disk -- never 17 GB on one click
                       "recommended": _name in ("seedvr2_3b_int8_convrot.safetensors", "seedvr2_ema_vae_fp16.safetensors")}
for _k, _v in REGISTRY.items():
    _v["file"] = _k


def fmt_bytes(n):
    n = int(n or 0)
    if n >= 1 << 30:
        return "%.1f GB" % (n / float(1 << 30))
    if n >= 1 << 20:
        return "%d MB" % round(n / float(1 << 20))
    return "%d KB" % max(1, round(n / float(1 << 10)))


def local_path(name):
    """The file on disk (any configured root, subfolders count) or None."""
    spec = REGISTRY.get(name)
    if spec is None:
        return None
    return PW.find_file(spec["folder"], spec["file"])


def label(name, path=None):
    """Combo entry: 'file \u00b7 size' (on disk: the real size); a missing
    registry file gets '\u00b7 not installed'."""
    spec = REGISTRY.get(name)
    if path is None:
        path = local_path(name) if spec else None
    if path:
        try:
            return name + " \u00b7 " + fmt_bytes(os.path.getsize(path))
        except OSError:
            return name
    if spec:
        return name + " \u00b7 " + fmt_bytes(spec["bytes"]) + " \u00b7 " + _MISSING
    return name


def strip(value):
    """Inverse of label(): exactly the tail it adds, nothing of the name."""
    return _LABEL_TAIL.sub("", str(value or "")).rstrip()


def status(set_name=None):
    """-> [{name, label, title, set, folder, installed, path, bytes, size, note, dir}]. Disk only."""
    _merge_rife()
    out = []
    for name, spec in REGISTRY.items():
        if set_name and spec["set"] != set_name:
            continue
        path = local_path(name)
        out.append({"name": name, "label": label(name, path), "title": spec["label"], "set": spec["set"],
                    "outdated": "", "rebuild": False, "optional": bool(spec.get("optional")),
                    "recommended": bool(spec.get("recommended")),
                    "folder": spec["folder"], "installed": bool(path), "path": path or "",
                    "bytes": spec["bytes"], "size": fmt_bytes(spec["bytes"]),
                    "note": spec["note"], "dir": PW.primary_dir(spec["folder"])})
    return out


def progress_fetch(report):
    """A _fetch for ensure_weights() that reports (done, total, phase)."""
    import urllib.request

    def _fetch(url, dest_part):
        req = urllib.request.Request(PW.https(url), headers=dict(PW.UA))
        with urllib.request.urlopen(req, timeout=PW.HTTP_TIMEOUT) as resp, open(dest_part, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            done, last, last_said = 0, -1.0, -1
            print("[PLS] Model door: downloading %s (%d MiB) from %s" % (os.path.basename(dest_part), total >> 20, url))
            report(done, total, "start")
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                pct = (done / total * 100.0) if total else 0.0
                if total and pct - last >= 1.0:            # one event per percent
                    last = pct
                    report(done, total, "run")
                if total and int(pct) // 10 > last_said:   # one console line per 10 %
                    last_said = int(pct) // 10
                    print("[PLS] Model door: %d%% (%d / %d MiB)" % (int(pct), done >> 20, total >> 20))
            report(done, total or done, "verify")
    return _fetch


def install(name, _fetch=None, _report=None):
    """Fetch + verify + place ONE registry file. -> local path."""
    _merge_rife()
    spec = REGISTRY.get(name)
    if spec is None:
        raise RuntimeError("unknown model: %r" % (name,))
    return PW.ensure_weights(spec["folder"], name, {name: spec}, tag="Model door (%s)" % spec["set"], _fetch=_fetch)


# ---------------------------------------------------------------- routes

def _send(name, done, total, phase):
    try:
        from server import PromptServer
        PromptServer.instance.send_sync(PROGRESS_EVENT, {"name": name, "done": done, "total": total,
                                                         "phase": phase})
    except Exception:
        pass


async def handle_status(request):
    from aiohttp import web
    want = request.query.get("set", "")
    if want and want not in SETS:
        return web.json_response({"ok": True, "models": []})
    return web.json_response({"ok": True, "models": status(want or None)})


async def handle_install(request):
    """POST {name}: the only way a model file arrives -- a click."""
    import asyncio
    from aiohttp import web
    try:
        data = await request.json()
    except Exception:
        data = {}
    name = strip(data.get("name"))
    _merge_rife()
    if PW.download_locked():
        return web.json_response({"ok": False, "error": PW.download_locked_text()}, status=403)
    if name not in REGISTRY:
        return web.json_response({"ok": False, "error": "unknown model: %s" % name})
    have = local_path(name)
    if have:
        return web.json_response({"ok": True, "path": have, "already": True})
    if name in _BUSY:
        return web.json_response({"ok": False, "error": "%s is already downloading" % name})
    _BUSY.add(name)
    try:
        import time as _time
        _t0 = _time.time()
        report = lambda d, t, ph: _send(name, d, t, ph)  # noqa: E731
        path = await asyncio.to_thread(install, name, progress_fetch(report), report)
        print("[PLS] Model door: %s installed in %.1f s -> %s" % (name, _time.time() - _t0, path))
    except Exception as e:
        _send(name, 0, 0, "error")
        return web.json_response({"ok": False, "error": str(e)})
    finally:
        _BUSY.discard(name)
    _send(name, 1, 1, "done")
    return web.json_response({"ok": True, "path": path})


def register_weights_routes():
    """GET /pls/weights/status, POST /pls/weights/install (bare + /api)."""
    try:
        from server import PromptServer
        app = PromptServer.instance.app
    except Exception as e:
        print("[PLS] Model door routes not registered: %s" % e)
        return 0
    n = 0
    for method, path, fn in (("GET", "/pls/weights/status", handle_status),
                             ("POST", "/pls/weights/install", handle_install)):
        for p in (path, "/api" + path):
            try:
                app.router.add_route(method, p, fn)
                n += 1
            except Exception as e:
                print("[PLS] could not add route %s %s: %s" % (method, p, e))
    return n
# v1216 (Frank 23.10., the first Control field run: "Auf jeden Fall braucht es wohl Modelle ... ausserdem muessen hier
# wohl auch offiziell Download-Optionen mit rein"): the control line joins the door. Apply Control picks a Fun
# ControlNet Union file in models/model_patches, Control Map's pose mode the SDPose checkpoint in models/checkpoints
# (its depth mode the DA3 files below). Pins from the Hugging Face API listings, 23.10.2026; every repo not gated.
# H3: Comfy-Org's repack of alibaba-pai's Fun ControlNet Union (MiniMax H3 Community License); 2.0 = 8 controls
# (canny, depth, HED, MLSD, pose, scribble, layout, gray) + inpainting, 1.0 = the first five. The 2.0 int8 is the one
# for a 16 GB card next to an int8 H3. Qwen-Image 2.1: Kijai's conversion of alibaba-pai's union (the image family).
_FUN = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/model_patches/"
_QFUN = "https://huggingface.co/Kijai/QwenImage_experimental/resolve/main/model_patches/"
for _name, _url, _bytes, _sha, _label, _note, _rec in (
        ("minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors", _FUN, 4531220608,
         "890af58bb350f0c2f6c409b1c67ea8f4196654297d037769b72de1e6f504c612", "H3 Fun ControlNet Union 2.0 int8",
         "Apply Control -- MiniMax H3, 8 controls + inpainting, int8 (the one for a 16 GB card; MiniMax H3 Community "
         "License)", True),
        ("minimax_h3_fun_controlnet_union_2.0_pruned_bf16.safetensors", _FUN, 8382288792,
         "cd48bdc11b1f1a3b934746c898d25123754b335c13d37d5c421f8373079282b4", "H3 Fun ControlNet Union 2.0 bf16",
         "Apply Control -- MiniMax H3, 8 controls + inpainting, bf16 (MiniMax H3 Community License)", False),
        ("minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors", _FUN, 2296635360,
         "9c645c0a308c8af361efd43b409710f6f8fec0db297c29503e141a84991fed0c", "H3 Fun ControlNet Union 1.0 int8",
         "Apply Control -- MiniMax H3, 5 controls (canny, depth, HED, MLSD, pose), int8 (MiniMax H3 Community "
         "License)", False),
        ("minimax_h3_fun_controlnet_union_pruned_bf16.safetensors", _FUN, 4222169456,
         "57fe1e64928a63a55e3cd4586b55cd5d0eb4980648b6f31e5d9dac16fe7f1c48", "H3 Fun ControlNet Union 1.0 bf16",
         "Apply Control -- MiniMax H3, 5 controls, bf16 (MiniMax H3 Community License)", False),
        ("qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors", _QFUN, 3779298944,
         "07aa961570ac0e03d4ca936aecd76854d077a33cde69b5092399afba01b3715d", "Qwen-Image 2.1 Fun ControlNet Union int8",
         "Apply Control -- Qwen-Image 2.1 (image family), int8 (Kijai's conversion of alibaba-pai's union; the repo "
         "names no licence)", True),
        ("qwen_image_2.1_fun_controlnet_union_bf16.safetensors", _QFUN, 7550977992,
         "1f9dea8384ba167084f39f624908f1dbf2ea8a223bb4a8108a77dafa9b55bfe1", "Qwen-Image 2.1 Fun ControlNet Union bf16",
         "Apply Control -- Qwen-Image 2.1 (image family), bf16 (Kijai's conversion of alibaba-pai's union)", False)):
    REGISTRY[_name] = {"set": "control", "folder": "model_patches", "label": _label, "url": _url + _name,
                       "bytes": _bytes, "sha256": _sha, "note": _note, "recommended": _rec}
REGISTRY["sdpose_wholebody_fp16.safetensors"] = {
    "set": "sdpose", "folder": "checkpoints", "label": "SDPose wholebody fp16",
    "url": "https://huggingface.co/Comfy-Org/SDPose/resolve/main/checkpoints/sdpose_wholebody_fp16.safetensors",
    "bytes": 1916645792, "sha256": "63d01f9a7494560693b24767f4469d59c9d3266b31ff0a253e74d1e611442721",
    "note": "Control Map pose -- whole-body keypoints from the frames (Comfy-Org's repack of teemosliang/SDPose-Wholebody, "
            "MIT)"}
# v394: Control Map's depth mode reads a Depth Anything 3 file (Comfy-Org's repack, models/geometry_estimation)
_DA3 = "https://huggingface.co/Comfy-Org/Depth-Anything-3/resolve/main/geometry_estimation/"
for _name, _label, _bytes, _sha, _note in (
        ("depth_anything_3_small.safetensors", "DA3-Small", 137254980,
         "9c0a53d157c5b315d4c82908a43d93eeaa3a9ac3de9d15c2b4242ef46b5508ad", "Control Map depth -- small and fast, Apache-2.0"),
        ("depth_anything_3_base.safetensors", "DA3-Base", 541524124,
         "418c0d2ea857e2d1215fa51baa46833f499a62eb2400ec63d337aa20d326414f", "Control Map depth, Apache-2.0"),
        ("depth_anything_3_mono_large.safetensors", "DA3-Mono-Large", 1336748056,
         "9b44eda5bedba5b4e125686fdb79d1db309c1b9785277576eb930f885b008f96",
         "Control Map depth -- the sharpest single-picture depth"),
        ("depth_anything_3_metric_large.safetensors", "DA3-Metric-Large", 1336748056,
         "b6eac00c060931fb15a97c1ba77d2e49d866b83ff7da631289279b3c1c0c99c8", "Control Map depth -- depth in metres")):
    REGISTRY[_name] = {"set": "da3", "folder": "geometry_estimation", "label": _label, "url": _DA3 + _name,
                       "bytes": _bytes, "sha256": _sha, "note": _note}


for _k, _v in REGISTRY.items():   # v394: the control / sdpose / da3 rows too
    _v.setdefault("file", _k)


def da3_choices(installed_names):
    """DA3 combo (Control Map's depth mode): the files on disk first (any name), then the known ones that are
    missing -- every entry labelled with its size."""
    have = list(installed_names)
    out = []
    for n in have:
        path = None
        try:
            import folder_paths
            path = folder_paths.get_full_path("geometry_estimation", n)
        except Exception:
            path = None
        out.append(label(n, path=path) if path else n)
    bases = {os.path.basename(n.replace("\\", "/")) for n in have}
    for name, spec in REGISTRY.items():
        if spec["set"] == "da3" and name not in bases:
            out.append(label(name, path=""))
    return out
