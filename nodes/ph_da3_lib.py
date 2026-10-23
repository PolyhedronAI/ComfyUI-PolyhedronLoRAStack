# -*- coding: ascii -*-
"""
Polyhedron Depth Anything 3 runner   (nodes/ph_da3_lib.py)
=========================================================
Loads and runs ComfyUI's Depth Anything 3 MODEL (comfy/ldm/depth_anything_3,
loaded with comfy.sd like every Polyhedron loader) without ComfyUI's DA3
NODES. The result is the same DA3_GEOMETRY dict those nodes pass around, so
both sides interoperate, with one difference on purpose:

  * the intrinsics are given at the INPUT size, the size the depth comes
    back at (ComfyUI keeps them at the model's working size, so its own
    point-cloud / mesh conversion of a multiview result unprojects with a
    focal length and centre that belong to another resolution). The dict
    says so: "intrinsics_size": "input".

Model files: models/geometry_estimation (the folder is registered here when
the Core does not know it). Small / Base: depth + confidence + camera poses
(multiview). Mono-Large / Metric-Large: depth + sky, one picture at a time.
"""

import os

import torch
import torch.nn.functional as F

FOLDER = "geometry_estimation"


def ensure_folder():
    try:
        import folder_paths
    except Exception:
        return False
    if FOLDER not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths[FOLDER] = (
            [os.path.join(folder_paths.models_dir, FOLDER)], set(folder_paths.supported_pt_extensions))
    return True


def da3_files():
    """DA3 files in models/geometry_estimation (MoGe shares the folder)."""
    if not ensure_folder():
        return []
    import folder_paths
    try:
        names = folder_paths.get_filename_list(FOLDER)
    except Exception:
        return []
    return [n for n in names if "depth_anything_3" in n.lower() or "da3" in n.lower()]


def load(name, weight_dtype="default"):
    """-> ComfyUI ModelPatcher around DepthAnything3Net."""
    ensure_folder()
    import comfy.sd
    import folder_paths
    path = folder_paths.get_full_path_or_raise(FOLDER, name)
    opts = {}
    if weight_dtype in ("fp16", "bf16", "fp32"):
        opts["dtype"] = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[weight_dtype]
    mp = comfy.sd.load_diffusion_model(path, model_options=opts)
    net = getattr(getattr(mp, "model", None), "diffusion_model", None)
    if type(net).__name__ != "DepthAnything3Net":
        raise ValueError("[PLS] %s is not a Depth Anything 3 model (it loads as %s)" % (name, type(net).__name__))
    return mp


def capabilities(mp):
    net = mp.model.diffusion_model
    return {"cam_dec": getattr(net, "cam_dec", None) is not None,
            "dualdpt": getattr(net, "head_type", "") == "dualdpt",
            "sky": bool(getattr(net, "has_sky", False)),
            "conf": bool(getattr(net, "has_conf", False))}


def _prep(frames, resolution, method):
    from comfy.ldm.depth_anything_3 import preprocess as P
    return P.preprocess_image(frames, process_res=int(resolution), method=method)


def _up(t, H, W):
    return F.interpolate(t.unsqueeze(1).float(), size=(H, W), mode="bilinear", align_corners=False).squeeze(1).cpu()


def run(mp, frames, resolution, method="upper_bound_resize", mode=None):
    """frames (B,H,W,3) -> DA3_GEOMETRY. mode: {"mode": "mono"} or
    {"mode": "multiview", "ref_view_strategy": ..., "pose_method": "cam_dec" | "ray_pose"}."""
    import comfy.model_management as mm
    mode = mode or {"mode": "mono"}
    frames = frames[..., :3]
    B, H, W = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    mm.load_model_gpu(mp)
    net = mp.model.diffusion_model
    dev = mm.get_torch_device()
    dt = net.dtype if getattr(net, "dtype", None) is not None else torch.float32
    geom = {"image": frames.cpu().float(), "mode": mode["mode"]}
    if mode["mode"] == "multiview" and B > 1:
        cap = capabilities(mp)
        pose = mode.get("pose_method", "cam_dec")
        if not (cap["cam_dec"] or cap["dualdpt"]):
            raise ValueError("[PLS] Depth Anything 3: multiview needs DA3-Small or DA3-Base (camera poses); "
                             "this model has none -- use it one picture at a time.")
        if pose == "cam_dec" and not cap["cam_dec"]:
            pose = "ray_pose"
        x = _prep(frames.to(dev), resolution, method).to(dtype=dt).unsqueeze(0)
        h, w = int(x.shape[-2]), int(x.shape[-1])
        with torch.no_grad():
            out = net(x, use_ray_pose=(pose == "ray_pose"), ref_view_strategy=mode.get("ref_view_strategy", "first"))
        geom["depth"] = _up(out["depth"], H, W)
        for src, dst in (("depth_conf", "confidence"), ("sky", "sky")):
            if src in out:
                geom[dst] = _up(out[src], H, W)
        if "extrinsics" in out and "intrinsics" in out:
            K = out["intrinsics"].float().cpu().clone()
            K[..., 0, :] *= W / float(w)                     # fx, 0, cx  -> input size
            K[..., 1, :] *= H / float(h)                     # 0, fy, cy
            geom["extrinsics"] = out["extrinsics"].float().cpu()
            geom["intrinsics"] = K
            geom["intrinsics_size"] = "input"
        geom["mode"] = "multiview"
        return geom
    depths, confs, skies = [], [], []
    for i in range(B):
        x = _prep(frames[i:i + 1].to(dev), resolution, method).to(dtype=dt)
        with torch.no_grad():
            out = net(x)
        depths.append(_up(out["depth"], H, W))
        if "depth_conf" in out:
            confs.append(_up(out["depth_conf"], H, W))
        if "sky" in out:
            skies.append(_up(out["sky"], H, W))
    geom["depth"] = torch.cat(depths, 0)
    if confs:
        geom["confidence"] = torch.cat(confs, 0)
    if skies:
        geom["sky"] = torch.cat(skies, 0)
    geom["mode"] = "mono"
    return geom
