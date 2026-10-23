# -*- coding: ascii -*-
"""
Polyhedron control runtime   (nodes/ph_control_core.py, v1131)
==============================================================
The model side of the control line, built on the comfy.* MODEL LIBRARY only -- no Core node module
(`nodes`, `comfy_extras`) is imported (guard v1063: the Core is GPL, the pack MIT).

  load_control_file(path)   a Fun ControlNet Union file -> ModelPatcher (video: MiniMax H3, image:
                            Qwen-Image 2.1); the model classes come from comfy.ldm.
  VideoControlPatch         steers a video model frame by frame (control / mask / source frames,
                            encoded once per latent shape with the model's VAE).
  ImageControlPatch         steers an image model, one control per batch item (one for all, or as
                            many as the batch; repeated over a CFG batch).
  draw_pose(frames, ...)    POSE_KEYPOINT frames -> IMAGE in the OpenPose / DWPose drawing convention
                            the control models were trained on (limb ellipses, joint dots, rainbow
                            hands, white face dots).
  sdpose_frames(...)        SDPose wholebody (a diffusion model with a heatmap head) -> POSE_KEYPOINT.
  depth_frames(...)         Depth Anything 3 (pack runner ph_da3_lib) -> near-bright depth IMAGE.

Patches chain: a second Apply Control wraps the first one's block patches, so pose + depth add up.
Nothing here downloads; missing files are the caller's message.
"""

import colorsys
import math
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NULL_CONTEXT = os.path.join(HERE, "data", "sdpose_null_context.safetensors")


# ------------------------------------------------------------------------------ loading
def _patch_ops(sd):
    """dtype + operations for a control file: quantized layers stay quantized (bf16 compute),
    otherwise the usual bf16 / fp32 pick for the device."""
    import comfy.model_management as mm
    import comfy.ops
    import comfy.utils
    quant = comfy.utils.detect_layer_quantization(sd, "")
    if quant is not None:
        return torch.bfloat16, comfy.ops.mixed_precision_ops(quant, torch.bfloat16)
    dev = mm.get_torch_device()
    dtype = mm.unet_dtype(model_params=-1, supported_dtypes=[torch.bfloat16, torch.float32],
                          weight_dtype=comfy.utils.weight_dtype(sd))
    cast = mm.unet_manual_cast(dtype, dev, supported_dtypes=[torch.bfloat16, torch.float32])
    return dtype, comfy.ops.pick_operations(dtype, cast, load_device=dev)


def _count_blocks(sd):
    n = 0
    while "control_blocks.%d.after_proj.weight" % n in sd:
        n += 1
    return n


def file_family(sd):
    """'video' (MiniMax H3 Fun union), 'image' (Qwen-Image 2.1 Fun union) or None."""
    if "control_img_in.weight" in sd and "control_blocks.0.img_mlp.out.weight" in sd:
        return "image"
    try:
        import comfy.ldm.minimax.controlnet as mc
        if mc.is_minimax_h3_fun_state_dict(sd):
            return "video"
    except Exception:
        pass
    return None


def build_control_model(sd, metadata=None):
    """state dict -> (family, torch module), the module built on comfy.ldm, weights not yet loaded."""
    import json
    import comfy.model_management as mm
    fam = file_family(sd)
    if fam is None:
        raise ValueError("[PLS] Apply Control: the file is not a Fun ControlNet Union (neither the video "
                         "nor the image family).")
    dtype, ops = _patch_ops(sd)
    off = mm.unet_offload_device()
    n = _count_blocks(sd)
    if fam == "image":
        import comfy.ldm.qwen_image21.model as qm
        inner = sd["control_img_in.weight"].shape[0]
        gate_up = sd.get("control_blocks.0.img_mlp.gate_up.weight", None)
        hidden = gate_up.shape[0] // 2 if gate_up is not None else sd["control_blocks.0.img_mlp.proj.weight"].shape[0]
        model = qm.QwenImage21FunControl(
            num_blocks=n, control_in_dim=129, inner_dim=inner,
            attention_head_dim=sd["control_blocks.0.attn.norm_q.weight"].shape[0],
            mlp_ratio=hidden // inner, fused_mlp=gate_up is not None,
            operations=ops, device=off, dtype=dtype)
        return fam, model
    import comfy.ldm.minimax.controlnet as mc
    layers = tuple(range(0, 50, 50 // n))                 # spread over the 50 base blocks (5 or 10 control blocks)
    meta = metadata or {}
    if "control_blocks_places" in meta:
        layers = tuple(json.loads(meta["control_blocks_places"]))
        if len(layers) != n:
            raise ValueError("[PLS] Apply Control: the file's control_blocks_places (%d) do not match its "
                             "%d control blocks." % (len(layers), n))
    qkv = sd["control_blocks.0.attn.qkv_proj.weight"]
    head = sd["control_blocks.0.attn.q_norm.weight"].shape[0]
    curves = meta.get("minimax_h3_fun_controlnet") == "adaln_basis"
    model = mc.MiniMaxH3FunControl(
        control_in_dim=49, injection_layers=layers,
        inpaint_post_norm=meta.get("inpaint_masked_pixel_mode") == "post_norm",
        hidden_size=sd["control_proj_in.weight"].shape[0],
        num_attention_heads=qkv.shape[0] // (3 * head), attention_head_dim=head,
        ffn_hidden_size=sd["control_blocks.0.mlp.fc1.weight"].shape[0] // 2,
        time_embed_dim=8 if curves else 2688, use_adaln_curves=curves,
        operations=ops, device=off, dtype=dtype)
    model.requires_grad_(False)
    return fam, model


def load_control_file(path):
    """-> ModelPatcher around the control model (the MODEL_PATCH type of the Core's patch loader)."""
    import comfy.model_management as mm
    import comfy.model_patcher
    import comfy.utils
    sd, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    _, model = build_control_model(sd, metadata)
    kw = {}
    try:
        import comfy.storage
        kw["fast_disk"] = comfy.storage.state_dict_fast_disk(sd)
    except Exception:
        pass
    mp = comfy.model_patcher.CoreModelPatcher(model, load_device=mm.get_torch_device(),
                                              offload_device=mm.unet_offload_device(), **kw)
    model.load_state_dict(sd, assign=mp.is_dynamic())
    return mp


# ------------------------------------------------------------------------------ patches
def _paused():
    try:
        import comfy.model_prefetch
        return comfy.model_prefetch.pause_malloc_graph()
    except Exception:
        import contextlib
        return contextlib.nullcontext()


class _BlockLink:
    """One base block's replacement: our before / after around the previous patch (or the block)."""

    def __init__(self, owner, block_index, previous):
        self.owner = owner
        self.block_index = block_index
        self.previous = previous

    def __call__(self, args, extra_args):
        with _paused():
            self.owner.before_block(self.block_index, args)
        out = extra_args["original_block"](args) if self.previous is None else self.previous(args, extra_args)
        with _paused():
            return self.owner.after_block(self.block_index, args, out)

    def to(self, device_or_dtype):
        self.owner.to(device_or_dtype)
        if hasattr(self.previous, "to"):
            self.previous = self.previous.to(device_or_dtype)
        return self

    def cleanup(self):
        self.owner.cleanup()
        if hasattr(self.previous, "cleanup"):
            self.previous.cleanup()

    def models(self):
        out = self.owner.models()
        if hasattr(self.previous, "models"):
            out = out + self.previous.models()
        return out


class _ControlBase:
    BLOCK_KIND = None

    def __init__(self, model_patch, vae, strength, sigma_start, sigma_end):
        self.model_patch = model_patch
        self.vae = vae
        self.strength = float(strength)
        self.sigma_start = float(sigma_start)
        self.sigma_end = float(sigma_end)
        self.active = False
        self.layers = None
        self.stream = None
        self.pristine = None

    def in_range(self, sigma):
        return self.sigma_end <= float(sigma) <= self.sigma_start

    def before_block(self, block_index, args):
        if self.active and block_index == self.layers[0]:
            self.pristine = args["img"].clone()          # the base block updates its input in place

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device):
            self._move(device_or_dtype)
            self.stream = None
        return self

    def _move(self, device):
        pass

    def cleanup(self):
        self.stream = None
        self.pristine = None
        self.active = False

    def models(self):
        return [self.model_patch]

    def register(self, model, layers):
        import comfy.patcher_extension
        self.layers = list(layers)
        model.add_wrapper(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, self.wrapper)
        for b in self.layers:
            prev = model.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {}).get(
                (self.BLOCK_KIND, b))
            model.set_model_patch_replace(_BlockLink(self, b, prev), "dit", self.BLOCK_KIND, b)


def _loaded():
    import comfy.model_management as mm
    return mm.loaded_models(only_currently_used=True)


def _reload(models):
    import comfy.model_management as mm
    mm.load_models_gpu(models)


class VideoControlPatch(_ControlBase):
    """Video family (MiniMax H3 Fun union): control / mask / source frames (N,3,H,W) in 0..1."""
    BLOCK_KIND = "double_block"

    def __init__(self, model_patch, vae, control, mask, source, strength, sigma_start, sigma_end):
        _ControlBase.__init__(self, model_patch, vae, strength, sigma_start, sigma_end)
        self.control = control
        self.mask = mask
        self.source = source
        self.latent = None
        self.latent_shape = None
        self.layers = list(model_patch.model.injection_layers)     # fixed by the control file

    def _frames(self, x, count, W, H):
        import comfy.utils
        idx = torch.arange(count, device=x.device).clamp(max=x.shape[0] - 1)   # short clips hold the last frame
        return comfy.utils.common_upscale(x[idx], W, H, "bilinear", "center")

    def _encode(self, frames, shape):
        lat = self.vae.encode(frames.movedim(1, -1)).to(torch.float32)
        if tuple(lat.shape) != tuple(shape):
            raise ValueError("[PLS] Apply Control: the VAE gives %s for the control, the latent is %s -- use the "
                             "model's own video VAE." % (tuple(lat.shape), tuple(shape)))
        return lat

    def prepare(self, shape):
        import torch.nn.functional as F
        import comfy.utils
        shape = tuple(shape)
        if self.latent is not None and self.latent_shape == shape:
            return
        lf, lh, lw = shape[2:]
        count = max((lf - 2) // 5, 0) * 17 + 5               # pixel frames behind lf latent frames
        sc = self.vae.spacial_compression_encode()
        W, H = lw * sc, lh * sc
        keep = _loaded()
        try:
            hint = None
            if self.control is not None:
                hint = self._encode(self._frames(self.control, count, W, H), shape)
            if self.mask is not None:
                m = (self.mask.reshape(-1, 1, self.mask.shape[-2], self.mask.shape[-1]) > 0.5).to(torch.float32)
                idx = torch.arange(count, device=m.device).clamp(max=m.shape[0] - 1)
                m = comfy.utils.common_upscale(m[idx], W, H, "bilinear", "center")
                vis = 1.0 - (m > 0.5).to(torch.float32)
                src = (torch.zeros(count, 3, H, W, dtype=vis.dtype, device=vis.device) if self.source is None
                       else self._frames(self.source, count, W, H))
                vis = vis.to(src.device)
                masked = src * vis
                if getattr(self.model_patch.model, "inpaint_post_norm", False):
                    from comfy.ldm.minimax.vae import IMAGENET_MEAN
                    masked = masked + (1.0 - vis) * torch.tensor(IMAGENET_MEAN, dtype=src.dtype,
                                                                 device=src.device).view(1, 3, 1, 1)
                mlat = self._encode(masked, shape)
                if hint is None:
                    hint = torch.zeros_like(mlat)
                vlat = F.interpolate(vis.squeeze(1)[None, None], size=(lf, lh, lw), mode="trilinear",
                                     align_corners=False)
                hint = torch.cat([hint, vlat.to(hint.device), mlat.to(hint.device)], dim=1)
        finally:
            _reload(keep)
        self.latent = hint
        self.latent_shape = shape

    def wrapper(self, executor, x, timestep, context, transformer_options={}, **kwargs):
        sig = transformer_options.get("sigmas")
        sigma = float(sig[0]) if sig is not None else float(timestep.flatten()[0]) / 1000.0
        self.active = self.in_range(sigma)
        self.stream = None
        if self.active:
            with _paused():
                self.prepare(x[0].shape)
        try:
            return executor(x, timestep, context, transformer_options, **kwargs)
        finally:
            self.stream = None
            self.pristine = None

    def after_block(self, block_index, args, out):
        if not self.active:
            return out
        cm = self.model_patch.model
        i = cm.injection_layers.index(block_index)
        if i == 0:
            self.latent = self.latent.to(out["img"].device)
            self.stream = cm.init_stream(self.pristine, self.latent, args["layout"], args["t_emb"])
            self.pristine = None
        self.stream, skip = cm.step(i, self.stream, args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                    transformer_options=args["transformer_options"])
        skip[args["layout"].audio_pos.to(skip.device)] = 0     # the audio tokens stay untouched
        out["img"].add_(skip, alpha=self.strength)
        return out

    def _move(self, device):
        if self.latent is not None:
            self.latent = self.latent.to(device)

    def cleanup(self):
        _ControlBase.cleanup(self)
        self.latent = None
        self.latent_shape = None

    def register(self, model):
        _ControlBase.register(self, model, self.model_patch.model.injection_layers)


class ImageControlPatch(_ControlBase):
    """Image family (Qwen-Image 2.1 Fun union): control / source (N,H,W,3), mask (N,H,W); one item for the
    whole batch, or one per batch item."""
    BLOCK_KIND = "single_block"

    def __init__(self, model_patch, vae, control, strength, source=None, mask=None, sigma_range=(float("inf"), 0.0)):
        _ControlBase.__init__(self, model_patch, vae, strength, sigma_range[0], sigma_range[1])
        self.image = control
        self.source = source
        self.mask = mask
        self.control = None

    def _item(self, image, source, mask, h, w):
        import comfy.latent_formats
        import comfy.utils
        sc = self.vae.spacial_compression_encode()
        W, H = w * sc, h * sc
        lf = comfy.latent_formats.QwenImage21()
        control = torch.zeros(1, lf.latent_channels, h, w)
        if image is not None:
            im = comfy.utils.common_upscale(image[None].movedim(-1, 1), W, H, "bicubic", "disabled")
            control = lf.process_in(self.vae.encode(im.movedim(1, -1))).float().cpu()
        regen = torch.ones(1, 1, H, W)
        if mask is not None:
            regen = (comfy.utils.common_upscale(mask[None, None].float().cpu(), W, H, "bilinear", "disabled")
                     >= 0.5).float()
        inp = torch.zeros_like(control)
        if source is not None:
            im = comfy.utils.common_upscale(source[None].movedim(-1, 1), W, H, "bicubic", "disabled").float().cpu()
            # regenerated pixels at mid-gray, the zero of the VAE's [-1, 1] input
            inp = lf.process_in(self.vae.encode((im * (1 - regen) + 0.5 * regen).movedim(1, -1))).float().cpu()
        keep = 1 - torch.nn.functional.interpolate(regen, size=(h, w), mode="nearest")
        return torch.cat([control, keep, inp], dim=1)                # 129 channels per token

    def items(self):
        masks = self.mask.reshape(-1, *self.mask.shape[-2:]) if self.mask is not None else None
        return max(int(self.image.shape[0]) if self.image is not None else 1,
                   int(self.source.shape[0]) if self.source is not None else 1,
                   int(masks.shape[0]) if masks is not None else 1), masks

    def prepare(self, h, w):
        n, masks = self.items()
        if self.control is not None and self.control.shape[-2:] == (h, w) and self.control.shape[0] in (n, ):
            return
        keep = _loaded()
        try:
            def at(t, i):
                return None if t is None else t[min(i, t.shape[0] - 1)]
            rows = [self._item(at(self.image, i), at(self.source, i), at(masks, i), h, w) for i in range(n)]
        finally:
            _reload(keep)
        self.control = torch.cat(rows, dim=0)

    def wrapper(self, executor, x, timestep, context, ref_latents, image_slots, transformer_options, **kwargs):
        sig = transformer_options.get("sigmas")
        sigma = float(sig.flatten()[0]) if sig is not None else float(timestep.flatten()[0])
        self.active = self.in_range(sigma)
        if self.active:
            with _paused():
                self.prepare(*x.shape[-2:])
        else:
            # outside the range our block links step aside, so the model can use its prefix cache
            dit = transformer_options.get("patches_replace", {}).get("dit", {})
            dit = {k: (p.previous if isinstance(p, _BlockLink) and p.owner is self else p) for k, p in dit.items()}
            dit = {k: p for k, p in dit.items() if p is not None}
            transformer_options = dict(transformer_options,
                                       patches_replace=dict(transformer_options.get("patches_replace", {}), dit=dit))
        try:
            return executor(x, timestep, context, ref_latents, image_slots, transformer_options, **kwargs)
        finally:
            self.stream = None
            self.pristine = None

    def after_block(self, block_index, args, out):
        if not self.active:
            return out
        cm = self.model_patch.model
        i = self.layers.index(block_index)
        if i == 0:
            B, n = int(out["img"].shape[0]), int(self.control.shape[0])
            if n > 1 and B != n:
                if B % n:
                    raise ValueError("[PLS] Apply Control: %d control images for a batch of %d -- give one image "
                                     "or as many as the batch (a multiple of it with CFG)." % (n, B))
                self.control = self.control.repeat(B // n, 1, 1, 1)
            self.control = self.control.to(out["img"].device, out["img"].dtype)
            self.stream = cm.init_stream(self.pristine, self.control.flatten(2).transpose(1, 2), args["prefix_len"])
            self.pristine = None
        self.stream, skip = cm.step(i, self.stream, args["mod"], args["pe"], args["attn_fn"], args["prefix_len"],
                                    args["transformer_options"])
        out["img"].add_(skip, alpha=self.strength)
        return out

    def _move(self, device):
        if self.control is not None:
            self.control = self.control.to(device)

    def cleanup(self):
        _ControlBase.cleanup(self)
        self.control = None

    def register(self, model):
        # the control blocks pair with every (base / control)-th base block
        base = len(model.get_model_object("diffusion_model").transformer_blocks)
        step = base // len(self.model_patch.model.control_blocks)
        _ControlBase.register(self, model, range(0, base, step))


# ------------------------------------------------------------------------------ pose drawing
# OpenPose 18-point body (0 nose, 1 neck, 2-4 right arm, 5-7 left arm, 8-10 right leg, 11-13 left leg,
# 14/15 eyes, 16/17 ears); limbs and colours in the OpenPose / DWPose convention.
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
         (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
N_HEAD_LIMBS = 5
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
          (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
          (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
HEAD_POINTS = {0, 14, 15, 16, 17}


def _pts(flat, n):
    if not flat:
        return np.zeros((n, 2), np.float32), np.zeros(n, np.float32)
    a = np.asarray(flat, np.float32).reshape(-1, 3)[:n]
    if a.shape[0] < n:
        a = np.concatenate([a, np.zeros((n - a.shape[0], 3), np.float32)])
    return a[:, :2], a[:, 2]


def _limb_polygon(p, q, half_width):
    """The limb as an ellipse polygon around the segment p-q: half axes (half length, half_width) + 0.5,
    the angle in whole degrees, a vertex every 4 degrees, vertices on whole pixels."""
    (x1, y1), (x2, y2) = p, q
    cx, cy = int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)
    length = math.hypot(x2 - x1, y2 - y1)
    ang = math.radians(int(math.degrees(math.atan2(y1 - y2, x1 - x2))) % 360)
    a, b = int(length / 2) + 0.5, half_width + 0.5
    t = np.radians(np.minimum(np.arange(0, 364, 4), 360))
    ca, sa = math.cos(ang), math.sin(ang)
    xs = np.rint(cx + a * np.cos(t) * ca - b * np.sin(t) * sa)
    ys = np.rint(cy + a * np.cos(t) * sa + b * np.sin(t) * ca)
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def _disc_offsets(r):
    n = int(math.ceil(r))
    oy, ox = np.mgrid[-n:n + 1, -n:n + 1]
    keep = ox * ox + oy * oy <= r * r
    return ox[keep], oy[keep]


def _stamp(canvas, xs, ys, r, color):
    """Discs of radius r at the integer points (xs, ys)."""
    if len(xs) == 0:
        return
    ox, oy = _disc_offsets(r)
    X = np.asarray(xs, np.int64)[:, None] + ox[None]
    Y = np.asarray(ys, np.int64)[:, None] + oy[None]
    h, w = canvas.shape[:2]
    ok = (X >= 0) & (X < w) & (Y >= 0) & (Y < h)
    canvas[Y[ok], X[ok]] = color


def _segment(x0, y0, x1, y1):
    """Integer points of the segment (Bresenham)."""
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err, xs, ys = dx - dy, [], []
    while True:
        xs.append(x0)
        ys.append(y0)
        if x0 == x1 and y0 == y1:
            return xs, ys
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def draw_pose(frames, hands=True, face=False, feet=False, stick_width=4, point_size=3, threshold=0.3, head=True,
              body=True, dot=4, hand_width=2, hand_dot=4):
    """POSE_KEYPOINT frames -> IMAGE (N,H,W,3) float, black background. Order: limbs, joint dots, feet,
    hands (lines, then dots), face dots."""
    from PIL import Image, ImageDraw
    if not frames:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    H, W = int(frames[0]["canvas_height"]), int(frames[0]["canvas_width"])
    out = np.zeros((len(frames), H, W, 3), np.uint8)

    def inside(x, y):
        return 0 <= x < W and 0 <= y < H

    core = len(LIMBS) - N_HEAD_LIMBS
    for fi, fr in enumerate(frames):
        cv = out[fi]
        for person in fr.get("people", []):
            bk, bs = _pts(person.get("pose_keypoints_2d"), 18)
            if body or head:
                img = Image.fromarray(cv)
                d = ImageDraw.Draw(img)
                for li, (a, b) in enumerate(LIMBS):
                    if (li >= core and not head) or (li < core and not body):
                        continue
                    if bs[a] < threshold or bs[b] < threshold or math.hypot(*(bk[a] - bk[b])) < 1:
                        continue
                    d.polygon(_limb_polygon(bk[a], bk[b], stick_width), fill=COLORS[li % len(COLORS)])
                cv[:] = np.asarray(img)
                for i in range(18):
                    if (i in HEAD_POINTS and not head) or (i not in HEAD_POINTS and i != 1 and not body):
                        continue
                    x, y = int(bk[i][0]), int(bk[i][1])
                    if bs[i] >= threshold and inside(x, y):
                        _stamp(cv, [x], [y], dot, COLORS[i % len(COLORS)])
            if feet:
                fk, fs = _pts(person.get("foot_keypoints_2d"), 6)
                for i in range(6):
                    x, y = int(fk[i][0]), int(fk[i][1])
                    if fs[i] >= threshold and inside(x, y):
                        _stamp(cv, [x], [y], dot, COLORS[(18 + i) % len(COLORS)])
            if hands:
                for key in ("hand_right_keypoints_2d", "hand_left_keypoints_2d"):
                    hk, hs = _pts(person.get(key), 21)
                    for ei, (a, b) in enumerate(HAND_EDGES):
                        x1, y1, x2, y2 = int(hk[a][0]), int(hk[a][1]), int(hk[b][0]), int(hk[b][1])
                        if hs[a] < threshold or hs[b] < threshold or min(x1, y1, x2, y2) <= 0:
                            continue
                        if inside(x1, y1) and inside(x2, y2):
                            r, g, bb = colorsys.hsv_to_rgb(ei / float(len(HAND_EDGES)), 1.0, 1.0)
                            xs, ys = _segment(x1, y1, x2, y2)
                            col = (int(r * 255), int(g * 255), int(bb * 255))
                            if hand_width > 1:
                                _stamp(cv, xs, ys, hand_width / 2.0 + 0.5, col)
                            else:
                                cv[ys, xs] = col
                    for i in range(21):
                        x, y = int(hk[i][0]), int(hk[i][1])
                        if hs[i] >= threshold and x > 0 and y > 0 and inside(x, y):
                            _stamp(cv, [x], [y], hand_dot, (0, 0, 255))
            if face:
                ck, cs = _pts(person.get("face_keypoints_2d"), 68)
                for i in range(68):
                    x, y = int(ck[i][0]), int(ck[i][1])
                    if cs[i] >= threshold and x > 0 and y > 0 and inside(x, y):
                        _stamp(cv, [x], [y], point_size, (255, 255, 255))
    return torch.from_numpy(out).float() / 255.0


# ------------------------------------------------------------------------------ SDPose (pose from pixels)
# COCO-WholeBody (133 points, MMPose order) -> OpenPose: body 17 + an inserted neck = 18, then 6 feet,
# 68 face, 21 right hand, 21 left hand.
_MM_TO_OP = {0: 0, 1: 17, 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9, 8: 12, 9: 14, 10: 16, 11: 11, 12: 13, 13: 15,
             14: 2, 15: 1, 16: 4, 17: 3}          # OpenPose index -> index in the 134-point list (neck at 17)


def wholebody_to_openpose(kp, sc, W, H):
    """kp (133,2) pixels, sc (133,) -> one OpenPose person dict."""
    kp = np.asarray(kp, np.float32)
    sc = np.asarray(sc, np.float32)
    neck = (kp[5] + kp[6]) / 2.0
    ns = min(sc[5], sc[6]) if (sc[5] > 0.3 and sc[6] > 0.3) else 0.0
    k = np.concatenate([kp[:17], neck[None], kp[17:]], 0)
    s = np.concatenate([sc[:17], [ns], sc[17:]], 0)
    body_k = np.stack([k[_MM_TO_OP[i]] for i in range(18)])
    body_s = np.array([s[_MM_TO_OP[i]] for i in range(18)], np.float32)

    def flat(a, b):
        return np.stack([a[:, 0], a[:, 1], b], 1).flatten().tolist()
    face_k = np.concatenate([k[24:92], body_k[[14, 15]]], 0)
    face_s = np.concatenate([s[24:92], body_s[[14, 15]]], 0)
    return {"pose_keypoints_2d": flat(body_k, body_s), "foot_keypoints_2d": flat(k[18:24], s[18:24]),
            "face_keypoints_2d": flat(face_k, face_s), "hand_right_keypoints_2d": flat(k[92:113], s[92:113]),
            "hand_left_keypoints_2d": flat(k[113:134], s[113:134])}


def null_context():
    """The conditioning SDPose was trained with: the frozen text encoder's output for an empty prompt
    (2 tokens x 1024), stored as data next to this module."""
    import comfy.model_management as mm
    import comfy.utils
    t = comfy.utils.load_torch_file(NULL_CONTEXT, safe_load=True)["context"]
    return [[t.to(mm.get_torch_device()).float(), {}]]


def sdpose_frames(model, vae, image, batch=16):
    """image (N,H,W,3) -> POSE_KEYPOINT frames (one person per frame, the whole image)."""
    import comfy.sample
    import comfy.utils
    dm = model.model.diffusion_model
    if not hasattr(dm, "heatmap_head"):
        raise ValueError("[PLS] Control Map: this checkpoint has no pose head -- pose needs the SDPose "
                         "wholebody checkpoint (models/checkpoints).")
    head = dm.heatmap_head
    mw, mh = int(head.heatmap_size[0]) * 4, int(head.heatmap_size[1]) * 4
    N, H, W = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
    feat = {}

    def grab(h, hsp, transformer_options):
        if h.shape[1] == 640:                 # the wholebody head reads the last 640-channel decoder feature
            feat["x"] = h.clone()
        return h, hsp

    m = model.clone()
    m.model_options["transformer_options"] = {"patches": {"output_block_patch": [grab]}}
    ctx = null_context()
    frames = []
    pbar = comfy.utils.ProgressBar(N)
    for a in range(0, N, int(batch)):
        part = image[a:a + int(batch), ..., :3]
        method = "area" if (mh <= H and mw <= W) else "bilinear"
        x = comfy.utils.common_upscale(part.movedim(-1, 1).float(), mw, mh, upscale_method=method,
                                       crop="disabled").movedim(1, -1)
        lat = vae.encode(x)
        feat.clear()
        comfy.sample.sample(m, noise=torch.zeros_like(lat), steps=1, cfg=1.0, sampler_name="euler",
                            scheduler="simple", positive=ctx, negative=ctx, latent_image=lat, disable_noise=True,
                            disable_pbar=True)
        if "x" not in feat:
            raise RuntimeError("[PLS] Control Map: SDPose gave no 640-channel feature -- not an SDPose checkpoint?")
        kps, scs = head(feat["x"])
        for kp, sc in zip(kps, scs):
            kp = np.array(kp, np.float32).copy()
            bad = kp[..., 0] < 0
            kp[..., 0] = kp[..., 0] * (W / float(mw))
            kp[..., 1] = kp[..., 1] * (H / float(mh))
            kp[bad] = -1
            frames.append({"canvas_width": W, "canvas_height": H,
                           "people": [wholebody_to_openpose(kp, np.asarray(sc, np.float32), W, H)]})
        pbar.update(int(part.shape[0]))
    return frames


# ------------------------------------------------------------------------------ Depth Anything 3
def da3_lib():
    try:
        from . import ph_da3_lib
    except ImportError:
        import ph_da3_lib
    return ph_da3_lib


def depth_frames(model_patcher, image, resolution=504):
    """image (N,H,W,3) -> depth IMAGE, near = bright, per frame between its 1 % / 99 % depth (sky black)."""
    from comfy.ldm.depth_anything_3 import preprocess as P
    g = da3_lib().run(model_patcher, image[..., :3], int(resolution), "upper_bound_resize", {"mode": "mono"})
    depth, sky = g["depth"], g.get("sky")
    norm = torch.stack([P.normalize_depth_v2_style(depth[i], sky[i] if sky is not None else None)
                        for i in range(depth.shape[0])], 0)
    return norm.clamp(0, 1).unsqueeze(-1).repeat(1, 1, 1, 3).float().contiguous()
