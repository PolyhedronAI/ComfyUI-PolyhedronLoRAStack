# -*- coding: ascii -*-
"""v1048 -- MiniMax H3 quality nodes, taken over from the MiniMax H3 Director
ecosystem (Frank 01.10.: "take everything sensible from the example node for
the best possible quality").

  ULSH3LatentUpscale   the LEARNED H3 latent upscaler (3D net): generate small,
                       lift the latent, refine at the target size with a second
                       sampler pass (low denoise). Skips the slow decode ->
                       pixel upscale -> encode round trip of the ~5B H3 VAE and
                       avoids the ghosting of a plain bilinear latent resize.
                       The AV latent's AUDIO stream passes through untouched.
  ULSFreeVRAM          a pass-through that unloads the models and empties the
                       CUDA cache at a chosen point (before a second pass, an
                       upscale, a face fix) -- the Director's "clear VRAM
                       before the second sample".
  ULSH3SemanticBridge  an optional adapter on the H3 text conditioning (a small
                       distilled MLP: layout, space, counting -- or BUNNY:
                       action ownership, several characters). Experimental:
                       distilled on FL2VA; on ref2va it is forced-compatible.

Third-party code (licences in docs/third_party/):
  * the 3D upscaler network, its weight-format detection and the H3 latent
    statistics: LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler (MIT,
    Copyright (c) 2026 LBH-123-AI). Changes: the unused attention block and
    the einops dependency removed; fp32 only (see below); the AV latent split
    and repacked around it.
  * the fp32 rule, the AV handling and the Semantic Bridge math:
    AIMixer/ComfyUI_MiniMaxH3_Director (Apache-2.0). The student formula is
    from speach1sdef178/MiniMax-H3-Semantic-Bridge; the weights are NOT
    shipped -- the user downloads them.

fp32 ONLY: the Director measured that this net's GroupNorm in bf16 can
collapse to NaN / mud and decode as a brown static frame; the upscaler's own
default is fp16. We do not offer the risky precisions.
"""
import gc
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

VAE_DOWNSAMPLE = 16
UPSCALE_FOLDER = "latent_upscale_models"
BRIDGE_FOLDER = "semantic_bridge"
NO_UPSCALER = "(put an H3 latent upscaler into models/latent_upscale_models)"
NO_ADAPTER = "(put a Semantic Bridge adapter into models/semantic_bridge)"
SIZE_MODES = ("scale", "width x height", "megapixels")

# --------------------------------------------------------------- H3 latent statistics (LBH-123-AI, MIT)
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
]


def _norm(device, dtype):
    m = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    s = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return m, s


# --------------------------------------------------------------- the 3D network (LBH-123-AI, MIT)
def _gn(ch):
    return nn.GroupNorm(32, ch)


def _zero(m):
    for p in m.parameters():
        p.detach().zero_()
    return m


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(_gn(channels), nn.SiLU(), nn.Conv3d(channels, self.out_channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels))
        self.out_norm = _gn(self.out_channels)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(p=dropout),
                                        _zero(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)))
        self.skip = nn.Conv3d(channels, self.out_channels, 1) if self.out_channels != channels else nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        e = self.emb_layers(emb).type(h.dtype)
        while e.dim() < h.dim():
            e = e[..., None]
        scale, shift = torch.chunk(e, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        return self.skip(x) + self.out_layers(h)


class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.norm = _gn(channels)
        self.dwconv = nn.Conv3d(channels, channels, kernel_size=(kernel_size, 1, 1), padding=(pad, 0, 0), groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class LatentResizer3D(nn.Module):
    """24-channel H3 video latent [B, C, T, H, W] -> [B, C, T, H', W']."""

    CHUNK = 32

    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12, channels=512, dropout=0.1,
                 temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        emb = 64
        self.embed = nn.Sequential(nn.Linear(1, emb), nn.SiLU(), nn.Linear(emb, emb))

        def blocks(n):
            out = nn.ModuleList()
            for b in range(n):
                out.append(ResBlockEmb3D(channels, emb, dropout))
                if temporal_every > 0 and b % temporal_every == 0:
                    out.append(TemporalConv(channels, temporal_kernel))
            return out
        self.in_blocks = blocks(in_blocks)
        self.out_blocks = blocks(out_blocks)
        self.norm_out = _gn(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def temporal_kernel(self):
        for b in self.in_blocks:
            if isinstance(b, TemporalConv):
                return int(b.dwconv.weight.shape[2])
        return 0

    def forward(self, x, scale, size, chunking=True):
        if tuple(size) == tuple(x.shape[-3:]):
            return x
        B, C, T, H, W = x.shape
        if not chunking or T <= self.CHUNK:
            return self._seg(x, scale, size)
        # temporal chunks with replicate padding and weighted overlap (no end-frame flicker)
        ov = self.temporal_kernel()
        xp = F.pad(x, (0, 0, 0, 0, ov, ov), mode="replicate")
        out = torch.zeros(B, C, T, size[-2], size[-1], device=x.device, dtype=x.dtype)
        wsum = torch.zeros(1, 1, T, 1, 1, device=x.device, dtype=x.dtype)
        start = 0
        while start < T:
            s0, s1 = start, min(T, start + self.CHUNK)
            o0, o1 = max(0, s0 - ov), min(T, s1 + ov)
            lo, hi = max(0, o0 - ov), min(T + 2 * ov, o1 + ov)
            seg = self._seg(xp[:, :, lo:hi].contiguous(), scale, (hi - lo, size[-2], size[-1]))
            a = (o0 + ov) - lo
            valid = seg[:, :, a:a + (o1 - o0)]
            n = o1 - o0
            w = torch.ones(n, device=x.device, dtype=x.dtype)
            if s0 > o0:
                k = s0 - o0
                w[:k] = torch.arange(1, k + 1, device=x.device, dtype=x.dtype) / (k + 1)
            if o1 > s1:
                k = o1 - s1
                w[-k:] = torch.arange(k, 0, -1, device=x.device, dtype=x.dtype) / (k + 1)
            out[:, :, o0:o1] += valid * w.view(1, 1, n, 1, 1)
            wsum[:, :, o0:o1] += w.view(1, 1, n, 1, 1)
            start += self.CHUNK
            del seg, valid
        return out / wsum.clamp(min=1e-8)

    def _seg(self, x, scale, size):
        emb = self.embed(torch.tensor([[float(scale) - 1.0]], dtype=x.dtype, device=x.device))
        x = self.conv_in(x)
        for b in self.in_blocks:
            x = b(x, emb.expand(x.shape[0], -1)) if isinstance(b, ResBlockEmb3D) else b(x)
        x = F.interpolate(x, size=tuple(size), mode="trilinear", align_corners=False)
        for b in self.out_blocks:
            x = b(x, emb.expand(x.shape[0], -1)) if isinstance(b, ResBlockEmb3D) else b(x)
        return self.conv_out(F.silu(self.norm_out(x)))


def detect_arch(sd):
    """The net's shape from its weights (LBH-123-AI's detection, MIT)."""
    cfg = {"in_channels": 24, "in_blocks": 12, "out_blocks": 12, "channels": 512,
           "dropout": 0.1, "temporal_every": 2, "temporal_kernel": 5}
    if "conv_in.weight" in sd:
        cfg["in_channels"] = int(sd["conv_in.weight"].shape[1])
        cfg["channels"] = int(sd["conv_in.weight"].shape[0])
    ins, outs, tmp = set(), set(), False
    for k in sd:
        m = re.match(r"in_blocks\.(\d+)\.in_layers\.", k)
        if m:
            ins.add(int(m.group(1)))
        m = re.match(r"out_blocks\.(\d+)\.in_layers\.", k)
        if m:
            outs.add(int(m.group(1)))
        if k.endswith("dwconv.weight"):
            tmp = True
            cfg["temporal_kernel"] = int(sd[k].shape[2])
    if ins:
        cfg["in_blocks"] = len(ins)
    if outs:
        cfg["out_blocks"] = len(outs)
    if not tmp:
        cfg["temporal_every"] = 0
    return cfg


def _state_dict(path):
    if path.lower().endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path, device="cpu")
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and isinstance(sd.get("model"), dict):
        sd = sd["model"]
    if isinstance(sd, dict) and isinstance(sd.get("state_dict"), dict):
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        raise ValueError("not a state dict: %s" % os.path.basename(path))
    f8 = getattr(torch, "float8_e4m3fn", None)
    sd = {k: (v.to(torch.float16) if f8 is not None and v.dtype == f8 else v) for k, v in sd.items()}
    if any(k.startswith("upscaler.") for k in sd):
        sd = {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    return sd


_UP_CACHE = {}


def load_upscaler(path):
    net = _UP_CACHE.get(path)
    if net is not None:
        return net
    sd = _state_dict(path)
    cfg = detect_arch(sd)
    if cfg["in_channels"] != 24:
        raise ValueError("%s is not a MiniMax H3 latent upscaler (it works on %d channels, H3 has 24) -- "
                         "the folder is shared with other models' upscalers" % (os.path.basename(path), cfg["in_channels"]))
    net = LatentResizer3D(**cfg)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:
        raise ValueError("%s does not fit the 3D H3 upscaler (%d weight(s) missing, e.g. %s) -- the 2D variant is not supported"
                         % (os.path.basename(path), len(missing), missing[0]))
    net.eval().requires_grad_(False)
    _UP_CACHE.clear()                  # one at a time -- it is not small
    _UP_CACHE[path] = net
    return net


# --------------------------------------------------------------- pure size math
def target_latent_hw(h_in, w_in, mode, scale=2.0, width=1344, height=768, megapixels=1.0, align=32):
    """-> (h_out, w_out, effective scale) for a latent of h_in x w_in (x16 px).
    Upscaling only; the pixel size snaps to `align` (32 = the H3 canvas)."""
    d = VAE_DOWNSAMPLE
    if mode == "scale":
        wp, hp, eff = w_in * d * float(scale), h_in * d * float(scale), float(scale)
    elif mode == "width x height":
        wp, hp = float(width), float(height)
        eff = (wp / (w_in * d) + hp / (h_in * d)) / 2.0
    elif mode == "megapixels":
        px = float(megapixels) * 1024 * 1024
        ar = w_in / float(h_in)
        hp = (px / ar) ** 0.5
        wp = hp * ar
        eff = (wp / (w_in * d) + hp / (h_in * d)) / 2.0
    else:
        raise ValueError("size_mode %r is not one of %s" % (mode, ", ".join(SIZE_MODES)))
    a = max(1, int(align))
    wpa, hpa = round(wp / a) * a, round(hp / a) * a
    w_out = max(1, int(round(wpa / d)))
    h_out = max(1, int(round(hpa / d)))
    if w_out < w_in or h_out < h_in:
        raise ValueError("the H3 latent upscaler only enlarges: %dx%d px -> %dx%d px"
                         % (w_in * d, h_in * d, w_out * d, h_out * d))
    return h_out, w_out, max(1.0, eff)


def split_av(samples):
    """-> (video [B,C,T,H,W], rest list, how) -- the AV NestedTensor's streams,
    a plain video tensor, or an image latent [B,C,H,W]."""
    if getattr(samples, "is_nested", False) or (not torch.is_tensor(samples) and hasattr(samples, "unbind")):
        parts = list(samples.unbind())
        return parts[0], parts[1:], "nested"
    if torch.is_tensor(samples) and samples.dim() == 5:
        return samples, [], "video"
    if torch.is_tensor(samples) and samples.dim() == 4:
        return samples.unsqueeze(2), [], "image"
    raise ValueError("H3 latent upscale: a video latent [B,C,T,H,W] or the H3 AV latent is needed, got %r"
                     % (tuple(getattr(samples, "shape", ())),))


def join_av(video, rest, how):
    if how == "nested":
        import comfy.nested_tensor
        return comfy.nested_tensor.NestedTensor(tuple([video] + list(rest)))
    if how == "image":
        return video.squeeze(2)
    return video


def upscale_video_latent(net, video, h_out, w_out, scale, chunking=True, device=None):
    """The lift itself, fp32, on `device` (default: CUDA when there is one)."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float32
    net = net.to(device=dev, dtype=dt)
    mean, std = _norm(dev, dt)
    try:
        with torch.inference_mode():
            x = (video.to(device=dev, dtype=dt) - mean) / std
            y = net(x, scale, (int(video.shape[2]), int(h_out), int(w_out)), chunking=chunking)
            y = y * std + mean
        if not torch.isfinite(y).all():
            raise ValueError("the upscaler returned NaN/inf -- the weights do not fit or are broken")
        return y.to(device="cpu", dtype=video.dtype).contiguous()
    finally:
        net.to("cpu")


def strip_keyframes(positive):
    """-> (the conditioning without minimax_keyframes / minimax_frame_count, how many
    keyframe lists were dropped). A copy; the input stays as it was."""
    if positive is None:
        return None, 0
    out, n = [], 0
    for item in positive:
        meta = item[1] if len(item) > 1 and isinstance(item[1], dict) else {}
        if "minimax_keyframes" in meta:
            n += 1
            meta = {k: v for k, v in meta.items() if k not in ("minimax_keyframes", "minimax_frame_count")}
        out.append([item[0], meta] + list(item[2:]))
    return out, n


def _files(folder):
    try:
        import folder_paths
        if folder not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(folder, os.path.join(folder_paths.models_dir, folder))
        names = [n for n in folder_paths.get_filename_list(folder) if n.lower().endswith((".safetensors", ".pt", ".pth"))]
    except Exception:
        names = []
    return names


def upscaler_names():
    names = _files(UPSCALE_FOLDER)
    # H3 ones first (the folder is shared with LTX2 and others)
    names.sort(key=lambda n: (0 if ("h3" in n.lower() or "minimax" in n.lower()) else 1, n.lower()))
    return names or [NO_UPSCALER]


def _full(folder, name):
    import folder_paths
    p = folder_paths.get_full_path(folder, name)
    if not p or not os.path.isfile(p):
        raise ValueError("model file not found: %s/%s" % (folder, name))
    return p


def _free(unload=True):
    try:
        import comfy.model_management as mm
        if unload:
            mm.unload_all_models()
        mm.soft_empty_cache(True)
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    gc.collect()


class ULSH3LatentUpscale:
    """Lift the MiniMax H3 latent with the learned 3D upscaler."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "The H3 AV latent of the FIRST pass (sampled small). Its audio "
                                                 "passes through untouched."}),
                "upscaler": (upscaler_names(), {"tooltip": "The learned H3 latent upscaler (3D), from "
                    "models/latent_upscale_models -- e.g. LBH-123-AI/Minimax_h3_latent_Upscaler on Hugging Face. "
                    "H3 files are listed first; the folder is shared with other models' upscalers."}),
                "size_mode": (list(SIZE_MODES), {"default": "scale", "tooltip": "How the new size is given: a "
                    "factor, a width x height in pixels, or megapixels (the aspect stays)."}),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05,
                                    "tooltip": "size_mode = scale: the factor (1.0 - 4.0)."}),
                "width": ("INT", {"default": 1344, "min": 64, "max": 8192, "step": 32,
                                  "tooltip": "size_mode = width x height: the target width in pixels."}),
                "height": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 32,
                                   "tooltip": "size_mode = width x height: the target height in pixels."}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.05,
                                         "tooltip": "size_mode = megapixels: the target size; the aspect stays."}),
                "align": ("INT", {"default": 32, "min": 16, "max": 256, "step": 16,
                                  "tooltip": "The pixel grid the size snaps to. 32 = the H3 canvas -- keep it."}),
                "temporal_chunking": ("BOOLEAN", {"default": True, "tooltip": "Lift long clips in 32-step chunks "
                    "with blended overlaps (less VRAM, no flicker at the clip's end). Off: one pass over the whole "
                    "clip (short clips)."}),
                "free_vram_first": ("BOOLEAN", {"default": True, "tooltip": "Unload the models and empty the CUDA "
                    "cache BEFORE the lift (the sampler of the first pass is done) -- the upscaler runs in fp32."}),
            },
            # v1049: a SOCKET (the widget order is untouched)
            "optional": {
                "positive": ("CONDITIONING", {"tooltip": "The first pass's positive -- it comes out at `positive` "
                    "WITHOUT keyframes (a first/last frame was pinned at the first pass's size and is in the latent "
                    "already; at the new size it would not fit). References stay. Wire the second pass's sampler "
                    "to that output."}),
            },
        }

    # v1049: positive APPENDED (output slots are positional in saved workflows)
    RETURN_TYPES = ("LATENT", "STRING", "CONDITIONING")
    RETURN_NAMES = ("latent", "info", "positive")
    FUNCTION = "run"
    CATEGORY = "Polyhedron/MiniMax"
    DESCRIPTION = ("The learned MiniMax H3 latent upscaler (3D): sample SMALL, lift the latent here, then a second "
                   "sampler pass at the target size with a low denoise (the Director: 3 steps, beta, 0.2) -- faster than sampling big, "
                   "no VAE round trip, no ghosting of a plain latent resize. The audio stream passes through. "
                   "Runs in fp32 (bf16 can collapse to a brown frame). Weights: LBH-123-AI/Minimax_h3_latent_Upscaler "
                   "(MIT) into models/latent_upscale_models.")

    def run(self, latent, upscaler, size_mode, scale, width, height, megapixels, align,
            temporal_chunking, free_vram_first, positive=None):
        tag = "[PLS] H3 Latent Upscale"
        pos, dropped = strip_keyframes(positive)
        if not upscaler or upscaler.startswith("("):
            raise ValueError("%s: no upscaler file -- put the H3 latent upscaler (e.g. LBH-123-AI/"
                             "Minimax_h3_latent_Upscaler, the 3D .safetensors) into models/latent_upscale_models "
                             "and press R in the browser" % tag)
        video, rest, how = split_av(latent["samples"])
        b, c, t, h_in, w_in = video.shape
        if c != 24:
            raise ValueError("%s: the latent has %d channels -- this is not a MiniMax H3 latent (24)" % (tag, c))
        h_out, w_out, eff = target_latent_hw(h_in, w_in, size_mode, scale, width, height, megapixels, align)
        if (h_out, w_out) == (h_in, w_in):
            info = "%s: %dx%d px is already the target size -- passed through" % (tag, w_in * 16, h_in * 16)
            print(info)
            return (latent, info, pos)
        if free_vram_first:
            _free(True)
        net = load_upscaler(_full(UPSCALE_FOLDER, upscaler))
        up = upscale_video_latent(net, video, h_out, w_out, eff, chunking=bool(temporal_chunking))
        out = {k: v for k, v in latent.items() if k not in ("samples", "noise_mask")}
        out["samples"] = join_av(up, rest, how)
        _free(False)
        info = ("%s: %dx%d px -> %dx%d px (latent %dx%d -> %dx%d, x%.2f, %d time steps%s) with %s"
                % (tag, w_in * 16, h_in * 16, w_out * 16, h_out * 16, w_in, h_in, w_out, h_out, eff, t,
                   ", audio passed through" if how == "nested" else "", upscaler))
        if dropped:
            info += "\n  positive: %d keyframe(s) dropped for the second pass (they are in the latent)" % dropped
        info += ("\n  next: a second sampler pass at this size -- the Director's setting: 3 steps, beta, "
                 "denoise 0.2, the same model, LoRAs and prompt")
        print(info.replace("\n", "\n[PLS] "))
        return (out, info, pos)


# --------------------------------------------------------------- Free VRAM
class _AnyType(str):
    def __ne__(self, other):
        return False


_any = _AnyType("*")


class ULSFreeVRAM:
    """A pass-through that frees the GPU at that point of the graph."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (_any, {"tooltip": "Anything -- it comes out unchanged. Wire it where the GPU should "
                                            "be freed (e.g. the first pass's latent before the upscale)."}),
                "mode": (["unload models + empty cache", "empty cache only"], {"default": "unload models + empty cache",
                         "tooltip": "unload: every model leaves the GPU (the next node loads its own again); "
                                    "empty cache: only the free blocks go back to the driver."}),
            },
        }

    RETURN_TYPES = (_any,)
    RETURN_NAMES = ("value",)
    FUNCTION = "run"
    CATEGORY = "Polyhedron/Utils"
    DESCRIPTION = ("Frees the GPU at this point of the graph and passes its input through: before a second "
                   "sampler pass, an upscale or a face fix (the Director's 'clear VRAM before the second sample').")

    # no IS_CHANGED: a node that always runs would make everything after it run
    # again (no cache). Cached input = cached downstream = nothing to free.

    def run(self, value, mode):
        before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        _free(mode.startswith("unload"))
        after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        print("[PLS] Free VRAM (%s): %.2f GB -> %.2f GB allocated" % (mode, before / 2 ** 30, after / 2 ** 30))
        return (value,)


# --------------------------------------------------------------- Semantic Bridge (AIMixer, Apache-2.0)
HIDDEN_DIM, STUDENT_DIM = 5120, 512


class SemanticStudent(nn.Module):
    """5120 -> 512 SiLU -> 512 SiLU -> 5120 (fc1/fc2/fc3)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(HIDDEN_DIM, STUDENT_DIM)
        self.fc2 = nn.Linear(STUDENT_DIM, STUDENT_DIM)
        self.fc3 = nn.Linear(STUDENT_DIM, HIDDEN_DIM)

    def forward(self, x):
        return self.fc3(F.silu(self.fc2(F.silu(self.fc1(x)))))


def bridge_hidden(hidden, student, alpha, match_mag=True):
    """C = H + alpha * (S' - H), S = student(rms_norm(H)), S' = S at H's norm."""
    if hidden.dim() != 3 or hidden.shape[-1] != HIDDEN_DIM:
        raise ValueError("the Semantic Bridge needs H3 text tokens [B, T, %d], got %r" % (HIDDEN_DIM, tuple(hidden.shape)))
    x = hidden.float()
    st = student.to(device=x.device, dtype=torch.float32)
    with torch.no_grad():
        s = st(x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6))
        if match_mag:
            s = s * (x.norm(dim=-1, keepdim=True) / (s.norm(dim=-1, keepdim=True) + 1e-8))
        out = x + float(alpha) * (s - x)
    return out.to(dtype=hidden.dtype)


_SB_CACHE = {}


def load_student(path):
    st = _SB_CACHE.get(path)
    if st is None:
        st = SemanticStudent()
        st.load_state_dict(_state_dict_plain(path), strict=True)
        st.eval().requires_grad_(False)
        _SB_CACHE.clear()
        _SB_CACHE[path] = st
    return st


def _state_dict_plain(path):
    if path.lower().endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device="cpu")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return sd["state_dict"] if isinstance(sd, dict) and isinstance(sd.get("state_dict"), dict) else sd


def _door():
    try:
        from . import ph_world_weights as WW
    except ImportError:  # pragma: no cover - direct import in the guards
        import ph_world_weights as WW
    return WW


def adapter_names():
    """v1204: the adapters on disk (bare names, as saved workflows carry them) + the door's registry adapters that are
    NOT on disk as 'name . size . not installed' -- picking one is what makes the node's line offer its Download
    (Frank 21.10.: an offer appears only when the option is really chosen)."""
    have = sorted(_files(BRIDGE_FOLDER))
    try:
        WW = _door()
        missing = [WW.label(n, None) for n, spec in WW.REGISTRY.items() if spec.get("set") == "semantic_bridge" and n not in have]
    except Exception:
        missing = []
    return (have + missing) or [NO_ADAPTER]


class ULSH3SemanticBridge:
    """An adapter on the H3 text tokens (experimental)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"tooltip": "The positive from CLIP Text Encode (the minimax CLIP)."}),
                "adapter": (adapter_names(), {"tooltip": "From models/semantic_bridge: the original (speach1sdef178/"
                    "MiniMax-H3-Semantic-Bridge: layout, space, counting) or BUNNY (JOKER141/BUNNY_H3_Conditioning_Bridge: "
                    "who does what, several characters). One bridge only -- never chain two."}),
                "alpha": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "How far the tokens move toward the adapter's version (0 = off). 0.15 is the "
                                               "authors' default."}),
                "magnitude_match": ("BOOLEAN", {"default": True, "tooltip": "Keep each token's length (recommended)."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "info")
    FUNCTION = "run"
    CATEGORY = "Polyhedron/MiniMax"
    DESCRIPTION = ("EXPERIMENTAL: a small distilled adapter rewrites the H3 text tokens (H + alpha * (S - H)). The "
                   "original leans to layout, space and counting; BUNNY to who does what with several characters. "
                   "Distilled on FL2VA -- on ref2va it is forced-compatible, compare with it off. Weights are not "
                   "shipped: put them into models/semantic_bridge.")

    @classmethod
    def VALIDATE_INPUTS(cls, adapter=None):
        """v1204: a picked adapter changes its label after the Download ('... not installed' -> its bare name) -- only
        this input leaves Core's membership check; run() strips the label and refuses a file that is not there."""
        return True

    def run(self, conditioning, adapter, alpha, magnitude_match):
        tag = "[PLS] H3 Semantic Bridge"
        try:
            adapter = _door().strip(adapter)
        except Exception:
            pass
        if adapter and not adapter.startswith("(") and adapter not in _files(BRIDGE_FOLDER):
            raise ValueError("%s: %s is not installed -- the Download on the node's line fetches it (into "
                             "models/%s), or pick an adapter that is on disk" % (tag, adapter, BRIDGE_FOLDER))
        if not adapter or adapter.startswith("("):
            raise ValueError("%s: no adapter -- put the Semantic Bridge weights into models/semantic_bridge "
                             "and press R in the browser (or bypass this node)" % tag)
        if float(alpha) <= 0:
            info = "%s: alpha 0 -- passed through" % tag
            print(info)
            return (conditioning, info)
        st = load_student(_full(BRIDGE_FOLDER, adapter))
        out, n = [], 0
        for item in conditioning:
            hidden, meta = item[0], (item[1] if len(item) > 1 else {})
            if isinstance(meta, dict) and meta.get("pls_semantic_bridge"):
                out.append(item)
                continue
            m2 = dict(meta) if isinstance(meta, dict) else {}
            m2["pls_semantic_bridge"] = adapter
            out.append([bridge_hidden(hidden, st, alpha, bool(magnitude_match)), m2] + list(item[2:]))
            n += 1
        info = "%s: %s, alpha %.2f, %d conditioning item(s)%s" % (
            tag, adapter, float(alpha), n, "" if n else " -- already bridged, passed through")
        print(info)
        return (out, info)


NODE_CLASS_MAPPINGS = {"ULSH3LatentUpscale": ULSH3LatentUpscale, "ULSFreeVRAM": ULSFreeVRAM,
                       "ULSH3SemanticBridge": ULSH3SemanticBridge}
