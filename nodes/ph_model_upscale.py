"""Polyhedron model upscale (v1062) -- an UPSCALE_MODEL over an IMAGE batch.

Written here instead of calling the core's ImageUpscaleWithModel node class
(Frank: the suite calls no Core node classes, only the comfy.* model library;
Core is GPL, the pack MIT). The pieces are the model library's own:
comfy.model_management (load the model's patcher, OOM test),
comfy.utils.tiled_scale (512 tiles, 32 overlap; on an out-of-memory error the
tile halves, down to 128). A fourth channel (RGBA) is taken off before the
model and scaled back on bilinearly, so a 3-channel model never sees alpha.

test_v1062 compares the result with the core node on a real spandrel model
(random weights): RGB on both cores, RGBA where the core handles it.

Pure ASCII.
"""
import torch

TILE = 512
OVERLAP = 32
MIN_TILE = 128


def _is_oom(exc):
    import comfy.model_management as mm
    fn = getattr(mm, "is_oom", None)
    if fn is not None:
        return bool(fn(exc))
    oom = getattr(mm, "OOM_EXCEPTION", torch.cuda.OutOfMemoryError)
    return isinstance(exc, oom)


def upscale_with_model(upscale_model, image):
    """IMAGE [B,H,W,C] -> IMAGE [B,H*s,W*s,C] through upscale_model (a spandrel
    descriptor with a .patcher, as the upscale-model loaders hand it on)."""
    import comfy.model_management as mm
    import comfy.utils
    alpha = None
    if image.shape[-1] == 4:
        alpha = image[..., 3:4]
        image = image[..., :3]
    scale = float(upscale_model.scale)
    device = upscale_model.patcher.load_device
    need = (512 * 512 * 3) * image.element_size() * max(scale, 1.0) * 384.0
    need += image.nelement() * image.element_size()
    mm.load_models_gpu([upscale_model.patcher], memory_required=need, force_full_load=True)
    x = image.movedim(-1, -3).to(device)
    out_device = mm.intermediate_device()
    tile = TILE
    while True:
        try:
            steps = x.shape[0] * comfy.utils.get_tiled_scale_steps(x.shape[3], x.shape[2], tile_x=tile,
                                                                   tile_y=tile, overlap=OVERLAP)
            pbar = comfy.utils.ProgressBar(steps)
            s = comfy.utils.tiled_scale(x, lambda a: upscale_model(a.float()), tile_x=tile, tile_y=tile,
                                        overlap=OVERLAP, upscale_amount=scale, pbar=pbar, output_device=out_device)
            break
        except Exception as exc:
            if not _is_oom(exc):
                raise
            tile //= 2
            if tile < MIN_TILE:
                raise
            print("[PLS] model upscale: out of memory -- tile %d" % tile)
    s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0).to(mm.intermediate_dtype())
    if alpha is not None:
        a = comfy.utils.common_upscale(alpha.movedim(-1, -3).to(s), s.shape[2], s.shape[1], "bilinear", "disabled")
        s = torch.cat((s, a.movedim(-3, -1)), dim=-1)
    return s


class ModelUpscaler:
    """The shape the call sites used (`(part,) = X.upscale(model, image)`)."""

    def upscale(self, upscale_model, image):
        return (upscale_with_model(upscale_model, image),)
