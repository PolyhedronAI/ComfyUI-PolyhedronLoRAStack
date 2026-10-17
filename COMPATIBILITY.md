# Compatibility — Polyhedron Suite

## Runtime

- **Python** 3.10 or newer (`requires-python`).
- **ComfyUI**: checked on 0.39.2 with frontend 1.53.10, and on an older pinned
  core (frontend 1.49.6). The nodes call no ComfyUI core *node* classes, only the
  `comfy.*` model library; the few rules they need (the Flux2 empty latent and
  schedule, the tiled model upscale) are written in this pack.
- **Required packages: none.** Optional: `opencv-python` (video thumbnails in
  the Media Loader), `av` (PyAV, shipped by current ComfyUI builds), ffmpeg on
  the PATH or via `imageio-ffmpeg` (the Cine Timeline's strips, waveforms and
  export). A missing optional package costs one feature and never blocks a load.
- **Downloads** happen only on a click, over https, checked by size and sha256
  (Interpolate's RIFE checkpoints, the Sampler's preview decoders). A run never
  downloads, and the pack never installs packages.

## Renderers

Every node works under the classic canvas and under **Nodes 2.0** (the Vue
renderer). Panels that the classic canvas paints have a DOM view under Nodes
2.0; saved workflows load the same in both.

## Loading

Every node group imports on its own: a breaking ComfyUI change in one group logs
that group and skips it, the rest of the pack registers. Routes are registered
under the bare path and the `/api` alias.

## Running with `--listen`

Bound beyond localhost, the routes that read folders or files the client names
answer 403: the Media Loader's folder routes, and a Cine Timeline clip "read
where it lies" (a dropped copy in `input/` still works). Set
`ULS_ALLOW_LAN_BROWSE=1` to open them deliberately.

## Wan Bridge and kijai's WanVideoWrapper

The Bridge patches internal functions of WanVideoWrapper (`load_weights`,
`set_lora_params`, `remove_lora_from_module`). The patches are idempotent and
marker-based; when an expected symbol is missing after a WanVideoWrapper update
the Bridge says so once in the console instead of doing nothing silently.
`PLS_BRIDGE_VERBOSE=1` turns on its detailed log.

## What breaks first on updates

`comfy.lora.*` and `comfy.ldm.modules.attention` (Stack, Engine, NAG,
Attention), WanVideoWrapper internals (Bridge), and optional attention wheels
(SageAttention, Kitchen) — the Attention node lists only the modes that load on
the machine.

## Vendored third-party code

`nodes/vfi/rife_arch.py` — the IFNet frame-interpolation network from
**Practical-RIFE** (hzwer), by way of **ComfyUI-Frame-Interpolation**
(Fannovel16). MIT licence, copied verbatim, not modified. Attribution and
reasoning in `nodes/vfi/NOTICE.md`. Used by ⬡ Polyhedron Interpolate.

`nodes/vendor/taehv/taehv.py` — the TAEHV tiny video autoencoder from
**taehv** (madebyollin, Ollin Boer Bohan). MIT licence, copied verbatim, not
modified; the LICENSE file sits next to it, provenance in
`nodes/vendor/taehv/SOURCE.md`. Used by ⬡ Polyhedron Sampler for the
"Video · TAE (taeh3)" live preview.
