# v389 -- 3.89.0: the Cine Timeline, every node up to date, the fox reworked, the legacy out

**Restart ComfyUI.** 41 nodes (one new: ⬡ Polyhedron Cine Timeline). No node was
renamed or removed; old workflows load unchanged. One widget was appended at the
end of a node (Load CLIP: `weight_dtype`), which saved workflows tolerate by design.
Baselines: `NODE_IDS_baseline_v389.txt`, `WIDGET_ORDER_baseline_v389.txt`.

## New: ⬡ Polyhedron Cine Timeline

A whole short H3 film in one node: shots on a time axis, the official H3 prompt
they make, and the rendered clips as tracks you can cut, heal, retime, unlink
(picture / sound) and export. It fills the CLIP Text Encode as a form
(`external_mode = fill {pos_N}`): the fixed prompt parts stay in the encoder's
segments, the timeline places them and carries what changes per shot. Runs land
in one folder per day (`output/<date>/PH_<time>_<project>_whole_*.mp4`) and on the
timeline's tracks; a new run opens a new track on top, as tall as V1, showing
its real frames and its waveform -- also after a reload. See the README chapter.

## Every node brought up to date

All 40 nodes of v385 carry the current development state, among others:

- **CLIP Text Encode** -- `external_mode = fill {pos_N}`.
- **Load CLIP** -- `weight_dtype` (appended). **Load Model / Load VAE / VAE Codec /
  Empty Latent / Power Upscale** call no ComfyUI core node classes any more; the few
  rules they need (Flux2 empty latent and schedule, the tiled model upscale) are
  written in the pack.
- **Interpolate** -- a run never downloads: a missing RIFE checkpoint is offered by
  the node with a **Download** button (https, size and sha256 checked).
- **Sampler, LoRA Stack / Engine, Save, Media Loader** -- the fixes since v385.

## Fixed

- **Two routes that never existed in this release**: the Seed's noise preview
  (`/uls/noise/preview`) and the Empty Latent's presets (`/pls/latent_presets`)
  were called by their frontends since v371 and answered 404. They now have their
  own module, `nodes/ph_basics_routes.py`. A guard now checks every endpoint any
  frontend file calls against the routes this build registers.
- **Media Loader routes**: a media folder pinned at a drive root (`D:\`) answered
  404; `/uls/media/proc_count` now stands behind the LAN lock like the other folder
  routes; videos are streamed (`FileResponse`) instead of being read whole into
  memory on the server's event loop; thumbnails and building / deleting a sequence
  run in worker threads.
- **LoRA Stack routes**: the LoRA preview video is streamed the same way; the LoRA
  list is built in a worker thread (2000 LoRAs on a network drive held the server
  for seconds).
- The TAE preview install refuses while downloads are locked.
- **Cine Timeline paths**: bound beyond localhost (`--listen`), a clip "read where it
  lies" (an absolute path) is refused unless `ULS_ALLOW_LAN_BROWSE=1` opens it -- the
  same lock as the Media Loader's folder routes; a dropped copy in `input/` still
  works. A copy's relative path can no longer climb out of `input/` (`../`).

## Removed (legacy)

- `uls_preview_gen.py`, the standalone CLI preview generator, and with it the
  `cli` extra (Pillow, requests) and the start-up note about them.
- `docs/`: the v372 user manual (it described the pack of fourteen releases ago --
  the README is the current documentation) and the changelog archive (the history
  is in git).
- `MAINTAINING.md` and the rule it stated (shared files are never edited); the
  release guard no longer pins `uls_routes.py` byte for byte.
- `COMPATIBILITY.md` rewritten: current, short, English.
- Version-number pins inside three guards (the version triple is checked once).
- Two images only the old manual used; one unused developer probe.
- The registration file's comments: the release-by-release narration is gone, the
  code is unchanged.

## Example workflow: the fox, reworked

`example_workflows/polyhedron_minimax_h3_text_to_video.json` is a complete H3 film
workflow built around the Cine Timeline: three shots of a red fox in a snowy
clearing, upscale and interpolation branches behind switches, filters, the token
and LoRA diagnostics. Still built only from this pack. The LoRA Stack ships empty;
the template has no NAG node (H3 has no cross-attention -- the in-canvas note says
why). The workflow is the one of a real run: `assets/fox_h3_run.mp4` is that film,
rendered from exactly this graph (it also carries the workflow -- drop it on the
canvas), and the template's preview image is a frame of it.

## Not in this release

The Stack's group **Bake** (a feature, not a fix) stays out, as before.

## Checked

See the release notes of the upload: GATE-1, pyflakes, `node --check`, the full
suite, the server on ComfyUI 0.39.2 / frontend 1.53.10, every node created fresh
under both renderers with a lossless save -> load, and the fox in both renderers.
