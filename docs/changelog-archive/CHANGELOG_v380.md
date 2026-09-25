# v380 -- Power Upscale refine for every model, progress everywhere, Filter auto modes

This is the first feature release since the pack returned to the Registry with
3.79.0. It carries the work of the internal cuts v965-v1013 for the nodes that
are public. No node was added or removed; every change to a saved workflow is
an append (see "Loading old workflows" below).

**After updating, reload the ComfyUI tab with Ctrl+Shift+R** (new widgets on
two nodes, and a lot of new frontend code).

## Power Upscale

- **Refine order for every model.** Two new widgets at the end of the node:
  `refine_order` = `after pixel` (default: ESRGAN first, the model refines on
  the big canvas), `before pixel` (the model refines first, ESRGAN runs ONCE
  behind the last decode) or `off` (no model pass, pixel path only, scaled by
  `final_upscale_by`). `audio_stream` = keep / denoise -- only read by joint
  audio-video models, ignored otherwise.
- **`sigmas` input reaches the tile refine too.** Flow-curve models start
  exactly at the requested denoise; eps-curve models run the last
  round(steps x denoise) steps of the curve.
- **Exact start.** `denoise 0.30` used to begin at sigma 0.469; it now begins
  where the curve says. `cfg` steps in 0.01.
- **Encode cache.** One entry, keyed by a content fingerprint: repeating a run
  on the same input skips ESRGAN + encode (~8 minutes saved on a 4K frame in
  the measurement). **Pixel cache** on top of it (35 % of free RAM, full
  precision or nothing).
- **The clock.** A plan before the first silent second, a heartbeat every
  15 s (console, HUD and bar), learned rates (EMA) that survive restarts.
  VRAM peaks per phase in the console. The VAE is read, not assumed.
- **A verdict for every model** in the status line.
- **Old saves load unchanged.** The two DOM widgets at the end of
  `widgets_values` broke the length check on load and every slider came back
  shifted; the tail is now split off by type before the marker decides.

## Progress bars and console tempo

New module `nodes/ph_progress.py`: a green bar, a console line with tempo and
ETA (at most every 3 s for countable work, a heartbeat every 15 s for a silent
call), and from the second run of a size class an estimate up front
("estimate ~0:05 (learned)"). Instrumented in this release: Power Upscale,
VAE Codec, Interpolate, Save (per frame + flush), Media Loader (PyAV/cv2),
LoRA merge, Fast Upscale, Load Model / CLIP / VAE (MB on disk), Outpaint and
Reference encode, Filter, Composite and Analyzer (bar only).

The learned rates live in `<ComfyUI>/user/polyhedron/rates.json`. Harmless,
deletable; interrupted runs learn nothing.

## Filter

- **Auto modes** (`auto_mode`): auto levels / auto colour / both, computed on
  the preview. **Pick white**: click a neutral in the preview, the temperature
  and tint follow.
- **Detail**: `sharpen_threshold` keeps flat areas out of the sharpen;
  `detail_amount` with a new optional `detail_source` input blends detail from
  a second image (a sharper pass, a different denoise).

## LoRA Stack

- **Group strength** (v981): one multiplier per group, applied to model and
  CLIP weights alike; the group's CLIP weight is read from the row first.
- **Group energy cap** (v983): an overlap-neutral cap per group, measured on
  the loaded LoRAs (one Gram product per layer, cached per file set).
- **Schedule-bound LoRAs** (v989, `nodes/uls_sched_loras.py`): a LoRA that
  only works on its own trained sigma grid (HyperFlow 8-step today) is
  recognised by file name; the Sampler says so when it is active without that
  grid connected, instead of rendering the soft or burnt result that makes a
  LoRA look broken.
- **Merge policy / Analyzer** (v985/v986): the Analyzer reports what it
  measured and what it did not; LoRAs with mixed key naming merge
  (`nodes/uls_merge_check.py`, `nodes/uls_overlap_math.py`).
- The suite's own popups no longer get the browser's context menu on top of
  them (v989).

## Media Loader

- Grid order: newest first / oldest first / name (natural), with `ctime`
  carried by the listing (v968); exact mtime ties are broken by natural name
  (v969); one more button in the pager bar; AVIF is listed without being
  decoded.

## Sigma schedules

- Curve preview and the terminal-zero rule for typed grids (v976/v984). The
  Sigma List node itself is **not** part of this release; its class rides
  along in `wan_sigma_schedule.py` unregistered (no menu entry, no route).

## CLIP Text Encode

- **Search row with live marks** (v992): find text in the prompt, matches
  marked as you type.
- **Token Counter** (v997): a per-block map of the prompt.

## Frontends

The frontend files of this release load on ComfyUI frontend **1.49.6** (the
pair this tree is measured on) and on **1.53.6** (v965: the widget base class
changed there and every pack node failed to create; one pack now serves both).

## Loading old workflows

`WIDGET_ORDER_baseline_v380.txt` replaces v375. Exactly two lines changed,
both appended at the end:

- `ULSFilter` + `auto_mode, sharpen_threshold, detail_amount`
- `ULSPowerUpscale` + `refine_order, audio_stream`

`NODE_IDS_baseline_v375.txt` is unchanged: 37 nodes, no id renamed.

## Not in this release

The group **Bake** (write one group into a single LoRA file) stays internal;
the public `uls_node.js` / `uls_stack_dom.js` hand no bake callback to the
group popup, so the section is not rendered.

## Tests

`tests/` and `tools/` stay in the repository and out of the Registry package
(`.comfyignore`). 184 guard scripts run green against this tree
(`python tests/test_x.py`, one file at a time). Three public guards were
re-grounded rather than loosened: v302 (the group's CLIP weights are now
collected in `_group_scaled`), v577 (census 35: `ULSSigmaList` joins the
statically guarded set), v633 and the two v968 guards read the public route
module `ph_media_routes.py` / `ph_filter_routes.py`. The version-triple pins
(v348, v351, v352) move to 3.80.0 / v380.
