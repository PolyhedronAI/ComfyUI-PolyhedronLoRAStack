# v397 -- 3.97.0: the first -> last frame showcase shows its template's run and both keyframes; the CTE's search row stays in its slot; handbook v397

**Ctrl+F5** (no node, widget or route change). Carries v396 below.

## First -> last frame showcase

- `assets/flf_sitdown_run.mp4` is now the run of `polyhedron_minimax_h3_flf_sitdown` as it ships (Apply Control 0.9,
  canny blended as `difference`). v396 showed an earlier run of the same scene with other control settings (0.8,
  canny `over`), while the README called it this template's run. Picture and sound are the render unchanged; the
  template is embedded, so dropping the clip on the canvas loads it.
- The README shows the first frame, a frame of the run (she is sitting down) and the last frame side by side
  (`assets/flf_sitdown_first.jpg`, `flf_sitdown_run.jpg`, `flf_sitdown_last.jpg`); before, only the run's poster.
- `docs/third_party/flf_sitdown_media.md` names the run's settings and the two new pictures.

## CLIP Text Encode: the search row stays in its slot

On the classic canvas the search row lay over the widget below it (`use_negative`, or `pos_1` in fill mode): the
frontend places a DOM widget 10 px inside its slot and sizes it to the slot minus twice that, and the row is a fixed
34 px. The row now asks for margin 0 and brings the classic widgets' 15 px sides itself (dropped under Nodes 2.0,
where the other rows have none). Measured at zoom 0.75 / 1.0 / 1.3: the row ends 4 px above the next widget. Guard
`tests/test_v1223_cte_search_slot.py`.

## Documentation

- Handbook **v397** (`docs/Polyhedron_Suite_Documentation_v397.pdf`, 125 pages): Part VII gets chapter 6, *Seams --
  editing a film of scenes* (the cut list and repeated heads, Alt, the carry following the cut, "both scenes"
  anchors, snap to scenes, with two new pictures); chapter 8 (Cuts and the Interpolate) says the cut_guard threshold
  is off with the cuts pin; the track table names Alt and the snap arrow. Part V notes the same for `cut_guard`. Part
  X's recipe adds the template's run (MOTION 0.44, half of the way at 1.19 s). The CTE search picture is retaken.
- README: the track table names Alt and the snap arrow; the handbook link points to v397.

---

# v396 -- 3.96.0: clean seams in the Cine Timeline, interpolation that leaves fast motion alone, a first -> last frame template

**ComfyUI restart** + Ctrl+F5. No new node (52), no widget of an existing node moved; one new route
(`POST /pls/cine/snap`). Carries v395 below.

## Cine Timeline: the seams of a scene film

Scenes that continue each other share frames: a motion context repeats the end of scene k-1 as the head of scene k.
The prompt's own cut list always knew that; a cut list edited by hand did not.

- An edited cut list never shows a repeated head: a piece that starts inside it is held at its end (the plan's
  warnings say so); a blend on a seam that would take head frames moves so it runs on the piece's own frames. Hold
  **Alt** while cutting to place a cut inside the head on purpose.
- Every piece remembers the take it was cut on. A new take of the scene moves it by the change of the head; a cut
  list made for another scene count falls back to the prompt's pieces. Both are named.
- The carry follows the cut: where the cut ends scene k-1 is where scene k continues (also for `@clip:k:last`).
  The end is handed to the ONE queued run that reads it, so a fine cut still never re-renders the graph.
- An anchor ON a scene boundary can be ticked **both scenes**: it is the last frame of the scene before and the
  first of the scene after -- two clips that meet on one picture. Unticked: as before (the scene after only).
- One rule for the join: an own first-frame anchor that wins over the carry now warns that the scene before ends
  free (unless it is ticked both scenes). Two anchors on one frame of the rendered scene stop the run by name.
- **snap to scenes** (toolbar, and an arrow in every track head): V1 cuts where the prompt cuts again, and every
  track's clip whose name says its scene (`..._scene_<k>_...`, `_whole_`) goes back to the start of its scene, its
  repeated head measured on that file and left out. Clips without a scene number and locked tracks stay.
- The anchor's pill and row show its role in the scene that renders it (first / last / both / guide).
- A scene that carries finds the clip of the scene before also when the browser tab was in the background or
  closed: it looks in today's and yesterday's output folder for the newest clip a Save wired to `clip_prefix`
  wrote, before it starts free.

## MiniMax Continue

- Warns when `context_frames` pins fewer tail frames than the Cine Timeline lengthened the scene by, with the value
  to set.

## Interpolate: fast motion is not a cut

- With the `cuts` pin wired (a Cine Timeline's cuts -- even when the shot has no cut) the `cut_guard` threshold is
  off: only the cuts the timeline asked for are held, found around their time. Before, a fast camera move (pair
  differences of 0.10-0.15 per frame, measured) reached the 0.1 threshold and every such pair was held -- the clip
  ran at 24 fps in the move and at 48 around it. The console says when the threshold is off. Without the pin
  nothing changes. The tooltip is rewritten.

## The model lines no longer freeze the page

- The model-status route computes in a thread: loading a workflow with many model lines (Control Map, Apply Control,
  the Download door) no longer blocks the server's event loop while it walks the model folders.
- Inside that route a folder walk is kept for a moment (a hit while the file is there, a miss for 5 s); every other
  caller -- a run, an install -- looks fresh. A file copied in by hand shows after 5 s at the latest.
- The frontend asks once per burst: one status request per set within 1.5 s, one list refresh 400 ms after the last
  ask, one check per node line per burst.

## New template: first -> last frame, she sits down

`polyhedron_minimax_h3_flf_sitdown` -- a first and a last frame (she stands, she has sat down on the chair) plus
pose, depth and canny from one video, blended to one control (Control Fit -> Control Map x3 -> Blend -> Apply
Control), Transition Check after the decode. The two keyframes, the pose video and the original clip (Pexels
11066122, scaled to 720x1280) are in `assets/flf_sitdown/`; the README shows the run
(`assets/flf_sitdown_run.mp4`, the template embedded). Credits: `docs/third_party/flf_sitdown_media.md`.

## Guards

New: `test_v1219_overlap`, `test_v1220_ui_freeze`, `test_v1222_interp_motion`. Brought to the new code (the promise
unchanged): `test_v1046`, `test_v1083`, `test_v1150`, `test_v1160`, `test_v1199`, `test_v1211`.

---

# v395 -- 3.95.0: first -> last frame, documented

**Ctrl+F5** (no node, widget or route change). Carries v394 below -- an install from v393 gets both.

## README

The Cine Timeline section has a new paragraph, *First -> last frame: motion, not a dissolve*: when the model
paints a real way between two keyframes (same camera, light and background; the last picture made from the
first), the official FL2VA rule the timeline's `info` repeats (describe the way, one shot, what each guide shows),
the warning for an anchor in the last quarter second, and where a pose video and the Transition Check come in.

## Handbook

`docs/Polyhedron_Suite_Documentation_v395.pdf` (124 pages): Part VII, *The two picture modes*, gets the same
rule as a note and points to Part X (chapters 5 and 6: Transition Check, and a first -> last frame run with a
pose video, measured).

---

# v394 -- 3.94.0: control, blend and the first -> last frame check

**ComfyUI restart** + Ctrl+F5. Five new nodes (47 -> 52), no widget of an existing node moved, no new route.
Carries v393 below.

## Control Fit, Control Map, Apply Control

A clip (or a picture) steers the run: Control Fit brings it to the run's length, size and frame rate; Control
Map turns it into pose (SDPose, whole body), depth (Depth Anything 3), canny, gray, scribble, soft edges or a
keypoint skeleton; Apply Control puts a Fun ControlNet Union on the model (strength, start, end). The control
and the first / last frame keyframes share one run. Needs a current ComfyUI (master).

- The three model files come through the pack's download door: listed in the combos with their size, a missing
  one turns the node's line orange and offers the Download of exactly that file (https, sha256 + size pinned,
  huggingface.co: Comfy-Org's MiniMax H3 Fun ControlNet Union 2.0 / 1.0, Kijai's Qwen-Image 2.1 union,
  Comfy-Org's SDPose and Depth Anything 3 repacks). A run never downloads; files placed by hand are listed too.
- The line follows what the node needs NOW: Control Map shows the pose model in pose mode, the depth model in
  depth mode, nothing for canny / gray / scribble / soft edges. Its states read plainly: nothing picked, not on
  disk (with the Download), on disk from the door, your own file, a picked file that is gone.
- The download notice closes once the file is done (it stayed on "1 KB of 1 KB" before).
- Control Map can show the map in the node (`preview`, an animated WebP); the `keypoints` output carries the pose.

## Polyhedron Blend

Up to three layers on a base, each with its own mode and opacity: the Photoshop set plus `over (black = clear)`
for lines and skeletons on black. Pictures and clips mix -- a single picture lies over every frame, a shorter
clip holds its last frame or loops, the result has the base's length or the longest. E.g. depth + pose + canny
as ONE control video.

## Transition Check

After a first -> last frame run: is the pair close enough to share a way, is the clip a MOTION, a DISSOLVE (A
faded into B) or a SNAP, where does half of the way fall, how close did the clip's ends come to the keyframes.
Text, a score and a report picture.

## Save + keypoints, the placeholder refused

- Polyhedron Save takes `keypoints` (POSE_KEYPOINT): written as `<name>_keypoints.json` (OpenPose format).
- A Media Loader that finds no file sends its placeholder frame; the Cine Timeline (first, last frame, anchors),
  MiniMax Keyframes, MiniMax Reference, MiniMax Continue and Control Fit now refuse it by name instead of
  rendering the checkerboard into the clip.
- Cine Timeline: the official FL2VA prompt rules in the info (describe the way, one shot), a warning when an
  anchor sits in the last quarter second (with the second that IS the last frame).

## Handbook

`docs/Polyhedron_Suite_Documentation_v394.pdf` (124 pages): new Part X -- Control Fit, Control Map and the Save
templates, Apply Control and the model door, Polyhedron Blend, Transition Check, and a recipe (first -> last frame
with a pose video, measured).

## Tests

Guards `test_v1130_control`, `test_v1215_transition`, `test_v1216_control_door`, `test_v1217_line_preview`,
`test_v1218_blend_save`. The Core-backed guards no longer name a machine path: they read `PLS_CORE_ROOT` (or
`PLS_CORE_MASTER` / `PLS_CORE_PIN` / `PLS_CORE_PY`) and fall back to the ComfyUI install above the pack.
Baselines NODE_IDS / WIDGET_ORDER v394.

---

# v393 -- 3.93.0: the live monitor cuts on the frame

**Ctrl+F5** (no node, widget or route change). Carries everything of v392 below -- an install from v391 gets both.

## Cine Timeline -- the live monitor

At a cut the monitor could show one frame of the clip that the film leaves out (the grid overhang past the cut)
before the next scene took over. At a scene that continues the one before (motion carry), the next scene goes on
from the frame before it -- the same moment twice, a stutter at the seam. The export was always clean.

- The monitor's clock counts the frame the clip on screen shows (it ran half a frame behind it).
- A clip that already shows a frame past its piece is not drawn; the monitor keeps its last picture until the next
  piece is there.

Measured in the browser with two barcoded clips: the frame past the cut in 4 of 5 runs before, in none of 12 after.
Guard `test_v1214_live_seam`.

## Handbook

`docs/Polyhedron_Suite_Documentation_v393.pdf` (116 pages): Part VII shows the two picture modes on the timeline
(Keyframes with its anchors, Reference with its `@image` tags) and the CONTINUE film after both runs.
Part VII names the three reference photos (authors and licences).

## The five fox templates, rendered

`assets/fox_t2v`, `fox_refs`, `fox_keyframes`, `fox_continue_film`, `fox_continue_refs` (`.mp4` + poster `.jpg`):
one run of each template, shown in the README under Example workflows. Picture and sound are the renders as
they came out (not re-encoded); the embedded workflow is the template as shipped in `example_workflows/`.
Reference photos: `docs/third_party/fox_reference_photos.md`.

---

# v392 -- 3.92.0: the H3 film chain complete -- scenes that continue, cuts that stay cuts, a restore pass

**Restart ComfyUI, then Ctrl+F5.** 47 nodes: six are new (below), none renamed or removed. Widgets are only
APPENDED (MiniMax Reference `ref_noise`; Power Upscale `pixel_speed` and the restore dials), old workflows load and
run unchanged; baselines NODE_IDS and WIDGET_ORDER v392.

## Six new nodes -- the H3 chain of the fox templates

- **⬡ Polyhedron MiniMax Continue** -- a scene that continues the one before: its last frames and their sound held
  in the AV latent (from the timeline's `tail` / `tail_audio`); pictures placed between first and last frame come in
  as guides (the timeline's `anchors`).
- **⬡ Polyhedron MiniMax Keyframes** -- first / last frame anchors for fl2va (`<Picture n>`; the timeline writes
  the official alignment line into the prompt).
- **⬡ Polyhedron MiniMax Audio Lock** -- a given sound track held, the picture made to it.
- **⬡ Polyhedron MiniMax Drift Control** -- the held start of a continued scene masked anew at every step.
- **⬡ Polyhedron H3 Latent Upscale** -- the learned 3D latent upscaler on the first pass's latent (LBH-123-AI, MIT).
- **⬡ Polyhedron MiniMax Second Pass Hold** -- the second pass of a continued scene keeps its join at full size.

Third-party code and its licences ship in `docs/third_party/` (MIT, Apache-2.0). No weights ship; the model door
offers the H3 latent upscaler (659 MB, pinned) when the node runs and the file is missing.

## Five templates

`example_workflows/polyhedron_minimax_h3_fox_t2v`, `_refs`, `_keyframes`, `_continue`, `_continue_refs`: the fox as
text only, with three reference pictures, with three keyframes, in two clips where the second continues the first,
and those two clips with the three reference pictures. The original `polyhedron_minimax_h3_text_to_video` stays;
two of its groups are fitted to their nodes again (the timeline and the Reference grew). Neutral
upscale (Universal Neutral, light sharpening after the interpolation), the timeline's cuts wired into the
Interpolate. Audio Lock and Second Pass ship bypassed in all five, Drift Control in both CONTINUE templates. The reference pictures
are not shipped -- load any three photos (the note in group ② says which loader takes which).

## ⬡ Polyhedron Cine Timeline

- **Continuation with motion and sound.** `carry_mode` gains **motion**: a scene that carries begins with the last
  frames of the scene before -- 5, 22 or 39 film frames (`head_frames`) -- and their sound. Two new outputs,
  `tail` and `tail_audio`, hand them on; the scene is compiled that much longer, the later shots move back by the
  head, and the film leaves the repeated frames out at the seam.
- **`cuts` output** (output 12, appended): the times of the cuts inside the clip this run renders, as the compiled
  scene places them. Wire it into the Interpolate (below).
- **Tracks:** one track per render, clips between tracks, *takes -> tracks* per render; NEW marks the latest render;
  clean cuts in the film and a monitor that does not flicker; one law for "how many film frames is this clip",
  measured against ffmpeg. The fixes of the Cine audit (confirmed bugs, risks, optimisations).

## ⬡ Polyhedron Interpolate

- **New pin `cuts`** (appended after `video`), from the timeline: around each cut the timeline asked for (+-0.25 s)
  the pair where the picture really jumps is held instead of interpolated -- no ghost dissolve at a cut, no threshold
  to guess. Fast motion raises a whole window and is not held; a painted dissolve has no single jump and is
  interpolated. `interp_info` names where each cut was found.
- **`cut_guard` defaults to 0.1** (was 0 = off) for clips without a timeline. Saved workflows keep their own value.

## ⬡ Polyhedron Power Upscale

- **Restore path** (`restore_model`, default **off**): a restoration model such as Core's SeedVR2 family runs next
  to the refine (`restore_order`, `restore_color`, `restore_overlap`, `restore_frames`). The four SeedVR2 files
  (Comfy-Org's repack, Apache-2.0) are offered by the model door only when one is picked; it clears the card first,
  keeps a temporal floor, steps down on out-of-memory, learns its VAE tile rung per card and size, and reports a plan
  with estimates and heartbeats.
- **`pixel_speed`**: how the pixel upscale model runs, with an fp16-aware tile estimate; the console names the pixel
  model that really runs the final pass; the final canvas lands even (no odd edge for the video encoder); a carried
  scene survives the joint refine (pins re-encoded, the canvas on the model's patch grid).
- With a curve on `sigmas` the refine reads neither `steps` nor `scheduler` -- it starts at sigma = denoise and runs
  every curve point below it. Those four dials now hide while a curve is wired, and `denoise` names what it buys:
  "denoise · 2 steps on the curve". The count comes from the run's own function (new route `POST /pls/pu_sigma_run`
  in the sigma route module); a slow check keeps it true in both renderers (no draw hook -- the node paints nothing).

## ⬡ Polyhedron Media Loader

- "Load from where it is" never copies into `input`. A drop whose path the browser hides, and that is not found in a
  recently used folder, opens the folder picker (its 📁 button is the native dialog) instead of falling back to a copy.

## ⬡ Polyhedron MiniMax Reference

- `ref_noise` (appended, default 0.001 = Core's behaviour, bit for bit): how much noise the reference pictures carry
  into the model -- higher is a looser hold.

## Also

- ⬡ Polyhedron Attention: SageAttention's Triton kernels address with 32 bits; a call that would pass 2^31 elements
  is handed to the model's own backend before the launch (said once), the rest of the run stays on sage.
- Download offers appear only when the node really needs the file now; buttons are tinted so they read as buttons.

## Model door

Three sets now: `rife` (Interpolate), `restore` (Power Upscale, four optional files) and `h3_latent_upscale` (H3
Latent Upscale, one file). Every row https, sha256-pinned, sized; a line under the node offers exactly the files it
needs now -- nothing while it is bypassed, an optional file only while it is picked.

## Guards

New: `test_v1170_new_corner`, `test_v1184_sage_int32`, `test_v1185_head_frames`, `test_v1190_new_is_latest`,
`test_v1195_film_frames`, `test_v1198_one_track`, `test_v1201_ref_noise`, `test_v1211_cuts`,
`test_v1212_pu_curve_steps` (route read from `ph_sigma_routes.py`), `test_v1160_routes_served`; for the H3 chain and
the restore path `test_v964_minimax_keyframes`, `test_v965_core_sprung`, `test_v1017_h3_prompt`, `test_v1022_field_floor_js`,
`test_v1048_h3_quality` (Q6 re-grounded: only H3 Latent Upscale of its module is registered here),
`test_v1049_carry`, `test_v1084_rest_of_table`, `test_v1178_av_mask`, `test_v1179_drift`, `test_v1180_pass_hold`,
`test_v1186_pixel_speed`, `test_v1194_restore`, `test_v1199_restore_pick`, `test_v1202_line_verdict`,
`test_v1203_wire_report`, `test_v1205_even_canvas`, `test_v1206`-`test_v1210` (restore). RE-GROUNDED with the reason
in the file: `test_v365_public_build` (47 node ids; the door serves exactly rife, restore, h3_latent_upscale, each set
owned by a registered node, no import of a module this build does not ship). Brought to the
internal state: the Cine guards v1034-v1165, `test_v1016_minimax_ref_all_inputs`, `test_v791_interpolate_video`
(pins appended). `test_v888_hide_inactive` lifts the curve helpers next to `_applyModeState`.

## Handbook and README

- `docs/Polyhedron_Suite_Documentation_v392.pdf` (115 pages): new Part VII Cine Timeline, Part VIII Workflow
  essentials, Part IX Model profiles and recipes; a page on how to read model notes (MiniMax H3, WAN 2.x, others) with
  evidence marks; Part II addenda (Stack controls added since, model notes per chapter); CLIP Text Encode Search and
  `fill {pos_N}`; Power Upscale refine order, restore, curve steps; the Interpolate's cuts. Parts I and II are carried
  over unchanged except the pin-zone sentence and the version.
- README: the pin zone never copies; the Interpolate's cuts; scenes that continue and the timeline's new outputs;
  refine order, restore and curve steps on the Power Upscale; the six H3 nodes; the five templates; the handbook link.
