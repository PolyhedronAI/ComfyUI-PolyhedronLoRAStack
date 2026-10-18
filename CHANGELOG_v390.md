# v390 -- 3.90.0: the Cine Timeline renders ticked scenes, replaces clips, plays on Space; carried scenes upscale

**Restart ComfyUI, then reload the page.** 41 nodes, none added, renamed or removed; no widget moved, old
workflows load unchanged (baselines stay `NODE_IDS_baseline_v389.txt`, `WIDGET_ORDER_baseline_v389.txt`).

## ⬡ Polyhedron Cine Timeline

- **Ticked scenes.** From two scenes on, every scene has a tick on the Scenes row. "render all scenes" becomes
  "render ticked scenes (n)" and queues only those, in order; the split run (Ctrl+Enter with the all-scenes choice)
  skips the unticked ones too. Saved with the workflow; a scene added later starts ticked.
- **new render -> [new track | replace].** The old checkbox is an explicit choice. `replace`: a new run takes the
  place of its scene's clip in V1 (the old files stay on disk, the tracks stay as they are).
- **The green NEW.** A V1 clip that replaced an earlier run's clip of its place carries a small green NEW (the
  tooltip says so). One run writes several files (raw, upscaled, interpolated) -- that alone is no replacement: the
  stack list now keeps a run history (which files belong to which run). Lists from before v390 get an estimate.
- **Space** plays / pauses the selected timeline (never while typing, never with two timelines selected).
- **The monitor takes the clip's shape** (portrait stays portrait, letterboxed) and follows the column's width.
- **Resizing holds together:** a floor width of 760 px; the shot editor, sound and monitor columns shrink instead of
  spilling out of the node.
- **Reference mode** lists the pictures wired at the MiniMax Reference that the timeline's prompt flows into, as
  `<Picture n>` in wired order, when the timeline has none of its own.

## ⬡ Polyhedron Power Upscale

- **Carried scenes upscale.** A scene that starts on the last frame of the scene before (carry) has that frame
  pinned in its conditioning, encoded at the render size. The joint refine reused it on its larger canvas and died
  (shape mismatch). The pin is now encoded again on the refine canvas, from the frame that stands there -- the
  seam to the scene before stays exact.
- **The joint refine canvas sits on the model's patch grid** (VAE factor x the transformer's patch, read from the
  model; MiniMax H3: 16 x 2 = 32). The /16 canvas gave odd latent sides, which a pinned frame cannot be cut into
  patches on. Runs without a pin land on the same grid (a few pixels smaller than before).

## Guards

New: `test_v1162_upscale_pins`, `test_v1163_scene_ticks`, `test_v1164_joint_grid`, `test_v1165_new_mark`.
Updated: `test_v1004_h3_refine` (the fake model carries H3's patch), `test_v1080_cine_basin` (the choice).
