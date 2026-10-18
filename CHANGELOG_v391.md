# v391 -- 3.91.0: the carry reads the clip at its frame rate (the seams of a scene film)

**Restart ComfyUI.** 41 nodes, nothing added, renamed or removed; no widget moved, old workflows load unchanged
(baselines stay v389).

## ⬡ Polyhedron Cine Timeline -- the carry

A scene that *carries* begins on the last frame the film shows of the scene before. The timeline counts film
frames at 24 fps, but the newest take of a scene is often the interpolated final -- 48 fps, 281 frames -- and the
timeline asked that clip for frame 140 BY NUMBER: its middle. Measured on a five-scene film: every scene began on
the middle of the scene before, at all four seams alike. Now the film frame goes through the clip's own rate
(140 at 48 fps -> frame 280, the last), for the carry and for the `@clip:K:...` anchors alike; the run's info names
the clip's rate and frame. A clip at 24 fps is unchanged.

After the update: render the carrying scenes again (tick them, `new render -> replace`, "render ticked scenes"),
in order -- each then starts on the true end of the scene before.

## Guards

New `test_v1168_carry_fps` (a real 48 fps clip: the carry frame is its last picture; by number it was the middle).
`test_v1083_film_truth` and `test_v1051_timeline_images` follow the new info line.
