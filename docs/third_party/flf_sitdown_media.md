# Media of the first -> last frame template (she sits down)

The template `polyhedron_minimax_h3_flf_sitdown` loads three files from
[`assets/flf_sitdown/`](../../assets/flf_sitdown/) (copy them into your ComfyUI
`input` folder). All of them, and the original clip beside them, come from one
stock video:

| | |
|---|---|
| Source | Pexels video **11066122** — [pexels.com/video/11066122](https://www.pexels.com/video/11066122/) (a young woman in a photo studio sits down on a chair; portrait 2160x3840, 29.97 fps, 15.7 s) |
| Licence | [Pexels licence](https://www.pexels.com/license/): free to use and to modify; attribution not required but given here |

| File | What it is | Changes |
|---|---|---|
| `flf_A_stand.png` | first frame of the template: she stands in front of the chair | frame 97 of the source, scaled and cropped to 768x1344 |
| `flf_B_sit.png` | last frame of the template: she has sat down | frame 157 of the source, scaled and cropped to 768x1344 |
| `flf_pose_sitdown_x1.mp4` | the template's pose video: the sit-down at real speed (0.6 s standing, 2.0 s sitting down, 1.8 s seated), 107 frames at 24 fps | cut from the source around the sit-down (frames 97-157) at real speed, scaled and cropped to 768x1344, 24 fps |
| `pexels_11066122_original_720p.mp4` | the source clip, for reference | scaled to 720x1280, re-encoded (sound kept) |

`assets/flf_sitdown_run.mp4` (and its poster `flf_sitdown_run.jpg`) is a new
render by MiniMax H3 guided by the keyframes and the pose video -- this template's
run as it ships (Apply Control 0.9 until 60 %, canny blended as `difference`); it
carries the template in its metadata. `assets/flf_sitdown_first.jpg` and
`flf_sitdown_last.jpg` are the two keyframes scaled to 540 px for the README.

The person shown is a model in a stock video. Use these files for trying the
template; do not use them to depict her in a way that is offensive or that
suggests she endorses anything.
