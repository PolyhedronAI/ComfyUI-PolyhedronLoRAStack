# v383 -- 3.83.0: MiniMax Reference takes every H3 input, Reference Board, readable CLIP Text Encode under replace, honest Power Upscale clock

Backend + frontend. Restart ComfyUI, then reload the browser page (Ctrl+Shift+R).
One new node (Reference Board, 37 -> 38). No node id renamed or removed;
widgets and outputs only APPENDED (baselines NODE_IDS v375 -> v383,
WIDGET_ORDER v380 -> v383). Old workflows load unchanged.

## MiniMax Reference: every input MiniMax H3 takes

MiniMax's Ref2VA limits are 9 images, 3 videos and 3 audios. The node took 3
images. Now:

| input | becomes |
|---|---|
| `image_1..9` (+ `megapixels_1..9`) | `<Picture i>`, each with its own budget |
| `video_1..3` | `<Video k>` -- straight from the Media Loader |
| `video_audio_1..3` | the soundtrack of video_n, its own `<Audio j>` |
| `audio_1..3` | `<Audio j>` after all soundtracks |
| `audio_vae` | needed only when audio is wired |
| `first_frame` | pinned at frame 0, never denoised |
| `refs` | the Reference Board (below) |

- **Real frame rate:** a 30 fps reference video is resampled to H3's 24 fps
  (Core assumes 24 and runs it 25 % slow).
- **Names instead of numbers:** write `@fox` in the prompt and `fox = image_2`
  in the new `tags` line (every slot name is a tag by itself: `@image_3`,
  `@video_1`). The node writes the live `<Picture n>` / `<Video k>` /
  `<Audio j>` in before the encode, so rewiring never breaks the prompt. An
  unknown tag or a tag on an unwired slot stops the run by name, before
  anything is encoded.
- **Prompt check** in `info` and the console (reported, never rewritten):
  references beyond what is wired, wired references the prompt never names,
  weight brackets `(x:1.3)` (not parsed by H3), `//` rubrics reaching the
  model, shot timing gaps, unclosed dialogue, length above ~700 tokens.
- **New output `prompt_text`** (appended): the prompt as it reached the
  encoder, tags resolved.
- **`first_frame` next to references:** Core overwrites a keyframe's latents
  with the references' when both are present; the node merges them, so both
  arrive.
- **The node grows with what is wired:** a fresh node shows one spare slot
  per kind; every wired slot stays, plus one spare. Old saves heal on load.
- Both text fields (`prompt`, `tags`) get the same 64 px on both renderers;
  a wired prompt keeps only a narrow row for its socket.

## New node: Reference Board

Up to 9 images, 3 videos and 3 audios as CARDS in one node: thumbnail, `@tag`,
role (subject, scene, style, first frame, ...; voice, music, ambience, ...),
retention (the official markers), budget, and `own` sound for videos. Add
with **+ add** or by dropping files; they go through ComfyUI's own
`/upload/image` into `input/pls_board` -- the pack opens no new route.

Outputs: `refs` -> MiniMax Reference `refs` (fills its FREE slots in board
order, a wired slot stays the wire's, the board's tags work in the prompt);
`definitions` -- a draft of the official `subject_definitions` /
`retention_analysis` lines in `@tag` form; `info`.

## CLIP Text Encode: external_mode = replace, readable

With `pos_external` (or `neg_external`) wired and the mode on `replace`, your
own segments are not encoded. They now FOLD into one grey bar ("2 own
segments -- not encoded while external_mode is replace"); a click shows them,
greyed. Your text is untouched and saved with the workflow.

- The green field shows the wired text AS IT CAME, with its `//` rubrics and
  line breaks, and grows with it (up to 640 px). Outside replace it is the
  small 96 px field again.
- The word counter counts only what is encoded.
- Nodes 2.0: the green field and the fold bar are the real elements now
  (before, Nodes 2.0 replaced the green field with an empty stand-in).

## Power Upscale: the clock stops saying "~0:00 left" for minutes

- The learned phase estimates (encode / step / decode) feed the run clock, so
  "stage left" uses them before anything is measured.
- Step n opens with step n-1's real duration as its plan.
- Past the plan the heartbeat says so: "+2:31 over the plan of ~2:03 --
  still working".

## Load CLIP / VAE Codec: MiniMax Music3

- Load CLIP loads the Music3 text encoder (`minimax_music3` -> Core's
  `minimax` type); Load VAE names the 128-channel 1-D music VAE.
- VAE Codec decodes a MUSIC latent to the AUDIO output through Core's own
  `vae_decode_audio` (before, it "decoded" it into a meaningless IMAGE).

## Checked

- Full test suite green in this cut; new guards
  `test_v1016_minimax_ref_all_inputs`, `test_v1016_minimax_ref_js`,
  `test_v1019_reference_board`, `test_v1019_reference_board_js`,
  `test_v619_external`; `test_v365_public_build` pins 38 nodes and the
  route-free promise for the Board and its carriers.
- Carrier modules (no node of their own): `nodes/h3_prompt.py`,
  `nodes/h3_kf_refs.py`, `nodes/cine_clip.py`.
