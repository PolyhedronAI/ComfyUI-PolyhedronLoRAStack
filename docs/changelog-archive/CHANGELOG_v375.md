# CHANGELOG v375 -- the two open issues, and a MiniMax H3 starter template

Carries the internal cut **v957**. No new node, no new route; 37 nodes,
`nodes/uls_routes.py` still bcc4d8c4 (thirteenth release running).

## Issue #3 (bytimer): "Press R to update loras, but the new downloaded loras can not be seen"
The pack reads the LoRA folder through its own route `/uls/list`, once, at page
load. "R" / Refresh reloads the core node definitions and tells extensions
through the `refreshComboInNodes` hook -- which nothing here implemented, so a
LoRA downloaded while ComfyUI runs only appeared after a restart.
`refreshLoraList()` now drops the loaded flag, the metadata cache and the
preview cache and fetches again; the `Polyhedron.stack` extension implements
the hook. Measured: page open with 2 LoRAs, a third file dropped into
models/loras, Refresh -> the new name is in the picker.

## Issue #2 (HuntingSuccubus): "video save preview not producing audio, totally mute"
The preview element starts muted because a browser refuses to autoplay a clip
WITH sound -- that stays. But there was no way to switch sound on at all. A
sound button now sits in the transport bar next to the loop toggle (video
only); the state persists on the node, switching it on unmutes and resumes.
Its tooltip names the other possible cause: a clip only carries sound when
audio was wired and the preset has an audio track.

## New template: polyhedron_minimax_h3_text_to_video
The fifth starter workflow, and the first built from THIS PACK ALONE -- no
other node pack at all. Load Model -> Load CLIP -> two VAEs (H3 writes picture
and sound), LoRA Engine -> Attention -> NAG -> Sampler -> VAE decode -> Save,
Empty Latent in H3 AV mode, the Seed node feeding both seed and noise. At CFG 1
the negative prompt feeds NAG instead of the sampler; four in-canvas notes say
why and what to change for image-to-video (ref2v + MiniMax Reference). The
three H3 turbo LoRAs ship as DISABLED example rows in the Engine. Same fox
prompt as the other four templates; the thumbnail is a frame from that run.
