# v382 -- 3.82.0: saved workflows keep their values on the new ComfyUI frontend; H3 template corrected

Frontend fix. Restart ComfyUI, then reload the browser page (Ctrl+Shift+R).
No node, input or widget changed; old workflows load as before.

## What was wrong

On current ComfyUI (frontend 1.53.x), saving a workflow and opening it again
moved the values of two nodes into the wrong fields:

- **Load CLIP**: `type = minimax` came back as `default`, the CLIP file name
  landed in `device`.
- **CLIP Text Encode**: the prompt landed in `strip_newlines`, the other
  settings shifted with it.

Both nodes show their fields in a different order than the one they are
saved in. Up to frontend 1.49 they put the fields back in save order inside
`serialize()`. Frontend 1.53 no longer calls `serialize()` on a node when it
saves the graph; it writes the fields itself in the order they are shown.

Measured with the bundled template `polyhedron_minimax_h3_text_to_video`:
2 of 17 nodes changed by one save -> reload on 1.53.6, 0 on 1.49.6.

## The fix

`web/js/ph_save_compat.js` (new): both nodes rebuild their saved values in
save order through `onSerialize`, which 1.53 still calls. On 1.49 the old
path already ran at that moment, so nothing is touched there and the saved
file stays byte for byte what it was.

If a workflow was already saved on 1.53 with an earlier version, open it
once, check Load CLIP (`type`) and CLIP Text Encode, and save again.

## Checked

- Save -> reload on frontend 1.53.6 and 1.49.6: 0 changed nodes in all
  bundled templates.
- New tool `tools/browser_roundtrip_probe.py` measures exactly this.
- New guard `tests/test_v1021_save_compat_js.py`: the 1.53 path writes save
  order, the 1.49 path stays untouched, a failing step never breaks a save.

## H3 template corrected

`example_workflows/polyhedron_minimax_h3_text_to_video.json`:

- **NAG removed.** The old note said NAG steers the attention away from the
  negative. On MiniMax H3 it cannot: NAG works inside a cross-attention and
  H3 has none (text, video and audio run as one joint sequence). The NAG
  node already refuses H3 and passes the model through unchanged, so the
  node did nothing in this template. Attention now feeds the Sampler
  directly; the note "No negative on H3" says why.
- **Live preview = Video - TAE (taeh3).** It was "Still - ComfyUI"; taeh3 reads
  the H3 AV latent, taew2_1 cannot.

Loaded in both frontends, save -> reload clean, accepted by the server.
