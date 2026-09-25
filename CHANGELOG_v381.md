# v381 -- Sage Attention 3: no more black images (issue #5)

Backend-only. Restart ComfyUI. No widget, baseline or node id changes.

## The report

Issue #5: with Polyhedron Attention set to `sage3 blackwell (fp4)`, KSamplers
produced black images; DisTorch Sage Patch and Attention Optimizer did not.

## What was wrong, read at the source

- The SageAttention 3 kernel (`sageattention3_blackwell/api.cu`) exists for
  head dim **64 and 128** only.
- Its Python wrapper has **no sequence-length check**, yet ComfyUI Core sends
  every call with **N <= 1024** to pytorch -- that is every cross-attention
  call of an image model. Our router called the kernel at every length and
  trusted it to raise. The wheel it was validated on did; a public wheel on
  image geometry answers with NaN instead, and one NaN attention call is a
  black image. `fallback` only catches exceptions.
- `sageattn3_blackwell(..., attn_mask=None)` accepts a mask and never reads
  it; we handed masks to a kernel that drops them.

The node was validated on one wheel (our own sm_120a build) and one
geometry (video, 39168 tokens, dim 128). The public field is every
Blackwell wheel on every geometry, which nobody can test in advance.

## What changed

- Calls with N <= 1024, head dim not in {64, 128}, or a mask are **passed
  through by name** to the model's own backend -- the rest of the run stays
  on Sage 3. That is exactly what Core and the other wrappers do.
- **Finite gate**: the first call on each (seq, head dim, dtype) geometry is
  checked for finite values (one sync per geometry, not per call). A wheel
  that answers NaN on some geometry gets one console line naming it, and that
  geometry runs on the model's own backend for the rest of the run. Other
  geometries stay on Sage 3. Worst case for an unknown wheel: speed on one
  geometry, never a black image.
- The capability probe launches head dim 64 as well and says when a wheel
  serves 128 but not 64.

Guard `tests/test_v1015_sage3_geometry.py` (G1-G6, driven with a fake
kernel; six mutations caught).

## If you were hit by #5

Update, restart ComfyUI, run again. Please post the node's console lines
(`live_check` verdict, "not every call is ... to serve", and any
"failed mid-run (Sage3NonFinite ...)") together with your wheel's source and
version and your torch version -- that tells us which geometry your wheel
does not serve.
