# CHANGELOG v374 -- Nodes 2.0 parity + the arrears since v908

Carries the internal cuts **v909 - v956** (12.09.2026). `nodes/uls_routes.py` is
untouched (bcc4d8c4, twelfth round); the four route modules stay as they are,
`ph_sampler_routes.py` gains ONE route. 37 nodes, unchanged.

## Nodes 2.0 (Vue renderer) -- every node works under both renderers
ComfyUI's Vue renderer ("Modern Node Design" / Nodes 2.0) is on by default in
new Desktop and Cloud installs since frontend 1.41. Until now the painted nodes
of this package showed a "needs the LiteGraph renderer" notice there. Now:
- `uls_vue_views.js` -- the switchboard: one registration for a node's Nodes 2.0
  view (a DOM widget the frontend places into its own widget grid).
- `uls_stack_dom.js`, `uls_engine_dom.js` -- the Stack and Engine as DOM panels:
  same rows, group colours, Apply pill, order arrows, trigger words, weights,
  DARE/Trim, painted row metrics (28 px, no gap), no hint line.
- `uls_extras_dom.js` -- CLIP Text Encode word band, Int chips, Filter A/B +
  preset chips, the Load nodes' status line.
- `uls_vue_hidden.js` -- widgets hidden on the canvas stay hidden under Nodes
  2.0 (430 widgets audited, 0 mismatches).
- `uls_vue_parity.js` -- size floor, field tint AND field height bridged from the
  painted node (CLIP Text Encode auto-fit grows and shrinks), a growth rule for
  DOM panels, disabled fields dimmed like LiteGraph.
- the "control after generate" row (fixed / increment / decrement / randomize)
  under every seed -- Nodes 2.0 folds it into a dice button; the row is back,
  wired to the same widget (Seed, Sampler, Power Upscale).
- `uls_live_preview.js` -- **the Sampler's in-node live preview, which was
  MISSING from the public package** (`uls_sampler.py` has streamed the frames
  since v365; no public file listened). Classic and Nodes 2.0 (through the
  frontend's own preview door).
- `ph_seed.js` -- noise preview and seed scrub visible under Nodes 2.0; scrubbing,
  clicking and Roll no longer pin the control to fixed (only "Reuse last" does).
- `ph_empty_latent.js` -- "Custom" listed in the preset combo (no red frame).
- `ph_power_upscale.js` -- no click wall over the process view (v939).
- `uls_compat.js` -- the renderer notice only for what is really unavailable.
Measured on Frank's workflows in the sandbox: all 37 nodes classic vs Nodes 2.0,
classic pixel-identical to the v943 reference crops (P0), 0 click walls.

## Backend arrears (internal v909 - v935)
- Stack / Engine: Apply pill **Auto / Bypass / Baked** (v913-v917, baked is the
  measured default on H3), merge policy + LoRA conversion + AdaLN rebase +
  PDD apply/math carriers (`uls_merge_policy.py`, `uls_lora_convert.py`,
  `uls_adaln_rebase.py`, `uls_pdd_apply.py`, `uls_pdd_math.py`).
- Sampler: taeh3 preview decoder for MiniMax H3 (v918, vendored
  `nodes/vendor/taehv/` -- MIT, madebyollin, verbatim, see its SOURCE.md),
  preview diet (v920) and the `tae_warm` route ported into
  `ph_sampler_routes.py`.
- NAG (v919), Noise (v916), Inspector notice (v910, `ph_inspector_toast.js`),
  Empty Latent and weights helpers, Token Counter toast (v908/v911).

## Guards
Suite **130 -> 162 / 162**; 79 internal guards adopted,
run green here as they are; public-only guards re-grounded where a public
module differs (v302: eleventh merge fallback, v310: shared tooltip lines,
v365: fourth sampler route, v920: routes live in ph_sampler_routes.py, v941 /
v943: no Everywhere / internal-only nodes, v934 / v937: harness stubs).
Baselines regenerated as v374 -- WIDGET_ORDER content-identical to v373 (no
widget changed), NODE_IDS unchanged.

## Not in this cut
The handbook stays the v372 edition (89 pages). Internal-only nodes stay
internal (Cutout, Mask Editor, Layers, Composite, Outpaint, Reference,
Vectorize, Everywhere, the 3D / light branch).
