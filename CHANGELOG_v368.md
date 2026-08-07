# CHANGELOG v368 — three more nodes: Power Upscale, Fast Upscale, Interpolate

Version 3.68.0 · **17 → 20 nodes** · no shared file re-opened.

Files added: `nodes/ph_power_upscale.py`, `nodes/ph_fast_upscale.py`,
`nodes/ph_interpolate.py`, `nodes/uls_tile_math.py`, `nodes/vfi/` (package,
2 modules + NOTICE), `web/js/ph_power_upscale.js`, `web/js/ph_interpolate.js`,
`tools/make_widget_baseline.py`, `WIDGET_ORDER_baseline_v368.txt`, and 42
guards under `tests/`.

Files changed: `__init__.py` (one new group), `pyproject.toml`,
`web/js/uls_compat.js`, `README.md`, `COMPATIBILITY.md`,
`tests/test_v365_public_build.py`, `tests/test_v348_merge_math.py`,
`tests/test_v351_pickframe_v3.py`, `tests/test_v352_inflate_v3.py`.

---

## 1. The three nodes

**⬡ Polyhedron Power Upscale** — MoE-aware tiled upscaler. Clean-room tile
geometry (`uls_tile_math.py`, guard-proved weight sum == 1) with core
delegation for the ESRGAN pass, the sampling loop and the Wan sigma shift.

**⬡ Polyhedron Fast Upscale** — the fast path beside it: optional ESRGAN, then
resize, no diffusion refine. Imports its helpers *from* Power Upscale, so there
is one source of truth rather than two that drift.

**⬡ Polyhedron Interpolate** — frame interpolation with the vendored RIFE
IFNet, plus this pack's own rate/timeline handling around it.

## 2. What the cut did NOT have to touch

`nodes/uls_routes.py` stays **bcc4d8c47e500c892619c22d51f7cbc6** — the seventh
release in a row it has not been re-opened. `ph_media_routes.py` and
`ph_sampler_routes.py` are equally untouched.

**None of the three nodes needs a server route.** Power Upscale reports tile
progress through `PromptServer.send_sync`, which needs no endpoint; the two JS
files import ComfyUI core only and call no `fetch`. This is now pinned in
`test_v365_public_build.py`: if one of them ever grows a route handler, the
guard fails and demands it be a declared act in its own module.

Three of the four carrier modules were already here from v365
(`ph_runclock`, `ph_logmute`, `ph_weights`, and the four symbols Power Upscale
imports from `uls_sampler`). Only `uls_tile_math.py` was missing.

## 3. Vendored third-party code — declared

`nodes/vfi/rife_arch.py` is **not ours**: it is the MIT IFNet from
Practical-RIFE (hzwer), by way of ComfyUI-Frame-Interpolation (Fannovel16),
copied byte-for-byte on purpose so that comparisons against reference runs
stay meaningful. `nodes/vfi/NOTICE.md` carries the attribution the MIT licence
asks for, and README plus COMPATIBILITY.md name it as well. Nothing in that
folder gets "improved" — every correction lives outside the engine, in
`ph_interpolate.py`, where it can be guarded.

## 4. Guards — 43 → 85

47 guards in the internal tree read these three files. Each was run here and
taken **only if it ran honestly green**; not one was weakened to fit.

- **42 taken.**
- **5 not taken**, because they read files this build does not contain
  (`ph_palette.js`, `ph_widget_hydrate.js`, `ph_basics.py`, `ph_dino_engine.py`,
  `ph_bgr_engine.py`): `test_v600_suite`, `test_v805_pin_migration`,
  `test_v829_interp_diamond`, `test_v835_weights_only`, `test_v836_rife_door`.

### The serialisation baseline, newly introduced here

`test_v577_widget_order` and `test_v791_interpolate_video` failed at first for
a different reason: they need a `WIDGET_ORDER_baseline_*.txt`, and this build
had none. That is a missing artifact, not a structural mismatch, so it was
supplied rather than dropped — the build now carries the same positional-canon
protection the internal tree has, which matters most exactly where saved
workflows live in other people's hands.

`tools/make_widget_baseline.py` generates it by **importing the scan out of the
guard itself** rather than re-implementing it; a second implementation would be
a second truth, and the second truth is always the one that is wrong.

**Declared re-grounding:** `test_v577_widget_order` carried `if guarded < 36`,
a census of the *internal* tree. This tree has 20 node classes, one declared
dynamic, so 19 is **full** coverage here — the floor was re-grounded to 19 and
stays maximally tight. The law itself (nothing leaves the guarded set
unannounced; no widget order shifts) is untouched.

Mutation-tested, three landed and were caught: a widget slipped into the middle
of Interpolate's `INPUT_TYPES`; a node removed from the baseline; a node made
silently dynamic. A fourth attempt was a **blank** — it did not move the
scanner at all, and is recorded as such rather than counted as a catch.

## 5. Version triple

`pyproject` 3.68.0 · banner `Polyhedron Suite  v368` · `PLUGIN_VERSION v368`,
with the pins in `test_v348`, `test_v351` and `test_v352` carried along.
`CHANGELOG_v367.md` rotated into `docs/changelog-archive/`.

## 6. Documentation — Part V

`docs/Polyhedron_Suite_Documentation_v368.pdf`, **82 pages in five parts and an
appendix** (was 75 in four). The v367 PDF is replaced.

**Part V — Upscale & Interpolate**, 7 pages: Fast Upscale; Power Upscale over
two pages (pins and stages, then the controls); the process view and how to read
its header line; Interpolate; and a closing section on choosing between the
three and the order that works.

Parts I and II are lifted unchanged from the v367 file and verified page by page
(**56/56 textually identical**). Parts III and IV are re-rendered and verified
**16/16 identical** once the declared footer change v367 → v368 is normalised
away. Contents, cover version and the Appendix page number follow; bookmarks go
from 40 to 46.

Five new screenshots ship in `assets/` and are referenced from the README node
chapters as well: the two whole nodes, the head of Power Upscale in
`High + Low`, its two upscale-model loaders, and a four-moment plate of one run
showing stages H, L and P with the estimate falling and the map switching from
tiles to chunks.

## 7. The serialisation guard learns to see a built combo list

Shipped here and in the internal tree (as v850) in the same sitting, because the
gate is the same file in both.

`_is_widget` recognised a combo list written inline or **named**, but not one
**built**: `([SAME_AS_HIGH] + list(comfy.samplers.KSampler.SAMPLERS), {...})` is
an `ast.BinOp`, which no branch covered, so the scanner filed a **real widget as
a socket**. Two nodes were affected here — `ULSPowerUpscale` and `ULSSampler`,
both losing `sampler_low` and `scheduler_low` from the baseline silently. For
Power Upscale the baseline diverged from the truth at **slot 15 onward**, so ten
slots were compared against the wrong names.

**Measured before it was changed**, because the correction inserts entries in the
middle of a node — the exact shape this gate exists to forbid. Both nodes publish
their true serialisation order in the frontend, and the patched scan reproduces
both exactly: `ORDER_CANON` **25/25**, `ORDER_V404` **19/19** (after discounting
core's injected `control_after_generate`). The baseline was **blind, not wrong**
— no saved workflow ever shifted, only the yardstick was short. Nothing to heal,
nothing to migrate.

`WIDGET_ORDER_baseline_v368.txt` regenerated as a declared act; the generator now
also carries `CANON` lines forward, after a first attempt downgraded a
canon-guarded node to a bare `DYNAMIC` line and stayed green — coverage lost
silently, in a cut about coverage lost silently. Caught by diffing old against
new instead of trusting the green.

Mutation-tested, both landed: patch reverted → the gate fails; a widget inserted
mid-`INPUT_TYPES` → *BREAKS THE SERIALISATION LAW at widget index 10*.
