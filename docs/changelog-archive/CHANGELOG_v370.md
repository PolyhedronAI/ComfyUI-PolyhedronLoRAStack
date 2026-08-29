# v370 — Power Upscale: the low expert gets its own sigma shift, and the twins sit together

Carries the internal v851 + v852 cuts. The Sampler has had `sigma_shift_low`
since v366; the Power Upscale applied ONE shift to both stages. That gap is
closed, and the new dial sits where the node's other twins sit — directly under
its HIGH partner.

## sigma_shift_low (from v851)

- NEW widget on `ULSPowerUpscale`, default **-1.0**, range -1..20, appended
  LAST in `INPUT_TYPES` so no saved workflow is renumbered (#577: widget values
  serialise BY INDEX).
- **-1** = *same as high*, which is exactly what the node did before, bit for
  bit · **0** = off for stage L alone · positive = its own shift.
- The sentinel resolves through the Sampler's `_resolve_low_shift`, **imported,
  not copied** — one semantics, one place. A guard rejects a local
  re-implementation structurally.
- **Always honoured.** Unlike the Sampler there is no continuous-handoff mode
  here in which one schedule serves both experts, so this can never go inert
  and grows no honesty line.
- Two traps, both guarded: applying the HIGH shift **rebinds** `model`, so
  stage L's source is captured BEFORE it, or a "low" shift would land on an
  already-shifted model. And without a wired `model_low`, stage L falls back to
  the `model` input — a low shift there needs a SECOND clone off the RAW model,
  or the loop hands it the high-shifted one silently. With the sentinel
  untouched and no wire, nothing extra is built at all.

## The display re-sort (from v852)

`sigma_shift_low` renders directly under `sigma_shift` instead of at the panel
bottom. A display re-sort is the most expensive mistake this node knows — the
live frontend serialises in DISPLAY order, so a careless move shifts every old
save by a slot and the self-heal net then MASKS half the damage by repairing
shifted values into plausible defaults. Nothing crashes; the node just runs
differently. So it was done with the full ceremony:

1. the order being left is frozen as `DISPLAY_LEGACY_V851`, beside the existing
   `DISPLAY_LEGACY_V587`. Neither table may ever be edited again.
2. the type fingerprint now separates FOUR layouts instead of two: slots
   13/14 + 16/17 answer "canon or some display order", 10/18 + 19/25 answer
   "which display era", 19/21 + 24/26 separate the previous display from
   today's. Every verdict rests on TWO witness pairs, so one corrupt value can
   neither flip it nor degrade it to "unknown".
3. the guard was rewritten in the same cut.

Also closed on the way: a stray save in the order displayed TODAY had no map at
all, because the current order never had a table. It is a verdict now and maps
back. The two historic tables share ONE mapper instead of two copies of the
same walk.

## Guards

Suite **84 -> 86/86**. Two new:

- `test_v851_upscale_shift_low.py` (7/7) — serialisation law, baseline
  agreement, the sentinel DEFAULT (the number that decides whether old
  workflows change), resolver imported not copied, the sentinel table written
  from the SPEC, order of application proved with a recording
  `_apply_sigma_shift`, five fallback cases run end to end, and the real
  frontend load chain.
- `test_v852_display_resort.py` (6/6) — canon untouched, display is a
  permutation with every HIGH/LOW twin adjacent (checked as a RULE over all
  seven pairs), both historic tables verbatim, the fingerprint's witness sets
  really draw the lines they claim, a save from all four eras comes home with
  the mapping branch LIFTED out of `configure()`, and a marked save
  short-circuits the fingerprint.

`test_v546` amended (pins the new display order and both frozen tables),
`test_v589` now runs the REAL load chain instead of a hand-built shortcut, and
both amendments are declared rather than quietly applied.

`WIDGET_ORDER_baseline_v368.txt` -> **v370**: 20 classes, 1 dynamic, unchanged;
exactly one row differs and the old list is a strict prefix of the new one.

## Unchanged

No route was added or touched — `uls_routes.py` is byte-identical for the NINTH
round running, and so are `ph_media_routes.py` and `ph_sampler_routes.py`. Node
count stays **20**.

## Installing

Backend AND frontend changed: restart ComfyUI, then Ctrl+F5. Existing
workflows open unchanged, with the new dial at -1.
