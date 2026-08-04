# v366 -- sigma_shift_low: a per-expert sigma shift for High + Low

## New

* **Polyhedron Sampler: `sigma_shift_low`.** The LOW-noise expert can now
  carry its OWN flow-matching shift. `-1` = *same as high* (the default --
  existing workflows run unchanged), `0` = off for the LOW expert, any
  positive value is its own shift. `sigma_shift 8.0` + `sigma_shift_low 5.0`
  replaces a graph that fed the two experts through two separate
  ModelSamplingSD3 nodes (the Wan 2.2 8/5 convention).
* The widget sits directly under `sigma_shift`, greys out in Single mode and
  in `Continuous` (where both experts share ONE schedule built from the HIGH
  expert -- the console says so if you set a value there), and bites in
  `Wan MoE parity`. An external SIGMAS curve keeps owning the schedule.
* Saved workflows from every earlier version load unchanged -- the node
  heals older widget arrays by appending the *same as high* default. Two
  long-standing gaps in that healing chain (very old save formats from the
  v4xx era) were fixed along the way.

## Documentation

* README: the Sampler section now describes `sigma_shift_low`; the old note
  ("the shift applies to both experts alike") is gone.
* `docs/Polyhedron_Suite_Documentation_v366.pdf` replaces the v365 file --
  same 74 pages, with Part IV's sigma-shift section updated for the new
  widget.

## Guards

* New `tests/test_v839_sigma_shift_low.py` (numbered after the internal
  line, like every carried guard): pins the canon append, drives the
  resolver and the apply split, drives the JS heal chain and the greying in
  all three states. Mutation-tested.

## Version

3.65.0 -> 3.66.0
