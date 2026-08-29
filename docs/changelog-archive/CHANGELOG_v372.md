# v372 — Attention, NAG, Filter, Audio Stretch — and everything the tree owed

Version triple `3.72.0` / banner `Polyhedron Suite  v372` / `PLUGIN_VERSION v372`.
**33 → 37 nodes.** 227 → 309 files.

This release carries **v371 in it**. That package was built but never
published — the live tree stood at v370 when this cut started, confirmed by
pulling it. Same situation as v366/v367, and the reason the first thing this
build did was fetch the live tree instead of trusting the folder.

## What is new

**⬡ Polyhedron Attention** — swaps the attention backend, and can swap it per
step range. The dropdown offers only what really loads on the machine, because
an option that fails when picked is worse than one that was never offered.
Includes `comfy kitchen int8` and a `live_check` that probes before the run.

**⬡ Polyhedron NAG** — Normalized Attention Guidance on a conditioning. Works
inside attention, so it stays useful where CFG is turned down or off.

**⬡ Polyhedron Filter** — colour grading whose preview *is* the grade: the
browser applies the same `.cube` the backend will.

**⬡ Polyhedron Audio Stretch** — retimes a soundtrack to a measured target,
sharing one implementation with the Interpolate node's `audio_mode`.

## The fourth route module

The Filter is the first node in the public build since v365 that genuinely
needs endpoints — its live preview must read the same LUT file the backend
grades with. The three routes went into **`nodes/ph_filter_routes.py`**,
following the rule the Media Loader and Sampler already set: one new module,
one registration call, and no shared route file is ever reopened.

`nodes/uls_routes.py` is therefore **still `bcc4d8c4`** — the tenth release
running without a byte changing.

`test_v365_public_build` now pins that mechanically: the routes must live in
the Filter's own module, `__init__` must call it, `ph_filter.js` must not call
a path the module does not serve, and the module must not *import* the other
three. That last check was a substring search in its first draft and went red
on the new module's own header, which names the three files in order to explain
why it does not use them. Prose is not a dependency — it checks imports now,
and the mutation was run to prove it still bites.

## Everything else the tree owed

The public tree had been built from an internal state older than v894. All 44
diverging files were classified individually before anything was touched:

* **31 were arrears** and were lifted: thirteen node modules, nine frontend
  files, nine guards (all of them internally re-grounded in v885/v887/v894).
* **10 are public's own** and were left alone: `README.md` (public's is the
  manual, 1282 lines against 239), `uls_routes.py`, `COMPATIBILITY.md`,
  `install.py`, `requirements.txt`, `uls_compat.js` (its only difference is
  the version pin), and four guards that read `ph_media_routes.py` or are
  re-grounded on the public import style.
* **3 were edited rather than replaced**: `__init__.py`, `pyproject.toml`, and
  `test_v577_widget_order.py` — a hybrid. Public keeps its own census; the
  internal **v901 schema scanner** was lifted, because that is the wound that
  in v900 shipped a package in which a node did not exist at all.

## Three modules that were missing, and how they were found

Lifting a file is not lifting a node. A scan of every sibling import in
`nodes/` after the lift turned up three modules that the public tree never had:

* **`uls_pick_frame_core.py`** — `wan_frame_inflate.py` imports it *hard*. The
  package would have died on import.
* **`ph_audio_stretch.py` + `uls_audio_math.py`** — `ph_interpolate.py` imports
  them lazily for `audio_mode="stretch"`. This one is worse than a crash: the
  package would have started green and failed in the field on the first stretch
  run. Silent degradation is the failure class this tree spends its guards on.

Registering `ULSAudioStretch` follows from the second: the file is in the tree
either way, it has its own guard, it opens no route — and an unregistered node
class is dead code the serialisation scanner reports on every run.

## The manual grew a Part VI

`docs/Polyhedron_Suite_Documentation_v372.pdf` — **89 pages** (was 83),
replacing the v369 file in the same release. New **Part VI — Attention &
grading**, six pages: the attention machinery and the step split, the backend
list, NAG, the Filter, Audio Stretch. The old parts are UNMOVED (III on 59, IV
on 69, V on 76); only the appendix slid from 83 to 89. The four new README
chapters carry the same figures from `assets/v6_*.png`.

## Census, measured not derived

`test_v577`'s public floor is **34**, not the 35 arithmetic suggested. 37 node
classes, three declared dynamic, no CANON row (the node that owns it does not
exist here). The gate said 34 on the first run and the gate was right.

## Verification

py_compile **187/0** · pyflakes undefined **0** · node --check **23/0** ·
suite **128/128** · 44 image references with no missing target ·
`uls_routes.py` **bcc4d8c4** unchanged · baselines regenerated to **v372**
(33 → 37 classes, exactly four added, none lost, no prefix violation) ·
ZIP == tree.
