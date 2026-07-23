# CHANGELOG v364 — Polyhedron Suite (public build)

Catch-up release. The public build was forked from the internal line at v666
and has been carrying two nodes (Media Loader, Save) plus the old Stack core
since v362. Everything below is an internal fix that had not reached the
public tree yet. No new nodes; the next nodes still arrive one at a time as
described in MAINTAINING.md.

Note: v363 was built but never uploaded. Its single change (the token toast)
is contained in this release, so v363 is skipped on the public branch.

## Fixed — trigger words: caption statistics were read as triggers (internal v579/v580)

`ss_tag_frequency` in a LoRA header is kohya's DATASET CAPTION STATISTICS —
every tag that ever appeared in a training caption, with its count. The
flattener treated it as a trigger list. Measured in the field on eleven LoRAs
whose headers carry tag frequencies:

    TRIGGERS : 1710 / 512  (334.0%)   <- 20 caption fragments x 11 LoRAs
    TRIGGERS :   38 / 512  (  7.4%)   <- same LoRAs, filename fallback

The tag soup beat the correct answer, because the header step short-circuited
the filename step. The Inspector hid it: it printed `triggers[0]` truncated to
the column width, so a twenty-tag soup showed up as one innocent word.

- `_looks_like_trigger_list()` — a real trigger list is short (<= 6 tags,
  <= 96 chars). Anything wider is caption statistics wearing a trigger's coat.
- `_cap_tags()` — no exit from the flattener may be unbounded. Two of them
  were: a non-JSON string and a non-dict value came back verbatim.
- One console notice per LoRA per process, naming the file and the tag count,
  and pointing at the row's trigger field as the way to pin a real trigger.
- Trigger PROVENANCE is now reported (v580): a value derived from the filename
  says that it is a guess instead of presenting itself as fact.

## Changed — token limit notice is amber, and says itself once (internal v492)

`web/js/uls_token_toast.js` raised the over-limit notice with severity
`"error"` — red and sticky. It is a budget warning, not a failure, so it is
now `"warn"` (amber). `onExecuted` fires on every run and the notice is
sticky, so identical notices used to stack up; a per-node signature over
(state, worst count, limit) re-notifies only when something actually changes.

## Changed — Media Loader preview mechanics rebuilt (internal v683/v684/v692)

`web/js/ph_media_loader.js` only. The route surface is unchanged: the same
four endpoints as before, no new imports beyond ComfyUI core, so
`nodes/ph_media_routes.py` is untouched.

- Deferred dimensions: opening a listing no longer probes every file. The
  probe follows what is actually on screen.
- Lazy thumbnails: one IntersectionObserver per loader UI, created on demand.
  Tile URLs are parked rather than assigned, so a large folder no longer
  fires hundreds of requests at once; discarded tiles are unobserved.
- `renderGrid` is now the single funnel every path goes through (refresh,
  solo, return-to-tiles), which is what made the deferred paths safe.
- Locate no longer fails silently: if the target is not in the list after the
  attempt, it says so, shows pending work, and stops spinning on a dead thumb.

## Added — one door to ComfyUI's versioned node API (internal v577/v580)

`nodes/ph_comfyapi.py` (new file). Six `*_v3.py` nodes each did
`from comfy_api.latest import io`. By ComfyUI's own documentation `latest` is
the API version still under development — changed without warning. A broken
import is caught by the existing per-node isolation; a CHANGED SIGNATURE is
not, because the import succeeds and the error only surfaces when ComfyUI
builds its node list.

The door pins to `v0_0_2`, measured rather than guessed: the console line it
prints on load reports which pinned versions the running ComfyUI actually
offers. It falls back through older pins to `latest`, so a ComfyUI without
`v0_0_2` loses nothing.

## Changed — node descriptions

The `*_v3` nodes and the sigma schedule carry the internal descriptions, which
explain what a node is for rather than restating its name.

## Housekeeping

- `uls_preview_gen.py`: f-strings without placeholders removed (lint only).
- `CHANGELOG_v362.md` moved to `docs/changelog-archive/`.
- Version triple raised together: `pyproject` 3.64.0, `__init__` banner
  `Polyhedron Suite  v364`, `uls_compat.js` `PLUGIN_VERSION = "v364"`.

## Deliberately NOT changed

`nodes/uls_routes.py` stays byte-identical — the Media Loader routes live in
their own module. `README.md`, `__init__.py`, `install.py`, `requirements.txt`
and `COMPATIBILITY.md` keep their public wording: the internal versions
describe vendored trimesh, the assimp FBX engine and the batch pipeline, none
of which ship here. `web/js/uls_node.js` keeps its public console line.
