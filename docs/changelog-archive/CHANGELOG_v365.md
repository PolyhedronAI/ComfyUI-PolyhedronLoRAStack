# CHANGELOG v365 -- Polyhedron Suite (public build)

Two nodes join the published pack, and the Media Loader catches up with the
internal line. Built to the rule in MAINTAINING.md: **a group owns its own
files**. `nodes/uls_routes.py` is byte-identical to the v362 release
(`bcc4d8c47e500c892619c22d51f7cbc6`) for the fourth release running, and the
new guard now pins that fact instead of trusting it.

## New nodes

### Polyhedron Sampler (`ULSSampler`)
Classic-wiring KSampler base: a real denoise field, the full sampler and
scheduler lists, `add_noise`. The sampling loop is delegated to ComfyUI core
(`comfy.sample` / `comfy.samplers` / `latent_preview`), so a changed core API
degrades the node instead of felling the pack.

* New files: `nodes/uls_sampler.py`, `web/js/uls_sampler.js`.
* New support modules the node needs: `nodes/ph_logmute.py`,
  `nodes/ph_runclock.py`, `nodes/ph_weights.py` (the one download door --
  `.part` staging, sha256 verification, atomic rename).
* New route module `nodes/ph_sampler_routes.py` with three endpoints, each
  registered under the bare path **and** the `/api` alias:
  `POST /pls/sampler/preview_mode` (switch the live preview decoder while a
  render runs -- preview only, it never touches the output latent),
  `POST /pls/sampler/tae_status` (is the TAE preview decoder on disk?),
  `POST /pls/sampler/tae_install` (fetch the pinned TAE through the one door).
  Every handler imports the node lazily, so a broken node cannot stop the
  module from loading.

### Polyhedron CLIP Text Encode (`ULSCLIPTextEncode`)
Reuses the CORE encoder and the Token Counter's own `_count_tokens`, so the
token figure has one source of truth. New files: `nodes/ph_clip_encode.py`,
`web/js/ph_clip_encode.js`. No server routes, no house imports in the JS.

## Media Loader -- deferred dimensions

The published loader carried the v362 listing; the internal line rebuilt it.
The frontend now asks for the page it is about to draw instead of probing the
whole folder, so a thousand-file folder costs a dozen probes.

* `nodes/ph_media_loader.py` and `web/js/ph_media_loader.js` pulled to the
  current internal state.
* `nodes/ph_media_routes.py` gains `GET /uls/media/dims` plus
  `_scan_media_fast` / `_media_dims_for` / `handle_media_dims`;
  `handle_media_list` honours `dims=0`. **The default is unchanged**, so every
  existing caller keeps its old contract. Routes: 13 -> 14.
* `nodes/uls_stack_node.py` and `nodes/uls_resolve_inspector.py` pulled to the
  current internal state as well.

## Guards

* New `tests/test_v365_public_build.py` -- the build's own law, mutation-tested
  six ways: `uls_routes.py` byte-identical and free of sampler routes; the
  sampler module registers exactly its three routes, with the `/api` alias and
  lazy node imports; `/uls/media/dims` served and called; both nodes registered
  behind their guarded flags; 17 nodes; the version triple agrees; the new
  module is pure ASCII.
* Version triple pins in `test_v348` / `test_v351` / `test_v352` moved to v365.
* 41 guards, all green.

## Not changed on purpose

* `nodes/uls_routes.py` -- upstream's file (see above).
* `install.py` / `requirements.txt` -- the internal versions describe vendored
  trimesh and the assimp engine, neither of which exists in this build.
* `COMPATIBILITY.md` -- the internal version adds a GroundingDINO section for
  nodes this build does not ship.
* `web/js/uls_node.js` and the three media guards -- measured identical to the
  internal versions apart from the published naming and a whitespace-tolerant
  route check, both of which this build is right about.

## Documentation

The manual grows with the pack, and the README follows it.

* `docs/Polyhedron_Suite_Documentation_v365.pdf` replaces the v362 file --
  **74 pages in four parts**. Parts I and II are carried over unchanged
  (verified page for page); **Part III -- CLIP Text Encode** (9 pages) and
  **Part IV -- Sampler** (7 pages) are new, illustrated throughout.
* README: the new nodes get full sections, the Media Loader gains the
  **Mini** view, and the part table at the top lists four parts instead of
  two. Eleven new screenshots in `assets/`.

## Version

3.64.0 -> 3.65.0 (banner `Polyhedron Suite  v365`, `PLUGIN_VERSION v365`).
`CHANGELOG_v364.md` moved to `docs/changelog-archive/`.
