# Maintaining this repository

This pack grows by **adding files, not by editing shared ones**. That single
rule is what keeps a release reviewable: the diff of a new node should show
new files plus a handful of registration lines, and nothing else.

## Layout

```
__init__.py               registration only — imports, mappings, route calls
nodes/
  uls_*.py, wan_*.py      LoRA Stack group
  uls_routes.py           LoRA Stack server routes
  ph_media_loader.py      Media I/O group ─┐
  ph_media_util.py                         │  self-contained:
  ph_save.py                               │  no Stack module imports these,
  ph_save_util.py                          │  and these import no Stack module
  ph_media_routes.py      its own routes ──┘
  ph_sampler_routes.py    Sampler routes — same pattern, own module
  ph_filter_routes.py     Filter routes — same pattern, own module (v372)
web/js/
  uls_*.js                Stack frontend
  ph_media_loader.js      Media I/O frontend
  ph_save.js
docs/
  Polyhedron_Suite_*.pdf  the manual (six parts + appendix)
  changelog-archive/      past release notes
```

## The four rules

**1. A node group owns its own files.**
Node, helpers, routes and frontend live in files that only that group uses.
`nodes/ph_media_routes.py` exists for exactly this reason: the Media Loader
needs server endpoints, and putting them into `uls_routes.py` would mean
every future release touches a file the Stack also depends on. It carries
its own copies of the few small path helpers (`_within`,
`_stream_part_to_disk`, `_cleanup_names`, `_hint_cv2_once`) so it imports
nothing from the Stack side.

**2. Registration is guarded and grouped.**
Every group in `__init__.py` is a `try/except` around its imports with an
`_OK` flag, then a matching `if _OK:` block that fills the mappings. A
failing group prints one actionable line and is skipped — the rest of the
pack still loads. Adding a node means adding to a group, or adding a group.

**3. Optional dependencies are imported lazily, with a fallback.**
`dependencies = []` in `pyproject.toml` is a promise: installing this pack
must never pull anything in. Pillow, OpenCV and PyAV are imported inside the
function that needs them; when one is missing the feature degrades and says
so once. Never move such an import to module level.

**4. Guards are single scripts.**
Files in `tests/` are standalone: a `main()` that prints `PASS: …` and exits
`0`, or prints `FAIL: …` and exits `1`. Run one with `python tests/<file>`,
or the lot with a shell loop:

```bash
for t in tests/test_*.py; do python "$t" || echo "RED: $t"; done
```

They are source-text and behaviour checks with no ComfyUI import, so they
run in a bare checkout. A guard that cannot fail is not a guard — when you
add one, break the thing it protects once and confirm it goes red.

## The name: what may change, what may not

The pack is called **Polyhedron Suite**. That name is a *label* — it appears
in the startup banner, the registry listing, the README and the frontend
menu. Three things carry the same words but are **identity**, and renaming
them breaks installs or saved workflows:

| Identity — never rename | Why |
| --- | --- |
| `name = "polyhedron-lora-stack"` (pyproject) | the Comfy Registry package id; a new id is a new package, and existing installs stop updating |
| `PublisherId = "polyhedron"` | the other half of that id |
| `NODE_CLASS_MAPPINGS` keys (`UltimateLoraStack`, `ULSMediaLoader`, …) | every saved workflow stores these; renaming one orphans the node in user graphs |
| The repository / folder name `ComfyUI-PolyhedronLoRAStack` | clone paths, and the raw.githubusercontent URLs for icon and banner |
| Route prefixes `/uls/...` | the frontend/backend contract |

Node *display* names are a separate matter: **⬡ Polyhedron LoRA Stack** is
the name of one node, not of the pack. It keeps its name.

If the label ever changes again, three places must move together or the
version guards go red — pyproject `version`, the `__init__.py` banner, and
`PLUGIN_VERSION` in `web/js/uls_compat.js`. One more must follow by hand:
the ComfyUI-Manager search term quoted in the README installation section
has to match the registry `DisplayName`, or nobody finds the pack.

## Adding the next node — the checklist

1. Drop `nodes/ph_<name>.py` (plus a `ph_<name>_util.py` if it needs one).
2. If it needs server endpoints: `nodes/ph_<name>_routes.py` with its own
   `register_<name>_routes()`. Do not extend `uls_routes.py`.
3. If it has a UI: `web/js/ph_<name>.js`. It may import only ComfyUI core
   (`../../scripts/app.js`, `../../scripts/api.js`).
4. `__init__.py`: one guarded import, one `NODE_CLASS_MAPPINGS` entry, one
   display name, and — if step 2 applied — one registration call.
5. `tests/test_<name>.py` in the script style above.
6. Bump `version` in `pyproject.toml` (the publish workflow triggers on that
   file) and add `CHANGELOG_v<n>.md`; move the previous one into
   `docs/changelog-archive/`.
7. Documentation: update `README.md` and the manual in `docs/`.

## Release check

```bash
python -m compileall -q nodes __init__.py     # every module parses
node --check web/js/<changed>.js              # every changed script parses
for t in tests/test_*.py; do python "$t" || echo "RED: $t"; done
git diff --stat                               # shared files should be absent
```

That last line is the important one: if `uls_routes.py` or a Stack node
shows up in the diff of a release that only adds a node, something went into
the wrong file.
