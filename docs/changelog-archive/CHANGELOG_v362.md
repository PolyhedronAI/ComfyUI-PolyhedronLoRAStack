# v362 — Media I/O: Polyhedron Media Loader & Polyhedron Save

Two new nodes join the pack. Nothing else changed: every Stack node, its
routes and its frontend are byte-identical to v361.

## New — ⬡ Polyhedron Media Loader

A pinned-folder media browser that loads images, video and audio from
**any** folder on the machine running ComfyUI — not just `input/`.

- **Pin a folder** by browsing it, typing a path, using the native file
  dialog, or dropping a file onto the node (two drop zones: copy into
  `input/`, or load the file where it lies and pin its folder).
- **Tile grid** with live hover preview — images pop up large, videos play
  muted in their tile, audio plays on hover; plus a filterable list view.
- **Video trim** with a scrubber, timecode fields and a fixed-length window;
  **audio pairing** with its own trim, and a documented hierarchy for which
  medium is the master.
- **Batch modes**: check tiles and load them as ONE clip (*Video frames*),
  or feed them in one per run (*Separate files*, swept by the node's own
  **▶ Run all** button). Name filters, sort orders, size rules and saved
  sequences included.
- **12 outputs** — `image`, `mask`, `video`, `audio`, `video_audio`,
  `frame_count`, `fps`, `width`, `height`, `filename`, `batch_info`,
  `video_path`.

## New — ⬡ Polyhedron Save

One output node for stills, video and audio. It picks the writer from what
is actually wired, embeds the workflow so the file can be dragged back into
ComfyUI, and drops a control still beside every clip (an `.mp4` cannot carry
a workflow, a PNG can).

- Backends: ComfyUI's native VIDEO path (H.264 MP4), Pillow (GIF/WebP), and
  PyAV for the masters native cannot do (H.265 grain-tuned, ProRes 422 HQ,
  ProRes 4444 + alpha, FFV1, VP9).
- `filename_prefix` is a **path** with tokens — `%date:yyyy-MM-dd%`,
  `%width%`, `%Node.widget%`, counters — so a run can file itself into
  dated subfolders.

## Structure — how this was added (and how the next node will be)

The two nodes are a **self-contained group**:

| What | Where |
| --- | --- |
| Nodes | `nodes/ph_media_loader.py`, `nodes/ph_save.py` |
| Their helpers | `nodes/ph_media_util.py`, `nodes/ph_save_util.py` |
| Their server routes | `nodes/ph_media_routes.py` (own module, own registration call) |
| Their frontend | `web/js/ph_media_loader.js`, `web/js/ph_save.js` |
| Registration | one guarded import group + two mapping entries in `__init__.py` |

`nodes/uls_routes.py`, every Stack node and every Stack `.js` file are
**untouched**. See `MAINTAINING.md`.

## Renamed — the pack is now "Polyhedron Suite"

Label only. The startup banner, the Comfy Registry `DisplayName`, the
frontend menu entry, the README and the manual now read **Polyhedron
Suite**; the two halves are Media I/O and LoRA Stack.

Nothing that identifies the package moved: the registry id stays
`polyhedron-lora-stack`, the publisher stays `polyhedron`, the repository
stays `ComfyUI-PolyhedronLoRAStack`, every `NODE_CLASS_MAPPINGS` key is
unchanged and so is every route — **existing workflows and existing
installs are unaffected, and an installed pack updates in place.** The node
⬡ Polyhedron LoRA Stack keeps its own name; only the pack around it was
renamed.

One thing to know when searching: ComfyUI-Manager lists the pack under its
display name, so search for **"Polyhedron Suite"** from this release on.

## Housekeeping

- Release notes moved from the repository root into
  `docs/changelog-archive/`; the root keeps only the current one.
- `docs/` now carries the combined manual
  `Polyhedron_Suite_Documentation_v362.pdf` (Part I Media, Part II Stack —
  the v313 Stack manual is included unchanged as Part II).

## Dependencies

Still zero required installs. The new nodes use Pillow, OpenCV and PyAV
**lazily**, each with a working fallback and a one-line hint when a feature
needs one. Recommended extras are listed in `requirements.txt`.
