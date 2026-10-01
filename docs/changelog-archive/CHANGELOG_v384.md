# v384 -- 3.84.0: Sigma List and Show Text online, curve plots for Sigma Curve / Dual Sigma Curve, search in the CLIP Text Encode

Backend + frontend. Restart ComfyUI, then reload the browser page (Ctrl+Shift+R).
Two new nodes (38 -> 40). No node id renamed or removed, no widget moved
(baselines NODE_IDS v383 -> v384, WIDGET_ORDER v383 -> v384: two lines
added, none changed). Old workflows load unchanged.

## Why

A real MiniMax H3 workflow built with this suite used two nodes the public
build did not register: **Sigma List** and **Show Text**. Opened with 3.83.0
it showed four red "missing node" boxes. Checking every frontend file against
the public nodes then found two more pieces that were missing online: the
curve plot of Sigma Curve / Dual Sigma Curve and the search row of the CLIP
Text Encode.

## New: Polyhedron Sigma List

An explicit SIGMAS grid instead of a computed curve -- the schedule a model or
an accelerator LoRA was trained on (presets, e.g. `hyperflow_8step_h3_raw`),
or your own comma-separated list, with `shift` and `enforce_terminal_zero`.
The node draws the grid the run will use. A typo in the list shows up in the
node as the run's own refusal, before you queue anything.

## New: Polyhedron Show Text

Shows any text that reaches it -- an `info` output, a resolved prompt -- on
the canvas, sized to its content, with the Note's colour row.

## Sigma Curve / Dual Sigma Curve: the curve is drawn

Both nodes now draw their curve in the node. It is computed by the same
function the node's run uses, so the picture cannot drift from the result.

## CLIP Text Encode: search

A search row under `segments`: every hit of a word, half sentence or
paragraph is marked live in the visible prompt fields; Enter / Shift+Enter or
the arrows jump between hits, Esc clears, Ctrl+F inside a prompt field opens
this search. The text itself is never touched.

## Routes

The Sigma List and the curve plots ask the backend for their numbers (the
frontend never calculates). Their three endpoints live in a fifth route
module of their own, `nodes/ph_sigma_routes.py`:

- `GET  /pls/sigma_presets`
- `POST /pls/sigma_preview`
- `POST /pls/sigma_curve_preview`

None takes a path or touches a file. `uls_routes.py` and the other route
modules are unchanged. Show Text and the search open no route.

## Checked

- New guards `test_v971_sigma_list`, `test_v972_sigma_presets`,
  `test_v974_sigma_list_ui`, `test_v976_sigma_curve`, `test_v984_sigma_curves`,
  `test_v990_show_text_fit`, `test_v992_cte_search` (the three route guards
  read `ph_sigma_routes.py` in the public build); `test_v365_public_build`
  pins 40 nodes and the fifth route module.
- The workflow that showed the gap: all 28 node types present, loads in both
  renderers without a missing node.
