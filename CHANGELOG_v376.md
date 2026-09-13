# v376 -- Nodes 2.0: the Filter's Reset in the title bar; AnySwitch status line; the switch point learns a header

Frontend-only release (Ctrl+F5 after replacing the folder). 37 nodes, no new
route, no widget change -- both baselines unchanged. Carries the internal cuts
v959 and v962 for the classes that ship here.

## Filter: Reset where the painted node has it (v962)

Under Nodes 2.0 the Filter's Reset sat in a row at the foot (v374), because
the frontend offers no hook for the title bar. Measured (frontend 1.49.6):
the Vue title bar is `.lg-node-header > div`, a flex row with the title at
the left; an element appended there sits at the right and survives a title
change, a resize and a redraw. The switch point (`web/js/uls_vue_views.js`)
gets an optional `spec.header(node, chip)`: built once, appended while the
view is shown, put back if a Vue re-render dropped it, removed on leave;
pointer events on the chip do not bubble into the drag handle. A header-only
spec adds no widget row -- the Filter node is 24 px shorter under Nodes 2.0.

## AnySwitch / AnySwitchInv: the status line after a run (v959)

The line at the foot ("-> A (image)" etc.) is painted in onDrawForeground,
which Nodes 2.0 never calls -- so it was simply absent there, and no fresh-node
sheet could show it. Now a view through the switch point; `ph_switch.js`
exports `switchStatusLine(node)` and paints from it too, so the two
renderers read one function. Both classes leave `VUE_USABLE` in
`web/js/uls_compat.js` (a viewed class is classified through its view).

## Guards

- NEW tests/test_v959_after_run_views.py (public scope: the switch view, the
  shared reader, the views rebuild nothing) and
  tests/test_v962_vue_header_chip.py (the switch point on a fake DOM: chip on
  show, put back after a re-render, removed on leave, no row for header-only,
  pointerdown stopped; Filter registers header-only).
- tests/test_v941_extras.py re-grounded (Filter via spec.header; the six v941
  classes still register); tests/test_v943_parity_p0.py "today" pin 6 / 8 ->
  4 / 10.

Version triple 3.76.0 / "Polyhedron Suite  v376" / PLUGIN_VERSION v376.
`nodes/uls_routes.py` untouched (bcc4d8c4). Handbook stays the v372 edition.
