# v367 -- LAN browse lock (opt-in) + event-loop cleanup

## Changed

* **Media routes: locked under `--listen`, open as ever on localhost.** The
  Media Loader's browse family anchors on a client-supplied folder -- the
  whole point on your own machine, but bound to the LAN it would let every
  client on the network read arbitrary directories, upload into them, and
  pop native dialogs on the host. From this release the 11 path-anchored
  routes answer **403** while ComfyUI runs with a non-local bind
  (`--listen`), unless you set **`ULS_ALLOW_LAN_BROWSE=1`** to open them
  deliberately. On the default localhost bind nothing changes --
  byte-for-byte the same behaviour, and an unknown bind reads as local
  (this lock can never break a normal setup). The three routes that operate
  inside the managed sequence-project folder stay open.
* `ph_sampler_routes.py`: `asyncio.get_event_loop()` ->
  `get_running_loop()` (deprecated since Python 3.12).

## Guards

* New `tests/test_v840_lan_lock.py` (numbered after the internal line):
  drives the lock policy through the full bind/override matrix, drives the
  403 door, pins the 11-gated/3-open wiring and the tree-wide
  get_event_loop ban. Mutation-tested (four landed).

## Documentation

* `docs/Polyhedron_Suite_Documentation_v367.pdf` replaces the v366 file --
  **75 pages**. Part IV section 2 gains *The low expert's own shift* (what
  the -1 sentinel means, the 8/5 case, and where the widget bites), and a
  new one-page **Appendix -- Running on a network** explains the browse
  lock, the 403, and the `ULS_ALLOW_LAN_BROWSE` override. Parts I-III are
  carried over unchanged (verified page for page).

## Version

3.66.0 -> 3.67.0
