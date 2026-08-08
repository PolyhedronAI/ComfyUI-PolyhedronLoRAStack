# CHANGELOG v369 — documentation: what the encoder actually receives

Version 3.69.0 · **documentation only** · no node source changed, no guard
logic changed.

Files touched: `docs/Polyhedron_Suite_Documentation_v369.pdf` (**new**,
replaces the v368 file), `assets/cte_comments_before.png` and
`assets/cte_comments_after.png` (**new**), `README.md`, `pyproject.toml`,
`web/js/uls_compat.js`, `__init__.py`, and the version pins in
`tests/test_v348_merge_math.py`, `tests/test_v351_pickframe_v3.py`,
`tests/test_v352_inflate_v3.py`, `tests/test_v365_public_build.py`.

---

## 1. The change

`strip_comments` was documented in words and illustrated with a screenshot of
the node showing a higher word count. That explains the *effect* but not the
*mechanism* — a reader still had to picture what the encoder ends up with.

Part III section 3 and the matching README chapter now show it directly, as a
before and after:

* **What you write** — the prompt with its `//` headings and notes.
* **What the encoder receives** — every marked line gone, the rest joined by
  the separator.

The same example runs through both panels, so the difference between them *is*
the feature. The earlier screenshot stays: it makes the second point, that
switching the control off turns those headings into 127 words instead of 99.

The two panels are drawn rather than cropped. The node's own text box scrolls,
so a screenshot of this example is always cut off part-way through — the last
heading would have appeared without the line it introduces, which is exactly
the thing the picture needs to show. The colours are sampled from the real box
(`#2F4F2F` background, `#DDDDCB` text, `#CBAA63` comments), so it reads as the
node it describes.

## 2. Documentation

**83 pages** in five parts and an appendix, up from 82. Part III grows from 9
pages to 10 — section 3 now runs over three pages, so sections 4, 5 and 6 move
down by one and everything from Part IV onward shifts by a page.

Verified page by page rather than assumed:

* Parts I and II lifted unchanged — **56/56 textually identical**.
* Part IV, Part V and the Appendix — **15/15 identical** at their new page
  numbers, once the declared footer change v368 → v369 is normalised away.
* Part III pages 1–3 unchanged; the growth is where the figures were added.
* Contents, cover version, bookmark targets and the appendix page number all
  follow. Bookmarks stay at 46; no blank page anywhere.

## 3. Unchanged on purpose

No node source, no guard logic, no baseline. `nodes/uls_routes.py` remains
**bcc4d8c47e500c892619c22d51f7cbc6** — the eighth release in a row it has not
been re-opened. `WIDGET_ORDER_baseline_v368.txt` stays as it is: nothing about
any node's widgets changed, and regenerating a baseline without a reason to is
how a real shift gets waved through later.
