# v377 — Media Loader: AVIF, and a grid you can order

This release answers [#4](https://github.com/PolyhedronAI/ComfyUI-PolyhedronLoRAStack/issues/4).

## AVIF

`.avif` is now a recognized image format, in both the single-file and the
image-batch path.

Listing a format and *decoding* it are two different things, so the node no
longer assumes: it asks Pillow whether an AVIF decoder is actually present
(feature registry first, extension registry second — never a guess from the
version string). If the file cannot be decoded, the Media Loader says which
file and what to do about it — Pillow 11.3 or newer, or the
`pillow-avif-plugin` package — instead of letting a bare
`UnidentifiedImageError` through.

WebP was already supported and still is.

## One extension law

The recognized-extension lists used to exist twice, once in the loader and
once in the media routes, kept in step by a comment. That is why a format
could end up half-supported. They now live in a single place that both sides
read, and a guard refuses any second copy.

## The tile grid has an order of its own

The grid was fixed to newest-first with no way to change it — and because it
is the most visible list in the node, it was easy to read as the order a
batch would *run* in. It never was: that order has always been set in
**▦▶ Batch…**, which defaults to Number (natural name order).

The grid now carries its own order, cycled from a handle at the right of the
pager row it already had:

**Newest first** (default) · **Number** · **A–Z** · **Oldest first** ·
**Date created**

Two things worth knowing:

- **Nothing changes for existing workflows.** The default is the behaviour the
  grid always had, and the setting is stored per node, outside the widget
  values — loading an older workflow looks exactly as it did.
- **The grid orders the browser, not the run.** The batch order stays
  independent, and the handle's tooltip says so.

Every grid order is built from the same ordering rules the batch uses, so the
grid can never display an order the backend would not reproduce.

## Fixed: "Date created" sorted by modification time

The frontend's copy of the ordering rules read the modification time for both
time-based modes, because the folder listing never carried a creation time.
The backend had always used the real creation time. The result: in "Date
created" mode, the Batch dialog previewed one order while the run used
another. The listing now carries the creation time and both sides agree.

Thanks to **@TheNextLevel** for the report — the pointer to how core's
`LoadImage` filters by MIME type was what made the AVIF case clear-cut.
