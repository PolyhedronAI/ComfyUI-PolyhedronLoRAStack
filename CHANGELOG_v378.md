# v378 — the folder listing no longer asks the disk

A follow-up to [v377](https://github.com/PolyhedronAI/ComfyUI-PolyhedronLoRAStack/blob/main/docs/changelog-archive/CHANGELOG_v377.md),
found while field-testing it.

## What was wrong

Files that share a modification time down to the second were left in whatever
order the operating system handed them over. The listing sorted by
modification time and nothing else, and that sort is stable — so on an exact
tie, the file system decided.

Most folders never notice: timestamps differ. But a batch of files unpacked
from one archive, copied in one operation, or written by one export all end up
sharing a minute, and then the grid showed an order that nothing in the data
justified. Worse, it was not reproducible: the same folder could list
differently on another disk, after a copy, or after a directory rebuild, with
nothing having changed.

## What changed

The listing now sorts by modification time **and then by natural name** —
the same name ordering the grid's `Number` mode and the batch pipeline
already use. Folders whose timestamps differ are completely unaffected; the
second key is only ever reached on an exact tie.

The practical effect: a freshly unpacked set of files now appears in sensible
name order instead of an arbitrary one, and the same folder always lists the
same way.

## Note for anyone testing sort orders

Timestamps frequently do **not** survive being unpacked from an archive —
many extractors stamp every file with the moment of extraction. If you want
to check the time-based grid orders, use a folder that grew naturally rather
than one you just unpacked.
