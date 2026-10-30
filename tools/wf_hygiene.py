#!/usr/bin/env python3
"""wf_hygiene.py -- v1247: a workflow that leaves the bench carries no bench paths.

The Ritter-Test workflows of 28.10. reached Frank with the Media Loader pinned to the build machine's
folder (/home/claude/srv/ComfyUI/input) -- "Cannot read this folder" on his computer. This check walks
every string of a workflow JSON (widgets, properties, extras) and names the ones that point into a
build sandbox. Use it before any workflow is handed out; the guard test_v1247 runs it on the pack's
example_workflows.

    python3 -I tools/wf_hygiene.py FILE_OR_FOLDER [...]     exit 0 = clean, 1 = findings
"""
import json
import os
import re
import sys

# paths that exist on a build machine, never on a user's: the sandbox home, its temp, a container root
BENCH = re.compile(r"(/home/claude\b|(?<![A-Za-z])/tmp/|/root/|/mnt/user-data|/workspace/[a-z]|\bsandbox\b)")


def strings(obj, path="$"):
    """Yield (json_path, string) for every string anywhere in obj."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from strings(v, "%s.%s" % (path, k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from strings(v, "%s[%d]" % (path, i))
    elif isinstance(obj, str):
        yield path, obj


def findings(wf):
    """[(json_path, string)] for every bench path in a loaded workflow."""
    out = []
    for p, s in strings(wf):
        if BENCH.search(s):
            out.append((p, s))
    return out


def check_file(path):
    with open(path, encoding="utf-8") as fh:
        wf = json.load(fh)
    return findings(wf)


def main(argv):
    files = []
    for a in argv:
        if os.path.isdir(a):
            for r, _, fs in os.walk(a):
                files += [os.path.join(r, f) for f in sorted(fs) if f.endswith(".json")]
        else:
            files.append(a)
    bad = 0
    for f in files:
        try:
            hits = check_file(f)
        except Exception as e:
            print("SKIP %s (%s)" % (f, e))
            continue
        for p, s in hits:
            bad += 1
            print("%s  %s  %r" % (f, p, s[:120]))
    print("wf_hygiene: %d file(s), %d bench path(s)" % (len(files), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
