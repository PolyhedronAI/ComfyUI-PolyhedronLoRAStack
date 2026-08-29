"""Guard v578 -- the NODE IDENTITY gate.

WHY THIS EXISTS: a rename came up, and a rename is where node packs die.

There are SEVEN names in a ComfyUI pack, and they have wildly different blast
radii. Exactly ONE of them is load-bearing:

  node_id  ("ULSVAE")        -> stands in EVERY saved workflow as {"type": ...}.
                                Move it and every workflow that used the node
                                turns into a red "missing node" box. FOREVER.
  display_name ("Polyhedron VAE Codec")  -> cosmetic. LiteGraph only serialises
                                a node's title when the USER overrode it, so a
                                changed display name simply shows up in old
                                workflows. FREE.
  CATEGORY                   -> menu position. FREE.
  python class name          -> internal. Free, but pointless churn.
  folder name                -> the /extensions/<folder>/ URL for WEB_DIRECTORY.
                                Measured: this pack has ZERO absolute paths with
                                the pack name in its JS, so it survives - as long
                                as the OLD folder is removed (two folders = the
                                same node_id registered twice).
  pyproject `name`           -> the REGISTRY SLUG. A new slug is a NEW registry
                                entry; the old one orphans and installed users
                                stop getting updates.
  GitHub repo name           -> GitHub redirects, but the Registry's Repository
                                field and the Manager DB point at it.

THE LAW, in one line:
    RENAME EVERYTHING YOU SEE. NEVER RENAME WHAT THE FILE STORES.

This gate enforces the second half. Display names, categories and the pack's
own banner may change as often as they like -- the node_ids are pinned, and any
drift fails loudly with the reason spelled out.

IF a node_id ever genuinely must change, there IS a safe road and it is NOT to
edit the mapping:
    1. register the NEW node_id,
    2. keep the OLD node_id registered, pointing at the same class, and set
       DEPRECATED = True on it. ComfyUI hides deprecated nodes from the search
       but keeps them fully working in existing workflows.
    3. regenerate this baseline in that cut, and say it in the changelog.
That road is a declared act. This gate makes sure it stays one.

Also checks that no two nodes share a display name -- two identical labels in
the search box is a real defect, and it is exactly the confusion that started
this cut ("Polyhedron VAE" next to "Polyhedron Load VAE").

Script-style: exit 0 = pass.
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _newest_baseline():
    """AMENDED IN v580 (1st amendment): the baseline FILENAME was a hard-coded
    text pin ("NODE_IDS_baseline_v579.txt"), so every cut had to remember to
    hand-edit this guard. A guard that must be hand-edited on a schedule will
    one day be hand-edited wrong.

    Lesson 1 of the handover, applied to a guard that exists to enforce lesson 1:
    pin the STRUCTURE, not the string. Same amendment landed in
    test_v577_widget_order.py in the same cut.

    AMENDED IN v581 (2nd amendment): two baselines in the tree used to be taken
    silently by max(). Its twin in test_v577 documented that case as "an
    ambiguity, not a convenience" and then tolerated it anyway; this one did not
    even document it. A stale baseline is a second memory, and a gate with two
    memories has none. Two now fails. The vMISSING sentinel is gone with it: it
    turned "the baseline is missing" into a filename that _read() would fail on
    LATER, one layer away from the truth. Missing fails HERE, where it happened.
    """
    hits = sorted(glob.glob(os.path.join(ROOT, "NODE_IDS_baseline_v*.txt")))
    if not hits:
        _fail("no NODE_IDS baseline found (NODE_IDS_baseline_v*.txt) - the gate "
              "has no memory to compare against")
    if len(hits) > 1:
        _fail(f"{len(hits)} NODE_IDS baselines present "
              f"({', '.join(os.path.basename(h) for h in hits)}) - exactly one "
              f"per cut. Delete the stale one; an ambiguity is not a convenience")

    def _v(p):
        stem = os.path.basename(p).rsplit("_v", 1)[-1]
        return int("".join(c for c in stem if c.isdigit()) or 0)
    return max(hits, key=_v)


BASELINE = None   # resolved below, after _fail is defined -- see v577's twin


def _fail(msg):
    print("[test_v578_node_ids] FAIL: " + msg)
    sys.exit(1)


def _registered():
    """{node_id: display_name} exactly as __init__.py registers them."""
    src = open(os.path.join(ROOT, "__init__.py"), encoding="utf-8").read()
    ids = set(re.findall(r'NODE_CLASS_MAPPINGS\[\s*"([^"]+)"\s*\]\s*=', src))
    ids |= set(re.findall(r'^\s*"(\w+)":\s*\w+,\s*$', src, re.M))
    dsp = dict(re.findall(
        r'NODE_DISPLAY_NAME_MAPPINGS\[\s*"([^"]+)"\s*\]\s*=\s*"([^"]+)"', src))
    dsp.update(dict(re.findall(r'^\s*"(\w+)":\s*"(\u2b21[^"]*)"', src, re.M)))
    return {i: dsp.get(i, "") for i in ids if i in dsp}


def _baseline():
    p = os.path.join(ROOT, BASELINE)
    if not os.path.isfile(p):
        _fail(f"{BASELINE} is missing - the gate has no memory of what the "
              f"saved workflows call these nodes")
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def main():
    global BASELINE
    BASELINE = os.path.basename(_newest_baseline())

    now = _registered()
    want = _baseline()

    if not want:
        _fail(f"{BASELINE} holds no node_id - it is empty or malformed")

    # --- 1: no node_id may VANISH ---------------------------------------------
    gone = [i for i in want if i not in now]
    if gone:
        _fail(f"node_id(s) GONE from the registration: {', '.join(sorted(gone))}.\n"
              f"       Every saved workflow that used them now shows a red "
              f"'missing node' box. If this is a rename, do NOT edit the "
              f"mapping: register the new id AND keep the old one alive with "
              f"DEPRECATED = True (ComfyUI hides it from search, old workflows "
              f"keep running). Then regenerate {BASELINE} and say it in the "
              f"changelog.")

    # --- 2: a NEW node_id is a declared act -----------------------------------
    fresh = sorted(set(now) - set(want))
    if fresh:
        _fail(f"new node_id(s) not in the baseline: {', '.join(fresh)}. A new "
              f"node is a declared act - regenerate {BASELINE} in THIS cut so "
              f"the gate carries it from here on.")

    # --- 3: display names must be unique --------------------------------------
    seen = {}
    for nid, name in sorted(now.items()):
        if not name:
            continue
        if name in seen:
            _fail(f"two nodes carry the SAME display name {name!r}: "
                  f"{seen[name]} and {nid}. In the search box they are "
                  f"indistinguishable. Display names are free to change - so "
                  f"change one.")
        seen[name] = nid

    # --- 4: every registered node must HAVE a display name --------------------
    nameless = sorted(i for i, n in now.items() if not n)
    if nameless:
        _fail(f"registered without a display name: {', '.join(nameless)} - "
              f"ComfyUI would fall back to the raw node_id in the menu")

    print(f"[test_v578_node_ids] PASS: {len(now)} node_ids pinned (the string "
          f"every saved workflow stores), {len(seen)} display names, all "
          f"unique. Rename what you see; never rename what the file stores.")
    sys.exit(0)


if __name__ == "__main__":
    main()
