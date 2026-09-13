#!/usr/bin/env python3
"""v957 -- the two field issues from the public repo.

#3 (bytimer): a LoRA downloaded while ComfyUI runs did not show up on "R" /
   Refresh. The list is fetched once at page load; the frontend announces a
   refresh through the `refreshComboInNodes` extension hook, which the pack did
   not implement. Now: the hook drops metaCache/previewCache and re-reads.
#2 (HuntingSuccubus): the Save preview is muted by design (a browser blocks
   unmuted autoplay) but offered no way to turn sound on. Now: a sound button
   next to the loop button, video only, state persisted on the node like loop.
"""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS = os.path.join(ROOT, "web", "js")
FAILS = []
def read(n):
    with open(os.path.join(JS, n), encoding="utf-8") as f: return f.read()
def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m)
    if not c: FAILS.append(m)

def main():
    print("v957: issue #3 (refresh) + issue #2 (preview sound)")
    n = read("uls_node.js"); s = read("ph_save.js")

    # ── issue #3 ───────────────────────────────────────────────────────
    check("export async function refreshLoraList()" in n, "S  refreshLoraList exported")
    fn = n.split("export async function refreshLoraList() {", 1)[1].split("\n}", 1)[0]
    check("_loraListLoaded = false;" in fn and "metaCache.clear();" in fn
          and "previewCache.clear();" in fn and "await loadLoraList();" in fn,
          "S  it drops the loaded flag AND both caches, then fetches again")
    ext = n.split('name: "Polyhedron.stack",', 1)[1].split("async setup()", 1)[0]
    check(re.search(r"async refreshComboInNodes\(\) \{[^}]*await refreshLoraList\(\);", ext) is not None,
          "S  the Polyhedron.stack extension implements refreshComboInNodes")
    check(n.count("loadLoraList();\n") >= 1 and "if (_loraListLoading) return;" in n,
          "S  the one-shot load at page start is untouched")

    # ── issue #2 ───────────────────────────────────────────────────────
    check("this._phSound = false;" in s, "S  the sound state defaults to OFF and persists on the node")
    check(re.search(r"if \(isVideo\) \{\s*soundBtn = document\.createElement\(\"button\"\)", s) is not None,
          "S  the sound button exists for video only")
    check('el.muted = true;                        // browser autoplay requirement' in s,
          "S  the element still STARTS muted (a browser blocks unmuted autoplay)")
    hook = s.split("this._phApplySound = () => {", 1)[1].split("};", 1)[0]
    check("el.muted = !this._phSound;" in hook and "el.play()" in hook,
          "S  the hook unmutes and resumes playback")
    check("if (soundBtn) bar.appendChild(soundBtn);" in s, "S  the button sits in the transport bar")
    check(re.search(r"soundBtn\.title = this\._phSound[\s\S]{0,200}audio was wired and the preset carries an audio track", s) is not None,
          "S  the tooltip says what silence can also mean (no audio wired / preset without audio)")

    print("v957: %s" % ("PASS" if not FAILS else "FAIL (%d)" % len(FAILS)))
    return 0 if not FAILS else 1

if __name__ == "__main__": sys.exit(main())
