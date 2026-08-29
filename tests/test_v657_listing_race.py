#!/usr/bin/env python3
"""test_v657_listing_race -- the drop-vs-focus listing race, DRIVEN.

The field wound (v655 regression): dropping a file from the Explorer gives
the window focus, the silent focus probe starts and fetches a PRE-upload
listing; the upload's own re-read applies the fresh listing and selects the
new tile; then the probe's stale response lands, "differs" by signature,
overwrites the listing and the new tile vanishes -- first drop "didn't
take", the second sat inside the 5 s debounce and worked.

What runs:
  * _focusReread DRIVEN VERBATIM under node with a controllable fetch:
      - stale scenario: while the probe's fetch is in flight, a newer
        listing fetch takes a ticket (this._listSeq bumps, exactly what
        refreshGrid does) -> the probe's response must be DROPPED (files
        untouched, no render);
      - fresh scenario: no newer ticket -> the changed listing applies;
      - busy scenario: _busyCount > 0 -> the probe never even fetches.
    node absent -> that half is SKIP-AS-PASS.
  * statics: BOTH listing fetches take a ticket (exactly two bumps of
    _listSeq), both check their ticket after the await, the busy
    suppression line exists.

MUTATIONS (wound injected into a COPY, catch proven): M1 the probe loses
its ticket check (the race returns -- driven catch), M2 the busy
suppression falls (static), M3 refreshGrid loses its ticket bump (the
probe's ticket can never go stale -- static count).
"""
import io
import os
import re
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOADER_JS = os.path.join(ROOT, "web", "js", "ph_media_loader.js")


def _fail(msg):
    print("[test_v657_listing_race] FAIL: " + msg)
    sys.exit(1)


def _need(cond, msg):
    if not cond:
        _fail(msg)


def _extract(src):
    m = re.search(r"    async _focusReread\(\) \{.*?\n    \}", src, flags=re.S)
    _need(m is not None, "_focusReread not found")
    f = re.search(r"function _filesSig\(files\) \{.*?\n\}", src, flags=re.S)
    _need(f is not None, "_filesSig not found")
    return f.group(0), m.group(0)


_DRIVER = r"""
%s
let resolveFetch = null;
const api = { fetchApi: () => new Promise((res) => { resolveFetch = res; }) };
class Host {
%s
    renderGrid() { this.rendered = (this.rendered || 0) + 1; }
    // v906 GUARD MAINTENANCE, declared. _focusReread gained three neighbours
    // when the loader learned that a listing is also news about what is GONE.
    // This guard's promise is UNCHANGED -- it is about staleness tickets, not
    // about pruning -- but its stand-in has to carry the new calls or it dies
    // of a NameError and reports a healthy tree as broken. That is exactly the
    // failure _lift.py was written about (v685/v839 stood red for versions).
    // The prune returns nothing here on purpose: this scenario is about which
    // RESPONSE wins, so nothing may vanish and perturb the render count.
    _pruneVanished() { this.pruned = (this.pruned || 0) + 1; return []; }
    _renderPreview() { this.repainted = (this.repainted || 0) + 1; }
    _announceVanished() { this.announced = (this.announced || 0) + 1; }
}
async function scenario(bumpDuringFlight, busy) {
    const h = new Host();
    h.folder = "F";
    h._files = [{name:"old.png", size:1, mtime:1}];
    h._lastFocusReread = 0;
    h._busyCount = busy ? 1 : 0;
    h.rendered = 0;
    const p = h._focusReread();
    if (busy) { await p; return { fetched: resolveFetch !== null, applied: false, rendered: h.rendered }; }
    // the probe is now awaiting its fetch
    if (bumpDuringFlight) h._listSeq = (h._listSeq || 0) + 1;   // a refreshGrid started
    resolveFetch({ ok: true, json: async () => ({ ok: true,
        files: [{name:"old.png", size:1, mtime:1}, {name:"new.png", size:2, mtime:2}] }) });
    await p;
    return { applied: h._files.length === 2, rendered: h.rendered };
}
(async () => {
    let ok = true;
    const stale = await scenario(true, false);
    ok = ok && stale.applied === false && stale.rendered === 0;   // stale response DROPPED
    resolveFetch = null;
    const fresh = await scenario(false, false);
    ok = ok && fresh.applied === true && fresh.rendered === 1;    // fresh response applies
    resolveFetch = null;
    const busy = await scenario(false, true);
    ok = ok && busy.fetched === false;                            // busy: never even fetches
    console.log(ok ? "RACE-PASS" : "RACE-FAIL");
})();
"""


def run_driven(src, strict=True):
    try:
        subprocess.run(["node", "--version"], capture_output=True, timeout=10)
    except Exception:
        if strict:
            print("[test_v657_listing_race] note: node absent -- driven half "
                  "skipped as pass; statics + static mutations still ran.")
        return None
    sig, method = _extract(src)
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(_DRIVER % (sig, method))
        path = f.name
    try:
        res = subprocess.run(["node", path], capture_output=True, text=True,
                             timeout=30)
        return "RACE-PASS" in (res.stdout or "")
    finally:
        os.remove(path)


def run_static(src):
    _need(src.count("this._listSeq = (this._listSeq || 0) + 1") == 2,
          "BOTH listing fetches must take a ticket (exactly two bumps)")
    _need(src.count("seq !== this._listSeq) return;") == 2,
          "both fetches must check their ticket after the await")
    _need("if (this._busyCount) return;   // an upload/refresh is running"
          in src,
          "the focus probe must be suppressed while busy")


def run_mutations():
    s = open(LOADER_JS, encoding="utf-8").read()

    def wounded(old, new, name):
        _need(s.count(old) == 1, "mutation target not unique: " + name)
        return s.replace(old, new)

    m1 = wounded(
        "        if (seq !== this._listSeq) return;   // a newer listing fetch started -> stale, drop\n",
        "", "M1")
    m2 = wounded(
        "        if (this._busyCount) return;   // an upload/refresh is running -- its own re-read is authoritative\n",
        "", "M2")
    m3 = wounded(
        "        const seq = (this._listSeq = (this._listSeq || 0) + 1);\n        try {",
        "        const seq = (this._listSeq || 0);\n        try {", "M3")
    caught = 0
    need = 3
    r = run_driven(m1, strict=False)
    if r is None:
        need -= 1
    elif r is False:
        caught += 1
    else:
        _fail("mutation ticket-check NOT caught")
    for wounded_src, name in ((m2, "busy"), (m3, "bump")):
        bit = False
        try:
            with redirect_stdout(io.StringIO()):   # the probe's FAIL print is intentional
                run_static(wounded_src)
        except SystemExit:
            bit = True
        if bit:
            caught += 1
        else:
            _fail("mutation " + name + " NOT caught")
    _need(caught == need, "mutation coverage incomplete")


def main():
    src = open(LOADER_JS, encoding="utf-8").read()
    r = run_driven(src)
    _need(r is not False, "driven race scenarios failed under node")
    run_static(src)
    run_mutations()
    print("[test_v657_listing_race] PASS: stale probe response dropped, "
          "fresh applies, busy never fetches (driven verbatim under node), "
          "two-ticket statics hold, mutations caught")


if __name__ == "__main__":
    main()
