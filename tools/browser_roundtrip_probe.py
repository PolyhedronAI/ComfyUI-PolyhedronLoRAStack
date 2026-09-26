#!/usr/bin/env python3
# -*- coding: ascii -*-
"""browser_roundtrip_probe.py -- save -> reload on a LIVE frontend (v1021).

Loads a workflow, reads every node's serialisable widget values, serialises
the graph the way the frontend saves it, loads that result again and reads
the values once more. Any node whose values differ after the round trip is
named with both lists.

Why it exists: on Core master's frontend 1.53.6 graph.serialize() stopped
calling node.serialize(), and three nodes that show their rows in a display
order (Load CLIP, CLIP Text Encode, Reference) came back with every value in
the wrong row -- P0/P1 look at a node, not at what it writes. Run it on every
frontend the pack is meant for (see web/js/ph_save_compat.js).

Needs a running ComfyUI with the pack, Python playwright and a Chromium
(PLAYWRIGHT_BROWSERS_PATH). Exit 0 = no node changed.

  python3 tools/browser_roundtrip_probe.py --workflow <file.json> [--url http://127.0.0.1:8188/]
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8188/")
    ap.add_argument("--workflow", required=True, action="append",
                    help="workflow JSON (repeatable)")
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    bad = 0
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1600, "height": 1000})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(args.url, wait_until="networkidle", timeout=180000)
        pg.wait_for_function("window.app && window.app.graph", timeout=120000)
        pg.wait_for_timeout(2500)
        pg.keyboard.press("Escape")          # 1.53.6 opens its template dialog on a fresh profile
        fe = pg.evaluate("() => window.__COMFYUI_FRONTEND_VERSION__ || '?'")
        for path in args.workflow:
            wf = json.load(open(path, encoding="utf-8"))
            r = pg.evaluate("""async (wf) => {
              const live = () => { const out = {};
                for (const n of window.app.graph._nodes)
                  out[n.id + ' ' + n.type] = (n.widgets || [])
                    .filter((w) => w && w.serialize !== false && (typeof w.value !== 'object' || w.value === null))
                    .map((w) => w.name + '=' + JSON.stringify(w.value));
                return out; };
              await window.app.loadGraphData(wf);
              await new Promise((r) => setTimeout(r, 800));
              const a = live();
              const saved = JSON.parse(JSON.stringify(window.app.graph.serialize()));
              await window.app.loadGraphData(saved);
              await new Promise((r) => setTimeout(r, 800));
              const b = live();
              const diffs = [];
              for (const k of Object.keys(a))
                if (JSON.stringify(a[k]) !== JSON.stringify(b[k])) diffs.push([k, a[k], b[k] || null]);
              return {nodes: Object.keys(a).length, diffs};
            }""", wf)
            print("%s  frontend %s  %d node(s), %d changed by save -> reload"
                  % (path.split("/")[-1], fe, r["nodes"], len(r["diffs"])))
            for k, before, after in r["diffs"]:
                bad += 1
                print("  CHANGED %s" % k)
                print("    before: %s" % ", ".join(before)[:600])
                print("    after : %s" % ", ".join(after or [])[:600])
        if errs:
            print("page errors: %s" % errs[:3])
        b.close()
    print("browser_roundtrip_probe: %s" % ("PASS" if not bad else "%d node(s) CHANGED" % bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
