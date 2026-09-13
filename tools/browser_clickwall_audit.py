#!/usr/bin/env python3
"""browser_clickwall_audit.py -- v936: every Polyhedron node, clicked under the
classic renderer. The general form of the v934 lesson (S0, 10.09.2026).

WHY
---
The frontend wraps every DOM widget in a container of its own. It leaves the
page only when the WIDGET is hidden; hiding the inner element leaves a
transparent, clickable container behind -- over the painted node, and often
far past it (a negative height falls back to full canvas height). No source
guard can see that. This audit can: it places every node class of the tree on
a canvas, pans onto each, and asks the browser what a click would hit.

WHAT IT REPORTS, per node, in three states (freshly inserted, after idling;
the same graph saved and LOADED as a workflow -- a container that never had a
valid height falls back to full canvas height, so the two differ; and after
Nodes 2.0 on -> off without reload):
  WALL    a DOM widget whose container is clickable while its content is
          hidden, empty, or much smaller than the container
  EXTENT  such a container reaching past the node
plus a 7x7 grid over the body and three points below it: canvas, real DOM
content, or the bare container (the click-wall signature).

Needs a running ComfyUI with this pack (--url), Python playwright and a
Chromium it can find (PLAYWRIGHT_BROWSERS_PATH). It CLEARS the graph and
toggles Comfy.VueNodes.Enabled (restored to off). Bench tool, not a pack
requirement. Exit 0 = no wall, 1 = walls found, 2 = environment missing.

Known and open when this tool was written: ULSPowerUpscale (its result and
process panes hide only their inner element).
"""
import argparse
import asyncio
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "http://127.0.0.1:8188/"


def node_classes():
    """Every class of the tree, from its NODE_IDS baseline (newest)."""
    files = sorted(glob.glob(os.path.join(ROOT, "NODE_IDS_baseline_v*.txt")))
    out = []
    for line in open(files[-1], encoding="utf-8"):
        t = line.strip()
        if t and not t.startswith("#"):
            out.append(t.split()[0])
    return out

MEASURE = r"""
(idx) => {
  const g = app.canvas, n = app.graph._nodes[idx];
  const cls = n.comfyClass || n.type;
  // pan so the node body sits at (240, 190) in canvas coordinates
  g.ds.scale = 1;
  g.ds.offset = [240 - n.pos[0], 190 - n.pos[1]];
  g.setDirty(true, true);
  return cls;
}
"""

READ = r"""
(idx) => {
  const g = app.canvas, n = app.graph._nodes[idx];
  const cr = g.canvas.getBoundingClientRect();
  const toClient = (x, y) => [cr.left + (x + g.ds.offset[0]) * g.ds.scale,
                              cr.top + (y + g.ds.offset[1]) * g.ds.scale];
  const [L, T] = toClient(n.pos[0], n.pos[1]);
  const [R, B] = toClient(n.pos[0] + n.size[0], n.pos[1] + n.size[1]);
  const rect = el => { const r = el.getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; };
  const vis = el => { const s = getComputedStyle(el); return s.display !== 'none' && s.visibility !== 'hidden'; };
  const dom = [];
  for (const w of (n.widgets || [])) {
    const el = w.element;
    if (!el) continue;
    const c = el.parentElement;
    const cs = c ? getComputedStyle(c) : null;
    const crc = c ? rect(c) : null, erc = rect(el);
    const cArea = crc ? crc[2] * crc[3] : 0, eArea = erc[2] * erc[3];
    const clickable = !!(c && cs.pointerEvents !== 'none' && cs.display !== 'none' && cArea > 1);
    const hiddenContent = !vis(el) || eArea < 1;
    const smaller = crc && erc[3] + 24 < crc[3];
    dom.push({
      name: w.name, type: w.type, hidden: !!w.hidden,
      container: crc && crc.map(Math.round), pe: cs && cs.pointerEvents, cdisp: cs && cs.display,
      element: erc.map(Math.round), edisp: getComputedStyle(el).display,
      wall: clickable && (hiddenContent || smaller),
      why: clickable ? (hiddenContent ? 'content hidden/empty' : (smaller ? 'container taller than content' : '')) : '',
      extent: !!(crc && clickable && (crc[1] + crc[3] > B + 10 || crc[0] + crc[2] > R + 10)),
    });
  }
  // the grid: what would a click hit?
  const hits = {canvas: 0, content: 0, container: 0, other: 0};
  const containerHits = [], otherHits = new Set();
  const pts = [];
  for (let i = 0; i < 7; i++) for (let j = 0; j < 7; j++)
    pts.push([L + 6 + (R - L - 12) * i / 6, T + 6 + (B - T - 12) * j / 6, 'body']);
  for (const dy of [20, 60, 140]) pts.push([(L + R) / 2, B + dy, 'below']);
  const below = {canvas: 0, container: 0, other: 0};
  for (const [x, y, where] of pts) {
    const el = document.elementFromPoint(x, y);
    let kind = 'other';
    if (el === g.canvas) kind = 'canvas';
    else if (el && el.classList && el.classList.contains('dom-widget')) kind = 'container';
    else if (el && el.closest && el.closest('.dom-widget')) kind = 'content';
    if (where === 'body') {
      hits[kind]++;
      if (kind === 'container') containerHits.push([Math.round(x - L), Math.round(y - T)]);
      if (kind === 'other' && el) otherHits.add(el.tagName + '.' + String(el.className).slice(0, 30));
    } else {
      below[kind === 'content' ? 'other' : kind] = (below[kind === 'content' ? 'other' : kind] || 0) + 1;
    }
  }
  return {size: [Math.round(n.size[0]), Math.round(n.size[1])], dom, hits,
          containerHits: containerHits.slice(0, 4), other: [...otherHits].slice(0, 3), below};
}
"""


async def settle(page, ms=250):
    await page.wait_for_timeout(ms)


async def sweep(page, count):
    out = []
    for i in range(count):
        cls = await page.evaluate(MEASURE, i)
        # The overlay places DOM-widget containers on canvas DRAWS. A pan that
        # is read too early measures the previous position -- the first draft
        # found Power Upscale's wall in one layout and missed it in another.
        # Two draws and time for each, then read.
        for _ in range(2):
            await settle(page, 250)
            await page.evaluate("app.canvas.setDirty(true, true)")
        await settle(page, 250)
        r = await page.evaluate(READ, i)
        r["cls"] = cls
        out.append(r)
    return out


async def main():
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch()
        page = await b.new_page(viewport={"width": 1600, "height": 1000})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)[:160]))
        await page.goto(URL)
        await page.wait_for_function("window.app && app.graph && app.canvas", timeout=90000)
        await page.wait_for_timeout(2500)
        await page.evaluate("app.extensionManager.setting.set('Comfy.TutorialCompleted', true)")
        for _ in range(3):
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(250)
        await page.evaluate("app.extensionManager.setting.set('Comfy.VueNodes.Enabled', false)")
        missing = await page.evaluate("""(classes) => {
          app.graph.clear();
          const miss = [];
          classes.forEach((c, i) => {
            const n = LiteGraph.createNode(c);
            if (!n) { miss.push(c); return; }
            n.pos = [(i % 8) * 900, Math.floor(i / 8) * 1400];
            app.graph.add(n);
          });
          return miss; }""", node_classes())
        if missing:
            print(json.dumps({"state": "missing_classes", "classes": missing}))
        await page.wait_for_timeout(4000)          # deferred setups + idle
        n = await page.evaluate("app.graph._nodes.length")
        a = await sweep(page, n)
        print(json.dumps({"state": "classic_fresh", "nodes": a}))
        # A container that once had a valid height keeps it; one that never
        # had one falls back to full canvas height. Freshly inserted nodes and
        # nodes from a LOADED workflow therefore differ -- and loaded workflows
        # are how the suite is used. Measure that path too.
        # MINIMAL entries -- type and position, no size, no widget values --
        # the way tools/build_inventory-style workflows and many shared
        # workflows arrive. This is the load path on which S0 (10.09.) saw
        # Power Upscale's wall at once; a full serialize() round trip carries
        # sizes and did not reproduce it.
        await page.evaluate("""async () => {
          const full = app.graph.serialize();
          const wf = {last_node_id: full.last_node_id, last_link_id: 0,
                      nodes: full.nodes.map((n, i) => ({id: n.id, type: n.type,
                        pos: n.pos, flags: {}, order: i, mode: 0,
                        properties: {}})),
                      links: [], groups: [], config: {}, extra: {}, version: 0.4};
          await app.loadGraphData(wf); }""")
        await page.wait_for_timeout(4000)
        n = await page.evaluate("app.graph._nodes.length")
        lw = await sweep(page, n)
        print(json.dumps({"state": "classic_loaded_workflow", "nodes": lw}))
        await page.evaluate("app.extensionManager.setting.set('Comfy.VueNodes.Enabled', true)")
        await page.wait_for_timeout(3500)
        await page.evaluate("app.extensionManager.setting.set('Comfy.VueNodes.Enabled', false)")
        await page.wait_for_timeout(3500)
        c = await sweep(page, n)
        print(json.dumps({"state": "classic_after_roundtrip", "nodes": c}))
        print(json.dumps({"state": "errors", "errors": errors[:10]}))
        await b.close()


RESULTS = []


def verdict(lines):
    walls = {}
    for line in lines:
        d = json.loads(line)
        if "nodes" not in d:
            continue
        for n in d["nodes"]:
            for w in n["dom"]:
                if w["wall"] or w["extent"]:
                    walls.setdefault(n["cls"], set()).add(
                        "%s: %s%s (%s)" % (w["name"], w["why"] or "clickable",
                                           ", EXTENT" if w["extent"] else "",
                                           d["state"]))
    return walls


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--json", help="also write the raw per-node lines here")
    args = ap.parse_args()
    URL = args.url
    try:
        import playwright  # noqa: F401
    except ImportError:
        print("browser_clickwall_audit: playwright not installed -- SKIPPED")
        sys.exit(2)
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        asyncio.run(main())
    lines = [ln for ln in buf.getvalue().splitlines() if ln.startswith("{")]
    if args.json:
        open(args.json, "w").write("\n".join(lines) + "\n")
    walls = verdict(lines)
    count = 0
    for ln in lines:
        d = json.loads(ln)
        if d.get("state") == "classic_fresh":
            count = len(d["nodes"])
        if d.get("state") == "missing_classes":
            print("  NOTE classes the frontend does not know: %s" % d["classes"])
        if d.get("state") == "errors" and d["errors"]:
            print("  NOTE page errors: %s" % d["errors"])
    for cls in sorted(walls):
        print("  WALL %s" % cls)
        for w in sorted(walls[cls]):
            print("       %s" % w)
    print("browser_clickwall_audit: %d node(s) measured, %d with a click wall -- %s"
          % (count, len(walls), "FAIL" if walls else "PASS"))
    sys.exit(1 if walls else 0)
