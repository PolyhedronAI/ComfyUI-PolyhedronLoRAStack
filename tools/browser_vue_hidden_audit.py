#!/usr/bin/env python3
"""browser_vue_hidden_audit.py -- v940: under Nodes 2.0, is every widget that
the classic view hides hidden too?

The classic renderer reads widget.hidden, the Vue renderer widget.options.hidden
(frontend 1.49.6). This audit places every class of the NODE_IDS baseline,
switches Nodes 2.0 on, and checks every widget of every pack node:
options.hidden must equal the classic hidden state (widget.hidden, type
"hidden", a "pls-hidden-" type, or -- v947 -- a computeSize that reports no
height, for a widget without a mounted element). It also looks for the Mask Editor's five
internal stores as visible text in its Vue node. Two paths: nodes inserted
fresh, and the same graph loaded as a minimal workflow.

Needs a running ComfyUI with this pack (--url), Python playwright and a
Chromium (PLAYWRIGHT_BROWSERS_PATH). It CLEARS the graph and toggles
Comfy.VueNodes.Enabled (restored to off). Exit 0 = PASS, 1 = FAIL, 2 = env.
"""
import argparse
import asyncio
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORES = ["mask_store", "points_store", "path_store", "layers_store", "bg_key"]

CHECK = r"""(stores) => {
  const bad = [], seen = {nodes: 0, widgets: 0, hidden: 0};
  const pack = n => { const c = String(n.comfyClass || n.type || ''); return c.startsWith('ULS') || c === 'UltimateLoraStack'; };
  for (const n of app.graph._nodes) {
    if (!pack(n)) continue; seen.nodes++;
    for (const w of n.widgets || []) {
      seen.widgets++;
      // v947: the same four forms uls_vue_hidden.classicHidden reads -- the fourth
      // is a row the classic layout gives NO height (and only for widgets without
      // a mounted element, exactly as the rule has it). A mirror that knows three
      // of four is how the Empty Latent pill stayed hidden from this audit while
      // P1 kept reporting it.
      let zero = false;
      if (!w.element && typeof w.computeSize === 'function') {
        try { const h = (w.computeSize(0) || [])[1]; zero = typeof h === 'number' && isFinite(h) && h <= 0; }
        catch (e) { zero = false; }
      }
      const ch = !!(w.hidden || w.type === 'hidden' || String(w.type || '').startsWith('pls-hidden-') || zero);
      if (ch) seen.hidden++;
      if (ch !== !!(w.options && w.options.hidden)) bad.push(n.type + ':' + w.name + (ch ? ' (hidden classic, shown in Nodes 2.0)' : ' (shown classic, hidden in Nodes 2.0)'));
    }
  }
  const me = app.graph._nodes.find(n => n.type === 'ULSMaskEditor');
  const el = me && document.querySelector(`[data-node-id="${me.id}"]`);
  const txt = el ? el.innerText : '';
  const meRegistered = !!(LiteGraph.registered_node_types && LiteGraph.registered_node_types['ULSMaskEditor']);
  return {bad, seen, stores: stores.filter(s => txt.includes(s)), meFound: !!el, meRegistered};
}"""


def classes():
    files = sorted(glob.glob(os.path.join(ROOT, "NODE_IDS_baseline_v*.txt")))
    return [l.split()[0] for l in open(files[-1], encoding="utf-8")
            if l.strip() and not l.startswith("#")]


async def run(url):
    from playwright.async_api import async_playwright
    out = {}
    async with async_playwright() as p:
        b = await p.chromium.launch()
        page = await b.new_page(viewport={"width": 1600, "height": 1000})
        await page.goto(url)
        await page.wait_for_function("window.app && app.graph && app.canvas", timeout=90000)
        await page.wait_for_timeout(2500)
        await page.evaluate("app.extensionManager.setting.set('Comfy.TutorialCompleted', true)")
        for _ in range(3):
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(250)
        await page.evaluate("app.extensionManager.setting.set('Comfy.VueNodes.Enabled', false)")
        await page.evaluate("""(cls) => { app.graph.clear();
          cls.forEach((c, i) => { const n = LiteGraph.createNode(c); if (!n) return;
            n.pos = [(i % 8) * 900, Math.floor(i / 8) * 1400]; app.graph.add(n); }); }""", classes())
        await page.wait_for_timeout(3000)
        await page.evaluate("app.extensionManager.setting.set('Comfy.VueNodes.Enabled', true)")
        await page.wait_for_timeout(3500)
        await page.evaluate("""() => { const me = app.graph._nodes.find(n => n.type === 'ULSMaskEditor');
          if (me) { app.canvas.ds.offset = [100 - me.pos[0], 100 - me.pos[1]]; app.canvas.setDirty(true, true); } }""")
        await page.wait_for_timeout(1500)
        out["fresh"] = await page.evaluate(CHECK, STORES)
        await page.evaluate("""async () => { const full = app.graph.serialize();
          const wf = {last_node_id: full.last_node_id, last_link_id: 0,
            nodes: full.nodes.map((n, i) => ({id: n.id, type: n.type, pos: n.pos, flags: {}, order: i, mode: 0, properties: {}})),
            links: [], groups: [], config: {}, extra: {}, version: 0.4};
          await app.loadGraphData(wf); }""")
        await page.wait_for_timeout(4000)
        await page.evaluate("""() => { const me = app.graph._nodes.find(n => n.type === 'ULSMaskEditor');
          if (me) { app.canvas.ds.offset = [100 - me.pos[0], 100 - me.pos[1]]; app.canvas.setDirty(true, true); } }""")
        await page.wait_for_timeout(1500)
        out["loaded"] = await page.evaluate(CHECK, STORES)
        await page.evaluate("app.extensionManager.setting.set('Comfy.VueNodes.Enabled', false)")
        await b.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8188/")
    a = ap.parse_args()
    try:
        import playwright  # noqa: F401
    except ImportError:
        print("browser_vue_hidden_audit: playwright not installed -- SKIPPED")
        return 2
    res = asyncio.run(run(a.url))
    fail = 0
    for state, r in res.items():
        # public build (v374): the Mask Editor is an internal-only node -- its
        # store check applies only where the class is registered at all
        ok = not r["bad"] and not r["stores"] and (r["meFound"] or not r.get("meRegistered", True))
        fail += 0 if ok else 1
        print("  %s %-7s %d pack nodes, %d widgets, %d hidden classic | mismatches %d | "
              "Mask Editor stores visible: %s"
              % ("PASS" if ok else "FAIL", state, r["seen"]["nodes"], r["seen"]["widgets"],
                 r["seen"]["hidden"], len(r["bad"]), r["stores"] or "none"))
        for x in r["bad"][:8]:
            print("       %s" % x)
    print("browser_vue_hidden_audit: %s" % ("FAIL" if fail else "PASS"))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
