#!/usr/bin/env python3
"""browser_parity_sizes.py -- P1 of the Nodes 2.0 parity concept.

Every node class of the NODE_IDS baseline under the classic renderer and under
Nodes 2.0 (switched on BEFORE loading -- Frank's path), at three sizes, each
reached by a REAL mouse drag on the south-east resize grip, so that the floor
of the respective renderer applies (LiteGraph: computeSize plus the pack's
onResize clamps; Nodes 2.0: inline min-width and measured content height):

  N  normal  -- the size from Frank's workflow; Nodes 2.0 is dragged to the
               same outer size as the classic node
  G  gathered -- the grip dragged to 10 x 10 px: shows the effective floor
  S  spread  -- outer size N + 400 px wide, + 500 px high

Per state it writes the outer size, the P0 metrics (rows, fields, buttons,
widgets), field and header colours and a photo of the node. Scenes and state
fillers come from browser_parity_sheet.py (one source).

    python3 tools/browser_parity_sizes.py --out /tmp/p1 \\
        --workflow Polyhedron_Inventory_v921.json --stack lora_stack.json \\
        [--only ULSInt,ULSCutout]
"""
import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import browser_parity_sheet as p0  # noqa: E402

SPREAD = (400, 500)
VIEW = (2000, 2800)

GEOM = r"""(vue) => {
  const n = window.__n, g = app.canvas, cr = g.canvas.getBoundingClientRect();
  const T = LiteGraph.NODE_TITLE_HEIGHT || 30;
  if (vue) {
    const el = document.querySelector(`[data-node-id="${n.id}"]`);
    if (!el) return null;
    const r = el.getBoundingClientRect(), h = el.querySelector('[data-corner="SE"]');
    const hd = el.querySelector(`[data-testid="node-header-${n.id}"]`);
    const hr = h ? h.getBoundingClientRect() : null;
    return { rect: [r.left, r.top, r.width, r.height], headerH: hd ? hd.getBoundingClientRect().height : 0,
             grip: hr ? [hr.left + hr.width / 2, hr.top + hr.height / 2] : null,
             size: [n.size[0], n.size[1]], minW: el.style.getPropertyValue('min-width') || '' };
  }
  const s = g.ds.scale, o = g.ds.offset;
  const left = cr.left + (n.pos[0] + o[0]) * s, top = cr.top + (n.pos[1] - T + o[1]) * s;
  const w = n.size[0] * s, h = (n.size[1] + T) * s;
  // LiteGraph's own corner box (LGraphNode.inResizeCorner): below the output rows
  const rows = n.outputs ? n.outputs.length : 1;
  const off = (n.constructor.slot_start_y || 0) + rows * LiteGraph.NODE_SLOT_HEIGHT;
  const gy = n.pos[1] + Math.max(n.size[1] - 15, off) + 10;
  return { rect: [left, top, w, h], headerH: T, grip: [left + w - 5, cr.top + (gy + o[1]) * s],
           size: [n.size[0], n.size[1]] };
}"""

COLORS = r"""(vue) => {
  const n = window.__n;
  const parse = (c) => { const m = String(c || '').match(/rgba?\(([^)]+)\)/);
    if (!m) return null; const p = m[1].split(',').map(x => parseFloat(x));
    return { rgb: p.slice(0, 3).map(Math.round), a: p.length > 3 ? p[3] : 1 }; };
  const eff = (e) => { for (let x = e; x && x !== document.body; x = x.parentElement) {
      const c = parse(getComputedStyle(x).backgroundColor); if (c && c.a > 0.05) return c.rgb; }
    return null; };
  const host = vue ? document.querySelector(`[data-node-id="${n.id}"]`) : null;
  const tas = vue ? [...host.querySelectorAll('textarea')]
    : (n.widgets || []).flatMap(w => { const e = w.element || w.inputEl; if (!e) return [];
        return e.tagName === 'TEXTAREA' ? [e] : [...e.querySelectorAll('textarea')]; })
        .filter(e => e.offsetParent !== null);
  return { color: n.color || null, bgcolor: n.bgcolor || null,
           fields: tas.map(e => ({ y: Math.round(e.getBoundingClientRect().top), bg: eff(e) }))
                      .sort((a, b) => a.y - b.y) };
}"""


async def drag(page, grip, dx, dy, steps=12):
    x0, y0 = grip
    x1 = min(max(2, x0 + dx), VIEW[0] - 2)
    y1 = min(max(2, y0 + dy), VIEW[1] - 2)
    await page.mouse.move(x0, y0)
    await page.mouse.down()
    for i in range(1, steps + 1):
        await page.mouse.move(x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps)
        await page.wait_for_timeout(15)
    await page.mouse.up()
    await page.wait_for_timeout(500)
    return (x1 - x0, y1 - y0) != (dx, dy)


async def measure(page, vue, cls, state, out):
    await page.evaluate("() => app.canvas.setDirty(true, true)")
    await page.wait_for_timeout(900 if vue else 400)
    g = await page.evaluate(GEOM, vue)
    if not g:
        return None
    m = await page.evaluate(p0.METRICS, vue)
    c = await page.evaluate(COLORS, vue)
    x, y, w, h = g["rect"]
    clip = {"x": max(0, x - 4), "y": max(0, y - 4), "width": max(10, w + 8),
            "height": max(10, min(h + 8, VIEW[1] - max(0, y - 4)))}
    tag = "vue" if vue else "classic"
    path = os.path.join(out, "shots", "%s_%s_%s.png" % (cls, state, tag))
    await page.screenshot(path=path, clip=clip)
    return {"geom": g, "metrics": m, "colors": c, "crop": [clip["x"], clip["y"]], "shot": path}


async def run(url, only, inv, stack, out):
    from playwright.async_api import async_playwright
    res = {}
    todo = [c for c in p0.classes() if not only or c in only]
    os.makedirs(os.path.join(out, "shots"), exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch()
        page = await b.new_page(viewport={"width": VIEW[0], "height": VIEW[1]})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)[:200]))
        await page.goto(url)
        await page.wait_for_function("window.app && app.graph && app.canvas", timeout=90000)
        await page.wait_for_timeout(2500)
        await page.evaluate("app.extensionManager.setting.set('Comfy.TutorialCompleted', true)")
        for vue in (False, True):
            tag = "vue" if vue else "classic"
            await page.evaluate("v => app.extensionManager.setting.set('Comfy.VueNodes.Enabled', v)", vue)
            await page.reload()
            await page.wait_for_function("window.app && app.graph && app.canvas", timeout=90000)
            await page.wait_for_timeout(2500)
            for _ in range(3):
                await page.keyboard.press("Escape")
                await page.wait_for_timeout(200)
            for cls in todo:
                rec = res.setdefault(cls, {})
                try:
                    errors.clear()
                    await page.evaluate("wf => app.loadGraphData(wf)", p0.scene_for(cls, inv, stack))
                    await page.evaluate("() => { window.__n = app.graph._nodes[0]; app.canvas.ds.offset = [0, 0];"
                                        " app.canvas.ds.scale = 1; app.canvas.setDirty(true, true); }")
                    await page.wait_for_timeout(1400)
                    if cls in p0.STATES:
                        await page.evaluate("(src) => { const f = eval(src); f(window.__n); app.canvas.setDirty(true, true); }",
                                            p0.STATES[cls])
                        await page.wait_for_timeout(900)
                    await page.wait_for_timeout(1200 if vue else 400)
                    st = {}
                    g0 = await page.evaluate(GEOM, vue)
                    if not g0 or not g0["grip"]:
                        rec[tag] = {"error": "node or grip not found"}
                        print("  %s %-28s NO NODE/GRIP" % (tag, cls), flush=True)
                        continue
                    if vue and rec.get("classic", {}).get("N"):
                        cw, ch = rec["classic"]["N"]["geom"]["rect"][2:]
                        await drag(page, g0["grip"], cw - g0["rect"][2], ch - g0["rect"][3])
                    st["N"] = await measure(page, vue, cls, "N", out)
                    nw, nh = st["N"]["geom"]["rect"][2:]
                    if vue and rec.get("classic", {}).get("N"):
                        nw, nh = rec["classic"]["N"]["geom"]["rect"][2:]
                    g1 = st["N"]["geom"]
                    await drag(page, g1["grip"], (g1["rect"][0] + 10) - g1["grip"][0], (g1["rect"][1] + 10) - g1["grip"][1])
                    st["G"] = await measure(page, vue, cls, "G", out)
                    g2 = st["G"]["geom"]
                    clipped = await drag(page, g2["grip"], (g2["rect"][0] + nw + SPREAD[0]) - g2["grip"][0],
                                         (g2["rect"][1] + nh + SPREAD[1]) - g2["grip"][1])
                    st["S"] = await measure(page, vue, cls, "S", out)
                    st["S"]["clipped"] = clipped
                    st["pageerrors"] = list(errors)
                    rec[tag] = st
                    print("  %s %-28s N %dx%d  G %dx%d  S %dx%d%s" % (
                        tag, cls, *[round(v) for v in st["N"]["geom"]["rect"][2:]],
                        *[round(v) for v in st["G"]["geom"]["rect"][2:]],
                        *[round(v) for v in st["S"]["geom"]["rect"][2:]],
                        "  ERR %d" % len(errors) if errors else ""), flush=True)
                except Exception as e:
                    rec[tag] = {"error": str(e)[:200]}
                    print("  %s %-28s ERROR %s" % (tag, cls, str(e)[:160]), flush=True)
        await page.evaluate("app.extensionManager.setting.set('Comfy.VueNodes.Enabled', false)")
        await b.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8188/")
    ap.add_argument("--out", default="p1_out")
    ap.add_argument("--workflow")
    ap.add_argument("--stack")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    inv = json.load(open(a.workflow, encoding="utf-8")) if a.workflow else None
    stack = json.load(open(a.stack, encoding="utf-8")) if a.stack else None
    only = set(x for x in a.only.split(",") if x)
    res = asyncio.run(run(a.url, only, inv, stack, a.out))
    json.dump(res, open(os.path.join(a.out, "p1.json"), "w"), indent=1)
    print("browser_parity_sizes: %d nodes measured" % len(res))


if __name__ == "__main__":
    main()
