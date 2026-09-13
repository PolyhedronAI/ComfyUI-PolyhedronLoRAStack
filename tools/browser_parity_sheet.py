#!/usr/bin/env python3
"""browser_parity_sheet.py -- v943, P0 of the Nodes 2.0 parity concept.

Every node class of the NODE_IDS baseline, alone on the canvas at zoom 1,
photographed under the classic renderer and under Nodes 2.0 (switched on
BEFORE the workflow is loaded -- Frank's path), cropped to the node, side by
side on one sheet per node, with measured figures and a first classification
(Polyhedron_Pruefkonzept_Nodes2_Paritaet.md, section 4):

  F1 suite colour missing   F2 row too many   F3 row missing
  F4 node overstretched     F5 field overflows/oversized   F8 stale notice
  (F6 position and F7 frontend design are judged on the sheet, by eye)

It also writes the md5 of every CLASSIC crop (classic_md5.json). With --ref
it compares them against a frozen reference: a cut that must not change the
classic view has to reproduce every classic crop pixel for pixel.

States: a node's entry in --workflow (Frank's inventory) is loaded as that
node's scene; --stack gives the Stack its rows; STATES below fills nodes that
show nothing without data. Needs a running ComfyUI with this pack (--url),
Python playwright + Pillow and a Chromium (PLAYWRIGHT_BROWSERS_PATH). It
clears the graph and toggles Comfy.VueNodes.Enabled (restored to off).

    python3 tools/browser_parity_sheet.py --out /tmp/parity \\
        --workflow Polyhedron_Inventory_v921.json --stack lora_stack.json \\
        [--ref classic_md5.json] [--only ULSInt,ULSCutout]
"""
import argparse
import asyncio
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# LiteGraph's standard widget types -- the ones Nodes 2.0 renders as a labelled
# row. Custom types (section heads, DOM panels) are judged on the sheet.
STD_TYPES = {"number", "combo", "toggle", "text", "customtext", "string", "slider"}
# Folded INTO another widget by the frontend (the seed's control sits inside
# the number field) -- F7 by design, never a missing row.
FRONTEND_FOLDED = {"control_after_generate", "control_before_generate"}

# Content a fresh node does not show on its own. Run after the scene loads.
STATES = {
    "ULSCLIPTextEncode": """n => { const s = (k, v) => { const w = n.widgets.find(x => x.name === k); if (w) { w.value = v; w.callback?.(v, app.canvas, n); } };
        s('pos_1', 'a red fox jumps over the lazy dog, golden hour, film grain'); s('neg_1', 'blurry, lowres'); }""",
    "ULSInt": """n => { const st = n._phi; if (st) { n.widgets.find(w => w.name === 'value').value = 4; } }""",
    "ULSNote": """n => { const w = (n.widgets || []).find(x => x.type === 'customtext' || x.name === 'text'); if (w) w.value = 'A note, two lines\\nsecond line'; }""",
    "ULSAccelerator": """n => { const u = n._uls; if (!u) return; u.mode = 'DARE';
        Object.assign(u.rows[0], {name: 'test_a.safetensors', weight: 0.85});
        u.rows.push({enabled: true, name: 'test_b.safetensors', weight: 1.0, wClip: 0.8, group: '\\u2014', wHigh: 1, wLow: 1}); n._ulsSync(); }""",
}

METRICS = r"""(vue) => {
  const n = window.__n, g = app.canvas, cr = g.canvas.getBoundingClientRect();
  const T = LiteGraph.NODE_TITLE_HEIGHT || 30;
  let rect;
  if (vue) { const el = document.querySelector(`[data-node-id="${n.id}"]`);
    if (!el) return null; const r = el.getBoundingClientRect(); rect = [r.left, r.top, r.width, r.height]; }
  else rect = [cr.left + n.pos[0] + g.ds.offset[0], cr.top + n.pos[1] - T + g.ds.offset[1], n.size[0], n.size[1] + T];
  const rel = (r) => [Math.round(r.left - rect[0]), Math.round(r.top - rect[1]), Math.round(r.width), Math.round(r.height)];
  const classicHidden = (w) => !!(w.hidden || w.type === 'hidden' || String(w.type || '').startsWith('pls-hidden-')
      || (typeof w.computeSize === 'function' && (() => { try { const s = w.computeSize(n.size[0]); return s && s[1] <= 0; } catch (e) { return false; } })()));
  const out = { rect: rect.map(Math.round), size: [Math.round(n.size[0]), Math.round(n.size[1])],
                notice: (n.widgets || []).some(w => w.name === 'polyhedron_renderer_notice'), widgets: [], rows: [], fields: [] };
  for (const w of n.widgets || []) {
    const e = w.element;
    out.widgets.push({ name: w.name, label: w.label || w.name, type: String(w.type), hiddenClassic: classicHidden(w),
                       flagHidden: !!(w.hidden || w.type === 'hidden' || String(w.type || '').startsWith('pls-hidden-')),
                       optHidden: !!(w.options && w.options.hidden),
                       dom: e ? rel(e.getBoundingClientRect()) : null,
                       overflow: e ? (e.scrollHeight > e.clientHeight + 2 || e.scrollWidth > e.clientWidth + 2) : false });
  }
  const host = vue ? document.querySelector(`[data-node-id="${n.id}"]`) : null;
  const tas = vue ? [...host.querySelectorAll('textarea')]
                  : (n.widgets || []).flatMap(w => { const e = w.element || w.inputEl; if (!e) return [];
                      return e.tagName === 'TEXTAREA' ? [e] : [...e.querySelectorAll('textarea')]; })
                      .filter(e => e.offsetParent !== null);
  out.textareas = tas.map(e => ({ box: rel(e.getBoundingClientRect()), contentH: e.scrollHeight, overflow: e.scrollHeight > e.clientHeight + 2 }))
                     .sort((a, b) => a.box[1] - b.box[1]);
  if (vue) out.buttons = [...host.querySelectorAll('button')].map(b => b.textContent.trim()).filter(Boolean);
  if (vue) {
    const el = host;
    for (const row of el.querySelectorAll('[data-testid="node-widget"]')) {
      const lab = row.querySelector('[data-testid="widget-layout-field-label"]');
      const f = row.querySelector('textarea, input[type=text], input:not([type])');
      out.rows.push({ label: lab ? lab.textContent.trim() : '', box: rel(row.getBoundingClientRect()),
                      field: f ? rel(f.getBoundingClientRect()) : null,
                      fieldOverflow: f ? (f.scrollHeight > f.clientHeight + 2) : false,
                      fieldContentH: f && f.tagName === 'TEXTAREA' ? f.scrollHeight : null });
    }
  } else {
    for (const w of n.widgets || []) {
      const e = w.element || w.inputEl;
      if (e && e.tagName === 'TEXTAREA') out.fields.push({ name: w.name, box: rel(e.getBoundingClientRect()), contentH: e.scrollHeight });
    }
  }
  return out;
}"""


def classes():
    files = sorted(glob.glob(os.path.join(ROOT, "NODE_IDS_baseline_v*.txt")))
    return [l.split()[0] for l in open(files[-1], encoding="utf-8")
            if l.strip() and not l.startswith("#")]


def scene_for(cls, inv, stack):
    """A one-node workflow: the node's entry from a given workflow, else bare."""
    src = None
    if cls == "UltimateLoraStack" and stack:
        src = stack["nodes"][0]
    elif inv:
        src = next((n for n in inv.get("nodes", []) if n.get("type") == cls), None)
    node = {"id": 1, "type": cls, "pos": [300, 200], "flags": {}, "order": 0, "mode": 0,
            "inputs": [], "outputs": [], "properties": {}}
    if src:
        for k in ("widgets_values", "properties", "size", "_uls", "_engine", "inputs", "outputs"):
            if k in src:
                node[k] = json.loads(json.dumps(src[k]))
        for i in node.get("inputs", []):
            i["link"] = None
        for o in node.get("outputs", []):
            o["links"] = None
    return {"last_node_id": 1, "last_link_id": 0, "nodes": [node], "links": [], "groups": [],
            "config": {}, "extra": {"ds": {"scale": 1, "offset": [0, 0]}}, "version": 0.4}


async def capture(url, only, inv, stack, out):
    from playwright.async_api import async_playwright
    res = {}
    todo = [c for c in classes() if not only or c in only]
    async with async_playwright() as p:
        b = await p.chromium.launch()
        page = await b.new_page(viewport={"width": 2000, "height": 1900})
        await page.goto(url)
        await page.wait_for_function("window.app && app.graph && app.canvas", timeout=90000)
        await page.wait_for_timeout(2500)
        await page.evaluate("app.extensionManager.setting.set('Comfy.TutorialCompleted', true)")
        for _ in range(3):
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(250)
        for vue in (False, True):
            await page.evaluate("v => app.extensionManager.setting.set('Comfy.VueNodes.Enabled', v)", vue)
            await page.reload()
            await page.wait_for_function("window.app && app.graph && app.canvas", timeout=90000)
            await page.wait_for_timeout(2500)
            for _ in range(3):
                await page.keyboard.press("Escape")
                await page.wait_for_timeout(200)
            tag = "vue" if vue else "classic"
            os.makedirs(os.path.join(out, tag), exist_ok=True)
            for cls in todo:
                try:
                    await page.evaluate("wf => app.loadGraphData(wf)", scene_for(cls, inv, stack))
                    await page.evaluate("() => { window.__n = app.graph._nodes[0]; app.canvas.ds.offset = [0, 0]; app.canvas.ds.scale = 1; app.canvas.setDirty(true, true); }")
                    await page.wait_for_timeout(1400)
                    if cls in STATES:
                        await page.evaluate("(src) => { const f = eval(src); f(window.__n); app.canvas.setDirty(true, true); }", STATES[cls])
                        await page.wait_for_timeout(900)
                    await page.wait_for_timeout(1600 if vue else 600)
                    m = await page.evaluate(METRICS, vue)
                    if not m:
                        res.setdefault(cls, {})[tag] = None
                        continue
                    x, y, w, h = m["rect"]
                    clip = {"x": max(0, x - 4), "y": max(0, y - 4), "width": max(10, w + 8), "height": max(10, min(h + 8, 1890 - max(0, y - 4)))}
                    path = os.path.join(out, tag, cls + ".png")
                    await page.screenshot(path=path, clip=clip)
                    m["crop"] = [clip["x"], clip["y"]]
                    res.setdefault(cls, {})[tag] = m
                    print("  %s %-28s %dx%d" % (tag, cls, m["size"][0], m["size"][1]), flush=True)
                except Exception as e:
                    res.setdefault(cls, {})[tag] = None
                    print("  %s %-28s ERROR %s" % (tag, cls, str(e)[:120]), flush=True)
        await page.evaluate("app.extensionManager.setting.set('Comfy.VueNodes.Enabled', false)")
        await b.close()
    return res


def sample(img, pt):
    x, y = int(pt[0]), int(pt[1])
    if 0 <= x < img.width and 0 <= y < img.height:
        return img.getpixel((x, y))[:3]
    return None


def classify(cls, c, v, out):
    """First, automatic classification. The sheet is the evidence; this is the index."""
    from PIL import Image
    f = []
    if not c or not v:
        return [("ERR", "not captured in both renderers")]
    if v.get("notice"):
        f.append(("F8", "renderer notice shown under Nodes 2.0"))
    labels_vue = {r["label"] for r in v["rows"] if r["label"]}
    for w in c["widgets"]:
        lab = w["label"]
        if w["name"] == "polyhedron_renderer_notice" or w["dom"]:
            continue
        # a widget the classic view hides ONLY by a zero height carries no
        # flag -- Nodes 2.0 (options.hidden) shows it
        vw = next((x for x in v["widgets"] if x["name"] == w["name"]), None)
        if w["hiddenClassic"] and not w["flagHidden"] and vw and not vw["optHidden"]:
            f.append(("F2", "'%s' hidden in classic by zero height only -- Nodes 2.0 shows it" % w["name"]))
            continue
        if w["name"] in FRONTEND_FOLDED:
            continue
        if w["type"] == "button":         # a button shows its text IN the button, no label row
            if not w["hiddenClassic"] and lab not in v.get("buttons", []) and lab not in labels_vue:
                f.append(("F3", "button '%s' shown in classic, missing in Nodes 2.0" % lab))
            continue
        if w["type"] not in STD_TYPES:
            continue                      # custom widgets: judged on the sheet
        if w["hiddenClassic"] and lab in labels_vue:
            f.append(("F2", "row '%s' hidden in classic, shown in Nodes 2.0" % lab))
        if not w["hiddenClassic"] and lab not in labels_vue:
            f.append(("F3", "row '%s' shown in classic, missing in Nodes 2.0" % lab))
    hc, hv = c["size"][1], v["rect"][3]
    if hc > 0 and hv / float(hc + 30) > 1.25:
        f.append(("F4", "Nodes 2.0 %d px high vs classic %d px (+%d%%)" % (hv, hc + 30, round(100 * (hv / float(hc + 30) - 1)))))
    for r in v["rows"]:
        if r["fieldOverflow"]:
            f.append(("F5", "field in row '%s' overflows" % r["label"]))
    tc, tv = c.get("textareas", []), v.get("textareas", [])
    pairs = list(zip(tc, tv)) if len(tc) == len(tv) else []
    if len(tc) != len(tv):
        f.append(("F5", "%d text fields in classic, %d in Nodes 2.0" % (len(tc), len(tv))))
    for i, (a, b) in enumerate(pairs):
        if b["box"][3] > 1.6 * max(20, a["box"][3]) and b["box"][3] > b["contentH"] * 0.0 + a["box"][3] + 24:
            f.append(("F5", "text field %d: %d px in Nodes 2.0 vs %d px classic" % (i + 1, b["box"][3], a["box"][3])))
    try:
        ic = Image.open(os.path.join(out, "classic", cls + ".png")).convert("RGB")
        iv = Image.open(os.path.join(out, "vue", cls + ".png")).convert("RGB")
        for i, (a, b) in enumerate(pairs):
            # sample inside the field, right of the text start, below the first line
            # inside the field, clear of its border and of the first text line
            pc = sample(ic, (a["box"][0] + int(a["box"][2] * 0.7) + 4, a["box"][1] + int(a["box"][3] * 0.7) + 4))
            pv = sample(iv, (b["box"][0] + int(b["box"][2] * 0.7) + 4, b["box"][1] + int(b["box"][3] * 0.7) + 4))
            if not pc or not pv:
                continue
            sat = max(pc) - min(pc)
            dist = sum((x - y) ** 2 for x, y in zip(pc, pv)) ** 0.5
            if sat > 18 and dist > 20:
                f.append(("F1", "text field %d: classic tint rgb%s, Nodes 2.0 rgb%s" % (i + 1, tuple(pc), tuple(pv))))
    except Exception:
        pass
    return f


def sheet(cls, c, v, findings, out):
    from PIL import Image, ImageDraw
    pc, pv = os.path.join(out, "classic", cls + ".png"), os.path.join(out, "vue", cls + ".png")
    ic = Image.open(pc) if os.path.exists(pc) else Image.new("RGB", (200, 60))
    iv = Image.open(pv) if os.path.exists(pv) else Image.new("RGB", (200, 60))
    head = 26 + 16 * max(1, len(findings))
    W = ic.width + iv.width + 40
    H = max(ic.height, iv.height) + head + 30
    s = Image.new("RGB", (W, H), (22, 22, 26))
    d = ImageDraw.Draw(s)
    d.text((10, 6), "%s    classic %s    |    Nodes 2.0 %s" % (
        cls, "%dx%d" % (c["size"][0], c["size"][1] + 30) if c else "-",
        "%dx%d" % (v["rect"][2], v["rect"][3]) if v else "-"), fill=(235, 200, 110))
    y = 26
    for code, text in findings or [("ok", "no automatic finding -- judge the sheet by eye")]:
        col = (255, 120, 90) if code in ("F1", "F2", "F3", "F5", "F8") else (230, 190, 90) if code == "F4" else (140, 200, 140)
        d.text((10, y), "%s  %s" % (code, text), fill=col)
        y += 16
    d.text((10, head + 2), "classic (LiteGraph)", fill=(150, 150, 170))
    d.text((ic.width + 30, head + 2), "Nodes 2.0", fill=(150, 150, 170))
    s.paste(ic, (10, head + 20))
    s.paste(iv, (ic.width + 30, head + 20))
    os.makedirs(os.path.join(out, "sheets"), exist_ok=True)
    s.save(os.path.join(out, "sheets", cls + ".png"))


NOISE_PX = 32     # measured 11.09.: Media Loader's thumbnail differs by 15 px run to run


def compare_dirs(ref_dir, new_dir):
    """Classic crops against a reference run: same size and at most NOISE_PX
    differing pixels = unchanged. Prints every node that is not bit-identical."""
    from PIL import Image, ImageChops
    changed, noisy, same = [], [], 0
    for name in sorted(os.listdir(ref_dir)):
        if not name.endswith(".png"):
            continue
        pn = os.path.join(new_dir, name)
        if not os.path.exists(pn):
            changed.append((name[:-4], "missing"))
            continue
        a = Image.open(os.path.join(ref_dir, name)).convert("RGB")
        b = Image.open(pn).convert("RGB")
        if a.size != b.size:
            changed.append((name[:-4], "size %s -> %s" % (a.size, b.size)))
            continue
        d = ImageChops.difference(a, b)
        n = sum(1 for px in d.getdata() if px != (0, 0, 0)) if d.getbbox() else 0
        if n == 0:
            same += 1
        elif n <= NOISE_PX:
            noisy.append((name[:-4], n))
        else:
            changed.append((name[:-4], "%d px" % n))
    print("classic vs reference: %d bit-identical, %d within noise %s, %d CHANGED %s"
          % (same, len(noisy), noisy, len(changed), changed[:8]))
    return 1 if changed else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8188/")
    ap.add_argument("--out", default="parity_out")
    ap.add_argument("--workflow")
    ap.add_argument("--stack")
    ap.add_argument("--only", default="")
    ap.add_argument("--ref")
    ap.add_argument("--ref-dir", help="a previous run's classic/ directory: pixel compare "
                    "with a fixed noise allowance (image thumbnails are not bit-stable)")
    a = ap.parse_args()
    try:
        import playwright  # noqa: F401
        import PIL  # noqa: F401
    except ImportError:
        print("browser_parity_sheet: playwright/Pillow missing -- SKIPPED")
        return 2
    inv = json.load(open(a.workflow, encoding="utf-8")) if a.workflow else None
    stack = json.load(open(a.stack, encoding="utf-8")) if a.stack else None
    only = set(x for x in a.only.split(",") if x)
    os.makedirs(a.out, exist_ok=True)
    res = asyncio.run(capture(a.url, only, inv, stack, a.out))
    report, md5 = {}, {}
    for cls in sorted(res):
        c, v = res[cls].get("classic"), res[cls].get("vue")
        fnd = classify(cls, c, v, a.out)
        report[cls] = {"classic": c, "vue": v, "findings": fnd}
        sheet(cls, c, v, fnd, a.out)
        pc = os.path.join(a.out, "classic", cls + ".png")
        if os.path.exists(pc):
            md5[cls] = hashlib.md5(open(pc, "rb").read()).hexdigest()
    json.dump(report, open(os.path.join(a.out, "parity.json"), "w"), indent=1)
    json.dump(md5, open(os.path.join(a.out, "classic_md5.json"), "w"), indent=1, sort_keys=True)
    counts = {}
    for cls in report:
        for code, _ in report[cls]["findings"]:
            counts[code] = counts.get(code, 0) + 1
    print("browser_parity_sheet: %d nodes, findings by class: %s" % (len(report), dict(sorted(counts.items()))))
    rc = 0
    if a.ref_dir:
        rc = max(rc, compare_dirs(a.ref_dir, os.path.join(a.out, "classic")))
    if a.ref:
        ref = json.load(open(a.ref, encoding="utf-8"))
        changed = sorted(k for k in md5 if k in ref and md5[k] != ref[k])
        missing = sorted(k for k in ref if k not in md5)
        print("classic reference: %d compared, %d changed %s, %d missing %s" % (
            len([k for k in md5 if k in ref]), len(changed), changed[:8], len(missing), missing[:5]))
        rc = 1 if (changed or missing) else 0
    return rc


if __name__ == "__main__":
    sys.exit(main())
