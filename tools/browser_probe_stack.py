#!/usr/bin/env python3
"""browser_probe_stack.py -- v934: the LoRA Stack, clicked in a REAL browser.

WHY THIS EXISTS
---------------
v922-v928 passed every source guard and still killed the painted Stack under
the classic renderer: the frontend's container around the DOM panel stayed
clickable and swallowed every click (field 10.09.2026). A grep cannot see a
click wall. This probe can -- it drives a running ComfyUI through Chromium and
counts clicks that actually arrive.

WHAT IT PROVES (each line is a PASS/FAIL)
-----------------------------------------
  A  classic renderer, fresh:     no panel widget exists; "+" adds a row
  B  Nodes 2.0 on:                panel visible; its own "+" adds a row
  C  back to classic:             canvas under the pointer; "+" adds a row
  D  Nodes 2.0 on again:          panel visible again
  E  classic again:               "+" adds a row
  F  saved workflow:              the panel adds nothing to widgets_values
  G  reload with Nodes 2.0 on:    the panel appears on a loaded Stack
  H  classic (v937): painted \u21b5 inserts into a prompt field; painted order
     badge opens the shared input; (v939) \u21b5 right after that input still
     lands in the prompt field
  J  the Engine (v938): classic untouched; its Nodes 2.0 view drives modes,
     DARE variant, Apply, weight rules and the picker through shared code
  K  the Stack panel's picker shows the choice (v938)
  I  Nodes 2.0 (v937): panel \u21b5 and order badge through the SAME functions,
     warnings from checkConflicts, the weight-header explainer

REQUIREMENTS (not pack requirements -- this is a bench tool)
------------------------------------------------------------
A ComfyUI with this pack installed, running at --url (default
http://127.0.0.1:8188/), Python `playwright` and a Chromium it can find
(PLAYWRIGHT_BROWSERS_PATH). The probe CHANGES the Comfy.VueNodes.Enabled
setting of that ComfyUI and restores it to off at the end. Do not point it at
a session with unsaved work: it clears the graph.

Exit code 0 = all PASS, 1 = a FAIL, 2 = environment missing.
"""
import argparse
import asyncio
import json
import sys

HEADER_H, ROW_H = 130, 28      # mirrored from uls_node.js (PAD/ROW_H/HEADER_H)
ENGINE_HEADER_H = None           # read from uls_node.js below

STATE = r"""
() => {
  const n = window.__stack, g = app.canvas;
  const w = n.widgets?.find(x => x.name === 'uls_rows_dom');
  const lx = n.size[0] / 2;
  const ly = %d + n._uls.rows.length * %d + %d / 2;
  const r = g.canvas.getBoundingClientRect();
  const px = r.left + (n.pos[0] + lx) * g.ds.scale + g.ds.offset[0];
  const py = r.top + (n.pos[1] + ly) * g.ds.scale + g.ds.offset[1];
  const el = document.elementFromPoint(px, py);
  const root = w?.element;
  const rr = root ? root.getBoundingClientRect() : null;
  return {
    vue: LiteGraph.vueNodesMode === true,
    widget: !!w, hidden: w ? !!w.hidden : null,
    panelVisible: !!(rr && rr.width > 0 && rr.height > 0 && document.contains(root)),
    canvasUnderPointer: el === g.canvas,
    pt: [px, py], rows: n._uls.rows.length,
  };
}
""" % (HEADER_H, ROW_H, ROW_H)

results = []


def _engine_header_h():
    import os
    import re as _re
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "web", "js", "uls_node.js"), encoding="utf-8").read()
    m = _re.search(r"^const ENGINE_HEADER_H\s*=\s*(\d+)", src, _re.M)
    return int(m.group(1)) if m else 96


ENGINE_HEADER_H = _engine_header_h()


def verdict(ok, label):
    results.append(ok)
    print(("  PASS " if ok else "  FAIL ") + label)


async def st(page):
    return await page.evaluate(STATE)


async def click_painted_add(page):
    s = await st(page)
    await page.mouse.click(*s["pt"])
    await page.wait_for_timeout(400)
    return s["rows"], (await st(page))["rows"], s["canvasUnderPointer"]


async def set_vue(page, on):
    await page.evaluate(
        "v => app.extensionManager.setting.set('Comfy.VueNodes.Enabled', v)", on)
    await page.wait_for_timeout(1500)


async def run(url):
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch()
        page = await b.new_page(viewport={"width": 1600, "height": 1000})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        await page.goto(url)
        await page.wait_for_function("window.app && app.graph && app.canvas",
                                     timeout=90000)
        await page.wait_for_timeout(2500)
        await page.evaluate(
            "app.extensionManager.setting.set('Comfy.TutorialCompleted', true)")
        for _ in range(3):
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(250)
        await set_vue(page, False)
        await page.evaluate("""() => {
          app.graph.clear();
          const n = LiteGraph.createNode('UltimateLoraStack'); n.pos = [200, 150];
          app.graph.add(n); window.__stack = n;
          app.canvas.ds.offset = [0, 0]; app.canvas.ds.scale = 1;
          app.canvas.setDirty(true, true); }""")
        await page.wait_for_timeout(3000)

        s = await st(page)
        verdict(not s["widget"], "A  classic: no panel widget on the node")
        a, b2, canvas = await click_painted_add(page)
        verdict(canvas and b2 == a + 1,
                "A  classic: '+' reaches the painted node (rows %d -> %d)" % (a, b2))
        await page.wait_for_timeout(3000)           # idle: nothing may appear
        a, b2, canvas = await click_painted_add(page)
        verdict(canvas and b2 == a + 1,
                "A  classic after 3 s idle: still clickable (rows %d -> %d)" % (a, b2))

        await set_vue(page, True)
        s = await st(page)
        verdict(s["vue"] and s["widget"] and s["panelVisible"],
                "B  Nodes 2.0: panel attached and visible")
        before = s["rows"]
        add = page.locator(".uls-dom .uls-dom-add").first
        if await add.count():
            await add.click()
            await page.wait_for_timeout(400)
        verdict((await st(page))["rows"] == before + 1,
                "B  Nodes 2.0: the panel's own '+' adds a row")

        await set_vue(page, False)
        s = await st(page)
        verdict(s["widget"] and s["hidden"] and not s["panelVisible"],
                "C  back to classic: panel marked hidden")
        a, b2, canvas = await click_painted_add(page)
        verdict(canvas and b2 == a + 1,
                "C  back to classic: '+' reaches the painted node (rows %d -> %d)"
                % (a, b2))

        await set_vue(page, True)
        s = await st(page)
        verdict(s["panelVisible"] and not s["hidden"],
                "D  Nodes 2.0 again: panel visible again")

        wf = await page.evaluate("JSON.stringify(app.graph.serialize())")
        await set_vue(page, False)
        a, b2, canvas = await click_painted_add(page)
        verdict(canvas and b2 == a + 1,
                "E  classic again: '+' reaches the painted node (rows %d -> %d)"
                % (a, b2))

        node = [n for n in json.loads(wf)["nodes"]
                if n.get("type") == "UltimateLoraStack"][0]
        vals = node.get("widgets_values", [])
        verdict(len(vals) == 1,
                "F  workflow saved under Nodes 2.0: widgets_values holds only "
                "uls_config (%d entr%s)" % (len(vals), "y" if len(vals) == 1 else "ies"))

        await set_vue(page, True)
        await page.evaluate("wf => app.loadGraphData(JSON.parse(wf))", wf)
        await page.wait_for_timeout(2500)
        ok = await page.evaluate("""() => {
          const n = app.graph._nodes.find(x => x.type === 'UltimateLoraStack');
          window.__stack = n;
          const w = n?.widgets?.find(x => x.name === 'uls_rows_dom');
          const r = w?.element?.getBoundingClientRect();
          return !!(r && r.height > 0 && !w.hidden); }""")
        verdict(ok, "G  workflow loaded with Nodes 2.0 on: panel appears")

        await set_vue(page, False)

        # ---- v937 (S2): parity pieces, SHARED code, in both renderers -------
        await page.evaluate("""() => {
          app.graph.clear();
          const n = LiteGraph.createNode('UltimateLoraStack'); n.pos = [200, 150];
          app.graph.add(n); window.__stack = n;
          const t = LiteGraph.createNode('CLIPTextEncode'); t.pos = [760, 150];
          app.graph.add(t); window.__te = t;
          t.widgets.find(w => w.name === 'text').value = 'PROMPT_MARK';
          app.canvas.ds.offset = [0, 0]; app.canvas.ds.scale = 1;
          app.canvas.setDirty(true, true); }""")
        await page.wait_for_timeout(1500)
        await page.evaluate("""() => { const n = window.__stack;
          n._uls.rows[0].name = 'test_a.safetensors'; n._uls.rows[0].group = 'subject';
          n._ulsSync(); n.setDirtyCanvas(true, true); }""")
        await page.wait_for_timeout(500)
        PT = r"""([lx, ly]) => { const n = window.__stack, g = app.canvas;
            const r = g.canvas.getBoundingClientRect();
            return [r.left + (n.pos[0] + lx) * g.ds.scale + g.ds.offset[0],
                    r.top + (n.pos[1] + ly) * g.ds.scale + g.ds.offset[1]]; }"""
        FOCUS = """() => { const ta = [...document.querySelectorAll('textarea')]
            .find(x => x.value.startsWith('PROMPT_MARK'));
          if (!ta) return false; ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length);
          return true; }"""
        TEXT = """() => { const ta = [...document.querySelectorAll('textarea')]
            .find(x => x.value.startsWith('PROMPT_MARK')); return ta ? ta.value : null; }"""
        W = await page.evaluate("window.__stack.size[0]")
        grp_x = W - 8 - 18 - 4 - 72 - 4 - 50            # onMouseDown zone rules
        ins_x = grp_x - 4 - 28
        row_y = 130                                       # HEADER_H, row 0

        # H1 classic: the painted trigger button inserts into the prompt field
        focused = await page.evaluate(FOCUS)
        x, y = await page.evaluate(PT, [ins_x + 14, row_y + 14])
        await page.mouse.click(x, y)
        await page.wait_for_timeout(700)
        txt = await page.evaluate(TEXT)
        verdict(focused and txt and txt != "PROMPT_MARK" and "test" in txt,
                "H1 classic: painted \u21b5 inserts the trigger (%r)" % (txt,))
        # H2 classic: the painted order badge opens the SHARED input
        x, y = await page.evaluate(PT, [grp_x + 7, row_y + 8])
        await page.mouse.click(x, y)
        await page.wait_for_timeout(400)
        has_inp = await page.evaluate("!!document.querySelector('#uls-weight-input input')")
        if has_inp:
            await page.keyboard.type("3")
            await page.keyboard.press("Enter")
            await page.wait_for_timeout(300)
        got = await page.evaluate("(window.__stack._uls.groupOrder || {}).subject")
        verdict(has_inp and got == 3, "H2 classic: painted order badge -> input -> 3 (%r)" % (got,))
        # H3 (v939): right after the order input, \u21b5 must land in the prompt
        # field. Before v939 the removed order INPUT was the target and the
        # TEXTAREA setter threw "Illegal invocation" (measured 10.09.).
        before = await page.evaluate(TEXT)
        x, y = await page.evaluate(PT, [ins_x + 14, row_y + 14])
        await page.mouse.click(x, y)
        await page.wait_for_timeout(700)
        after = await page.evaluate(TEXT)
        verdict(after is not None and after != before and after.count("(test:1.00)") == 2,
                "H3 classic: \u21b5 right after the order input still lands in the prompt (%r)" % (after,))

        # I: the same pieces under Nodes 2.0, through the DOM panel
        await page.evaluate("() => { const t = window.__te; t.widgets.find(w => w.name === 'text').value = 'PROMPT_MARK'; }")
        await set_vue(page, True)
        await page.wait_for_timeout(800)
        n_trig = await page.locator(".uls-dom .uls-dom-trig").count()
        verdict(n_trig == 1, "I1 Nodes 2.0: one \u21b5 per row in the panel (%d)" % n_trig)
        focused = await page.evaluate(FOCUS)
        await page.locator(".uls-dom .uls-dom-trig").first.click()
        await page.wait_for_timeout(700)
        txt = await page.evaluate(TEXT)
        verdict(focused and txt and txt != "PROMPT_MARK" and "test" in txt,
                "I2 Nodes 2.0: panel \u21b5 inserts through the SAME function (%r)" % (txt,))
        badge = page.locator(".uls-dom .uls-dom-obadge").first
        btxt = await badge.text_content() if await badge.count() else None
        verdict(btxt == "3", "I3 Nodes 2.0: the badge shows the order set in the painted view (%r)" % (btxt,))
        if await badge.count():
            await badge.click()
            await page.wait_for_timeout(400)
            has_inp = await page.evaluate("!!document.querySelector('#uls-weight-input input')")
            dialog_open = await page.evaluate("!!document.querySelector('#uls-group-mode-popup, .uls-group-mode-popup')")
            if has_inp:
                await page.keyboard.press("Control+A")
                await page.keyboard.type("5")
                await page.keyboard.press("Enter")
                await page.wait_for_timeout(400)
            got = await page.evaluate("(window.__stack._uls.groupOrder || {}).subject")
            btxt = await page.locator(".uls-dom .uls-dom-obadge").first.text_content()
            verdict(has_inp and got == 5 and btxt == "5" and not dialog_open,
                    "I4 Nodes 2.0: badge -> SHARED input -> 5, panel redraws, group dialog stays shut (%r/%r)" % (got, btxt))
        await page.evaluate("""() => { const n = window.__stack;
          n._uls.rows[0].wHigh = 6; n._uls.rows[0].wLow = 6;
          n._uls.rows.push(Object.assign({}, n._uls.rows[0], {name: 'test_b.safetensors'}));
          n._ulsSync(); }""")
        await page.locator(".uls-dom .uls-dom-add").first.click()   # any commit redraws
        await page.wait_for_timeout(400)
        gw = await page.locator(".uls-dom .uls-dom-gwarn").count()
        verdict(gw >= 1, "I5 Nodes 2.0: global weight warning from checkConflicts (%d)" % gw)
        tip = await page.evaluate("document.querySelector('.uls-dom .uls-dom-whdr')?.title || ''")
        verdict(tip.startswith("Weight / CLIP Strength\nClick: model weight."),
                "I6 Nodes 2.0: weight-header explainer from the shared lines")
        await set_vue(page, False)

        # ---- v938 (S4): the Engine, both renderers; the Stack's picker -----
        await page.evaluate("""() => {
          app.graph.clear();
          const e = LiteGraph.createNode('ULSAccelerator'); e.pos = [200, 150];
          app.graph.add(e); window.__eng = e;
          const n = LiteGraph.createNode('UltimateLoraStack'); n.pos = [760, 150];
          app.graph.add(n); window.__stack = n;
          app.canvas.ds.offset = [0, 0]; app.canvas.ds.scale = 1;
          app.canvas.setDirty(true, true); }""")
        await page.wait_for_timeout(2000)
        EH = await page.evaluate("ENGINE_HDR => ENGINE_HDR", ENGINE_HEADER_H)
        async def eng_add_click():
            r0 = await page.evaluate("window.__eng._uls.rows.length")
            x, y = await page.evaluate("""(eh) => { const n = window.__eng, g = app.canvas;
              const r = g.canvas.getBoundingClientRect();
              const ly = eh + n._uls.rows.length * 28 + 14;
              return [r.left + (n.pos[0] + n.size[0] / 2) * g.ds.scale + g.ds.offset[0],
                      r.top + (n.pos[1] + ly) * g.ds.scale + g.ds.offset[1]]; }""", EH)
            await page.mouse.click(x, y)
            await page.wait_for_timeout(400)
            return r0, await page.evaluate("window.__eng._uls.rows.length")
        has_w = await page.evaluate("!!window.__eng.widgets.find(w => w.name === 'uls_engine_dom')")
        a, b2 = await eng_add_click()
        verdict(not has_w and b2 == a + 1,
                "J1 classic: Engine has no view widget; painted '+' works (%d -> %d)" % (a, b2))
        await set_vue(page, True)
        await page.wait_for_timeout(800)
        EV = "window.__eng.widgets.find(w => w.name === 'uls_engine_dom')"
        vis = await page.evaluate("(() => { const w = %s; if (!w) return false; const r = w.element.getBoundingClientRect(); return r.height > 0 && !w.hidden; })()" % EV)
        verdict(vis, "J2 Nodes 2.0: the Engine's view is attached and visible")
        eng = "window.__eng.widgets.find(w => w.name === 'uls_engine_dom').element"
        async def eng_click(sel, idx=0, shift=False):
            await page.evaluate("([sel, idx, shift]) => { const el = %s.querySelectorAll(sel)[idx]; el.dispatchEvent(new MouseEvent('click', {bubbles: true, shiftKey: shift, clientX: 400, clientY: 300})); }" % eng, [sel, idx, shift])
            await page.wait_for_timeout(250)
        await eng_click(".uls-eng-mode", 2)
        modev = await page.evaluate("[window.__eng._uls.mode, %s.querySelectorAll('.uls-eng-dv').length]" % eng)
        verdict(modev == ["DARE", 1], "J3 Nodes 2.0: D selects DARE and the CHAN/ELEM pill appears (%s)" % modev)
        await eng_click(".uls-eng-dv")
        dv = await page.evaluate("window.__eng._uls.dareVariant")
        a0 = await page.evaluate("window.__eng._uls.apply || 'auto'")
        await eng_click(".uls-eng-apply")
        a1 = await page.evaluate("window.__eng._uls.apply")
        verdict(dv == "element" and a1 != a0,
                "J4 Nodes 2.0: variant toggles (%s), Apply cycles (%s -> %s)" % (dv, a0, a1))
        await eng_click(".uls-dom-step", 1)
        await eng_click(".uls-dom-step", 1, shift=True)
        wts = await page.evaluate("[window.__eng._uls.rows[0].weight, window.__eng._uls.rows[0].wClip]")
        verdict(wts == [1.05, 1.1],
                "J5 Nodes 2.0: \u25b6 steps the weight, Shift+\u25b6 the CLIP strength, by the SHARED rule (%s)" % wts)
        await page.evaluate("() => %s.querySelector('.uls-dom-name').click()" % eng)
        await page.wait_for_timeout(800)
        item = page.locator("#uls-lora-select >> text=test_b").first   # the picker only
        if await item.count():
            await item.click()
            await page.wait_for_timeout(800)
        shown = await page.evaluate("%s.querySelector('.uls-dom-name').textContent" % eng)
        stn = await page.evaluate("window.__eng._uls.rows[0].name")
        verdict(stn == "test_b.safetensors" and shown == "test_b",
                "J6 Nodes 2.0: the Engine's picker -- state and view agree (%r / %r)" % (stn, shown))
        await page.evaluate("() => %s.querySelector('.uls-dom-add').click()" % eng)
        await page.wait_for_timeout(300)
        nr = await page.evaluate("window.__eng._uls.rows.length")
        cfg = await page.evaluate("JSON.parse(window.__eng.widgets.find(w => w.name === 'engine_config').value)")
        verdict(nr == 3 and cfg["mode"] == "DARE" and cfg["dare_variant"] == "element"
                and cfg["rows"][0]["name"] == "test_b.safetensors",
                "J7 Nodes 2.0: '+' adds a row; engine_config carries every change")
        # K: the Stack panel's picker shows the choice (v928 showed the old name)
        stack_name = page.locator(".uls-dom .uls-dom-name").first
        sv = "window.__stack.widgets.find(w => w.name === 'uls_rows_dom').element"
        await page.evaluate("() => %s.querySelector('.uls-dom-name').click()" % sv)
        await page.wait_for_timeout(800)
        item = page.locator("#uls-lora-select >> text=test_b").first   # the picker only
        if await item.count():
            await item.click()
            await page.wait_for_timeout(800)
        shown = await page.evaluate("%s.querySelector('.uls-dom-name').textContent" % sv)
        stn = await page.evaluate("window.__stack._uls.rows[0].name")
        verdict(stn == "test_b.safetensors" and shown == "test_b",
                "K  Nodes 2.0: the Stack's picker -- state and view agree (%r / %r)" % (stn, shown))
        await set_vue(page, False)
        await page.wait_for_timeout(600)
        a, b2 = await eng_add_click()
        hid = await page.evaluate("%s.hidden" % EV)
        verdict(hid and b2 == a + 1,
                "J8 back to classic: view hidden, painted '+' works again (%d -> %d)" % (a, b2))
        verdict(not errors, "no page errors (%s)" % ("; ".join(errors)[:200] or "none"))
        await b.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8188/")
    args = ap.parse_args()
    try:
        import playwright  # noqa: F401
    except ImportError:
        print("browser_probe_stack: playwright not installed -- SKIPPED")
        return 2
    asyncio.run(run(args.url))
    bad = results.count(False)
    print("browser_probe_stack: %s (%d/%d)" % (
        "PASS" if not bad else "FAIL", results.count(True), len(results)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
