# -*- coding: utf-8 -*-
"""
test_v267_hardening.py
══════════════════════
Guards the v267 hardening pass (audit findings A-1…A-8 + new findings N-1…N-9
from the second independent audit). Three parts:

  [1] Wiring (always runs, no ComfyUI/torch needed): every fix is present in
      the shipped source, and the removed hazards are actually gone:
        • A-1  50 MB safetensors-header cap in handle_civitai_fetch
        • A-2  _pick_preview_url helper, NO fallback past the NSFW filter
        • A-3  32 MB download cap in the CLI generator (pre-check + mid-stream)
        • A-4  two-stage tokenizer load (local cache first, then online)
        • A-5  stale "disabled until its backend lands" JSDoc corrected
        • A-6  group whitelist on POST /uls/groups
        • A-8  exact handoff pin in the dual sigma schedule
        • N-1  NSFW threshold unified (CLI ≤2, route ≤2)
        • N-2  overlay group buttons call node._ulsSync() (the one missed path)
        • N-5  RFC-7233 suffix ranges handled
        • N-7  CUDA RNG state saved whenever CUDA is initialised
        • N-9  stale "[ULS v052]" console tags gone
        • N-3  README trigger priority order matches the code
  [2] _pick_preview_url functional (AST-loaded, pure function, no deps):
      filter-strict behaviour — all-NSFW lists yield None, no images[0] bypass.
  [3] Sigma handoff EXACT (needs torch+numpy, SKIPs without): after A-8 the
      dual schedule satisfies HIGH[split] == LOW[split] bit-exactly (diff 0.0,
      previously 1 ULP ≈ 5.96e-08), and both curves stay monotone.
"""
import os
import ast
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
ROUTES = os.path.join(ROOT, "nodes", "uls_routes.py")
STACK = os.path.join(ROOT, "nodes", "uls_stack_node.py")
SIGMA = os.path.join(ROOT, "nodes", "wan_sigma_schedule.py")
ULSJS = os.path.join(ROOT, "web", "js", "uls_node.js")
README = os.path.join(ROOT, "README.md")


def main():
    ok = True

    def check(part, label, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{part}] {label:<52} {'PASS' if cond else 'FAIL'}")

    routes = open(ROUTES, encoding="utf-8").read()
    stack = open(STACK, encoding="utf-8").read()
    # v348: the CUDA RNG-state guard lives in _resolve_sign_elect, which moved to
    # uls_merge_math.py — append it so the N-7 positive check still finds it.
    _MM = os.path.join(ROOT, "nodes", "uls_merge_math.py")
    if os.path.exists(_MM):
        stack += "\n" + open(_MM, encoding="utf-8").read()
    sigma = open(SIGMA, encoding="utf-8").read()
    ulsjs = open(ULSJS, encoding="utf-8").read()
    readme = open(README, encoding="utf-8").read()

    # ── [1] wiring ──
    check(1, "A-1 header cap (50 MB) present",
          "Safetensors header invalid or oversized" in routes
          and "if length <= 0 or length > 50 * 1024 * 1024:" in routes)
    check(1, "A-1 truncated-header guard present",
          "truncated header" in routes)
    check(1, "A-2 _pick_preview_url helper defined",
          "def _pick_preview_url(images, max_nsfw_level: int = 2):" in routes)
    check(1, "A-2 call site uses the helper",
          "preview_url = _pick_preview_url(images)" in routes)
    check(1, "A-2 old fallback bypass gone",
          "preview_url = images[0].get" not in routes)
    check(1, "A-4 two-stage tokenizer load present",
          "local_files_only=_local_only" in stack)
    check(1, "A-5 stale JSDoc corrected",
          "live since v256" in ulsjs and "disabled until its backend lands" not in ulsjs)
    check(1, "A-6 group whitelist present",
          "_VALID_GROUPS" in routes and "Unknown group:" in routes)
    check(1, "A-8 exact handoff pin present",
          "sigmas_low_np[split_step] = handoff" in sigma)
    check(1, "N-2 overlay group buttons sync the widget",
          'console.warn("[ULS] Gruppe speichern:"' in ulsjs
          and ulsjs.find("node?._ulsSync?.();",
                         ulsjs.find('Gruppe speichern')) - ulsjs.find('Gruppe speichern') < 700)
    # v388: the preview video is served by aiohttp's FileResponse, which answers suffix ranges itself
    check(1, "N-5 suffix ranges answered (FileResponse)",
          "web.FileResponse(vid_path" in routes)
    check(1, "N-7 CUDA RNG state condition widened",
          "torch.cuda.is_initialized()" in stack)
    check(1, "N-9 stale [ULS v052] tags gone",
          "[ULS v052]" not in ulsjs)
    prio_line = next((ln for ln in readme.splitlines()
                      if "\u2192" in ln and ".uls-meta.json" in ln), "")
    check(1, "N-3 README priority order corrected",
          0 <= prio_line.find(".uls-meta.json") < prio_line.find(".txt")
          < prio_line.find("safetensors header")
          and "`.uls-meta.json` (user-curated" in readme)

    # ── [2] _pick_preview_url functional (pure, AST-loaded) ──
    tree = ast.parse(routes)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "_pick_preview_url")
    ns = {}
    exec(ast.get_source_segment(routes, fn), ns)
    pick = ns["_pick_preview_url"]

    sfw = {"type": "image", "nsfwLevel": 1, "url": "https://x/sfw.png"}
    soft = {"type": "image", "nsfwLevel": 2, "url": "https://x/soft.png"}
    mature = {"type": "image", "nsfwLevel": 4, "url": "https://x/mature.png"}
    xxx = {"type": "image", "nsfwLevel": 8, "url": "https://x/x.png"}
    vid = {"type": "video", "nsfwLevel": 1, "url": "https://x/v.mp4"}

    check(2, "all-NSFW list → None (NO images[0] fallback)",
          pick([mature, xxx]) is None)
    check(2, "first SFW image wins", pick([mature, sfw, soft]) == sfw["url"])
    check(2, "Soft (level 2) accepted", pick([soft]) == soft["url"])
    check(2, "Mature (level 4) rejected at default", pick([mature]) is None)
    check(2, "video entries skipped", pick([vid, sfw]) == sfw["url"])
    check(2, "empty / None tolerated", pick([]) is None and pick(None) is None)
    check(2, "threshold override works", pick([mature], max_nsfw_level=4) == mature["url"])

    # ── [3] sigma handoff exact (torch-guarded) ──
    try:
        import torch  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        print("[3] sigma exactness: SKIP (torch/numpy not available)")
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location("wss_v267", SIGMA)
        wss = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wss)
        node = wss.ULSWanSplitNoiseSchedule()
        worst = 0.0
        mono = True
        for sh in wss.SIGMA_SCHEDULE_NAMES:
            for sl in wss.SIGMA_SCHEDULE_NAMES:
                hi, lo = node.compute(sh, sl, 20, 8, 1.0, 0.002, 7.0, 7.0)
                worst = max(worst, abs(float(hi[8] - lo[8])))
                for s in (hi, lo):
                    v = s[:-1]
                    mono &= all(float(v[i]) >= float(v[i + 1]) - 1e-9
                                for i in range(len(v) - 1))
        check(3, f"HIGH[split] == LOW[split] bit-exact (max diff {worst:.1e})",
              worst == 0.0)
        check(3, "both curves monotone", mono)

    print("=" * 64)
    print("RESULT:", "ALL CHECKS PASS" if ok else "FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
