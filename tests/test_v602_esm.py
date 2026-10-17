"""Guard v602 -- the gate that lied, and the SyntaxError it waved through.

WHAT HAPPENED. v600 added six lines to ph_clip_encode.js. One of them was:

    const _configure = nodeType.prototype.onConfigure;

Twenty-five lines below sat v557's own, older line:

    const _configure = nodeType.prototype.configure;

Two `const`s, one scope. `SyntaxError: Identifier '_configure' has already been
declared`. **The module never loaded.** No tints, no visibility logic, six prompt
segments where two were asked for. The node did not misbehave -- it never ran at
all, and every symptom Frank reported was a corpse, not a bug.

AND GATE-2 SAID GREEN.

    node --check ph_clip_encode.js    ->  exit 0     (a lie)
    node --check ph_clip_encode.mjs   ->  SyntaxError (the truth)

`node --check` on a `.js` path parses as a CommonJS SCRIPT. ComfyUI loads these
files as ES MODULES -- they all begin with `import { app } from ...`. Different
parser, different rules, and the one the gate used was not the one the browser
uses. Twenty-one files checked, twenty-one green, one of them fatally broken.

**A gate that checks the wrong thing is worse than no gate.** No gate leaves you
careful. A green gate makes you confident, and confidence is what shipped this.

THE LAW: every file the browser will load as a module must PARSE as a module. This
guard runs the real parser, in the real mode, on every file -- and it is a guard,
not a shell line, so it cannot be forgotten the next time somebody re-types the
gate from memory.
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSDIR = os.path.join(ROOT, "web", "js")


def _fail(msg):
    print("[test_v602_esm] FAIL: %s" % msg)
    sys.exit(1)


if shutil.which("node") is None:
    print("[test_v602_esm] SKIP: node is not on PATH (the gate cannot run here)")
    sys.exit(0)


def _check_as_module(path):
    """Parse the file the way the BROWSER will: as an ES module.

    The .mjs copy is the whole trick. Same bytes, same parser binary, different
    goal symbol -- and that difference is the entire bug this guard exists for.
    """
    with tempfile.TemporaryDirectory() as tmp:
        mjs = os.path.join(tmp, "probe.mjs")
        shutil.copyfile(path, mjs)
        res = subprocess.run(["node", "--check", mjs],
                             capture_output=True, text=True, timeout=30)
        return res.returncode == 0, (res.stderr or "").strip()


files = []
for root, _dirs, names in os.walk(JSDIR):
    if "lib" in root.split(os.sep):
        continue                      # vendored bundles are not ours to police
    for n in sorted(names):
        if n.endswith(".js"):
            files.append(os.path.join(root, n))

if not files:
    _fail("no JS found -- the gate would pass by having nothing to check, which is the same "
          "failure in a different costume")

broken = []
for f in files:
    ok, err = _check_as_module(f)
    if not ok:
        head = [ln for ln in err.splitlines() if "Error" in ln]
        broken.append((os.path.relpath(f, ROOT), head[0] if head else err.splitlines()[0]))

if broken:
    print("[test_v602_esm] FAIL: %d file(s) do NOT parse as ES modules -- and the browser will "
          "load every one of them as an ES module, so each is DEAD on arrival, silently, with "
          "its whole extension:" % len(broken))
    for path, err in broken:
        print("    %-34s %s" % (path, err))
    print("  `node --check foo.js` parses a SCRIPT and will tell you these are fine. It is not "
          "the parser the browser uses. Check them as .mjs.")
    sys.exit(1)

# =========================================================================
# And prove the guard can actually SEE the bug it was built for. A guard that
# never fails is indistinguishable from a guard that cannot fail -- which is
# exactly what GATE-2 turned out to be.
# =========================================================================
victim = os.path.join(JSDIR, "ph_clip_encode.js")
src = open(victim, encoding="utf-8").read()
with tempfile.TemporaryDirectory() as tmp:
    hurt = os.path.join(tmp, "hurt.js")
    # Re-introduce the exact wound: a second `const _configure` in the same scope.
    open(hurt, "w", encoding="utf-8").write(
        src.replace("        const _cfgHeight = nodeType.prototype.onConfigure;",
                    "        const _configure = nodeType.prototype.onConfigure;", 1))
    ok, _err = _check_as_module(hurt)
    if ok:
        _fail("the self-test did not reproduce the wound -- this guard cannot see the bug it was "
              "written for, and is therefore decoration")

print("[test_v602_esm] PASS: all %d JS files parse as ES MODULES -- the mode the browser actually "
       "uses, not the CommonJS mode `node --check *.js` silently falls back to. The duplicate-const "
       "wound was re-injected and correctly caught, so this guard can see the thing it exists for. "
       "GATE-2 could not." % len(files))
sys.exit(0)
