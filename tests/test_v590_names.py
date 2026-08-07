"""v590 guard: NO NAME IS READ THAT WAS NEVER BOUND. Tree-wide.

Born from Frank's field crash of 2026-07-14. The final pass of a 17:51 run
died on its last breath:

    ph_power_upscale.py:765  if fit_s["s"] > 0.5 and mid:
    NameError: name 'fit_s' is not defined

v587 built the fit stopwatch into TWO functions and forgot the channel between
them: `fit_s` was declared inside _esrgan_resident and read inside
_esrgan_pass. Every pass carrying an upscale model - 'model + fit' (the
default), 'model only', 'model final' (Frank's driving list) - hit that line
and died. Only 'fit only' survived, because it returns early. The Power
Upscale node was dead in every model mode for THREE cuts (v587, v588, v589)
and 52 guards were green.

They were green because GATE-1 is `py_compile`, and py_compile checks SYNTAX.
`fit_s["s"]` is perfectly grammatical Python. Whether the name exists is a
question about SCOPE, and nothing in the chain asked it. This guard asks it,
for every function in the tree, on every cut, forever.

It is deliberately CONSERVATIVE: it binds comprehension targets, lambda
parameters and nested definitions into the enclosing function even where
Python would give them their own scope. That can miss an exotic offender; it
can never invent one. A guard that cries wolf gets switched off, and then it
guards nothing - the worst outcome of all. Star-imported modules are skipped
and reported, because there the question is genuinely unanswerable.

The checker proves it can fail before it is trusted (v581 §5): the broken
v587 shape and the fixed v590 shape are both run through it every time.
"""
import ast
import builtins
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# v590: self-count. 107 modules shipped in v589, this guard makes 108. If the
# scan ever sees a fraction of the tree - a broken path, a moved folder - it
# fails instead of reporting a comfortable green over a room it never entered
# (the v575 blindness, v581 §3).
_MIN_MODULES = 100

_DUNDERS = {"__file__", "__name__", "__doc__", "__package__", "__spec__",
            "__loader__", "__builtins__", "__path__", "__debug__", "__all__",
            "__version__", "__dict__", "__class__"}

_PROBE_BROKEN = '''
import time
def resident(x):
    fit_s = {"s": 0.0}
    fit_s["s"] += time.monotonic()
    return x

def outer(x):
    y = resident(x)
    if fit_s["s"] > 0.5:
        print(fit_s["s"])
    return y
'''

_PROBE_SOUND = '''
import time
def resident(x, fit_s=None):
    fit_s = {"s": 0.0} if fit_s is None else fit_s
    fit_s["s"] += time.monotonic()
    return x

def outer(x):
    fit_s = {"s": 0.0}
    y = resident(x, fit_s=fit_s)
    if fit_s["s"] > 0.5:
        print(fit_s["s"])
    return [n for n in range(3)] + [(lambda q: q)(1)]
'''


def _fail(msg):
    print(f"[test_v590_names] FAIL: {msg}")
    sys.exit(1)


def _is_scope(node):
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Lambda))


def _bindings(node, cross_scopes):
    """Names bound by `node`'s body.

    cross_scopes=False stays on this scope's own level (used for the module):
    an import inside a function does not bind at module level. cross_scopes=
    True descends into everything (used inside a function): conservative on
    purpose - comprehension targets and lambda params get folded in, so this
    can never invent an offender.
    """
    bound = set()
    stack = list(ast.iter_child_nodes(node))
    while stack:
        n = stack.pop()
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            bound.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                if a.name == "*":
                    bound.add("*")          # unanswerable - flagged by caller
                else:
                    bound.add(a.asname or a.name.split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            bound.update(n.names)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        if cross_scopes or not _is_scope(n):
            stack.extend(ast.iter_child_nodes(n))
    return bound


def _params(fn):
    a = fn.args
    out = {x.arg for x in (list(a.posonlyargs) + list(a.args)
                           + list(a.kwonlyargs))}
    if a.vararg:
        out.add(a.vararg.arg)
    if a.kwarg:
        out.add(a.kwarg.arg)
    return out


def _scan_source(src, label="<probe>"):
    """-> list of (function, name, line) read without ever being bound."""
    tree = ast.parse(src, filename=label)
    module = _bindings(tree, cross_scopes=False)
    if "*" in module:
        return []                            # star import: cannot prove anything
    visible = module | set(dir(builtins)) | _DUNDERS
    hits = []

    def walk(node, outer):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                own = outer | _params(child) | _bindings(child, cross_scopes=True)
                for sub in ast.walk(child):
                    if (isinstance(sub, ast.Name)
                            and isinstance(sub.ctx, ast.Load)
                            and sub.id not in own):
                        hits.append((child.name, sub.id, sub.lineno))
                walk(child, own)             # nested defs see the enclosing names
            else:
                walk(child, outer)

    walk(tree, visible)
    # one report per (function, name) - the first line is where to look
    seen, out = set(), []
    for fn, name, line in hits:
        if (fn, name) not in seen:
            seen.add((fn, name))
            out.append((fn, name, line))
    return out


def main():
    # ---- 1: the checker must prove it can fail (v581 §5) --------------------
    broken = _scan_source(_PROBE_BROKEN, "probe_broken")
    if not any(name == "fit_s" for _, name, _ in broken):
        _fail("the checker does not catch the v587 shape (bound in one scope, "
              "read in another) - it would have been green through the crash "
              "it exists to prevent")
    if _scan_source(_PROBE_SOUND, "probe_sound"):
        _fail("the checker flags the CORRECT shape (handed-in dict, "
              "comprehension, lambda) - a guard that cries wolf gets switched "
              "off, and then it guards nothing")

    # ---- 2: the tree ---------------------------------------------------------
    # v645: nodes/vendor is THIRD-PARTY (vendored trimesh) -- excluded like
    # web/js/lib in GATE-2; its property-setter pattern false-positives here.
    files = sorted(p for p in ROOT.rglob("*.py")
                   if "vendor" not in p.parts
                   if "__pycache__" not in p.parts)
    if len(files) < _MIN_MODULES:
        _fail(f"only {len(files)} modules found (expected >= {_MIN_MODULES}) - "
              f"the scan is looking at the wrong place, not at a smaller tree")

    offenders, skipped = [], []
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        try:
            src = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            _fail(f"{rel} is not readable as utf-8")
        try:
            tree = ast.parse(src, filename=rel)
        except SyntaxError as exc:
            _fail(f"{rel} does not parse: {exc}")   # GATE-1 territory, restated
        if "*" in _bindings(tree, cross_scopes=False):
            skipped.append(rel)
            continue
        for fn, name, line in _scan_source(src, rel):
            offenders.append((rel, fn, name, line))

    if offenders:
        for rel, fn, name, line in offenders:
            print(f"[test_v590_names]   {rel}:{line}  {fn}(): "
                  f"undefined name '{name}'")
        _fail(f"{len(offenders)} name(s) READ but never BOUND. This is the "
              f"v587-v589 crash class (NameError at runtime, on a path no "
              f"guard executes). py_compile passes these files: syntax is not "
              f"scope.")

    note = f", {len(skipped)} skipped (star import)" if skipped else ""
    print(f"PASS: v590 -- {len(files)} modules scanned, no name is read that "
          f"was never bound{note} (probes: broken shape caught, sound shape "
          f"clean)")


if __name__ == "__main__":
    main()
