# -*- coding: ascii -*-
"""Find the UnboundLocalError shape: a name bound by a CONDITIONAL import
inside a function, and read on a path that can skip that import.

WHY THIS EXISTS. An `import x.y` anywhere in a function body makes `x` LOCAL to
the WHOLE function -- Python decides that at compile time. Put it inside an
`if` or a `try` that some paths do not take, read `x` on one of those paths, and
the read raises UnboundLocalError. There is no warning: the module imports
fine, the function compiles fine, and it only dies when that path runs.

That is what v870 did to uls_sampler._initial_noise (measured 28.08.):

    if not add_noise:
        ...
        try:
            import comfy.nested_tensor          # binds `comfy` -- LOCALLY
            ...
    ...
    return comfy.sample.prepare_noise(...)      # add_noise=True path: unbound

The usual shape is SAFE and must not be flagged:

    try:
        import x
    except ImportError:
        return None                             # the skipping path terminates
    x.foo()                                     # so this is unreachable unbound

So this is a small dataflow pass, not a grep: it carries the set of names that
are DEFINITELY bound at each point, merges branches by intersection, and treats
a branch that returns/raises as contributing nothing to the merge.

ASCII only, standard library only.
"""

import ast


def _bound_by(node):
    """Names an import statement binds."""
    out = []
    if isinstance(node, ast.Import):
        for a in node.names:
            out.append(a.asname or a.name.split(".")[0])
    elif isinstance(node, ast.ImportFrom):
        for a in node.names:
            if a.name != "*":
                out.append(a.asname or a.name)
    return out


def _terminates(body, exits=()):
    """Does this block always leave the enclosing flow?

    Beyond return/raise/continue/break this counts a CALL to something that
    never comes back: sys.exit / os._exit, and any helper of the same module
    whose own body always terminates. Guards in this tree are written as
    `_fail(msg)` where `_fail` ends in sys.exit(1) -- without this, every such
    guard reads as a fall-through and the audit drowns in false positives.
    """
    for st in body:
        if isinstance(st, (ast.Return, ast.Raise, ast.Continue, ast.Break)):
            return True
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
            if _is_exit_call(st.value, exits):
                return True
        if isinstance(st, ast.If) and st.orelse:
            if _terminates(st.body, exits) and _terminates(st.orelse, exits):
                return True
        if isinstance(st, ast.Try):
            if (_terminates(st.body, exits)
                    and all(_terminates(h.body, exits) for h in st.handlers)):
                return True
    return False


def _is_exit_call(call, exits=()):
    f = call.func
    if isinstance(f, ast.Name):
        return f.id in exits
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        return (f.value.id, f.attr) in (("sys", "exit"), ("os", "_exit"))
    return False


def _module_exits(tree):
    """Names of module-level functions that never return normally.

    One pass, then a second so a helper calling another helper is caught -- two
    is enough for this tree and stays honest about what it does NOT prove.
    """
    exits = {"exit", "quit"}
    for _ in range(2):
        for fun in tree.body:
            if isinstance(fun, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if _terminates(fun.body, exits) and _always_leaves(fun.body,
                                                                  exits):
                    exits.add(fun.name)
    return exits


def _always_leaves(body, exits):
    """True when EVERY path out of this body raises or exits -- a plain
    `return` does come back to the caller, so it does not count here."""
    for st in body:
        if isinstance(st, ast.Return):
            return False
        if isinstance(st, ast.Raise):
            return True
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
            if _is_exit_call(st.value, exits):
                return True
        if isinstance(st, ast.If) and st.orelse:
            if (_always_leaves(st.body, exits)
                    and _always_leaves(st.orelse, exits)):
                return True
    return False


def _assigned(node):
    out = []
    for t in getattr(node, "targets", []) or ([node.target]
                                              if hasattr(node, "target") else []):
        for n in ast.walk(t):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                out.append(n.id)
    return out


class _Scan(object):
    """Walk one function body, tracking definitely-bound names."""

    def __init__(self, watched, exits=()):
        self.watched = set(watched)
        self.exits = set(exits)
        self.risky = []          # (name, read_lineno)

    def reads(self, node, safe, skip=()):
        for n in ast.walk(node):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.Lambda)):
                continue          # a nested scope has its own rules
            if (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                    and n.id in self.watched and n.id not in safe
                    and n.id not in skip):
                self.risky.append((n.id, n.lineno))

    def block(self, body, safe):
        """Returns the set of names definitely bound after this block."""
        safe = set(safe)
        for st in body:
            if isinstance(st, (ast.Import, ast.ImportFrom)):
                safe |= set(_bound_by(st))
                continue
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
                safe.add(st.name)
                continue
            if isinstance(st, ast.If):
                self.reads(st.test, safe)
                a = self.block(st.body, safe)
                b = self.block(st.orelse, safe) if st.orelse else set(safe)
                if _terminates(st.body, self.exits):
                    safe = b
                elif st.orelse and _terminates(st.orelse, self.exits):
                    safe = a
                else:
                    safe = a & b
                continue
            if isinstance(st, ast.Try):
                # The body may fail ANYWHERE, so on its own its bindings are
                # not certain. But the common fallback shape
                #     try:    from .x import Y
                #     except: from  x import Y
                # binds the same name on BOTH paths, and is safe. So: merge the
                # successful body with every handler that does NOT leave the
                # flow; handlers that return or raise contribute nothing,
                # because no path continues through them.
                a = self.block(st.body, safe)
                paths = [a]
                for h in st.handlers:
                    got = self.block(h.body, safe)
                    if not _terminates(h.body, self.exits):
                        paths.append(got)
                if not st.handlers:
                    paths.append(set(safe))     # bare try/finally: no cover
                after = set.intersection(*[set(p) for p in paths])
                if st.orelse:
                    after = self.block(st.orelse, after)
                if st.finalbody:
                    after = self.block(st.finalbody, after)
                safe = after
                continue
            if isinstance(st, (ast.For, ast.AsyncFor, ast.While)):
                self.reads(getattr(st, "iter", None) or st.test, safe)
                self.block(st.body, safe)      # may run zero times
                if st.orelse:
                    self.block(st.orelse, safe)
                continue
            if isinstance(st, (ast.With, ast.AsyncWith)):
                for item in st.items:
                    self.reads(item.context_expr, safe)
                safe = self.block(st.body, safe)
                continue
            self.reads(st, safe)
            safe |= set(_assigned(st))
        return safe


def scan_function(fun, exits=()):
    """Risky (name, lineno) pairs for one FunctionDef."""
    watched = set()
    for st in ast.walk(fun):
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and st is not fun:
            continue
        if isinstance(st, (ast.Import, ast.ImportFrom)):
            watched |= set(_bound_by(st))
    if not watched:
        return []
    # arguments are bound on entry
    args = fun.args
    start = set()
    for a in (list(args.args) + list(args.posonlyargs) + list(args.kwonlyargs)
              + ([args.vararg] if args.vararg else [])
              + ([args.kwarg] if args.kwarg else [])):
        start.add(a.arg)
    s = _Scan(watched, exits)
    s.block(fun.body, start)
    return s.risky


def scan_source(text):
    """Risky findings for a whole module: (function, name, lineno)."""
    out = []
    tree = ast.parse(text)
    exits = _module_exits(tree)
    for fun in ast.walk(tree):
        if isinstance(fun, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for name, line in scan_function(fun, exits):
                out.append((fun.name, name, line))
    return out
