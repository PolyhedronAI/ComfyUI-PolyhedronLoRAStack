"""Helper for the group guards (v1330, v1331): copy the group modules (or
mutated versions of them) into a temp tree that mirrors the pack's relative
imports -- web/js/<files> and scripts/app.js as a stub -- and run a driver
module that imports them for real. Not a test itself (no test_ prefix)."""
import json
import os
import shutil
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JS = os.path.join(ROOT, "web", "js")

APP_STUB = ("export const app = { registerExtension() {}, canvas: null, graph: null, "
            "extensionManager: null };\n")


def read(name):
    return open(os.path.join(JS, name), encoding="utf-8").read()


def run(files, driver, timeout=60):
    """files: {"ph_groups_core.js": src, ...}; driver: module source run from web/js."""
    tmp = tempfile.mkdtemp(prefix="pls_groups_")
    try:
        # web/js/x.js imports "../../scripts/app.js": from a/b/web/js that is a/b/scripts/app.js
        os.makedirs(os.path.join(tmp, "a", "b", "web", "js"))
        os.makedirs(os.path.join(tmp, "a", "b", "scripts"))
        with open(os.path.join(tmp, "a", "b", "scripts", "app.js"), "w") as f:
            f.write(APP_STUB)
        d = os.path.join(tmp, "a", "b", "web", "js")
        for name, src in files.items():
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                f.write(src)
        drv = os.path.join(d, "_driver.mjs")
        with open(drv, "w", encoding="utf-8") as f:
            f.write(driver)
        # the .js files must load as ES modules
        with open(os.path.join(tmp, "a", "b", "package.json"), "w") as f:
            f.write('{"type": "module"}')
        r = subprocess.run(["node", drv], capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return None, r.stderr[-600:]
        return json.loads(r.stdout), None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
