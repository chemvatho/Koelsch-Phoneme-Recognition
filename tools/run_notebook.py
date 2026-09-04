#!/usr/bin/env python3
"""Execute a notebook and report where it stops. Used to test the repo.

Two things are rewritten before execution, and neither changes what the notebook
does when a human runs it:

  * `!pip install ...` lines are commented out. On Colab they are the point; in
    a prepared local environment they would reinstall the world, and a resolver
    that decides to move torch mid-test invalidates every later result.
  * A cell tagged `skip-test` in its metadata is replaced by `pass`. That is for
    cells that would take GPU-hours (a full fine-tune) or spend money (a Gemini
    API call) -- the notebook is still checked up to and past them, and the
    report says which ones were skipped so the result is not oversold.

Exit status is 0 only if every executed cell ran clean.

    tools/run_notebook.py 02_corpus/02_corpus_statistics.ipynb [--timeout 900]
"""
import argparse
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

PIP = re.compile(r"^(\s*)(!\s*(pip|apt-get|apt)\b.*)$", re.M)


def prepare(nb):
    """Return (notebook, n_pip_lines_muted, [skipped cell indices])."""
    muted, skipped = 0, []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        if "skip-test" in cell.get("metadata", {}).get("tags", []):
            skipped.append(i)
            cell.source = "pass  # skipped by tools/run_notebook.py (skip-test)"
            continue
        new, n = PIP.subn(r"\1# [test] \2", cell.source)
        cell.source, muted = new, muted + n
    return nb, muted, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="per-cell timeout in seconds")
    ap.add_argument("--save", metavar="PATH",
                    help="write the executed notebook here (with outputs)")
    a = ap.parse_args()

    path = Path(a.notebook).resolve()
    nb, muted, skipped = prepare(nbformat.read(path, as_version=4))
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"── {path.name}: {n_code} code cells, {muted} pip line(s) muted, "
          f"{len(skipped)} skipped")

    # resources.metadata.path is the cwd the kernel starts in; the notebooks
    # locate the repo root from it via kolsch_paths.py, so it must be the
    # notebook's own directory, exactly as when a human opens it.
    client = NotebookClient(nb, timeout=a.timeout, kernel_name="python3",
                            allow_errors=False,
                            resources={"metadata": {"path": str(path.parent)}})
    t0 = time.time()
    try:
        client.execute()
    except CellExecutionError as e:
        dt = time.time() - t0
        bad = next((i for i, c in enumerate(nb.cells)
                    if c.cell_type == "code"
                    and any(o.get("output_type") == "error"
                            for o in c.get("outputs", []))), None)
        print(f"   FAIL after {dt:.0f}s at cell {bad}")
        print("   " + str(e).strip().splitlines()[-1][:300])
        if a.save:
            nbformat.write(nb, a.save)
        return 1
    dt = time.time() - t0
    print(f"   PASS in {dt:.0f}s" + (f"  (skipped {skipped})" if skipped else ""))
    if a.save:
        nbformat.write(nb, a.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
