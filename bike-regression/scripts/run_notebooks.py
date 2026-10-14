"""รัน Notebook ตามลำดับด้วย kernel ใหม่ทุกไฟล์ และเก็บ outputs/เวลารัน."""
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


def main():
    # kernelspec อยู่ใน environment นี้: ZIP ที่แตกใหม่ไม่อาศัย kernel ของ checkout เดิม
    subprocess.run([sys.executable, "-m", "ipykernel", "install", "--sys-prefix",
                    "--name", "seoul-bike", "--display-name", "Python (Seoul Bike)"], check=True)
    os.environ["JUPYTER_PATH"] = str(Path(sys.prefix) / "share" / "jupyter")
    out = ROOT / "verification"
    out.mkdir(exist_ok=True)
    summary = {"started_utc": datetime.now(timezone.utc).isoformat(),
               "python": platform.python_version(), "system": platform.platform(), "notebooks": []}
    for name in ["00_setup", "01_eda", "02_modeling"]:
        path = ROOT / "notebooks" / f"{name}.ipynb"
        nb = nbformat.read(path, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        print(f"START {name}", flush=True)
        start = time.perf_counter()
        def progress(cell, cell_index, **kwargs):
            if cell.cell_type == "code":
                print(f"  cell {cell_index}: {cell.source.splitlines()[0][:100]}", flush=True)
        client = NotebookClient(nb, timeout=3600, kernel_name="seoul-bike",
                                resources={"metadata": {"path": str(ROOT / "notebooks")}},
                                on_cell_start=progress)
        client.execute()
        nbformat.write(nb, path)
        seconds = time.perf_counter() - start
        logs = []
        for i, cell in enumerate(nb.cells):
            for output in cell.get("outputs", []):
                value = output.get("text") or output.get("data", {}).get("text/plain", "")
                if value:
                    logs.append(f"CELL {i}\n{value}")
        transcript = "\n\n".join(logs)
        (out / f"{name}-output.txt").write_text("\n".join(line.rstrip() for line in transcript.splitlines())+"\n", encoding="utf-8")
        summary["notebooks"].append({"name": name, "seconds": seconds,
                                      "code_cells": sum(c.cell_type == "code" for c in nb.cells), "status": "passed"})
        (out / "execution.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"PASS {name}: {seconds:.1f} s", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=(out / "environment-lock.txt").open("w"), check=True)


if __name__ == "__main__":
    main()
