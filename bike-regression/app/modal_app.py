"""Deploy the Gradio app in this folder to Modal (https://modal.com).

One-time login:  modal token new
Deploy:          modal deploy bike-regression/app/modal_app.py
Try before deploy (temporary URL, Ctrl+C to stop):  modal serve bike-regression/app/modal_app.py

The image installs requirements.txt and copies this folder (app.py, model.joblib,
model_metrics.json) to /root/app, so the container serves exactly the files in git.
"""
from pathlib import Path
import sys

import modal

APP_DIR = Path(__file__).resolve().parent
REMOTE_DIR = "/root/app"

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install_from_requirements(str(APP_DIR / "requirements.txt"))
    .add_local_dir(APP_DIR, remote_path=REMOTE_DIR,
                   ignore=["__pycache__", "**/__pycache__", ".gradio", "flagged", ".venv"])
)

app = modal.App("seoul-bike-demand", image=image)


def build_asgi(app_dir=REMOTE_DIR):
    """Mount the Gradio Blocks from app.py on a FastAPI app (also used for a local check)."""
    sys.path.insert(0, str(app_dir))
    import gradio as gr
    from fastapi import FastAPI
    from app import CSS, THEME, demo

    return gr.mount_gradio_app(FastAPI(), demo, path="/", theme=THEME, css=CSS)


# Gradio keeps per-session state in memory, so a single container must serve every
# request (max_containers=1). The container stops after 20 idle minutes; the next
# visit cold-starts it again in a few seconds.
@app.function(max_containers=1, scaledown_window=20 * 60, timeout=10 * 60)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label="seoul-bike-demand")
def web():
    return build_asgi()
