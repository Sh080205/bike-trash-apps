"""Deploy bike-regression/app/ ขึ้น Hugging Face Space (Gradio)

ก่อนรัน:
  1. สร้าง token แบบ Write ที่ https://huggingface.co/settings/tokens
  2. ตั้ง env HF_TOKEN=<token>  หรือรัน `hf auth login` ครั้งเดียว
รัน:
  python scripts/deploy_space.py --repo-id <hf-username>/seoul-bike-demand

ใช้ upload_folder แทน git push เพราะ model.joblib เป็นไฟล์ binary
(git push ธรรมดาไป HF จะถูกปฏิเสธถ้าไม่ใช้ LFS/Xet — upload_folder จัดการให้เอง)
"""
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

APP_DIR = Path(__file__).resolve().parents[1] / "bike-regression" / "app"
REQUIRED = ["app.py", "model.joblib", "requirements.txt", "README.md"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="เช่น your-username/seoul-bike-demand")
    parser.add_argument("--private", action="store_true", help="สร้าง Space แบบ private")
    args = parser.parse_args()

    missing = [f for f in REQUIRED if not (APP_DIR / f).exists()]
    if missing:
        sys.exit(f"ไฟล์ใน {APP_DIR} ไม่ครบ: {missing} (model.joblib สร้างจาก notebook หัวข้อ 9)")

    api = HfApi(token=os.environ.get("HF_TOKEN"))   # None → ใช้ token จาก `hf auth login`
    try:
        user = api.whoami()["name"]
    except Exception as e:
        sys.exit(f"ยืนยันตัวตนกับ Hugging Face ไม่ได้ ({e}) — ตั้ง HF_TOKEN หรือรัน `hf auth login` ก่อน")
    print("login เป็น:", user)

    url = api.create_repo(args.repo_id, repo_type="space", space_sdk="gradio",
                          private=args.private, exist_ok=True)
    print("Space:", url)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="space",
        folder_path=APP_DIR,
        ignore_patterns=["__pycache__/*", "*.pyc", ".gradio/*", "flagged/*"],
        commit_message="Deploy bike-regression app",
    )
    print(f"อัปโหลดเสร็จ → https://huggingface.co/spaces/{args.repo_id}")
    print("รอ build 1–3 นาที แล้วดู log ที่แท็บ Logs ถ้าแอปไม่ขึ้น")


if __name__ == "__main__":
    main()
