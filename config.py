"""ค่าคงที่ของ repo — scripts/ และ notebook import จากที่นี่
⚠️ app/ ห้าม import ไฟล์นี้ (app ต้อง self-contained สำหรับ Hugging Face Space)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- reproducibility ----
RANDOM_STATE = 42

# ---- bike-regression ----
# ใช้ config ของตัวเองที่ bike-regression/config.py (path, encoding, วันพายุ, ชื่อคอลัมน์)
BIKE_DIR = ROOT / "bike-regression"
BIKE_MODEL = BIKE_DIR / "app" / "model.joblib"

# ---- trash-classification ----
TRASH_DIR = ROOT / "trash-classification"
TRASH_HF_ID = "garythung/trashnet"
TRASH_IMAGES_DIR = TRASH_DIR / "data" / "images"
TRASH_LABELS_CSV = TRASH_DIR / "data" / "labels.csv"
TRASH_EXAMPLES_DIR = TRASH_DIR / "app" / "examples"
TRASH_TEST_FRAC = 0.2
TRASH_MAX_SIDE = 256           # resize ด้านยาว (px)
TRASH_JPEG_QUALITY = 85
TRASH_MAX_TOTAL_MB = 200       # เกินนี้ให้หยุดแล้วถามก่อน
TRASH_MODEL = TRASH_DIR / "app" / "model.joblib"
