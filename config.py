"""ค่าคงที่ของ repo — scripts/ และ notebook import จากที่นี่
⚠️ app/ ห้าม import ไฟล์นี้ (app ต้อง self-contained สำหรับ Hugging Face Space)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- reproducibility ----
RANDOM_STATE = 42

# ---- bike-regression ----
BIKE_DIR = ROOT / "bike-regression"
BIKE_RAW_CSV = BIKE_DIR / "data" / "SeoulBikeData.csv"
BIKE_SPLIT_DIR = BIKE_DIR / "data" / "splits"
BIKE_TRAIN_DATES = BIKE_SPLIT_DIR / "train_dates.csv"
BIKE_TEST_DATES = BIKE_SPLIT_DIR / "test_dates.csv"
BIKE_UCI_ID = 560
BIKE_TEST_FRAC = 0.2           # ⚠️ ต้องเท่ากับ seoul-bike-demand/config.py
BIKE_DATE_COL = "Date"
BIKE_TARGET_COL = "Rented Bike Count"
BIKE_RAW_ENCODING = None       # TODO: ใส่ encoding ที่ตรวจแล้ว (ไฟล์ต้นฉบับไม่ใช่ UTF-8)
BIKE_DATE_FORMAT = None        # TODO: ใส่รูปแบบวันที่ที่ตรวจแล้ว
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
