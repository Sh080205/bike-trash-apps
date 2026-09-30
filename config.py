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
BIKE_DATE_COL = "date"
BIKE_TARGET_COL = "rented_bike_count"
BIKE_COLUMN_MAP = {
    "Date": "date",
    "Rented Bike Count": "rented_bike_count",
    "Hour": "hour",
    "Temperature(°C)": "temp_c",
    "Humidity(%)": "humidity_pct",
    "Wind speed (m/s)": "wind_speed_ms",
    "Visibility (10m)": "visibility_10m",
    "Dew point temperature(°C)": "dew_point_c",
    "Solar Radiation (MJ/m2)": "solar_radiation_mj",
    "Rainfall(mm)": "rainfall_mm",
    "Snowfall (cm)": "snowfall_cm",
    "Seasons": "seasons",
    "Holiday": "holiday",
    "Functioning Day": "functioning_day",
}
BIKE_RAW_ENCODING = "cp1252"   # ตรวจแล้วใน notebook 3.1: ° = byte 0xB0 → utf-8 อ่านไม่ได้, cp1252 อ่านได้
BIKE_DATE_FORMAT = "%d/%m/%Y"  # ตรวจแล้วใน notebook 3.1: ส่วนแรกมีค่าถึง 31 → เป็นวัน/เดือน/ปี
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
