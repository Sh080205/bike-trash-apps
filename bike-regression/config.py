"""ค่าคงที่ของโปรเจกต์ — ทุกไฟล์ import จากที่นี่ ห้าม hard-code ซ้ำ"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- reproducibility ----
RANDOM_STATE = 42          # ใช้ทุกจุดที่มีการสุ่ม (split, CV, model)
TEST_FRAC = 0.2            # สัดส่วน "วัน" ที่เป็น test (ไม่ใช่สัดส่วนแถว)

# ---- ชนิดงาน ----
TASK = "regression"        # ทำนาย rented_bike_count (คัน/ชั่วโมง) — เหตุผลใน 02_modeling ขั้น 1

# ---- ขอบเขตพายุ: กำหนดจากประกาศ KMA ก่อนเหตุการณ์ ไม่ใช้ residual หรือคะแนน test ----
# รวมวันที่แจ้งฝน/ลมจากพายุบนบกในเขตเมืองหลวง; ไม่อ้างว่าเป็นวันขึ้นฝั่งหรือระดับเตือนภัยเดียวกัน
STORM_EVENTS = (
    {
        "name": "Prapiroon",
        "dates": ("2018-07-02",),
        "published": "2018-06-29",
        "source": "https://testweather.kma.go.kr/metropolitan/html/news/notice_view.jsp?articleno=9627&boardId=press2&pageNo=31",
        "basis": "2 ก.ค. แจ้งฝน/ลมจากอิทธิพลพายุในเขตเมืองหลวง; 30 มิ.ย.–1 ก.ค. ระบุเป็นมรสุม จึงไม่ตัดด้วยกฎพายุ",
    },
    {
        "name": "Soulik",
        "dates": ("2018-08-23", "2018-08-24"),
        "published": "2018-08-22",
        "source": "https://testweather.kma.go.kr/metropolitan/html/news/notice_view.jsp?articleno=9857&boardId=press2&pageNo=31",
        "basis": "23–24 ส.ค. แจ้งล่วงหน้าถึงผลกระทบพายุและฝน/ลมในโซล–อินชอน–คยองกี",
    },
    {
        "name": "Kong-rey",
        "dates": ("2018-10-05", "2018-10-06", "2018-10-07"),
        "published": "2018-10-04",
        "source": "https://www.kma.go.kr/metropolitan/html/news/notice_view.jsp?articleno=9956&boardId=press2&pageNo=13",
        "basis": "5–6 ต.ค. แจ้งฝนจากพายุ; 5–7 ต.ค. แจ้งลมแรงรวมบางพื้นที่บนบกของเขตเมืองหลวง จึงรวม 7 ต.ค. ด้วย",
    },
)
STORM_DATES = tuple(sorted({day for event in STORM_EVENTS for day in event["dates"]}))
MAGNUS_REVIEW_GAP_PP = 15.0  # จุดเปอร์เซ็นต์: เกณฑ์สอบทานจาก train ใน EDA 4.11 ไม่ใช่เกณฑ์ตัดแถว

# ---- paths ----
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "seoul+bike+sharing+demand" / "SeoulBikeData.csv"   # แตกมาจาก zip ของ UCI
# dataset ที่ใช้ทุก notebook: ไฟล์ UCI ที่ความชื้น 0% ว่าง 9 จาก 17 ช่อง (missing value) · สร้างด้วย scripts/make_dataset.py
DATA_CSV = DATA_DIR / "processed" / "SeoulBikeData_missing.csv"
SPLIT_DIR = DATA_DIR / "splits"
TRAIN_DATES = SPLIT_DIR / "train_dates.csv"
TEST_DATES = SPLIT_DIR / "test_dates.csv"
FIGURES_DIR = ROOT / "figures"
REPORTS_DIR = ROOT / "reports"

# ---- dataset ----
UCI_ID = 560
# ตรวจแล้วใน notebooks/01_eda.ipynb ขั้น 2: ไฟล์ไม่ใช่ UTF-8 (° = byte 0xB0) และวันที่เป็น dd/mm/yyyy
RAW_ENCODING = "cp1252"
DATE_FORMAT = "%d/%m/%Y"

# ชื่อคอลัมน์ดิบ → snake_case (ไม่มีหน่วย/อักขระพิเศษ) — ใช้ชื่อฝั่งขวาทุกที่หลังโหลด
COLUMN_MAP = {
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
DATE_COL = "date"
TARGET_COL = "rented_bike_count"
