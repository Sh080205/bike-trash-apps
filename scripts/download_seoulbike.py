"""ดาวน์โหลด/ตรวจ Seoul Bike Sharing Demand → bike-regression/data/SeoulBikeData.csv

รัน: python scripts/download_seoulbike.py
- ถ้ามีไฟล์อยู่แล้ว (commit ไว้ใน repo) → ตรวจความถูกต้องอย่างเดียว ไม่ดาวน์โหลดซ้ำ
- ถ้าไม่มี → ดาวน์โหลด zip จาก UCI แล้วแตกเฉพาะ SeoulBikeData.csv (ไบต์ดิบ ไม่แปลง encoding)
  ใช้ไฟล์เดียวกับ seoul-bike-demand/data/raw/ → split ได้วันชุดเดียวกัน
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip"
CSV_NAME = "SeoulBikeData.csv"


def download():
    print("ดาวน์โหลด", UCI_ZIP_URL)
    try:
        with urllib.request.urlopen(UCI_ZIP_URL, timeout=60) as resp:
            payload = resp.read()
    except OSError as e:
        raise SystemExit(f"ดาวน์โหลดไม่สำเร็จ ({e}) — ดาวน์โหลด zip จากหน้า UCI เองแล้ววาง "
                         f"{CSV_NAME} ไว้ที่ {config.BIKE_RAW_CSV}")
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        member = next((n for n in zf.namelist() if n.endswith(CSV_NAME)), None)
        if member is None:
            raise SystemExit(f"ไม่พบ {CSV_NAME} ใน zip: {zf.namelist()}")
        config.BIKE_RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        config.BIKE_RAW_CSV.write_bytes(zf.read(member))   # เก็บไบต์ดิบ → encoding เดิม (cp1252)


def verify():
    """ตรวจ encoding, ชื่อคอลัมน์, รูปแบบวันที่ และความครบ 365 วัน × 24 ชม."""
    df = pd.read_csv(config.BIKE_RAW_CSV, encoding=config.BIKE_RAW_ENCODING)
    assert list(df.columns) == list(config.BIKE_COLUMN_MAP), f"คอลัมน์ไม่ตรง config: {list(df.columns)}"
    df = df.rename(columns=config.BIKE_COLUMN_MAP)
    dates = pd.to_datetime(df[config.BIKE_DATE_COL], format=config.BIKE_DATE_FORMAT)  # ผิดรูปแบบ → error ทันที
    rows_per_day = dates.groupby(dates).size()
    print(f"shape: {df.shape} | วันที่: {dates.min().date()} ถึง {dates.max().date()} "
          f"| {dates.nunique()} วัน | แถว/วัน: {rows_per_day.min()}–{rows_per_day.max()}")
    assert df.shape == (8760, 14) and dates.nunique() == 365 and (rows_per_day == 24).all()
    print("ตรวจผ่าน:", config.BIKE_RAW_CSV.relative_to(config.ROOT))


def main():
    if not config.BIKE_RAW_CSV.exists():
        download()
    verify()


if __name__ == "__main__":
    main()
