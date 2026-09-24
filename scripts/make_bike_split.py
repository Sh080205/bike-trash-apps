"""แบ่ง train/test ตาม "วัน" → bike-regression/data/splits/{train,test}_dates.csv

รัน: python scripts/make_bike_split.py
⚠️ logic ต้องเหมือน seoul-bike-demand/scripts/make_split.py ทุกบรรทัด
   (คัดลอกไฟล์ ไม่ import ข้าม repo) — ตรวจด้วย diff test_dates.csv ของสอง repo
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def main():
    # TODO 1: โหลด config.BIKE_RAW_CSV (encoding ที่ตรวจแล้ว) และ parse วันที่
    # TODO 2: GroupShuffleSplit(n_splits=1, test_size=config.BIKE_TEST_FRAC,
    #         random_state=config.RANDOM_STATE) โดย groups = วันที่
    #   ⚠️ ห้ามสุ่มทีละแถว
    #   ⚠️ เรียงข้อมูลตามวันที่ก่อน split ให้เหมือน repo 1
    # TODO 3: บันทึกรายการวัน (ISO yyyy-mm-dd, เรียงแล้ว) ลง BIKE_TRAIN_DATES / BIKE_TEST_DATES
    # TODO 4: พิมพ์จำนวนวัน/แถว train, test + assert ไม่มีวันซ้ำสองฝั่ง
    raise NotImplementedError("TODO: make_bike_split.py")


if __name__ == "__main__":
    main()
