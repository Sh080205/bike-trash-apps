"""แบ่ง train/test ตาม "วัน" → bike-regression/data/splits/{train,test}_dates.csv

รัน: python scripts/make_bike_split.py
⚠️ logic เหมือน seoul-bike-demand/notebooks/01_eda.ipynb ขั้น 3 ทุกขั้น
   (เรียง date+hour → GroupShuffleSplit(test_size=0.2, random_state=42), groups=date)
   ตรวจแล้ว: test_dates.csv / train_dates.csv ได้ไฟล์ตรงกับ repo นั้นทุก byte
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def load_raw():
    """โหลด CSV ดิบด้วย encoding/รูปแบบวันที่/ชื่อคอลัมน์ที่ตรวจแล้วใน config"""
    df = pd.read_csv(config.BIKE_RAW_CSV, encoding=config.BIKE_RAW_ENCODING)
    if list(df.columns) != list(config.BIKE_COLUMN_MAP):
        raise ValueError(f"ชื่อคอลัมน์ไม่ตรงกับ config.BIKE_COLUMN_MAP: {list(df.columns)}")
    df = df.rename(columns=config.BIKE_COLUMN_MAP)
    df[config.BIKE_DATE_COL] = pd.to_datetime(df[config.BIKE_DATE_COL], format=config.BIKE_DATE_FORMAT)
    return df


def main():
    df = load_raw()
    date_col = config.BIKE_DATE_COL

    # เรียงก่อนสุ่ม → ลำดับแถวคงที่ ผลสุ่มจึงเหมือนเดิมทุกครั้ง (และเหมือน repo seoul-bike-demand)
    df = df.sort_values([date_col, "hour"]).reset_index(drop=True)
    # group = วัน → ทุกชั่วโมงของวันเดียวกันอยู่ฝั่งเดียว (ห้ามสุ่มทีละแถว: ชั่วโมงข้างเคียงคล้ายกัน = leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=config.BIKE_TEST_FRAC, random_state=config.RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, groups=df[date_col]))

    train_dates = np.sort(df.loc[train_idx, date_col].unique())
    test_dates = np.sort(df.loc[test_idx, date_col].unique())
    assert len(set(train_dates) & set(test_dates)) == 0, "มีวันที่อยู่ทั้ง train และ test"
    assert len(train_dates) + len(test_dates) == df[date_col].nunique()

    config.BIKE_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for dates, path in [(train_dates, config.BIKE_TRAIN_DATES), (test_dates, config.BIKE_TEST_DATES)]:
        pd.Series(pd.to_datetime(dates).strftime("%Y-%m-%d"), name="date").to_csv(path, index=False)

    n_days = df[date_col].nunique()
    print(f"train: {len(train_dates)} วัน ({len(train_dates) / n_days:.1%}) | {len(train_idx):,} แถว")
    print(f"test : {len(test_dates)} วัน ({len(test_dates) / n_days:.1%}) | {len(test_idx):,} แถว")
    print("บันทึก:", config.BIKE_TRAIN_DATES.relative_to(config.ROOT), ",",
          config.BIKE_TEST_DATES.relative_to(config.ROOT))


if __name__ == "__main__":
    main()
