"""สร้าง dataset ที่ใช้ในโปรเจกต์ (config.DATA_CSV) จากไฟล์ UCI (config.RAW_CSV)

ความชื้น 0% ในไฟล์ UCI มี 17 ช่อง → ลบให้เป็นช่องว่าง 9 ช่อง (สุ่มด้วย RANDOM_STATE) เพื่อฝึกจัดการ missing value
ส่วนอีก 8 ช่องคงเป็น 0 ไว้ให้ตรวจเป็น outlier (01_eda 4.10)
แก้เป็นข้อความทีละช่อง (ไม่ใช้ to_csv) → ส่วนอื่นของไฟล์เหมือนไฟล์ UCI ทุก byte

รัน: python scripts/make_dataset.py
"""
import sys
from pathlib import Path

# stdout ผ่าน pipe บน Windows ต้องรองรับข้อความไทยโดยไม่พึ่ง code page
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

N_MISSING = 9   # จำนวนช่องความชื้น 0% ที่ทำให้ว่าง (ประมาณครึ่งของ 17)


def main():
    raw = config.RAW_CSV.read_bytes()
    lines = raw.decode(config.RAW_ENCODING).splitlines(keepends=True)
    h = lines[0].rstrip("\r\n").split(",").index("Humidity(%)")   # ตำแหน่งคอลัมน์ความชื้น
    zero_rows = [i for i, line in enumerate(lines[1:], start=1) if line.rstrip("\r\n").split(",")[h] == "0"]
    blank = set(np.random.default_rng(config.RANDOM_STATE).choice(zero_rows, size=N_MISSING, replace=False).tolist())

    out = []
    for i, line in enumerate(lines):
        if i in blank:
            body = line.rstrip("\r\n")
            cells = body.split(",")
            cells[h] = ""                                   # 0 → ช่องว่าง = missing value
            line = ",".join(cells) + line[len(body):]       # คงตัวขึ้นบรรทัดเดิม
        out.append(line)
    new = "".join(out).encode(config.RAW_ENCODING)
    assert len(zero_rows) == 17 and len(raw) - len(new) == N_MISSING   # ลบไปแค่เลข 0 จำนวน N_MISSING ตัว

    config.DATA_CSV.parent.mkdir(exist_ok=True)
    config.DATA_CSV.write_bytes(new)
    print(f"{config.DATA_CSV.relative_to(config.ROOT).as_posix()}: ความชื้นว่าง {N_MISSING} ช่อง · ความชื้น 0% เหลือ {len(zero_rows) - N_MISSING} ช่อง")


if __name__ == "__main__":
    main()
