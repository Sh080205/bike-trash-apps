"""ดาวน์โหลด Seoul Bike Sharing Demand → bike-regression/data/SeoulBikeData.csv

รัน: python scripts/download_seoulbike.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def main():
    # TODO 1: ลอง ucimlrepo ก่อน — fetch_ucirepo(id=config.BIKE_UCI_ID)
    #   ⚠️ ตรวจว่าได้คอลัมน์ Date + target ครบ
    # TODO 2: ถ้าไม่ได้ → ดาวน์โหลด zip จากหน้า UCI แล้วแตกไฟล์
    #   ⚠️ ห้ามสมมติ encoding — ลองหลายแบบ แล้วเลือกแบบที่ "°C" อ่านถูก
    # TODO 3: พิมพ์ shape, ชื่อคอลัมน์, ช่วงวันที่, จำนวนวัน, แถวต่อวัน
    #   ⚠️ ตรวจรูปแบบวันที่ dd/mm vs mm/dd ด้วยวันที่ > 12
    # TODO 4: บันทึกลง config.BIKE_RAW_CSV แล้วอัปเดต BIKE_RAW_ENCODING / BIKE_DATE_FORMAT
    #   ⚠️ ผลลัพธ์ต้องเหมือนไฟล์ใน seoul-bike-demand (split จะได้ตรงกัน)
    raise NotImplementedError("TODO: download_seoulbike.py")


if __name__ == "__main__":
    main()
