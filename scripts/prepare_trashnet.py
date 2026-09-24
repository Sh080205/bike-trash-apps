"""เตรียม TrashNet → trash-classification/data/images/<class>/*.jpg + labels.csv

รัน: python scripts/prepare_trashnet.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def main():
    # TODO 1: datasets.load_dataset(config.TRASH_HF_ID)
    #   ⚠️ พิมพ์ชื่อ split, จำนวนภาพ และชื่อคลาสจริง (ClassLabel.names) ก่อนใช้ — ห้ามเดา
    # TODO 2: resize ด้านยาวเหลือ config.TRASH_MAX_SIDE px (รักษาอัตราส่วน)
    #         บันทึก JPEG quality=config.TRASH_JPEG_QUALITY ลง TRASH_IMAGES_DIR/<class>/
    #   ⚠️ แปลงเป็น RGB ก่อนบันทึก (บางภาพอาจเป็น RGBA/L)
    #   ⚠️ ตั้งชื่อไฟล์ให้ไม่ซ้ำและ deterministic
    # TODO 3: train/test 80/20 แบบ stratified ตามคลาส (random_state=config.RANDOM_STATE)
    #         เขียน labels.csv คอลัมน์: filename, label, split
    # TODO 4: พิมพ์ตารางจำนวนภาพต่อคลาส × split และขนาดโฟลเดอร์รวม
    #   ⚠️ ถ้าเกิน config.TRASH_MAX_TOTAL_MB → หยุดและแจ้งก่อน commit
    # TODO 5: คัดลอกภาพ **test** คลาสละ 1 รูปไป config.TRASH_EXAMPLES_DIR
    raise NotImplementedError("TODO: prepare_trashnet.py")


if __name__ == "__main__":
    main()
