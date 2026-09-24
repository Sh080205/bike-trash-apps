"""Gradio app: ทำนายจำนวนจักรยานที่ถูกเช่า (คัน/ชั่วโมง)

⚠️ self-contained — ห้าม import อะไรจากนอกโฟลเดอร์ app/ (คัดลอกไปเป็น HF Space ได้ทันที)
รัน: python bike-regression/app/app.py
"""
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

# TODO: รายการ input ≤ 8 feature ที่เลือกใน notebook (ชื่อ, ชนิด, ช่วงค่าที่ยอมรับ)
FEATURES = []


def load_model():
    """TODO: โหลด Pipeline จาก MODEL_PATH ครั้งเดียวตอนเริ่มแอป
    ⚠️ ถ้าไม่มีไฟล์ → คืน None และให้ UI แสดงข้อความ "ยังไม่มีโมเดล" (ห้าม crash)
    """
    raise NotImplementedError


def predict(*inputs):
    """TODO: ตรวจช่วงค่า input → สร้าง DataFrame ชื่อคอลัมน์ตรงกับตอน train → predict
    ⚠️ input ผิดช่วง → ข้อความชัดเจน
    ⚠️ ค่าทำนายติดลบได้ในบางโมเดล — ตัดสินใจว่าจะจัดการอย่างไร
    คืนข้อความรูปแบบ "X คัน/ชั่วโมง"
    """
    raise NotImplementedError


def build_ui():
    """TODO: gr.Interface / gr.Blocks พร้อม examples= (2–3 แถวตัวอย่าง)"""
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: build_ui().launch()
    raise NotImplementedError("TODO: app.py")
