"""Gradio app: จำแนกประเภทขยะจากรูปภาพ

⚠️ self-contained — import ได้เฉพาะไฟล์ในโฟลเดอร์ app/ (เช่น features.py)
รัน: python trash-classification/app/app.py
"""
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.joblib"
EXAMPLES_DIR = APP_DIR / "examples"

# TODO: import features ก่อน joblib.load — joblib ต้องหา class ใน module "features" เจอ
#       (ตอนรันจากนอกโฟลเดอร์ อาจต้องเพิ่ม APP_DIR เข้า sys.path)


def load_model():
    """TODO: โหลด Pipeline (feature extractor + model) ครั้งเดียวตอนเริ่มแอป
    ⚠️ ไม่มีไฟล์ → คืน None และแสดง "ยังไม่มีโมเดล" (ห้าม crash)
    """
    raise NotImplementedError


def predict(image):
    """TODO: คืน dict {คลาส: ความน่าจะเป็น} ทุกคลาส (สำหรับ gr.Label)
    ⚠️ ไม่มีรูป / รูปเสีย / โหมดสีแปลก → ข้อความชัดเจน
    ⚠️ ใช้ model.classes_ เป็นชื่อคลาส อย่า hard-code ลำดับ
    """
    raise NotImplementedError


def build_ui():
    """TODO: gr.Interface พร้อม examples= จากไฟล์ใน EXAMPLES_DIR"""
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: build_ui().launch()
    raise NotImplementedError("TODO: app.py")
