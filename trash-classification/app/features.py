"""Custom transformer สกัด feature จากภาพ

⚠️ class นี้ต้องอยู่ในไฟล์นี้เท่านั้น — notebook ต้อง `from features import ...`
   ถ้านิยาม class ใน notebook, joblib จะบันทึกชื่อ module เป็น __main__ แล้วแอปโหลดโมเดลไม่ได้
"""
from sklearn.base import BaseEstimator, TransformerMixin


class ImageFeatureExtractor(BaseEstimator, TransformerMixin):
    """TODO: รับรายการภาพ (path หรือ PIL.Image) → คืน array (n_samples, n_features)

    ไอเดีย feature (เลือก/ทดลองเอง): color histogram, สถิติสีต่อช่อง,
    ภาพย่อ flatten, texture/edge
    ⚠️ พารามิเตอร์ทั้งหมดรับผ่าน __init__ และเก็บชื่อเดียวกัน (กฎ sklearn — ไม่งั้น clone/GridSearch พัง)
    ⚠️ ขนาดภาพต้องปรับให้เท่ากันก่อนสกัด
    ⚠️ transform ต้องรับได้ทั้งตอน train (path) และตอนแอป (ภาพจาก Gradio)
    """

    def __init__(self):
        # TODO: พารามิเตอร์ เช่น ขนาดภาพ, จำนวน bin
        pass

    def fit(self, X, y=None):
        # TODO: ส่วนใหญ่ไม่ต้องเรียนรู้อะไร — return self
        raise NotImplementedError

    def transform(self, X):
        # TODO
        raise NotImplementedError
