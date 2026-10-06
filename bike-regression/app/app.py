"""Gradio app: ทำนายจำนวนจักรยานที่ถูกเช่า (คัน/ชั่วโมง) — Seoul Bike Sharing

⚠️ self-contained — ห้าม import อะไรจากนอกโฟลเดอร์ app/ (คัดลอกไปเป็น HF Space ได้ทันที)
รัน: python bike-regression/app/app.py
"""
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

SEASONS = ["Spring", "Summer", "Autumn", "Winter"]
SEASON_TH = {"Spring": "ใบไม้ผลิ (Spring)", "Summer": "ร้อน (Summer)",
             "Autumn": "ใบไม้ร่วง (Autumn)", "Winter": "หนาว (Winter)"}
DAY_TYPES = ["วันธรรมดา (จ.–ศ.)", "เสาร์–อาทิตย์"]

# input 8 ตัว — ชื่อ/ลำดับคอลัมน์ต้องตรงกับตอน train (notebook หัวข้อ 5.7 / 9)
# limits = ช่วงที่แอปยอมรับ · train_range = ช่วงที่โมเดลเคยเห็นใน train (นอกนี้ยังทำนายได้แต่เตือน)
FEATURES = {
    "hour":          {"label": "ชั่วโมง (0–23)",           "limits": (0, 23),    "train_range": (0, 23)},
    "temp_c":        {"label": "อุณหภูมิ (°C)",            "limits": (-20, 40),  "train_range": (-16.2, 39.4)},
    "humidity_pct":  {"label": "ความชื้นสัมพัทธ์ (%)",      "limits": (1, 100),   "train_range": (10, 98)},
    "rainfall_mm":   {"label": "ปริมาณฝน (มม./ชม.)",       "limits": (0, 40),    "train_range": (0, 35)},
    "wind_speed_ms": {"label": "ความเร็วลม (ม./วินาที)",    "limits": (0, 10),    "train_range": (0, 7.4)},
}
COLUMNS = ["hour", "temp_c", "humidity_pct", "rainfall_mm", "wind_speed_ms",
           "seasons", "is_weekend", "is_holiday"]


def load_model():
    """โหลด Pipeline ครั้งเดียวตอนเริ่มแอป · ไม่มีไฟล์/โหลดไม่ได้ → None (UI แสดงข้อความแทนการ crash)"""
    if not MODEL_PATH.exists():
        print(f"[warn] ไม่พบไฟล์โมเดล: {MODEL_PATH}")
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:  # เช่น เวอร์ชัน scikit-learn ไม่ตรงกับตอน train
        print(f"[warn] โหลดโมเดลไม่สำเร็จ: {e!r}")
        return None


MODEL = load_model()


def validate(values):
    """ตรวจ input ตัวเลข → (errors, warnings) เป็นรายการข้อความภาษาไทย"""
    errors, warnings = [], []
    for name, value in values.items():
        spec = FEATURES[name]
        lo, hi = spec["limits"]
        if value is None:
            errors.append(f"กรุณากรอก **{spec['label']}**")
        elif not lo <= value <= hi:
            errors.append(f"**{spec['label']}** ต้องอยู่ระหว่าง {lo} ถึง {hi} (ได้ {value})")
        else:
            t_lo, t_hi = spec["train_range"]
            if not t_lo <= value <= t_hi:
                warnings.append(f"{spec['label']} = {value} อยู่นอกช่วงข้อมูลที่ใช้ฝึก ({t_lo} ถึง {t_hi}) ผลอาจคลาดเคลื่อนมาก")
    if values.get("hour") is not None and float(values["hour"]) != int(values["hour"]):
        errors.append("**ชั่วโมง** ต้องเป็นจำนวนเต็ม")
    return errors, warnings


def predict(hour, temp_c, humidity_pct, rainfall_mm, wind_speed_ms, season, day_type, is_holiday):
    """คืนข้อความ Markdown รูปแบบ "X คัน/ชั่วโมง" หรือข้อความแจ้งข้อผิดพลาด"""
    if MODEL is None:
        return "⚠️ **ยังไม่มีโมเดล** — ต้องรัน notebook หัวข้อ 9 เพื่อสร้าง `app/model.joblib` ก่อน"

    numeric = {"hour": hour, "temp_c": temp_c, "humidity_pct": humidity_pct,
               "rainfall_mm": rainfall_mm, "wind_speed_ms": wind_speed_ms}
    errors, warnings = validate(numeric)
    if season not in SEASONS:
        errors.append("กรุณาเลือก **ฤดูกาล**")
    if day_type not in DAY_TYPES:
        errors.append("กรุณาเลือก **ประเภทวัน**")
    if errors:
        return "⚠️ **ข้อมูลไม่ถูกต้อง**\n\n" + "\n".join(f"- {e}" for e in errors)

    row = pd.DataFrame([{
        **{k: float(v) for k, v in numeric.items()},
        "hour": int(hour),
        "seasons": season,
        "is_weekend": int(day_type == DAY_TYPES[1]),
        "is_holiday": int(bool(is_holiday)),
    }], columns=COLUMNS)
    # loss="poisson" ทำนายค่าติดลบไม่ได้อยู่แล้ว · max(…, 0) กันไว้อีกชั้นเผื่อเปลี่ยนโมเดล
    pred = max(float(MODEL.predict(row)[0]), 0.0)

    text = f"## ≈ {pred:,.0f} คัน/ชั่วโมง"
    text += "\n\nโดยเฉลี่ยโมเดลทายคลาดราว 105 คัน/ชม. (MAE บนชุด test)"
    if rainfall_mm and rainfall_mm > 0:
        text += "\n\n☔ ช่วงฝนตกโมเดลคลาดเคลื่อนสูง (ราว 70% ของยอดจริง) เพราะมีข้อมูลฝนตกน้อย"
    if warnings:
        text += "\n\n" + "\n".join(f"- ⚠️ {w}" for w in warnings)
    return text


DESCRIPTION = """ทำนาย **จำนวนจักรยานสาธารณะที่ถูกเช่าทั้งเมืองโซลใน 1 ชั่วโมง** จากเวลาและสภาพอากาศ
(โมเดล HistGradientBoosting · ฝึกจาก Seoul Bike Sharing Demand, UCI, ปี 2017–2018)

ทำนายเฉพาะชั่วโมงที่ระบบ **เปิดให้บริการ** · กรอกค่าจากพยากรณ์อากาศ แล้วกด Submit หรือเลือกตัวอย่างด้านล่าง"""

ARTICLE = """**ข้อจำกัด:** ฝนตกและวันพิเศษ (พายุ, เทศกาลชูซอก) ทายคลาดมาก · ข้อมูลมีแค่ 1 ปีและเป็นยอดรวมทั้งเมือง ไม่ใช่รายสถานี

Dataset: Seoul Bike Sharing Demand [Dataset]. (2020). UCI Machine Learning Repository.
https://doi.org/10.24432/C5F62R — CC BY 4.0"""


def build_ui():
    f = FEATURES
    inputs = [
        gr.Slider(0, 23, value=8, step=1, label=f["hour"]["label"]),
        gr.Number(value=20.0, label=f["temp_c"]["label"], minimum=-20, maximum=40),
        gr.Slider(1, 100, value=55, step=1, label=f["humidity_pct"]["label"]),
        gr.Number(value=0.0, label=f["rainfall_mm"]["label"], minimum=0, maximum=40),
        gr.Number(value=1.5, label=f["wind_speed_ms"]["label"], minimum=0, maximum=10),
        gr.Dropdown(choices=[(SEASON_TH[s], s) for s in SEASONS], value="Summer", label="ฤดูกาล"),
        gr.Radio(choices=DAY_TYPES, value=DAY_TYPES[0], label="ประเภทวัน"),
        gr.Checkbox(value=False, label="เป็นวันหยุดนักขัตฤกษ์"),
    ]
    examples = [
        [8, 22.0, 55, 0.0, 1.5, "Summer", DAY_TYPES[0], False],    # เช้าวันทำงาน ฤดูร้อน → พีคเช้า
        [18, 15.0, 50, 0.0, 2.0, "Autumn", DAY_TYPES[1], False],   # เย็นวันเสาร์-อาทิตย์ ฤดูใบไม้ร่วง
        [18, 18.0, 90, 5.0, 2.5, "Spring", DAY_TYPES[0], False],   # เย็นวันทำงาน แต่ฝนตก
        [14, -3.0, 40, 0.0, 1.0, "Winter", DAY_TYPES[0], True],    # บ่ายฤดูหนาว วันหยุด
    ]
    return gr.Interface(
        fn=predict,
        inputs=inputs,
        outputs=gr.Markdown(label="ผลทำนาย"),
        examples=examples,
        title="🚲 Seoul Bike Demand — ทำนายยอดเช่าจักรยานรายชั่วโมง",
        description=DESCRIPTION,
        article=ARTICLE,
        flagging_mode="never",
    )


if __name__ == "__main__":
    build_ui().launch()
