# Seoul Bike Demand App

> วิชา: **Introduction to Data Science** · กำหนดส่ง **15 ต.ค. 2569 07:00**

**bike-regression** — ทำนายจำนวนจักรยานที่ถูกเช่า (คัน/ชั่วโมง) จากสภาพอากาศและเวลา

## 1. แนวคิด / ปัญหา / ประโยชน์

TODO: ปัญหาคืออะไร ใครได้ประโยชน์ นำไปใช้อย่างไร

## 2. แหล่งข้อมูล จำนวน และการแบ่ง

| | |
|---|---|
| Dataset | Seoul Bike Sharing Demand (UCI id 560) |
| จำนวนทั้งหมด | TODO แถว |
| train / test | TODO (แบ่งตาม **วัน**) |

TODO: อธิบายเหตุผลการแบ่ง (ทำไมต้องแบ่งตามวัน)

## 3. โมเดล ผลหลัก และข้อจำกัด

TODO: โมเดลที่เลือก + เหตุผล · MAE (คัน/ชม.) / RMSE / R² บน test · ข้อจำกัด

## 4. ติดตั้งและรันบนเครื่อง

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python scripts/download_seoulbike.py
python scripts/make_bike_split.py

python bike-regression/app/app.py
```

## 5. วิธีใช้ + ตัวอย่าง input

TODO: ภาพหน้าจอ + ตัวอย่าง input

## 6. URL แอป

TODO

## 7. โครงสร้างไฟล์

```
config.py                         ค่าคงที่ (seed, path) สำหรับ scripts + notebook
scripts/download_seoulbike.py     ดาวน์โหลด Seoul Bike
scripts/make_bike_split.py        แบ่ง train/test ตามวัน
bike-regression/data/             CSV + splits/
bike-regression/notebook/         notebook วิเคราะห์ + train
bike-regression/app/              Gradio app (self-contained, deploy ได้ทันที)
```

## 8. สมาชิก (เรียงตามรหัสนิสิต)

| รหัสนิสิต | ชื่อ-นามสกุล |
|---|---|
| TODO | TODO |
| TODO | TODO |
| TODO | TODO |
| TODO | TODO |
| TODO | TODO |

## Attribution

- **Seoul Bike Sharing Demand** [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5F62R — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
