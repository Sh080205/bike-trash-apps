# Bike & Trash Apps

> วิชา: **Introduction to Data Science** · กำหนดส่ง **15 ต.ค. 2569 07:00**

สองแอปในรีโปเดียว:

1. **bike-regression** — ทำนายจำนวนจักรยานที่ถูกเช่า (คัน/ชั่วโมง) จากสภาพอากาศและเวลา
2. **trash-classification** — จำแนกประเภทขยะจากรูปภาพ (6 คลาส)

## 1. แนวคิด / ปัญหา / ประโยชน์

### bike-regression
TODO: ปัญหาคืออะไร ใครได้ประโยชน์ นำไปใช้อย่างไร

### trash-classification
TODO

## 2. แหล่งข้อมูล จำนวน และการแบ่ง

| | bike-regression | trash-classification |
|---|---|---|
| Dataset | Seoul Bike Sharing Demand (UCI id 560) | TrashNet (`garythung/trashnet`) |
| จำนวนทั้งหมด | TODO แถว | TODO ภาพ |
| train / test | TODO (แบ่งตาม **วัน**) | TODO (stratified 80/20 ตามคลาส) |

TODO: อธิบายเหตุผลการแบ่ง (ทำไมต้องแบ่งตามวัน / ทำไม stratified)

## 3. โมเดล ผลหลัก และข้อจำกัด

### bike-regression
TODO: โมเดลที่เลือก + เหตุผล · MAE (คัน/ชม.) / RMSE / R² บน test · ข้อจำกัด

### trash-classification
TODO: โมเดล + feature ที่สกัด · accuracy, precision/recall/F1 รายคลาส (ระบุ macro/weighted) · คลาสที่สับสน · ข้อจำกัด

## 4. ติดตั้งและรันบนเครื่อง

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python scripts/download_seoulbike.py
python scripts/make_bike_split.py
python scripts/prepare_trashnet.py

python bike-regression/app/app.py
python trash-classification/app/app.py
```

## 5. วิธีใช้ + ตัวอย่าง input

TODO: ภาพหน้าจอ + ตัวอย่าง input ของแต่ละแอป

## 6. URL แอป

- bike-regression: TODO
- trash-classification: TODO

## 7. โครงสร้างไฟล์

```
config.py                         ค่าคงที่ (seed, path) สำหรับ scripts + notebook
scripts/download_seoulbike.py     ดาวน์โหลด Seoul Bike
scripts/make_bike_split.py        แบ่ง train/test ตามวัน
scripts/prepare_trashnet.py       ย่อภาพ + แบ่ง train/test + labels.csv
bike-regression/data/             CSV + splits/
bike-regression/notebook/         notebook วิเคราะห์ + train
bike-regression/app/              Gradio app (self-contained, deploy ได้ทันที)
trash-classification/data/        images/<class>/ + labels.csv
trash-classification/notebook/    notebook วิเคราะห์ + train
trash-classification/app/         Gradio app + features.py + examples/
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
- **TrashNet** — Gary Thung & Mindy Yang, https://github.com/garythung/trashnet (Hugging Face: https://huggingface.co/datasets/garythung/trashnet) — MIT License
