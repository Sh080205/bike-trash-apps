# Seoul Bike Demand

> วิชา: **Introduction to Data Science** · กำหนดส่ง **15 ต.ค. 2569 07:00**

**bike-regression** — ทำนายจำนวนจักรยานที่ถูกเช่า (คัน/ชั่วโมง) จากสภาพอากาศและเวลา งานทั้งหมดอยู่ในโฟลเดอร์ `bike-regression/`

## 1. แนวคิด / ปัญหา / ประโยชน์

- **ทำนายอะไร:** จำนวนจักรยาน Seoul Bike ที่ถูกเช่าทั้งเมืองในชั่วโมงหนึ่ง (`Rented Bike Count`) หน่วย **คัน/ชั่วโมง**
- **ตัวแปรนำเข้า:** ชั่วโมง วันที่ (→ วันในสัปดาห์ เดือน วันหยุดสุดสัปดาห์) ฤดู วันหยุดราชการ อุณหภูมิ ความชื้น ลม ทัศนวิสัย dew point แสงแดด ฝน หิมะ และฝนรวม 3 ชั่วโมงก่อนหน้าในวันเดียวกัน รวม 18 feature
- **ประโยชน์:** ผู้ให้บริการใช้คาดยอดเช่าล่วงหน้าเพื่อวางแผนกระจายจักรยานไปยังสถานี จัดรอบเติมจักรยาน และจัดคนตามช่วงเวลาที่ความต้องการสูง (เช่น เย็นวันทำงานฤดูร้อน) หรือลดลง (ฝนตก ฤดูหนาว)

## 2. แหล่งข้อมูล จำนวน และการแบ่ง

| | |
|---|---|
| Dataset | Seoul Bike Sharing Demand (UCI id 560, CC BY 4.0) |
| จำนวนทั้งหมด | 8,760 แถว (365 วัน × 24 ชม.) × 14 คอลัมน์ |
| train / test | train **292 วัน / 7,008 แถว** · test **73 วัน / 1,752 แถว** (แบ่งตาม **วัน**) |
| หลังกำหนดขอบเขต | train 6,672 ชม. / 278 วัน · test 1,656 ชม. / 69 วัน |

แบ่งด้วย `GroupShuffleSplit(test_size=0.2, random_state=42)` โดย group = วันที่ เพราะชั่วโมงในวันเดียวกันมีอากาศและพฤติกรรมคล้ายกัน ถ้าสุ่มทีละแถวจะมีชั่วโมงของวันเดียวกันอยู่ทั้ง train และ test ทำให้คะแนนดีเกินจริง · รายการวันอยู่ใน `bike-regression/data/splits/` · ขอบเขตโมเดล: ตัดชั่วโมงที่ระบบปิด (`Functioning Day = No`, ยอด = 0 เสมอ) และวันได้รับผลกระทบจากพายุตามประกาศ KMA ล่วงหน้า (`bike-regression/config.py`) ใช้กฎเดียวกันทั้ง train/test · ข้อมูลที่ notebook ใช้คือ `data/processed/SeoulBikeData_missing.csv` = ไฟล์ UCI ที่เว้นค่าความชื้น 0% ว่าง 9 ช่องเพื่อฝึกจัดการ missing value (ดู `bike-regression/data/README.md`)

## 3. โมเดล ผลหลัก และข้อจำกัด

- **โมเดล:** `HistGradientBoostingRegressor(learning_rate=0.1, max_iter=200, max_leaf_nodes=63, min_samples_leaf=5, l2_regularization=1.0)` ใน `Pipeline` เดียว (median impute + scale + one-hot → `SelectKBest(mutual_info, k=36)` → HGB)
- **วิธีเลือก:** เทียบ Linear, kNN, Random Forest, HGB (รวม feature selection และ PCA) ด้วย 5-fold `GroupKFold` ตามวันบน train เท่านั้น · HGB ได้ CV MAE ต่ำสุด **104.86 ± 3.96** (baseline ที่ใช้เวลาอย่างเดียว 398.4) เพราะจับความสัมพันธ์ไม่เชิงเส้นและปฏิสัมพันธ์ ชั่วโมง × ประเภทวัน × อากาศ ได้ดี
- **ผลบน test (1,656 ชม.):**

| เมตริก | ค่า | ความหมาย |
|---|---|---|
| **MAE (หลัก)** | **91.46 คัน/ชม.** | ทายคลาดเฉลี่ย ~91 คัน หรือ ~15% ของยอดเฉลี่ย test (615 คัน/ชม.) |
| RMSE | 149.06 คัน/ชม. | สูงกว่า MAE (1.63 เท่า) → มีบางชั่วโมงที่ผิดมาก |
| R² | 0.938 | อธิบายความแปรปรวนของยอดเช่าได้ ~94% |

  เลือก MAE เป็นเมตริกหลักเพราะอ่านเป็นจำนวนคันได้ตรง ๆ และไม่ถูกครอบงำด้วยชั่วโมงผิดปกติไม่กี่ชั่วโมง · ไม่ใช้ MAPE เพราะมียอด = 0
- **ข้อจำกัด:** ชั่วโมงฝนตกยังผิดมาก (MAE 143 คัน, 83% ของยอดเฉลี่ยกลุ่มนั้น) · ไม่ครอบคลุมวันพายุ (รวมวันพายุแล้ว MAE 95.48; เฉพาะชั่วโมงพายุ 257.65) · ข้อมูลปีเดียว เมืองเดียว ใช้อากาศที่วัดจริง ไม่ใช่พยากรณ์ · การเลือก feature, selector และพารามิเตอร์ทั้งหมดใช้ CV บนชุดฝึกเท่านั้น ไม่ได้ใช้ผล test ย้อนปรับหรือเลือกโมเดล — test ใช้ประเมินหลังตรึงโมเดลแล้ว (ระหว่างพัฒนามีการรันประเมิน test มากกว่าหนึ่งรอบ คะแนน test จึงอาจดีกว่าผลกับข้อมูลใหม่จริงเล็กน้อย)
- รายละเอียดทั้งหมดอยู่ใน `bike-regression/notebooks/02_modeling.ipynb` (ขั้น 13 = ผล test + error analysis, ขั้น 14 = บันทึกโมเดล)

## 4. ติดตั้งและรันบนเครื่อง

ทดสอบบน Windows 11 + Python 3.13

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

```bash
cd bike-regression
python scripts/run_notebooks.py      # รัน 00_setup → 01_eda → 02_modeling ตามลำดับ (~4 นาที) และบันทึก app/model.joblib
python app/app.py                    # เปิดแอปที่ http://127.0.0.1:7860
```

- เปิดดู/รันทีละ cell ได้ด้วย `jupyter notebook notebooks` (รันตามลำดับ 00 → 01 → 02)
- ข้อมูลทุกไฟล์แนบใน repo แล้ว ไม่ต้องดาวน์โหลด · `python scripts/make_dataset.py` สร้างไฟล์ processed ซ้ำจาก raw ได้
- รันเฉพาะแอป: `pip install -r bike-regression/app/requirements.txt` แล้ว `python bike-regression/app/app.py`

## 5. วิธีใช้ + ตัวอย่าง input

1. **เลือกเวลา:** กดช่องวันที่แล้วเลือกจากปฏิทิน (พิมพ์เองไม่ได้ จึงไม่มีวันที่ผิดรูปแบบ) และเลื่อนแถบชั่วโมง (0–23) · แอปคำนวณวันในสัปดาห์ วันธรรมดา/เสาร์-อาทิตย์ และฤดูจากวันที่ให้เอง (เหมือนในข้อมูล) · ช่องวันหยุดเลือก Holiday ให้อัตโนมัติถ้าเป็นวันหยุดราชการในชุดข้อมูล
2. **ปรับสภาพอากาศด้วยแถบเลื่อน:** อุณหภูมิ (−17 ถึง 40 °C) ความชื้น (10–100%) ฝน (0–35 มม.) ลม (0–7.5 m/s) · ช่องเพิ่มเติม (กดเปิด): ทัศนวิสัย (20–2,000 หน่วย 10 ม.) แสงแดด (0–3.5 MJ/m²) หิมะ (0–9 ซม.) ฝน 3 ชม.ก่อนหน้า (0–52 มม.) · ช่วงของแถบเลื่อนเท่ากับค่าที่โซลเคยมีจริงในข้อมูล · พิมพ์ตัวเลขในกล่องข้างแถบได้ · dew point คำนวณจากอุณหภูมิ + ความชื้น
3. **ดูผล:** ค่าทำนาย **คัน/ชั่วโมง** อัปเดตทันทีเมื่อเปลี่ยนค่าใดก็ได้ (หรือกด Predict demand →) พร้อมวัน/ชั่วโมง/ฤดูที่ใช้ และความคลาดเฉลี่ยบน test (MAE 91 คัน/ชม.) · ถ้าค่าไม่สมจริง (เช่น แสงแดดตอนกลางคืน หิมะในฤดูร้อน) จะมีคำเตือนสีเหลือง
4. หรือกดตัวอย่างใน *Try an example* เพื่อโหลดค่าและทำนายทันที

| ตัวอย่าง | วันที่ | ชม. | วันหยุด | อุณหภูมิ °C | ความชื้น % | ฝน มม. | ลม m/s | ทัศนวิสัย | แสงแดด | หิมะ | ฝน 3 ชม.ก่อน | ทำนาย (คัน/ชม.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| วันธรรมดา ฤดูร้อน เช้า | 2018-06-13 | 8 | No Holiday | 21 | 65 | 0 | 1.5 | 2000 | 0.9 | 0 | 0 | 2,145 |
| เสาร์ ฤดูร้อน เย็น | 2018-06-16 | 18 | No Holiday | 24 | 60 | 0 | 1.5 | 2000 | 0.5 | 0 | 0 | 2,233 |
| คืนฝนตก ฤดูร้อน | 2018-07-09 | 22 | No Holiday | 20 | 95 | 3 | 2 | 300 | 0 | 0 | 4 | 125 |
| วันหยุด (Liberation Day) | 2018-08-15 | 14 | Holiday | 30 | 60 | 0 | 2 | 2000 | 2.5 | 0 | 0 | 827 |
| บ่ายฤดูหนาว | 2018-01-17 | 15 | No Holiday | −3 | 40 | 0 | 1.5 | 2000 | 0.5 | 0 | 0 | 238 |

## 6. URL แอป

- **bike-regression:** https://sh080205--seoul-bike-demand.us-east.modal.run
- **image-classification (Pet Breed Classifier):** https://franknakub--pet-breed-classifier-ui.us-east.modal.run ([repository](https://github.com/fxanlx/pet-breed-classifier))

deploy บน [Modal](https://modal.com) ด้วย `bike-regression/app/modal_app.py` · ถ้าไม่มีคนใช้ 20 นาที แอปจะพักเครื่อง การเปิดครั้งแรกหลังจากนั้นใช้เวลาเริ่มประมาณ 5–10 วินาที · deploy ซ้ำ: `pip install modal` → `modal token new` → `modal deploy bike-regression/app/modal_app.py` (รายละเอียดใน `bike-regression/app/README.md`)

## 7. โครงสร้างไฟล์

```
requirements.txt                        แพ็กเกจสำหรับติดตั้ง notebook + scripts + แอป

bike-regression/config.py               ค่าคงที่ของงาน bike (path, encoding, วันพายุ, ชื่อคอลัมน์)
bike-regression/requirements.txt       แพ็กเกจ (pin เวอร์ชัน) ที่ 00_setup ตรวจ
bike-regression/data/raw/               ไฟล์ต้นฉบับจาก UCI (zip + CSV)
bike-regression/data/processed/         dataset ที่ notebook ใช้ (ความชื้นว่าง 9 ช่อง)
bike-regression/data/splits/            train_dates.csv / test_dates.csv (แบ่งตามวัน)
bike-regression/notebooks/              00_setup → 01_eda → 02_modeling
bike-regression/figures/                กราฟที่ notebook สร้าง
bike-regression/scripts/                make_dataset, run_notebooks, export_app_model, export_results
bike-regression/verification/           ผลและตารางจากรอบรันล่าสุด (results.json, *.csv, log)
bike-regression/app/                    Gradio app (self-contained): app.py, model.joblib, model_metrics.json, requirements.txt, modal_app.py (deploy)
```

## 8. สมาชิก (เรียงตามรหัสนิสิต)

| รหัสนิสิต | ชื่อ-นามสกุล |
|---|---|
| 6630300866 | นาย สกลภัทร ภิญโญ |
| 6730300281 | นาย นันทพัฒน์ วรรณลักษณ์ |
| 6730300591 | นาย ศุภวิชญ์ ไชยานุศักดิ์ |
| 6730300779 | นาย ธรรศธรณ์ เจียมตน |
| 6730300850 | นาย พัฒนเดช มะหิเมือง |

## Attribution

- **Seoul Bike Sharing Demand** [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5F62R — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
