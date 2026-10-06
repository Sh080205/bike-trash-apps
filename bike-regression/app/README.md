---
title: Seoul Bike Demand
emoji: 🚲
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.28.0
python_version: "3.13"
app_file: app.py
pinned: false
license: cc-by-4.0
short_description: ทำนายยอดเช่าจักรยานรายชั่วโมงในโซล
---

# Seoul Bike Demand

ทำนายจำนวนจักรยานสาธารณะที่ถูกเช่าทั้งเมืองโซลใน 1 ชั่วโมง (คัน/ชั่วโมง) จากเวลาและสภาพอากาศ 8 ค่า
โมเดล: Pipeline (one-hot ฤดู → HistGradientBoosting, poisson loss) · test MAE ≈ 104.5 คัน/ชม. · R² ≈ 0.918

Source code + notebook: https://github.com/Sh080205/bike-trash-apps

Dataset: Seoul Bike Sharing Demand [Dataset]. (2020). UCI Machine Learning Repository.
https://doi.org/10.24432/C5F62R — CC BY 4.0
