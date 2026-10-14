# Seoul Bike Demand app

Source code, data and notebooks: https://github.com/Sh080205/seoul-bike-demand-app/tree/main/bike-regression

Run this folder independently with Python 3.13:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open the local URL printed by Gradio. The folder contains the fitted
`model.joblib` pipeline and `model_metrics.json`; the app does not import
`config.py`, the notebooks or the data. On Modal, import `demo`, `THEME` and
`CSS` from `app.py` and pass the latter two to `gr.mount_gradio_app`.

`model.joblib` is saved by step 14 of
[`../notebooks/02_modeling.ipynb`](../notebooks/02_modeling.ipynb), which
selects the HGB configuration with GroupKFold on the train days and evaluates it
once on the fixed test days (in-scope test MAE 91.46 bikes/hour).
`../scripts/export_app_model.py` rebuilds the same file without the notebook. Inputs include the date and
additional weather fields because the fitted pipeline uses 18 features. Day
type must agree with the date. The app derives dew point from temperature and
humidity, using the same formula and rounding as the notebook scenarios.
