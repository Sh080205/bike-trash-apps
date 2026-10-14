# Seoul Bike Demand app

Run this folder independently with Python 3.13:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open the local URL printed by Gradio. The folder contains the fitted
`model.joblib` pipeline and `model_metrics.json`; the app does not import files
from the training repository. On Modal, import `demo`, `THEME` and `CSS` from
`app.py` and pass the latter two to `gr.mount_gradio_app`.

The pipeline was refit from the Seoul Bike training project's
`notebooks/02_modeling.ipynb` using the fixed train/test date lists and selected
HGB configuration. It reproduces the
notebook's in-scope test MAE of 91.46 bikes/hour. Inputs include the date and
additional weather fields because the fitted pipeline uses 18 features. Day
type must agree with the date. The app derives dew point from temperature and
humidity, using the same formula and rounding as the notebook scenarios.
