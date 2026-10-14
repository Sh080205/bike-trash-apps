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
`config.py`, the notebooks or the data.

## Deploy on Modal

`modal_app.py` builds a Python 3.13 image from `requirements.txt`, copies this
folder into it and mounts the Gradio app with `gr.mount_gradio_app`. From the
repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install modal
.\.venv\Scripts\modal.exe token new                                 # one-time login (opens the browser)
.\.venv\Scripts\modal.exe deploy bike-regression/app/modal_app.py    # prints the public URL
```

`modal serve bike-regression/app/modal_app.py` gives a temporary URL for testing
until Ctrl+C. The app runs in one container (Gradio keeps sessions in memory)
that sleeps after 20 idle minutes; the first visit afterwards takes a few
seconds to start.

`model.joblib` is saved by step 14 of
[`../notebooks/02_modeling.ipynb`](../notebooks/02_modeling.ipynb), which
selects the HGB configuration with GroupKFold on the train days and evaluates it
once on the fixed test days (in-scope test MAE 91.46 bikes/hour).
`../scripts/export_app_model.py` rebuilds the same file without the notebook. Inputs include the date and
additional weather fields because the fitted pipeline uses 18 features. Season,
day of week and weekend are derived from the date exactly as in the dataset, and
dew point from temperature and humidity with the same formula and rounding as the
notebook scenarios.
