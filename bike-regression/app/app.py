"""Standalone Gradio app for the selected Seoul bike demand model."""

from datetime import date as Date
from functools import lru_cache
from html import escape
import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.joblib"
METRICS_PATH = APP_DIR / "model_metrics.json"

# Same raw feature order as notebooks/02_modeling.ipynb and export_app_model.py.
FEATURES = [
    "hour", "temp_c", "humidity_pct", "wind_speed_ms", "visibility_10m",
    "dew_point_c", "solar_radiation_mj", "rainfall_mm", "snowfall_cm",
    "seasons", "day_of_week", "is_weekend", "is_holiday", "month",
    "hour_sin", "hour_cos", "is_raining", "rain_prev_3h",
]

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.green,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800;900&display=swap');
:root {
  --bike-ink: #173344;
  --bike-muted: #526877;
  --bike-surface: #ffffff;
  --bike-edge: #dce9e9;
  --bike-soft: #f1f9f5;
  --bike-accent: #168457;
}
.dark {
  --bike-ink: #eef7f7;
  --bike-muted: #b5cbd2;
  --bike-surface: #183039;
  --bike-edge: #35515a;
  --bike-soft: #172d35;
  --bike-accent: #75d9ab;
}
body, .gradio-container, .gradio-container * { font-family: 'Nunito', 'Segoe UI', sans-serif !important; }
html, body, gradio-app, .gradio-container { width: 100% !important; max-width: 100% !important; min-width: 0 !important; box-sizing: border-box !important; }
.gradio-container { max-width: 1180px !important; margin: auto !important; padding: 18px !important; }
.bike-hero {
  position: relative; overflow: hidden; border-radius: 28px; padding: 34px 38px 0;
  width: 100%; min-width: 0; box-sizing: border-box;
  background: linear-gradient(118deg, #087944 0%, #1a9d78 48%, #2485b7 100%);
  box-shadow: 0 22px 48px rgba(13, 83, 89, .18);
}
.bike-hero::before { content: ''; position: absolute; width: 390px; height: 390px; right: -110px; top: -210px; border: 1px solid rgba(255,255,255,.23); border-radius: 50%; }
.bike-hero::after { content: ''; position: absolute; width: 520px; height: 520px; right: -120px; top: -290px; border: 1px solid rgba(255,255,255,.15); border-radius: 50%; }
.bike-hero-content { position: relative; z-index: 1; max-width: 660px; }
.bike-eyebrow { color: #d5fff0 !important; font-size: 12px; letter-spacing: .16em; font-weight: 900; text-transform: uppercase; }
.bike-hero h1 { color: #fff !important; margin: 8px 0 10px; font-size: clamp(2.3rem, 5vw, 4.2rem); line-height: 1.04; letter-spacing: -.045em; font-weight: 900; }
.bike-hero p { color: #f0fff9 !important; font-size: 1.05rem; line-height: 1.55; max-width: 590px; margin: 0 0 20px; }
.bike-chips { display: flex; flex-wrap: wrap; gap: 9px; margin-bottom: 8px; }
.bike-chip { display: inline-flex; gap: 7px; align-items: baseline; border: 1px solid rgba(255,255,255,.4); border-radius: 999px; color: #fff !important; background: rgba(8,57,70,.23); padding: 8px 13px; font-size: .84rem; font-weight: 700; }
.bike-chip strong { color: #fff !important; font-size: 1rem; font-weight: 900; }
.bike-skyline { display: block; width: 100%; max-height: 125px; margin-top: 12px; }
.bike-heading { margin: 29px 0 14px; color: var(--bike-ink) !important; font-size: 1.35rem; font-weight: 900; }
.bike-steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.bike-step, .bike-panel, .bike-result {
  border: 1px solid var(--bike-edge); border-radius: 20px; background: var(--bike-surface);
  box-shadow: 0 8px 22px rgba(17, 68, 78, .055);
  min-width: 0; box-sizing: border-box;
}
.bike-step { padding: 16px 18px; color: var(--bike-ink) !important; }
.bike-step .num { display: inline-grid; place-items: center; width: 28px; height: 28px; border-radius: 9px; background: #ddf6e7; color: #116a49 !important; font-weight: 900; margin-right: 8px; }
.bike-step strong { color: var(--bike-ink) !important; font-weight: 900; }
.bike-step span { display: block; color: var(--bike-muted) !important; margin-top: 5px; font-size: .91rem; }
.main-grid { margin-top: 14px; gap: 18px !important; align-items: stretch !important; }
.bike-panel { padding: 20px !important; }
.bike-panel h3 { color: var(--bike-ink) !important; font-size: 1.15rem; font-weight: 900; margin: 2px 0 4px; }
.bike-panel p { color: var(--bike-muted) !important; margin: 0 0 11px; }
.bike-panel .gr-form { background: transparent !important; }
.bike-panel label, .bike-panel .block-title { color: var(--bike-ink) !important; font-weight: 800 !important; }
.bike-panel input, .bike-panel textarea { border-radius: 10px !important; }
.bike-panel .bike-subtitle { color: var(--bike-muted) !important; font-size: .88rem; }
.bike-predict { margin-top: 14px; border-radius: 14px !important; min-height: 48px; font-weight: 900 !important; background: linear-gradient(100deg, #0b8955, #168dad) !important; color: #fff !important; border: none !important; }
.bike-result { min-height: 322px; padding: 29px; position: sticky; top: 16px; background: linear-gradient(160deg, var(--bike-surface), var(--bike-soft)); }
.bike-result .label { color: var(--bike-muted) !important; text-transform: uppercase; letter-spacing: .12em; font-weight: 900; font-size: .76rem; }
.bike-result .value { color: var(--bike-ink) !important; font-size: clamp(3.2rem, 6vw, 5.5rem); line-height: 1.08; font-weight: 900; margin-top: 23px; letter-spacing: -.055em; }
.bike-result .unit { color: var(--bike-accent) !important; font-size: 1.2rem; font-weight: 900; }
.bike-result .detail { color: var(--bike-muted) !important; font-size: .92rem; margin-top: 16px; line-height: 1.5; }
.bike-result .line { height: 4px; width: 56px; border-radius: 4px; background: linear-gradient(90deg, #18a85f, #2697bf); margin-top: 25px; }
.bike-note { color: var(--bike-muted) !important; font-size: .84rem; line-height: 1.5; }
.bike-footer { color: var(--bike-muted) !important; text-align: center; margin: 28px 0 9px; font-size: .84rem; }
.bike-footer a { color: var(--bike-accent) !important; text-decoration: underline; }
.bike-examples { max-width: 100%; min-width: 0; overflow-x: auto; }
@media (max-width: 800px) {
  .gradio-container { padding: 10px !important; }
  .bike-hero { border-radius: 20px; padding: 24px 20px 0; }
  .bike-hero p { font-size: .96rem; }
  .bike-skyline { max-height: 85px; }
  .bike-steps { grid-template-columns: 1fr; }
  .main-grid { flex-direction: column !important; }
  .bike-panel { padding: 15px !important; }
  .bike-result { position: static; min-height: 250px; }
}
"""


@lru_cache(maxsize=1)
def load_model():
    """Load the fitted pipeline once; allow the UI to start without an artifact."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def load_metrics():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def validate(day, hour, season, day_type, holiday, temp_c, humidity_pct,
             rainfall_mm, wind_speed_ms, visibility_10m, solar_radiation_mj,
             snowfall_cm, rain_prev_3h):
    """Validate the inputs and return one row in the fitted model's feature order."""
    try:
        selected_date = Date.fromisoformat(str(day).strip())
    except (ValueError, TypeError) as exc:
        raise ValueError("Enter a date in YYYY-MM-DD format.") from exc

    if season not in {"Spring", "Summer", "Autumn", "Winter"}:
        raise ValueError("Select a valid season.")
    if day_type not in {"Weekday", "Weekend"}:
        raise ValueError("Select Weekday or Weekend.")
    if holiday not in {"No Holiday", "Holiday"}:
        raise ValueError("Select a holiday status.")
    weekday = selected_date.weekday()
    if (weekday >= 5) != (day_type == "Weekend"):
        raise ValueError("Day type must match the selected date.")

    def checked_number(value, name, lower, upper):
        try:
            number = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Enter a number for {name}.") from exc
        if not np.isfinite(number) or not lower <= number <= upper:
            raise ValueError(f"{name} must be between {lower:g} and {upper:g}.")
        return number

    hour = checked_number(hour, "Hour", 0, 23)
    if not hour.is_integer():
        raise ValueError("Hour must be a whole number.")
    temp_c = checked_number(temp_c, "Temperature", -35, 45)
    humidity_pct = checked_number(humidity_pct, "Humidity", 1, 100)
    rainfall_mm = checked_number(rainfall_mm, "Rainfall", 0, 100)
    wind_speed_ms = checked_number(wind_speed_ms, "Wind speed", 0, 35)
    visibility_10m = checked_number(visibility_10m, "Visibility", 0, 2000)
    solar_radiation_mj = checked_number(solar_radiation_mj, "Solar radiation", 0, 5)
    snowfall_cm = checked_number(snowfall_cm, "Snowfall", 0, 30)
    rain_prev_3h = checked_number(rain_prev_3h, "Previous 3h rainfall", 0, 300)

    # Same Magnus formula and one-decimal rounding as notebook scenario inputs.
    gamma = np.log(humidity_pct / 100) + 17.625 * temp_c / (243.04 + temp_c)
    dew_point_c = round(float(243.04 * gamma / (17.625 - gamma)), 1)
    hour = int(hour)
    values = {
        "hour": hour,
        "temp_c": temp_c,
        "humidity_pct": humidity_pct,
        "wind_speed_ms": wind_speed_ms,
        "visibility_10m": visibility_10m,
        "dew_point_c": dew_point_c,
        "solar_radiation_mj": solar_radiation_mj,
        "rainfall_mm": rainfall_mm,
        "snowfall_cm": snowfall_cm,
        "seasons": season,
        "day_of_week": weekday,
        "is_weekend": int(weekday >= 5),
        "is_holiday": int(holiday == "Holiday"),
        "month": selected_date.month,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "is_raining": int(rainfall_mm > 0),
        "rain_prev_3h": rain_prev_3h,
    }
    return pd.DataFrame([values], columns=FEATURES)


def predict_count(*inputs):
    model = load_model()
    if model is None:
        raise FileNotFoundError("The model file is missing. Export model.joblib first.")
    row = validate(*inputs)
    return int(np.round(max(0.0, float(model.predict(row)[0]))))


def result_card(value=None, detail=None):
    shown = "—" if value is None else f"{value:,}"
    detail = detail or "Set the conditions and select Predict demand."
    return (
        '<div class="bike-result">'
        '<div class="label">Estimated rentals</div>'
        f'<div class="value">{shown}</div>'
        '<div class="unit">bikes/hour</div>'
        f'<div class="detail">{escape(detail)}</div>'
        '<div class="line"></div>'
        '</div>'
    )


def predict(*inputs):
    try:
        return result_card(predict_count(*inputs), "Expected rentals for this hour across Seoul.")
    except (ValueError, FileNotFoundError) as exc:
        raise gr.Error(str(exc)) from exc


EXAMPLES = [
    ["2018-06-13", 8, "Summer", "Weekday", "No Holiday", 21, 65, 0, 1.5, 2000, 0.9, 0, 0],
    ["2018-07-09", 22, "Summer", "Weekday", "No Holiday", 20, 95, 3, 2, 300, 0, 0, 4],
    ["2018-01-17", 15, "Winter", "Weekday", "No Holiday", -3, 40, 0, 1.5, 2000, 0.5, 0, 0],
    ["2018-06-16", 18, "Summer", "Weekend", "No Holiday", 24, 60, 0, 1.5, 2000, 0.5, 0, 0],
]


def hero_html(metrics):
    chips = ""
    if metrics:
        chips = (
            '<div class="bike-chips">'
            f'<span class="bike-chip">Test MAE <strong>{metrics["test_mae"]:.2f}</strong> bikes/hour</span>'
            f'<span class="bike-chip">Test R² <strong>{metrics["test_r2"]:.3f}</strong></span>'
            f'<span class="bike-chip"><strong>{metrics["test_hours"]:,}</strong> test hours</span>'
            '</div>'
        )
    return f"""
    <div class="bike-hero">
      <div class="bike-hero-content">
        <div class="bike-eyebrow">Seoul · 서울 · Ttareungyi 따릉이</div>
        <h1>Seoul Bike Demand</h1>
        <p>Explore how time and weather shape city cycling. Estimate how many bikes are rented across Seoul in a selected hour.</p>
        {chips}
      </div>
      <svg class="bike-skyline" viewBox="0 0 1100 125" role="img" aria-label="Seoul skyline and a bridge over the Han River" xmlns="http://www.w3.org/2000/svg">
        <path d="M0 113 Q260 96 550 111 T1100 107 V125 H0Z" fill="#c5f8e8" opacity=".3"/>
        <path d="M0 95 H55 V66 H85 V43 H125 V95 H170 V60 H209 V95 H244 V74 H285 V95 H330 V52 H368 V95 H412 V69 H448 V95 H490 V74 H515 V38 H524 V17 H529 V5 H534 V17 H540 V38 H548 V74 H574 V95 H625 V50 H659 V95 H710 V70 H748 V95 H797 V58 H832 V95 H891 V78 H928 V95 H977 V67 H1015 V95 H1100" fill="#e8fff7" opacity=".43"/>
        <path d="M0 104 Q255 83 550 105 T1100 103" fill="none" stroke="#fff" stroke-width="3" opacity=".7"/>
        <path d="M690 103 L725 77 L933 77 L972 103 M745 77 V103 M950 79 V103" fill="none" stroke="#f5fff9" stroke-width="4" stroke-linecap="round" opacity=".85"/>
        <g fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><circle cx="334" cy="93" r="12"/><circle cx="380" cy="93" r="12"/><path d="M334 93 L352 70 L364 93 L380 93 L363 70 H352 M356 61 H367 M363 70 L373 62"/></g>
      </svg>
    </div>"""


def build_ui():
    metrics = load_metrics()
    score_note = (
        f"Its in-scope holdout test MAE is **{metrics['test_mae']:.2f} bikes/hour** "
        f"over **{metrics['test_hours']:,} hours**. "
        if metrics else "Test metrics are unavailable. "
    )
    with gr.Blocks(title="Seoul Bike Demand") as page:
        gr.HTML(hero_html(metrics))
        gr.HTML("""
          <h2 class="bike-heading">How to use</h2>
          <div class="bike-steps">
            <div class="bike-step"><span class="num">1</span><strong>Choose a time</strong><span>Set the date, hour and calendar details.</span></div>
            <div class="bike-step"><span class="num">2</span><strong>Add the weather</strong><span>Enter conditions measured in that hour.</span></div>
            <div class="bike-step"><span class="num">3</span><strong>See the estimate</strong><span>Get a citywide rentals estimate in bikes/hour.</span></div>
          </div>""")
        with gr.Row(elem_classes="main-grid"):
            with gr.Column(scale=3, min_width=320, elem_classes="bike-panel"):
                gr.HTML('<h3>Time</h3><p>Pick the hour you want to estimate.</p>')
                with gr.Row():
                    day = gr.Textbox(label="Date (YYYY-MM-DD)", value="2018-06-13")
                    hour = gr.Number(label="Hour (0–23)", value=18, precision=0)
                with gr.Row():
                    season = gr.Dropdown(["Spring", "Summer", "Autumn", "Winter"], label="Season", value="Summer")
                    day_type = gr.Dropdown(["Weekday", "Weekend"], label="Day type", value="Weekday")
                    holiday = gr.Dropdown(["No Holiday", "Holiday"], label="Holiday", value="No Holiday")
                gr.HTML('<h3 style="margin-top:24px">Weather</h3><p>Enter the conditions for that hour.</p>')
                with gr.Row():
                    temp = gr.Number(label="Temperature (°C)", value=24)
                    humidity = gr.Number(label="Humidity (%)", value=60)
                with gr.Row():
                    rainfall = gr.Number(label="Rainfall (mm)", value=0)
                    wind = gr.Number(label="Wind speed (m/s)", value=1.5)
                with gr.Accordion("Additional weather used by the model", open=False):
                    gr.Markdown("These measurements help the model match its training data. Visibility is recorded in units of 10 metres; 2,000 means 20 km. Dew point is calculated from temperature and humidity.", elem_classes="bike-subtitle")
                    with gr.Row():
                        visibility = gr.Number(label="Visibility (10 m units)", value=2000)
                        solar = gr.Number(label="Solar radiation (MJ/m²)", value=0.5)
                    with gr.Row():
                        snowfall = gr.Number(label="Snowfall (cm)", value=0)
                        rain_prev = gr.Number(label="Rain in previous 3 hours (mm)", value=0)
                button = gr.Button("Predict demand →", variant="primary", elem_classes="bike-predict")
            with gr.Column(scale=2, min_width=280):
                result = gr.HTML(result_card())
                gr.HTML('<p class="bike-note">This is a model estimate for all of Seoul, not an availability forecast for a specific station.</p>')

        inputs = [day, hour, season, day_type, holiday, temp, humidity, rainfall,
                  wind, visibility, solar, snowfall, rain_prev]
        button.click(fn=predict, inputs=inputs, outputs=result)
        page.load(fn=predict, inputs=inputs, outputs=result)
        gr.HTML('<h2 class="bike-heading">Try an example</h2><p class="bike-note">Select a row to load its conditions and prediction.</p>')
        with gr.Group(elem_classes="bike-examples"):
            gr.Examples(examples=EXAMPLES, inputs=inputs, outputs=result, fn=predict,
                        cache_examples=False, run_on_click=True)
        with gr.Accordion("About the model and limitations", open=False):
            about_text = (
                "**Model:** HistGradientBoostingRegressor trained on the Seoul Bike Sharing Demand dataset. "
                "The pipeline includes median imputation, scaling, encoding and feature selection. "
                + score_note
                + "The data covers one year in one city and uses measured weather. Accuracy with weather forecasts or in another year is untested. "
                "Closed-system hours and specified storm dates were excluded from training and the reported test score. "
                "Rainy hours had higher errors. The date determines month and day of week; day type must agree with it."
            )
            gr.Markdown(about_text)
        gr.HTML('''<div class="bike-footer">Data: <a href="https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand" target="_blank" rel="noopener">UCI Seoul Bike Sharing Demand</a> · <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noopener">CC BY 4.0</a></div>''')
    return page


demo = build_ui()


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
