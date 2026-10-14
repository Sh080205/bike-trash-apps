"""Standalone Gradio app for the selected Seoul bike demand model."""

from datetime import datetime
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

# Seasons in the dataset follow the month exactly (Dec-Feb = Winter, ...).
SEASON_BY_MONTH = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                   6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
# Public holidays flagged in the dataset (Dec 2017 - Nov 2018); only used to pre-select "Holiday".
DATASET_HOLIDAYS = {
    "2017-12-22", "2017-12-24", "2017-12-25", "2017-12-31", "2018-01-01", "2018-02-15",
    "2018-02-16", "2018-02-17", "2018-03-01", "2018-05-01", "2018-05-22", "2018-06-06",
    "2018-08-15", "2018-09-23", "2018-09-24", "2018-09-25", "2018-10-03", "2018-10-09",
}
# Observed range of each input over the training hours; values outside are extrapolation.
TRAIN_RANGES = {
    "Temperature": (-16.2, 39.4, "°C"),
    "Humidity": (10, 98, "%"),
    "Wind speed": (0, 7.4, "m/s"),
    "Visibility": (27, 2000, "× 10 m"),
    "Solar radiation": (0, 3.52, "MJ/m²"),
    "Rainfall": (0, 35, "mm"),
    "Snowfall": (0, 8.8, "cm"),
    "Previous 3h rainfall": (0, 52, "mm"),
}
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Slider limits cover the hourly values seen in the data (Seoul, Dec 2017 - Nov 2018),
# so every reachable input is a condition the city actually had.
WEATHER_SLIDERS = {
    "temp": dict(label="🌡️ Temperature (°C)", minimum=-17, maximum=40, step=0.5, value=24,
                 info="Seoul ranged from −17 to 40 °C."),
    "humidity": dict(label="💧 Humidity (%)", minimum=10, maximum=100, step=1, value=60),
    "rainfall": dict(label="🌧️ Rainfall this hour (mm)", minimum=0, maximum=35, step=0.1, value=0,
                     info="0 = dry · 3 mm ≈ steady rain"),
    "wind": dict(label="💨 Wind speed (m/s)", minimum=0, maximum=7.5, step=0.1, value=1.5),
    "visibility": dict(label="👁️ Visibility (× 10 m)", minimum=20, maximum=2000, step=10, value=2000,
                       info="2,000 = 20 km, a clear day."),
    "solar": dict(label="☀️ Solar radiation (MJ/m²)", minimum=0, maximum=3.5, step=0.01, value=0.5,
                  info="0 at night · up to ~3.5 at noon in summer"),
    "snowfall": dict(label="❄️ Snowfall (cm)", minimum=0, maximum=9, step=0.1, value=0),
    "rain_prev": dict(label="☔ Rain in previous 3 hours (mm)", minimum=0, maximum=52, step=0.1, value=0),
}


def weather_slider(name):
    return gr.Slider(elem_classes="bike-slider", **WEATHER_SLIDERS[name])


# The calendar is the only way to set the date: the text box becomes read-only and opens it.
DATE_PICKER_JS = """
() => {
  const wire = () => {
    const root = document.querySelector('#bike-date');
    const input = root && root.querySelector('input');
    const button = root && root.querySelector('button.calendar');
    if (!input || !button) return false;
    if (!input.dataset.bikeCalendar) {
      input.dataset.bikeCalendar = '1';
      input.readOnly = true;
      input.setAttribute('aria-label', 'Date, choose from the calendar');
      input.addEventListener('click', () => button.click());
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); button.click(); }
      });
    }
    return true;
  };
  if (!wire()) { const timer = setInterval(() => { if (wire()) clearInterval(timer); }, 200); }
}
"""

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
.bike-skyline { display: block; width: 100%; height: clamp(96px, 13vw, 150px); margin-top: 12px; overflow: hidden; }
.bike-rider { animation: bike-ride 22s linear infinite; }
.bike-bob { animation: bike-bob .45s ease-in-out infinite alternate; }
.bike-wheel { transform-box: fill-box; transform-origin: center; animation: bike-spin .8s linear infinite; }
.bike-wave { animation: bike-wave 7s ease-in-out infinite alternate; }
.bike-wave.slow { animation-duration: 11s; animation-direction: alternate-reverse; }
.bike-tower-light { animation: bike-blink 2.6s ease-in-out infinite; }
@keyframes bike-ride { from { transform: translateX(-90px); } to { transform: translateX(1160px); } }
@keyframes bike-bob { to { transform: translateY(-1.5px); } }
@keyframes bike-spin { to { transform: rotate(360deg); } }
@keyframes bike-wave { to { transform: translateX(-40px); } }
@keyframes bike-blink { 50% { opacity: .25; } }
@media (prefers-reduced-motion: reduce) {
  .bike-rider { animation: none; transform: translateX(330px); }
  .bike-bob, .bike-wheel, .bike-wave, .bike-tower-light { animation: none; }
}
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
.bike-panel .bike-subtitle { color: var(--bike-muted) !important; font-size: .88rem; padding: 0 8px; }
.bike-predict { margin-top: 14px; border-radius: 14px !important; min-height: 48px; font-weight: 900 !important; background: linear-gradient(100deg, #0b8955, #168dad) !important; color: #fff !important; border: none !important; }
.bike-result { min-height: 322px; padding: 29px; position: sticky; top: 16px; background: linear-gradient(160deg, var(--bike-surface), var(--bike-soft)); }
.bike-result .label { color: var(--bike-muted) !important; text-transform: uppercase; letter-spacing: .12em; font-weight: 900; font-size: .76rem; }
.bike-result .value { color: var(--bike-ink) !important; font-size: clamp(3.2rem, 6vw, 5.5rem); line-height: 1.08; font-weight: 900; margin-top: 23px; letter-spacing: -.055em; }
.bike-result .unit { color: var(--bike-accent) !important; font-size: 1.2rem; font-weight: 900; }
.bike-result .detail { color: var(--bike-muted) !important; font-size: .92rem; margin-top: 16px; line-height: 1.5; }
.bike-result .line { height: 4px; width: 56px; border-radius: 4px; background: linear-gradient(90deg, #18a85f, #2697bf); margin-top: 25px; }
#bike-date input { cursor: pointer; caret-color: transparent; font-weight: 800; color: var(--bike-ink) !important; }
#bike-date button.calendar { color: var(--bike-accent) !important; border-radius: 10px; }
#bike-date button.calendar:hover { background: var(--bike-soft) !important; }
.picker { border: 1px solid var(--bike-edge) !important; border-radius: 18px !important; background: var(--bike-surface) !important;
  box-shadow: 0 18px 40px rgba(13, 83, 89, .18) !important; padding: 12px !important; }
.picker .month-year { color: var(--bike-ink) !important; font-weight: 900 !important; }
.picker .nav-button { color: var(--bike-accent) !important; border-radius: 10px !important; font-weight: 900 !important; }
.picker .nav-button:hover { background: var(--bike-soft) !important; }
.picker .weekday { color: var(--bike-muted) !important; font-weight: 800 !important; }
.picker .day { border-radius: 10px !important; color: var(--bike-ink) !important; font-weight: 700 !important; }
.picker .day.other-month { color: var(--bike-muted) !important; opacity: .45; }
.picker .day:hover { background: var(--bike-soft) !important; }
.picker .day.selected { background: linear-gradient(135deg, #0b8955, #168dad) !important; color: #fff !important; }
.picker-actions > .action-button:first-child { display: none !important; }
.picker .action-button { border-radius: 10px !important; font-weight: 800 !important; }
.picker-actions-right .action-button:last-child { background: var(--bike-accent) !important; color: #fff !important; border-color: transparent !important; }
.dark .picker-actions-right .action-button:last-child { color: #0b2a20 !important; }
.bike-slider input[type=range] { accent-color: var(--bike-accent); }
.bike-slider input[type=number] { font-weight: 800; color: var(--bike-ink) !important; border-radius: 10px !important; }
.bike-result.bike-error .value { color: #b42318 !important; }
.bike-result.bike-error .detail { color: #b42318 !important; font-weight: 700; }
.bike-warn { margin: 14px 0 0; padding: 10px 14px 10px 30px; border-radius: 12px; background: #fff6e0; color: #7a4b00 !important; font-size: .86rem; line-height: 1.45; }
.dark .bike-warn { background: #3d3115; color: #ffd98a !important; }
.bike-note { color: var(--bike-muted) !important; font-size: .84rem; line-height: 1.5; }
.bike-footer { color: var(--bike-muted) !important; text-align: center; margin: 28px 0 9px; font-size: .84rem; }
.bike-footer a { color: var(--bike-accent) !important; text-decoration: underline; }
.bike-examples { max-width: 100%; min-width: 0; overflow-x: auto; }
@media (max-width: 800px) {
  .gradio-container { padding: 10px !important; }
  .bike-hero { border-radius: 20px; padding: 24px 20px 0; }
  .bike-hero p { font-size: .96rem; }
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


def parse_date(day):
    """The calendar sends YYYY-MM-DD (a time part, if any, is ignored)."""
    try:
        return datetime.strptime(str(day or "").strip()[:10], "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("Choose a date from the calendar.") from exc


def describe_date(day):
    """Season and day type are derived from the date, exactly as in the training data."""
    try:
        selected = parse_date(day)
    except ValueError:
        return "Choose a date to see its day of week and season."
    weekday = selected.weekday()
    kind = "weekend" if weekday >= 5 else "weekday"
    note = " · public holiday in the dataset" if selected.isoformat() in DATASET_HOLIDAYS else ""
    return f"**{DAY_NAMES[weekday]}** ({kind}) · season **{SEASON_BY_MONTH[selected.month]}**{note}"


def suggest_holiday(day):
    try:
        return "Holiday" if parse_date(day).isoformat() in DATASET_HOLIDAYS else "No Holiday"
    except ValueError:
        return gr.update()


def range_warnings(row):
    """Inputs outside the training range or physically unlikely: still predicted, but flagged."""
    values = {
        "Temperature": row["temp_c"], "Humidity": row["humidity_pct"],
        "Wind speed": row["wind_speed_ms"], "Visibility": row["visibility_10m"],
        "Solar radiation": row["solar_radiation_mj"], "Rainfall": row["rainfall_mm"],
        "Snowfall": row["snowfall_cm"], "Previous 3h rainfall": row["rain_prev_3h"],
    }
    notes = []
    for name, value in values.items():
        lower, upper, unit = TRAIN_RANGES[name]
        if not lower <= value <= upper:
            notes.append(f"{name} {value:g} {unit} is outside the training range "
                         f"({lower:g}–{upper:g} {unit}).")
    if row["solar_radiation_mj"] > 0 and (row["hour"] <= 5 or row["hour"] >= 21):
        notes.append("Solar radiation was always 0 between 21:00 and 05:00 in the data.")
    if row["snowfall_cm"] > 0 and row["seasons"] == "Summer":
        notes.append("Snowfall never occurred in summer in the data.")
    return notes


def validate(day, hour, holiday, temp_c, humidity_pct,
             rainfall_mm, wind_speed_ms, visibility_10m, solar_radiation_mj,
             snowfall_cm, rain_prev_3h):
    """Validate the inputs and return one row in the fitted model's feature order."""
    selected_date = parse_date(day)
    if holiday not in {"No Holiday", "Holiday"}:
        raise ValueError("Select a holiday status.")
    weekday = selected_date.weekday()
    season = SEASON_BY_MONTH[selected_date.month]

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
    return int(np.round(max(0.0, float(model.predict(row)[0])))), row.iloc[0]


def result_card(value=None, detail=None, notes=(), error=None):
    if error:
        shown, detail = "—", error
    else:
        shown = "—" if value is None else f"{value:,}"
        detail = detail or "Set the conditions and select Predict demand."
    notes_html = "".join(f"<li>{escape(n)}</li>" for n in notes)
    return (
        f'<div class="bike-result{" bike-error" if error else ""}">'
        '<div class="label">Estimated rentals</div>'
        f'<div class="value">{shown}</div>'
        '<div class="unit">bikes/hour</div>'
        f'<div class="detail">{escape(detail)}</div>'
        + (f'<ul class="bike-warn">{notes_html}</ul>' if notes_html else "")
        + '<div class="line"></div>'
        '</div>'
    )


def predict(*inputs):
    """Always redraw the card, so an input error never leaves an old estimate on screen."""
    try:
        value, row = predict_count(*inputs)
    except (ValueError, FileNotFoundError) as exc:
        return result_card(error=str(exc))
    when = (f"{DAY_NAMES[row['day_of_week']]} {row['hour']:02d}:00 · {row['seasons']}"
            + (" · public holiday" if row["is_holiday"] else ""))
    metrics = load_metrics()
    accuracy = (f" On unseen test days the model was off by {metrics['test_mae']:.0f} bikes/hour"
                " on average (MAE)." if metrics else "")
    detail = f"Expected rentals across Seoul for {when}.{accuracy}"
    return result_card(value, detail, range_warnings(row))


# date, hour, holiday, temperature, humidity, rainfall, wind, visibility, solar, snowfall, previous 3h rain
EXAMPLES = [
    ["2018-06-13", 8, "No Holiday", 21, 65, 0, 1.5, 2000, 0.9, 0, 0],
    ["2018-06-16", 18, "No Holiday", 24, 60, 0, 1.5, 2000, 0.5, 0, 0],
    ["2018-07-09", 22, "No Holiday", 20, 95, 3, 2, 300, 0, 0, 4],
    ["2018-08-15", 14, "Holiday", 30, 60, 0, 2, 2000, 2.5, 0, 0],
    ["2018-01-17", 15, "No Holiday", -3, 40, 0, 1.5, 2000, 0.5, 0, 0],
]
EXAMPLE_LABELS = [
    "Summer weekday · 08:00 commute",
    "Summer Saturday · 18:00 evening",
    "Rainy summer night · 22:00",
    "Liberation Day holiday · 14:00",
    "Cold winter afternoon · 15:00",
]


# Inline Seoul scene (no external images): skyline, N Seoul Tower on Namsan, the Han River
# with a bridge, and a Ttareungyi bike riding along the riverside path.
_WAVES = "M-80 128 q20 -5 40 0" + " t40 0" * 32
_ARCHES = "M610 121" + " q30 -14 60 0" * 8
_WHEEL = ('<g class="bike-wheel"><circle cx="{x}" cy="100" r="11"/>'
          '<path d="M{x} 89 V111 M{l} 100 H{r}" stroke-width="1.6"/></g>')
SEOUL_SVG = f"""
      <svg class="bike-skyline" viewBox="0 0 1100 150" preserveAspectRatio="xMidYMax slice" role="img" aria-label="Seoul skyline with N Seoul Tower, the Han River and a bike on the riverside path" xmlns="http://www.w3.org/2000/svg">
        <path d="M560 112 Q630 60 700 62 Q780 64 860 112 Z" fill="#e8fff7" opacity=".2"/>
        <g fill="#f4fffb" opacity=".88">
          <rect x="697" y="30" width="6" height="34" rx="2"/>
          <rect x="684" y="22" width="32" height="11" rx="5.5"/>
          <rect x="689" y="14" width="22" height="8" rx="4"/>
          <rect x="698.5" y="3" width="3" height="12" rx="1.5"/>
        </g>
        <circle class="bike-tower-light" cx="700" cy="3" r="2.6" fill="#ff8a7a"/>
        <path d="M0 112 V86 H40 V64 H78 V112 H96 V72 H130 V50 H150 V112 H176 V80 H214 V112 H238 V60 H270 V112 H300 V76 H336 V112 H362 V56 H392 V112 H420 V82 H456 V112 H484 V68 H516 V112 H560 V94 H600 V112 H640 V90 H676 V112 H724 V92 H760 V112 H800 V88 H840 V112 H868 V62 H902 V112 H930 V74 H966 V112 H992 V58 H1026 V112 H1050 V80 H1100 V112 Z" fill="#e8fff7" opacity=".4"/>
        <rect x="0" y="113" width="1100" height="37" fill="#b9ecff" opacity=".22"/>
        <path class="bike-wave" d="{_WAVES}" fill="none" stroke="#e6fbff" stroke-width="2" opacity=".55"/>
        <path class="bike-wave slow" d="{_WAVES.replace('M-80 128', 'M-60 140')}" fill="none" stroke="#e6fbff" stroke-width="2" opacity=".35"/>
        <path d="{_ARCHES} M610 121 H1090" fill="none" stroke="#f5fff9" stroke-width="3.5" stroke-linecap="round" opacity=".8"/>
        <path d="M0 112.5 H1100" stroke="#fff" stroke-width="3" opacity=".75"/>
        <g class="bike-rider"><g class="bike-bob" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
          {_WHEEL.format(x=0, l=-11, r=11)}{_WHEEL.format(x=46, l=35, r=57)}
          <path d="M0 100 L18 77 L30 100 L46 100 L29 77 H18 M22 68 H33 M29 77 L39 69"/>
        </g></g>
      </svg>"""


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
      {SEOUL_SVG}
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
            <div class="bike-step"><span class="num">1</span><strong>Choose a time</strong><span>Pick a date and hour; season and weekend are filled in for you.</span></div>
            <div class="bike-step"><span class="num">2</span><strong>Add the weather</strong><span>Enter conditions measured in that hour.</span></div>
            <div class="bike-step"><span class="num">3</span><strong>See the estimate</strong><span>The citywide estimate in bikes/hour updates as you change any value.</span></div>
          </div>""")
        with gr.Row(elem_classes="main-grid"):
            with gr.Column(scale=3, min_width=320, elem_classes="bike-panel"):
                gr.HTML('<h3>Time</h3><p>Pick the hour you want to estimate.</p>')
                with gr.Row():
                    day = gr.DateTime(label="📅 Date", value="2018-06-13", include_time=False, type="string",
                                      elem_id="bike-date", info="Click to choose from the calendar.")
                    holiday = gr.Dropdown(["No Holiday", "Holiday"], label="Public holiday", value="No Holiday",
                                          info="Pre-selected for holidays in the dataset.")
                hour = gr.Slider(0, 23, value=18, step=1, label="🕒 Hour of day (0–23)",
                                 info="18 means 18:00–18:59.", elem_classes="bike-slider")
                date_info = gr.Markdown(describe_date("2018-06-13"), elem_classes="bike-subtitle")
                gr.HTML('<h3 style="margin-top:24px">Weather</h3><p>Enter the conditions for that hour.</p>')
                with gr.Row():
                    temp = weather_slider("temp")
                    humidity = weather_slider("humidity")
                with gr.Row():
                    rainfall = weather_slider("rainfall")
                    wind = weather_slider("wind")
                with gr.Accordion("Additional weather used by the model", open=False):
                    gr.Markdown("These measurements help the model match its training data. Dew point is calculated from temperature and humidity.", elem_classes="bike-subtitle")
                    with gr.Row():
                        visibility = weather_slider("visibility")
                        solar = weather_slider("solar")
                    with gr.Row():
                        snowfall = weather_slider("snowfall")
                        rain_prev = weather_slider("rain_prev")
                button = gr.Button("Predict demand →", variant="primary", elem_classes="bike-predict")
            with gr.Column(scale=2, min_width=280):
                result = gr.HTML(result_card())
                gr.HTML('<p class="bike-note">This is a model estimate for all of Seoul, not an availability forecast for a specific station.</p>')

        inputs = [day, hour, holiday, temp, humidity, rainfall,
                  wind, visibility, solar, snowfall, rain_prev]
        day.change(fn=describe_date, inputs=day, outputs=date_info)
        day.change(fn=suggest_holiday, inputs=day, outputs=holiday)
        button.click(fn=predict, inputs=inputs, outputs=result)
        # Keep the estimate in step with the inputs; while a slider is dragged only the last value runs.
        gr.on(triggers=[component.change for component in inputs], fn=predict, inputs=inputs,
              outputs=result, trigger_mode="always_last", show_progress="hidden")
        page.load(fn=predict, inputs=inputs, outputs=result)
        page.load(fn=None, js=DATE_PICKER_JS)
        gr.HTML('<h2 class="bike-heading">Try an example</h2><p class="bike-note">Select a scenario to load its conditions and prediction.</p>')
        with gr.Group(elem_classes="bike-examples"):
            gr.Examples(examples=EXAMPLES, example_labels=EXAMPLE_LABELS, inputs=inputs,
                        outputs=result, fn=predict, cache_examples=False, run_on_click=True)
        with gr.Accordion("About the model and limitations", open=False):
            about_text = (
                "**Model:** HistGradientBoostingRegressor trained on the Seoul Bike Sharing Demand dataset. "
                "The pipeline includes median imputation, scaling, encoding and feature selection. "
                + score_note
                + "The data covers one year in one city and uses measured weather. Accuracy with weather forecasts or in another year is untested. "
                "Closed-system hours and specified storm dates were excluded from training and the reported test score. "
                "Rainy hours had higher errors (test MAE about 143 bikes/hour). "
                "The date determines month, day of week, weekend and season, the same way as in the dataset. "
                "Inputs outside the training range are still predicted but flagged, because the model has not seen such conditions."
            )
            gr.Markdown(about_text)
        gr.HTML('''<div class="bike-footer">Data: <a href="https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand" target="_blank" rel="noopener">UCI Seoul Bike Sharing Demand</a> · <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noopener">CC BY 4.0</a></div>''')
    return page


demo = build_ui()


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
