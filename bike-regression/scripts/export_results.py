"""เก็บค่าจาก kernel ของ 02_modeling เพื่อสอบทานรายงานจากรอบเดียวกัน."""
import hashlib
import json
import re
from pathlib import Path


def save_evidence(ns):
    root = ns["ROOT"]
    out = root / "verification"
    out.mkdir(exist_ok=True)
    tables = {}
    table_names = ["scope_table", "fold_table", "rain_feature_cv", "rain_fold_cv",
                   "k_table", "lasso_table", "fs_summary", "fs_results", "fs_choices",
                   "pca_table", "baselines", "compare", "tuned", "clo4", "scope_sensitivity",
                   "storm_day_sensitivity", "by_hour", "worst", "scenarios"]
    for name in table_names:
        value = ns[name]
        value.to_csv(out / f"{name}.csv", encoding="utf-8-sig")
        tables[name] = json.loads(value.reset_index().to_json(orient="records", date_format="iso"))
    for name, value in [("season_errors", ns["group_err"]("seasons")),
                        ("day_errors", ns["group_err"]("day_type")),
                        ("rain_errors", ns["group_err"](ns["np"].where(ns["res"]["rainfall_mm"] > 0, "ฝนตก", "ไม่มีฝน")))]:
        value.to_csv(out / f"{name}.csv", encoding="utf-8-sig")
        tables[name] = json.loads(value.reset_index().to_json(orient="records"))
    payload = {"best_model": ns["best_name"], "features": ns["FEATURES"],
               "preprocessed_dimensions": ns["fs_dimensions"], "test_metrics": ns["test_scores"],
               "test_unclipped_metrics": {"MAE": float(ns["mean_absolute_error"](ns["y_test"], ns["raw_pred"]))},
               "negative_predictions": int((ns["raw_pred"] < 0).sum()),
               "pca_90": int(ns["n90"]), "pca_95": int(ns["n95"]),
               "pca_cumulative": ns["cum"].tolist(),
               "search_parameters": {k: {p: re.sub(r" at 0x[0-9A-Fa-f]+", "", str(v)) for p, v in s.best_params_.items()} for k, s in ns["searches"].items()},
               "tables": tables,
               "sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [root / "config.py", root / "requirements.txt", *sorted((root / "figures").glob("*.png"))]}}
    (out / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("บันทึก verification/results.json และตาราง CSV จาก kernel รอบนี้แล้ว")
