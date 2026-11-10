"""
Streamlit UI for the Tamil Nadu Climate-Responsive Seed & Irrigation
Decision Support System.

Wraps the CropIrrigationAdvisor (CatBoost crop model + NASA POWER climate
data + FAO-56 irrigation calculation) behind a simple form: pick a
district, month, soil test values, and whether the field has irrigation
access — get back ranked crop recommendations and an irrigation estimate.

This file is fully self-contained (the CropIrrigationAdvisor class and Kc
table are embedded directly) so it has no dependency on the other project
scripts being present.

Requires, in the same folder as this script:
    - outputs/catboost_crop_model.cbm
    - nasa_power_tamilnadu/tamilnadu_climate_monthly_agg_2015_2024.csv
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier

MODEL_PATH = Path("outputs/catboost_crop_model.cbm")
CLIMATE_MONTHLY_PATH = Path("nasa_power_tamilnadu/tamilnadu_climate_monthly_agg_2015_2024.csv")

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}

CROP_KC_TABLE = {
    "rice":        {"initial": 1.05, "development": 1.10, "mid_season": 1.20, "late_season": 0.90},
    "cotton":      {"initial": 0.35, "development": 0.70, "mid_season": 1.18, "late_season": 0.60},
    "maize":       {"initial": 0.30, "development": 0.70, "mid_season": 1.20, "late_season": 0.60},
    "chickpea":    {"initial": 0.40, "development": 0.70, "mid_season": 1.00, "late_season": 0.35},
    "banana":      {"initial": 0.50, "development": 0.80, "mid_season": 1.10, "late_season": 1.00},
    "watermelon":  {"initial": 0.40, "development": 0.70, "mid_season": 1.00, "late_season": 0.75},
    "grapes":      {"initial": 0.30, "development": 0.60, "mid_season": 0.80, "late_season": 0.55},
    "lentil":      {"initial": 0.40, "development": 0.70, "mid_season": 1.10, "late_season": 0.30},
    "kidneybeans": {"initial": 0.40, "development": 0.70, "mid_season": 1.15, "late_season": 0.35},
    "muskmelon":   {"initial": 0.50, "development": 0.75, "mid_season": 1.05, "late_season": 0.75},
    "apple":       {"initial": 0.45, "development": 0.70, "mid_season": 0.95, "late_season": 0.70},
    "orange":      {"initial": 0.70, "development": 0.65, "mid_season": 0.65, "late_season": 0.70},
    "papaya":      {"initial": 0.50, "development": 0.75, "mid_season": 1.00, "late_season": 0.75},
    "mango":       {"initial": 0.60, "development": 0.75, "mid_season": 0.75, "late_season": 0.70},
    "coconut":     {"initial": 0.90, "development": 0.90, "mid_season": 0.90, "late_season": 0.90},
    "coffee":      {"initial": 0.90, "development": 0.95, "mid_season": 1.05, "late_season": 0.95},
    "pigeonpeas":  {"initial": 0.40, "development": 0.75, "mid_season": 1.15, "late_season": 0.65},
    "mothbeans":   {"initial": 0.40, "development": 0.70, "mid_season": 1.00, "late_season": 0.35},
    "mungbean":    {"initial": 0.40, "development": 0.70, "mid_season": 1.05, "late_season": 0.35},
    "blackgram":   {"initial": 0.40, "development": 0.70, "mid_season": 1.05, "late_season": 0.35},
    "pomegranate": {"initial": 0.50, "development": 0.70, "mid_season": 0.90, "late_season": 0.65},
    "jute":        {"initial": 0.40, "development": 0.75, "mid_season": 1.10, "late_season": 0.55},
}
APPROXIMATE_KC_CROPS = {"pigeonpeas", "mothbeans", "mungbean", "blackgram", "pomegranate", "jute"}
DEFAULT_KC_MID = 0.85


class CropIrrigationAdvisor:
    IRRIGATED_HUMIDITY_BOOST = 15.0
    IRRIGATED_HUMIDITY_CAP = 92.0
    IRRIGATED_EFFECTIVE_RAINFALL_DEFAULT = 230.0

    def __init__(self, model_path: Path, climate_path: Path):
        self.model = CatBoostClassifier()
        self.model.load_model(str(model_path))
        self.climate = pd.read_csv(climate_path)
        self.climate["year_month"] = pd.PeriodIndex(self.climate["year_month"], freq="M")
        self.districts = sorted(self.climate["district"].unique())

    def get_climate_normals(self, district: str, month: int, season_months: int = 4) -> dict:
        if district not in self.districts:
            raise ValueError(f"Unknown district '{district}'.")
        season_month_list = [((month - offset - 1) % 12) + 1 for offset in range(season_months)]
        district_data = self.climate[self.climate["district"] == district]
        single_month = district_data[district_data["year_month"].dt.month == month]
        if single_month.empty:
            raise ValueError(f"No climate data for {district}, month {month}")
        season_data = district_data[district_data["year_month"].dt.month.isin(season_month_list)].copy()
        season_data["year"] = season_data["year_month"].dt.year
        rainfall_by_year = season_data.groupby("year")["rainfall_total"].sum()
        return {
            "temperature": single_month["temp_mean"].mean(),
            "humidity": single_month["humidity_mean"].mean(),
            "rainfall": rainfall_by_year.mean(),
            "rainfall_month": single_month["rainfall_total"].mean(),
            "et0_total": single_month["et0_total"].mean(),
        }

    def recommend_crop(self, district, month, N, P, K, ph, top_k=5, irrigation_access: bool = False):
        climate = self.get_climate_normals(district, month)
        humidity_to_use = climate["humidity"]
        rainfall_to_use = climate["rainfall"]
        if irrigation_access:
            humidity_to_use = min(climate["humidity"] + self.IRRIGATED_HUMIDITY_BOOST, self.IRRIGATED_HUMIDITY_CAP)
            rainfall_to_use = self.IRRIGATED_EFFECTIVE_RAINFALL_DEFAULT
        features = pd.DataFrame([{
            "N": N, "P": P, "K": K, "temperature": climate["temperature"],
            "humidity": humidity_to_use, "ph": ph, "rainfall": rainfall_to_use,
        }])[FEATURE_COLS]
        probs = self.model.predict_proba(features)[0]
        classes = self.model.classes_
        ranking = (pd.DataFrame({"crop": classes, "probability": probs})
                   .sort_values("probability", ascending=False).head(top_k).reset_index(drop=True))
        climate_used = {**climate, "humidity_used": humidity_to_use, "rainfall_used": rainfall_to_use}
        return ranking, climate_used

    def estimate_irrigation_need(self, crop: str, district: str, month: int, growth_stage: str) -> dict:
        climate = self.get_climate_normals(district, month)
        crop_key = crop.lower()
        if crop_key in CROP_KC_TABLE:
            kc = CROP_KC_TABLE[crop_key].get(growth_stage, DEFAULT_KC_MID)
            is_approx = crop_key in APPROXIMATE_KC_CROPS
        else:
            kc = DEFAULT_KC_MID
            is_approx = True
        crop_water_need_mm = climate["et0_total"] * kc
        net_irrigation_mm = max(crop_water_need_mm - climate["rainfall_month"], 0)
        return {
            "kc_used": kc, "kc_is_approximate": is_approx,
            "et0_total_mm": round(climate["et0_total"], 1),
            "rainfall_month_mm": round(climate["rainfall_month"], 1),
            "crop_water_need_mm": round(crop_water_need_mm, 1),
            "net_irrigation_need_mm": round(net_irrigation_mm, 1),
        }


@st.cache_resource
def load_advisor():
    return CropIrrigationAdvisor(MODEL_PATH, CLIMATE_MONTHLY_PATH)


st.set_page_config(page_title="TN Crop & Irrigation Advisor", page_icon="🌾", layout="centered")
st.title("🌾 Climate-Responsive Seed & Irrigation Advisor")
st.caption("Tamil Nadu — CatBoost crop recommendation + NASA POWER climate data + FAO-56 irrigation estimate")

if not MODEL_PATH.exists() or not CLIMATE_MONTHLY_PATH.exists():
    st.error(
        "Required files not found. Make sure this script sits alongside:\n\n"
        f"- `{MODEL_PATH}`\n- `{CLIMATE_MONTHLY_PATH}`"
    )
    st.stop()

advisor = load_advisor()

with st.form("advisor_form"):
    col1, col2 = st.columns(2)
    with col1:
        district = st.selectbox("District", advisor.districts)
        month_num = st.selectbox("Month", list(MONTH_NAMES.keys()), format_func=lambda m: MONTH_NAMES[m])
        irrigation_access = st.checkbox(
            "Field has irrigation access (canal / flood-fed)",
            help="Raises effective humidity and water availability toward irrigated conditions.",
        )
    with col2:
        N = st.number_input("Nitrogen (N) — soil ratio", min_value=0, max_value=140, value=90)
        P = st.number_input("Phosphorous (P) — soil ratio", min_value=0, max_value=145, value=40)
        K = st.number_input("Potassium (K) — soil ratio", min_value=0, max_value=205, value=40)
        ph = st.number_input("Soil pH", min_value=3.5, max_value=9.5, value=6.5, step=0.1)
    growth_stage = st.selectbox("Growth stage (for irrigation estimate)",
                                 ["initial", "development", "mid_season", "late_season"], index=2)
    submitted = st.form_submit_button("Get Recommendation", type="primary")

if submitted:
    try:
        ranking, climate_used = advisor.recommend_crop(
            district, month_num, N, P, K, ph, top_k=5, irrigation_access=irrigation_access
        )
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.subheader("Recommended Crops")
    st.bar_chart(ranking.set_index("crop")["probability"])
    st.dataframe(
        ranking.assign(probability=lambda d: (d["probability"] * 100).round(1).astype(str) + "%"),
        hide_index=True, use_container_width=True,
    )

    top_crop = ranking.iloc[0]["crop"]
    top_prob = ranking.iloc[0]["probability"]
    if top_prob < 0.30:
        st.warning(f"The top recommendation ({top_crop}) has relatively low confidence ({top_prob:.1%}).")

    with st.expander("Climate values used for this recommendation"):
        st.write(
            f"- Temperature: **{climate_used['temperature']:.1f} °C**\n"
            f"- Humidity used: **{climate_used['humidity_used']:.1f}%**"
            + (" (boosted for irrigation access)" if irrigation_access else " (ambient)")
            + f"\n- Rainfall used (season-cumulative): **{climate_used['rainfall_used']:.1f} mm**"
            + f"\n- Reference evapotranspiration (ET0, this month): **{climate_used['et0_total']:.1f} mm**"
        )

    st.subheader(f"Irrigation Estimate — {top_crop.title()}")
    irrigation = advisor.estimate_irrigation_need(top_crop, district, month_num, growth_stage)
    m1, m2, m3 = st.columns(3)
    m1.metric("ET₀ this month", f"{irrigation['et0_total_mm']} mm")
    m2.metric("Rainfall this month", f"{irrigation['rainfall_month_mm']} mm")
    m3.metric("Net irrigation need", f"{irrigation['net_irrigation_need_mm']} mm")
    st.caption(
        f"Kc used: {irrigation['kc_used']} ({growth_stage.replace('_', ' ')} stage)"
        + (" — approximated." if irrigation["kc_is_approximate"] else " — FAO-56 Table 12 (Allen et al., 1998).")
    )
    st.caption("Net irrigation need = ET₀ × Kc − rainfall (FAO-56 water balance method). Decision-support estimate only.")
else:
    st.info("Fill in the form above and click **Get Recommendation** to see results.")
