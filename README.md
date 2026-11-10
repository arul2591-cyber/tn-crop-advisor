# TN Crop & Irrigation Advisor

Tamil Nadu Climate-Responsive Seed & Irrigation Decision Support System —
CatBoost crop recommendation + NASA POWER climate data + FAO-56 irrigation
estimate, packaged as a double-click Windows desktop app.

## Getting the .exe

Go to the **Actions** tab of this repository, open the latest **Build
Windows EXE** run, and download the `TN_Crop_Advisor-windows-exe` artifact
from the bottom of that run's page. Unzip it to get `TN_Crop_Advisor.exe`.

## IMPORTANT — climate data

`nasa_power_tamilnadu/tamilnadu_climate_monthly_agg_2015_2024.csv` in this
repo is currently a **placeholder** (synthetic values, correct shape only)
because the build environment that created this repo could not reach the
NASA POWER API. Replace it with your real file from
`fetch_nasa_power_tamilnadu.py`, commit, and push — the Actions workflow
will automatically rebuild the exe with your real data.

The model file `outputs/catboost_crop_model.cbm` IS real — trained on the
actual Kaggle Crop_recommendation.csv dataset (99.39% test accuracy,
bestIteration=277).

## Running it

Double-click `TN_Crop_Advisor.exe`. A console window opens briefly, then
your default browser opens automatically to `http://localhost:8501` with
the app running.
