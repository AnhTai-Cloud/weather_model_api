from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import os

import pandas as pd
import numpy as np
import joblib
import requests


# =========================
# CONFIG
# =========================
BUNDLE_PATH = "weather_ai_bundle.joblib"
HISTORY_CSV = "sensor_history.csv"
MIN_HISTORY = 24

# Vị trí mặc định: Hà Nội
LATITUDE = 21.0285
LONGITUDE = 105.8542
TIMEZONE = "Asia/Bangkok"


# =========================
# LOAD MODEL
# =========================
bundle = joblib.load(BUNDLE_PATH)

reg_model = bundle["reg_model"]
rain_model = bundle["rain_model"]
weather_model = bundle["weather_model"]
features = bundle["features"]

RAIN_THRESHOLD = bundle.get("rain_threshold", 0.35)

WEATHER_MAP = bundle.get(
    "weather_map",
    {
        0: "Nắng",
        1: "Âm u",
        2: "Mưa",
    }
)


# =========================
# APP
# =========================
app = FastAPI(title="Weather AI Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# INPUT MODELS
# =========================
class SensorRow(BaseModel):
    time: Optional[str] = None
    temperature: float
    humidity: float
    pressure: float
    light: float
    rain: float = 0


class IngestInput(BaseModel):
    time: Optional[str] = None
    temperature: float
    humidity: float
    pressure: float
    light: float
    rain: Optional[float] = 0
    gas: float = 0
    rain_raw: Optional[float] = None
    rain_state: Optional[str] = None
    gas_alarm: Optional[bool] = None
    rack_state: Optional[str] = None
    door_state: Optional[str] = None
    mode: Optional[str] = None
    period: Optional[str] = None


class PredictRequest(BaseModel):
    history: List[SensorRow] = Field(default_factory=list)


def dump_model(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj.dict()


# =========================
# HISTORY FUNCTIONS
# =========================
def normalize_sensor_row(data: IngestInput):
    row = dump_model(data)

    if row.get("time") is None:
        row["time"] = datetime.now().isoformat()

    rain_state = str(row.get("rain_state") or "").upper()

    if row.get("rain") is None:
        row["rain"] = 1 if rain_state in ["RAIN", "RAINING", "YES", "TRUE", "1"] else 0

    return {
        "time": row.get("time"),
        "temperature": float(row.get("temperature", 0)),
        "humidity": float(row.get("humidity", 0)),
        "pressure": float(row.get("pressure", 0)),
        "light": float(row.get("light", 0)),
        "rain": float(row.get("rain", 0)),
        "gas": float(row.get("gas", 0)),
        "rain_raw": row.get("rain_raw", None),
        "rain_state": row.get("rain_state", None),
        "gas_alarm": row.get("gas_alarm", None),
        "rack_state": row.get("rack_state", None),
        "door_state": row.get("door_state", None),
        "mode": row.get("mode", None),
        "period": row.get("period", None),
    }


def append_history(row):
    new_df = pd.DataFrame([row])

    if os.path.exists(HISTORY_CSV):
        old_df = pd.read_csv(HISTORY_CSV)
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df = new_df

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df.sort_values("time")
    df = df.drop_duplicates(subset=["time"], keep="last")
    df = df.tail(1000)

    df.to_csv(HISTORY_CSV, index=False)

    return df


def read_history(limit=24):
    if not os.path.exists(HISTORY_CSV):
        return pd.DataFrame()

    df = pd.read_csv(HISTORY_CSV)

    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df.sort_values("time")

    return df.tail(limit)


# =========================
# OPEN-METEO HISTORY
# =========================
def fetch_open_meteo_history(hours=72):
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timezone": TIMEZONE,
        "past_hours": hours,
        "forecast_hours": 0,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "shortwave_radiation",
            "rain"
        ])
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()

    data = response.json()
    hourly = data.get("hourly", {})

    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    hums = hourly.get("relative_humidity_2m", [])
    pressures = hourly.get("surface_pressure", [])
    radiation = hourly.get("shortwave_radiation", [])
    rains = hourly.get("rain", [])

    rows = []

    for i in range(len(times)):
        rain_value = float(rains[i] or 0)

        rows.append({
            "time": times[i],
            "temperature": float(temps[i] or 0),
            "humidity": float(hums[i] or 0),
            "pressure": float(pressures[i] or 0),
            "light": float(radiation[i] or 0),
            "rain": 1.0 if rain_value > 0 else 0.0,
            "source": "weather",
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df.sort_values("time")

    return df.tail(hours)


def prepare_sensor_history(input_history):
    sensor_df = pd.DataFrame(input_history)

    if sensor_df.empty:
        return sensor_df

    needed_cols = ["time", "temperature", "humidity", "pressure", "light", "rain"]

    for col in needed_cols:
        if col not in sensor_df.columns:
            if col == "time":
                sensor_df[col] = datetime.now().isoformat()
            else:
                sensor_df[col] = 0

    sensor_df = sensor_df[needed_cols]
    sensor_df["time"] = pd.to_datetime(sensor_df["time"], errors="coerce")
    sensor_df = sensor_df.dropna(subset=["time"])

    for col in ["temperature", "humidity", "pressure", "light", "rain"]:
        sensor_df[col] = pd.to_numeric(sensor_df[col], errors="coerce").fillna(0)

    sensor_df = sensor_df.sort_values("time")
    sensor_df["time_hour"] = sensor_df["time"].dt.floor("h")
    sensor_df = sensor_df.drop_duplicates(subset=["time_hour"], keep="last")
    sensor_df = sensor_df.drop(columns=["time_hour"])
    sensor_df["source"] = "sensor"

    return sensor_df


def complete_history_with_weather(input_history, required_samples=24):
    sensor_df = prepare_sensor_history(input_history)

    if len(sensor_df) >= required_samples:
        final_df = sensor_df.sort_values("time").tail(required_samples)

        return final_df, {
            "sensor_samples": int(len(sensor_df)),
            "external_samples": 0,
            "final_samples": int(len(final_df)),
        }

    try:
        weather_df = fetch_open_meteo_history(hours=72)
        weather_error = None
    except Exception as e:
        weather_df = pd.DataFrame()
        weather_error = str(e)

    if weather_df.empty:
        final_df = sensor_df.sort_values("time").tail(required_samples)

        return final_df, {
            "sensor_samples": int(len(sensor_df)),
            "external_samples": 0,
            "final_samples": int(len(final_df)),
            "weather_error": weather_error or "Không lấy được dữ liệu thời tiết.",
        }

    if not sensor_df.empty:
        sensor_df["time_hour"] = sensor_df["time"].dt.floor("h")
    else:
        sensor_df = pd.DataFrame(
            columns=["time", "temperature", "humidity", "pressure", "light", "rain", "source", "time_hour"]
        )

    weather_df["time_hour"] = weather_df["time"].dt.floor("h")

    sensor_hours = set(sensor_df["time_hour"].tolist()) if not sensor_df.empty else set()
    weather_fill = weather_df[~weather_df["time_hour"].isin(sensor_hours)].copy()

    combined = pd.concat(
        [
            weather_fill[["time", "temperature", "humidity", "pressure", "light", "rain", "source"]],
            sensor_df[["time", "temperature", "humidity", "pressure", "light", "rain", "source"]],
        ],
        ignore_index=True
    )

    combined["time"] = pd.to_datetime(combined["time"], errors="coerce")
    combined = combined.dropna(subset=["time"])
    combined["priority"] = combined["source"].apply(lambda x: 1 if x == "sensor" else 0)
    combined["time_hour"] = combined["time"].dt.floor("h")
    combined = combined.sort_values(["time_hour", "priority", "time"])
    combined = combined.drop_duplicates(subset=["time_hour"], keep="last")

    final_df = combined.sort_values("time").tail(required_samples)
    external_count = int((final_df["source"] == "weather").sum()) if "source" in final_df.columns else 0

    return final_df, {
        "sensor_samples": int(len(sensor_df)),
        "external_samples": external_count,
        "final_samples": int(len(final_df)),
        "weather_error": weather_error,
    }


# =========================
# FORECAST API
# =========================
def weather_code_to_label(code, rain_probability=0, rain_sum=0):
    code = int(code)
    rain_probability = float(rain_probability or 0)
    rain_sum = float(rain_sum or 0)

    if code == 0:
        return "Trời quang"

    if code in [1, 2]:
        return "Ít mây"

    if code == 3:
        return "Âm u"

    if code in [45, 48]:
        return "Sương mù"

    if code in [51, 53, 55, 56, 57]:
        return "Mưa phùn"

    if code in [61, 63, 65, 66, 67]:
        if rain_probability < 20 and rain_sum < 1:
            return "Ít mây"
        return "Mưa"

    if code in [80, 81, 82]:
        if rain_probability < 20 and rain_sum < 1:
            return "Ít mây"
        return "Mưa rào"

    if code in [95, 96, 99]:
        if rain_probability >= 60 or rain_sum >= 5:
            return "Dông"
        if rain_probability >= 25 or rain_sum >= 1:
            return "Mưa rào"
        return "Ít mây"

    return "Không rõ"

def weather_code_to_icon(code, rain_probability=0, rain_sum=0):
    code = int(code)
    rain_probability = float(rain_probability or 0)
    rain_sum = float(rain_sum or 0)

    if code == 0:
        return "sunny"

    if code in [1, 2]:
        return "partly_cloudy"

    if code == 3:
        return "cloudy"

    if code in [45, 48]:
        return "fog"

    if code in [51, 53, 55, 56, 57]:
        return "drizzle"

    if code in [61, 63, 65, 66, 67]:
        if rain_probability >= 50 or rain_sum >= 2:
            return "rain"
        return "partly_cloudy"

    if code in [80, 81, 82]:
        if rain_probability >= 50 or rain_sum >= 2:
            return "rain"
        return "partly_cloudy"

    if code in [95, 96, 99]:
        if rain_probability >= 60 or rain_sum >= 5:
            return "thunderstorm"
        if rain_probability >= 25 or rain_sum >= 1:
            return "rain"
        return "partly_cloudy"

    return "partly_cloudy"

def day_name_vi(date_text, index):
    if index == 0:
        return "Hôm nay"

    dt = pd.to_datetime(date_text)

    names = {
        0: "Thứ 2",
        1: "Thứ 3",
        2: "Thứ 4",
        3: "Thứ 5",
        4: "Thứ 6",
        5: "Thứ 7",
        6: "CN",
    }

    return names.get(dt.weekday(), "--")


def fetch_open_meteo_forecast(days=5):
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timezone": TIMEZONE,
        "forecast_days": days,
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "weather_code",
            "surface_pressure",
            "rain"
        ]),
        "daily": ",".join([
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max",
            "rain_sum"
        ])
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()

    data = response.json()

    current = data.get("current", {})
    daily = data.get("daily", {})

    dates = daily.get("time", [])
    codes = daily.get("weather_code", [])
    temp_max = daily.get("temperature_2m_max", [])
    temp_min = daily.get("temperature_2m_min", [])
    rain_prob = daily.get("precipitation_probability_max", [])
    rain_sum = daily.get("rain_sum", [])

    daily_result = []

    for i in range(len(dates)):
        code = int(codes[i] or 0)
        rain_probability_value = float(rain_prob[i] or 0)
        rain_sum_value = float(rain_sum[i] or 0)
    
        weather_label = weather_code_to_label(
            code,
            rain_probability=rain_probability_value,
            rain_sum=rain_sum_value,
        )
    
        weather_icon = weather_code_to_icon(
            code,
            rain_probability=rain_probability_value,
            rain_sum=rain_sum_value,
        )
    
        daily_result.append({
            "date": dates[i],
            "day_name": day_name_vi(dates[i], i),
            "weather_code": code,
            "weather": weather_label,
            "icon": weather_icon,
            "temp_max": float(temp_max[i] or 0),
            "temp_min": float(temp_min[i] or 0),
            "rain_probability": rain_probability_value,
            "rain_sum": rain_sum_value,
        })

        current_code = int(current.get("weather_code") or 0)
        current_rain = float(current.get("rain") or 0)
    
    current_weather_label = weather_code_to_label(
        current_code,
        rain_probability=0,
        rain_sum=current_rain,
    )
    
    current_weather_icon = weather_code_to_icon(
        current_code,
        rain_probability=0,
        rain_sum=current_rain,
    )
    
    return {
        "ok": True,
        "location": "Hanoi, Vietnam",
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timezone": TIMEZONE,
        "current": {
            "time": current.get("time"),
            "temperature": float(current.get("temperature_2m") or 0),
            "humidity": float(current.get("relative_humidity_2m") or 0),
            "pressure": float(current.get("surface_pressure") or 0),
            "rain": current_rain,
            "weather_code": current_code,
            "weather": current_weather_label,
            "icon": current_weather_icon,
        },
        "daily": daily_result,
    }


# =========================
# FEATURE ENGINEERING
# =========================
def build_features(history_rows):
    df = pd.DataFrame([dump_model(row) for row in history_rows])

    if "time" not in df.columns or df["time"].isna().all():
        df["time"] = pd.date_range(
            end=pd.Timestamp.now(),
            periods=len(df),
            freq="h"
        )
    else:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["time"] = df["time"].ffill().fillna(pd.Timestamp.now())

    df = df.sort_values("time").reset_index(drop=True)

    base_cols = ["temperature", "humidity", "pressure", "light", "rain"]

    for col in base_cols:
        if col not in df.columns:
            df[col] = 0

    lag_steps = [1, 2, 3, 6, 12, 24]
    rolling_windows = [3, 6, 12, 24]

    feature_blocks = []

    lag_features = {}
    for col in base_cols:
        for lag in lag_steps:
            lag_features[f"{col}_lag_{lag}"] = df[col].shift(lag)

    feature_blocks.append(pd.DataFrame(lag_features, index=df.index))

    rolling_features = {}
    for col in base_cols:
        for window in rolling_windows:
            rolling_features[f"{col}_roll_mean_{window}"] = df[col].rolling(window).mean()
            rolling_features[f"{col}_roll_max_{window}"] = df[col].rolling(window).max()
            rolling_features[f"{col}_roll_min_{window}"] = df[col].rolling(window).min()

    for window in rolling_windows:
        rolling_features[f"rain_roll_sum_{window}"] = df["rain"].rolling(window).sum()

    feature_blocks.append(pd.DataFrame(rolling_features, index=df.index))

    diff_features = {}
    for col in base_cols:
        for lag in [1, 3, 6, 12, 24]:
            diff_features[f"{col}_diff_{lag}"] = df[col] - df[col].shift(lag)

    feature_blocks.append(pd.DataFrame(diff_features, index=df.index))

    df = pd.concat([df] + feature_blocks, axis=1)

    df["pressure_drop_3h"] = (df.get("pressure_diff_3", 0) < 0).astype(int)
    df["pressure_drop_6h"] = (df.get("pressure_diff_6", 0) < 0).astype(int)
    df["humidity_rise_3h"] = (df.get("humidity_diff_3", 0) > 0).astype(int)
    df["humidity_rise_6h"] = (df.get("humidity_diff_6", 0) > 0).astype(int)
    df["light_drop_3h"] = (df.get("light_diff_3", 0) < 0).astype(int)
    df["light_drop_6h"] = (df.get("light_diff_6", 0) < 0).astype(int)

    df["bad_weather_signal_3h"] = (
        df["pressure_drop_3h"] +
        df["humidity_rise_3h"] +
        df["light_drop_3h"]
    )

    df["bad_weather_signal_6h"] = (
        df["pressure_drop_6h"] +
        df["humidity_rise_6h"] +
        df["light_drop_6h"]
    )

    for col in features:
        if col not in df.columns:
            df[col] = 0

    df = df.ffill().bfill().fillna(0)

    X_latest = df[features].tail(1)

    return X_latest


def decide_command(weather_label, rain_probability):
    if rain_probability >= RAIN_THRESHOLD:
        return "CLOSE"

    if weather_label == "Mưa":
        return "CLOSE"

    if weather_label == "Nắng":
        return "OPEN"

    return "CLOSE"


# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {
        "status": "Weather AI API is running",
        "model": BUNDLE_PATH,
        "history_file": HISTORY_CSV,
        "min_history": MIN_HISTORY,
        "forecast_endpoint": "/forecast?days=5",
        "predict_endpoint": "/predict",
    }


@app.post("/ingest")
def ingest(data: IngestInput):
    row = normalize_sensor_row(data)
    df = append_history(row)

    return {
        "ok": True,
        "history_count": len(df),
        "saved": row,
    }


@app.get("/history")
def get_history(limit: int = 24):
    df = read_history(limit)

    if df.empty:
        return {
            "ok": False,
            "error": "Chưa có dữ liệu lịch sử.",
            "history": [],
        }

    history = []

    for _, row in df.iterrows():
        history.append({
            "time": row["time"].isoformat(),
            "temperature": float(row["temperature"]),
            "humidity": float(row["humidity"]),
            "pressure": float(row["pressure"]),
            "light": float(row["light"]),
            "rain": float(row["rain"]),
        })

    return {
        "ok": True,
        "count": len(history),
        "history": history,
    }


@app.get("/meteo")
def get_meteo(hours: int = 24):
    if hours < 1:
        hours = 1
    if hours > 72:
        hours = 72

    df = fetch_open_meteo_history(hours=hours)

    data = []

    for _, row in df.iterrows():
        data.append({
            "time": row["time"].isoformat() if hasattr(row["time"], "isoformat") else str(row["time"]),
            "temperature": float(row["temperature"]),
            "humidity": float(row["humidity"]),
            "pressure": float(row["pressure"]),
            "light": float(row["light"]),
            "rain": float(row["rain"]),
            "source": row.get("source", "weather"),
        })

    return {
        "ok": True,
        "count": len(data),
        "history": data,
    }


@app.get("/forecast")
def get_forecast(days: int = 5):
    if days < 1:
        days = 1

    if days > 7:
        days = 7

    return fetch_open_meteo_forecast(days=days)


@app.post("/predict")
def predict(req: PredictRequest):
    input_history = [dump_model(row) for row in req.history]

    if len(input_history) < MIN_HISTORY:
        old_df = read_history(MIN_HISTORY)

        if not old_df.empty:
            old_history = old_df[
                ["time", "temperature", "humidity", "pressure", "light", "rain"]
            ].to_dict(orient="records")

            input_history = old_history + input_history

    hist_df, source_info = complete_history_with_weather(
        input_history,
        required_samples=MIN_HISTORY
    )

    hist_df = hist_df.sort_values("time").tail(MIN_HISTORY)

    if len(hist_df) < MIN_HISTORY:
        return {
            "ok": False,
            "error": "Chưa đủ dữ liệu lịch sử để dự đoán.",
            "required_samples": MIN_HISTORY,
            "received_samples": len(hist_df),
            "source_info": source_info,
        }

    history_rows = []

    for _, row in hist_df.iterrows():
        history_rows.append(
            SensorRow(
                time=row["time"].isoformat() if hasattr(row["time"], "isoformat") else str(row["time"]),
                temperature=float(row["temperature"]),
                humidity=float(row["humidity"]),
                pressure=float(row["pressure"]),
                light=float(row["light"]),
                rain=float(row["rain"]),
            )
        )

    X_latest = build_features(history_rows)

    reg_pred = reg_model.predict(X_latest)[0]

    rain_probability = float(rain_model.predict_proba(X_latest)[0][1])
    rain_pred = int(rain_probability >= RAIN_THRESHOLD)

    weather_pred_id = int(weather_model.predict(X_latest)[0])
    weather_proba = weather_model.predict_proba(X_latest)[0].tolist()
    weather_label = WEATHER_MAP.get(weather_pred_id, "Không rõ")

    command = decide_command(weather_label, rain_probability)

    history_used = []

    for _, row in hist_df.iterrows():
        history_used.append({
            "time": row["time"].isoformat() if hasattr(row["time"], "isoformat") else str(row["time"]),
            "temperature": float(row["temperature"]),
            "humidity": float(row["humidity"]),
            "pressure": float(row["pressure"]),
            "light": float(row["light"]),
            "rain": float(row["rain"]),
            "source": row.get("source", "unknown"),
        })

    return {
        "ok": True,
        "future_step": 3,
        "prediction": weather_label,
        "command": command,
        "rain_probability": rain_probability,
        "rain_pred": rain_pred,
        "weather_proba": {
            "Nắng": float(weather_proba[0]),
            "Âm u": float(weather_proba[1]),
            "Mưa": float(weather_proba[2]),
        },
        "forecast_3h": {
            "temperature": float(reg_pred[0]),
            "humidity": float(reg_pred[1]),
            "pressure": float(reg_pred[2]),
            "light": float(reg_pred[3]),
        },
        "source_info": source_info,
        "history_used": history_used,
    }
