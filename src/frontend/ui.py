import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

# --- CONFIGURATION ---
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

st.set_page_config(
    page_title="NYC Taxi Duration Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3204/3204081.png", width=80)
    st.header("🚕 NYC Taxi MLOps")
    st.markdown(
        "Predict the duration of your taxi ride in New York City using an **ONNX-optimized Random Forest** model."
    )
    st.divider()
    st.write("### 🛠️ Tech Stack")
    st.write("- **Frontend:** Streamlit")
    st.write("- **Backend:** FastAPI (Async)")
    st.write("- **Cache:** Redis")
    st.write("- **Model:** scikit-learn → ONNX")

# --- MAIN LAYOUT ---
st.title("🗽 NYC Taxi Trip Duration Predictor")
st.markdown("Enter your trip details below to get an instant AI-powered time estimate.")

col_input, col_display = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("📝 Trip Details")

    with st.container(border=True):
        st.markdown("**🕒 Date & Time**")
        date_col, time_col = st.columns(2)
        pickup_date = date_col.date_input("Date", datetime.now())
        pickup_time = time_col.time_input("Time", datetime.now())

    with st.container(border=True):
        st.markdown("**📍 Coordinates**")
        p_col1, p_col2 = st.columns(2)
        pickup_lat = p_col1.number_input("Pickup Lat", value=40.7580, format="%.4f")
        pickup_lon = p_col2.number_input("Pickup Lon", value=-73.9855, format="%.4f")

        d_col1, d_col2 = st.columns(2)
        dropoff_lat = d_col1.number_input("Dropoff Lat", value=40.7320, format="%.4f")
        dropoff_lon = d_col2.number_input("Dropoff Lon", value=-73.9960, format="%.4f")

    with st.container(border=True):
        st.markdown("**👥 Passengers**")
        passenger_count = st.slider(
            "Number of Passengers", min_value=1, max_value=6, value=1
        )

    predict_btn = st.button(
        "🚀 Estimate Duration", type="primary", use_container_width=True
    )

with col_display:
    st.subheader("🗺️ Trip Route")

    map_data = pd.DataFrame(
        {
            "lat": [pickup_lat, dropoff_lat],
            "lon": [pickup_lon, dropoff_lon],
            "type": ["Pickup", "Dropoff"],
        }
    )
    st.map(map_data, size=200, color="#ff4b4b", zoom=11)

    st.divider()

    if predict_btn:
        combined_datetime = f"{pickup_date} {pickup_time}"
        payload = {
            "pickup_datetime": combined_datetime,
            "pickup_longitude": pickup_lon,
            "pickup_latitude": pickup_lat,
            "dropoff_longitude": dropoff_lon,
            "dropoff_latitude": dropoff_lat,
            "passenger_count": passenger_count,
        }

        with st.spinner("⚡ Model is predicting..."):
            try:
                response = requests.post(API_URL, json=payload)

                if response.status_code == 200:
                    result = response.json()
                    minutes = result["predicted_duration_minutes"]
                    seconds = result["predicted_duration_seconds"]

                    st.success("✅ Prediction Successful!")

                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.metric(label="Estimated Time", value=f"{minutes} min")
                    with m_col2:
                        st.metric(label="Total Seconds", value=f"{seconds:.1f} sec")

                elif response.status_code == 422:
                    st.warning(
                        "⚠️ Invalid input data. Please check the coordinate boundaries (-90 to 90 for Lat)."
                    )
                else:
                    st.error(f"❌ API Error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Connection Error: Could not reach the API.")
                st.info(
                    "💡 Make sure FastAPI is running (`uvicorn src.api.main:app --workers 4`)"
                )
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
