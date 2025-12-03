import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import joblib
import os

# ==========================================
# CONFIGURATION
# ==========================================
# REPLACE THESE WITH YOUR GITHUB DETAILS
# To get the raw URL: Open image in GitHub -> Right Click "Download" -> Copy Link Address
GITHUB_USER = "YOUR_USERNAME" 
GITHUB_REPO = "YOUR_REPO"
IMAGE_NAME = "China%20Photo.jpg"  # %20 handles the space in the filename

# Construct the raw URL (Fallback to a default if not set up)
BACKGROUND_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/{IMAGE_NAME}"
# Fallback image just in case the link is broken during testing
DEFAULT_BG = "https://images.unsplash.com/photo-1625246333195-5848b42814b3?q=80&w=2074"

# Page configuration
st.set_page_config(
    page_title="Smart Irrigation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown(f"""
<style>
    /* Main Background */
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), 
                    url('{BACKGROUND_URL}'), 
                    url('{DEFAULT_BG}'); /* Fallback */
        background-size: cover;
        background-position: center; 
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* Text Styling for Contrast */
    [data-testid="stMetricValue"] {{
        font-size: 2.2rem;
        font-weight: bold;
        color: white; 
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    }}
    
    [data-testid="stMetricLabel"] {{
        color: #e2e8f0;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
    }}
    
    h1, h2, h3 {{
        color: #1f2937;
        font-weight: 800;
    }}
    
    /* Card Container Style */
    .css-card {{
        background: rgba(255, 255, 255, 0.90);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 20px;
        backdrop-filter: blur(5px);
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: rgba(248, 250, 252, 0.95);
        border-right: 1px solid rgba(0,0,0,0.1);
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# CROP DATA
# ==========================================
# Crop Coefficients (Kc) - mid-season values
CROP_DATA = {
    "Wheat": 1.15,
    "Rice": 1.20,
    "Maize (Corn)": 1.20,
    "Tomato": 1.15,
    "Potato": 1.15,
    "Cotton": 1.15,
    "Sugarcane": 1.25,
    "Soybean": 1.15,
    "Onion": 1.05,
    "Grapes": 0.85,
    "Citrus": 0.70,
    "Reference Crop (Grass)": 1.00
}

# ==========================================
# LOAD MODELS
# ==========================================
@st.cache_resource
def load_trained_models():
    try:
        if os.path.exists('best_et_model.pkl') and os.path.exists('scaler.pkl'):
            svm = joblib.load('best_et_model.pkl')
            scaler = joblib.load('scaler.pkl')
            return svm, scaler, True, "SVM Model"
        else:
            return None, None, False, "FAO-56 Equation"
    except Exception as e:
        return None, None, False, "FAO-56 Equation"

svm_model, scaler_model, models_loaded, method = load_trained_models()

# ==========================================
# CALCULATION ENGINE
# ==========================================
def calculate_reference_et(t_max, t_min, rh, ws, solar):
    """Calculates Reference ET (ET0)"""
    if models_loaded and svm_model is not None:
        input_data = np.array([[t_max, t_min, rh, ws, solar]])
        input_scaled = scaler_model.transform(input_data)
        et = svm_model.predict(input_scaled)[0]
        return max(0, min(15, et))
    else:
        # Fallback Physics Equation
        T_mean = (t_max + t_min) / 2
        gamma = 0.067
        delta = 4098 * (0.6108 * np.exp((17.27 * T_mean) / (T_mean + 237.3))) / ((T_mean + 237.3) ** 2)
        es = (0.6108 * np.exp((17.27 * t_max)/(t_max + 237.3)) + 0.6108 * np.exp((17.27 * t_min)/(t_min + 237.3))) / 2
        ea = es * (rh / 100)
        Rn = 0.77 * solar
        numerator = (0.408 * delta * Rn) + (gamma * (900 / (T_mean + 273)) * ws * (es - ea))
        denominator = delta + (gamma * (1 + 0.34 * ws))
        return max(0, min(15, numerator / denominator))

# ==========================================
# UI: HEADER
# ==========================================
st.markdown(f"""
<div class='css-card' style='text-align: center;'>
    <h1 style='color: #2563eb; margin: 0;'>🌾 Smart Crop Water Planner</h1>
    <div style='display: flex; justify-content: center; gap: 20px; margin-top: 10px;'>
        <span style='background: #e0f2fe; padding: 5px 15px; border-radius: 20px; color: #0284c7; font-weight: bold;'>
             Method: {method}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# UI: SIDEBAR (INPUTS)
# ==========================================
with st.sidebar:
    st.header("1. Farm Details")
    crop_type = st.selectbox("Select Crop Type", list(CROP_DATA.keys()), index=0)
    kc_value = CROP_DATA[crop_type]
    st.caption(f"Crop Coefficient ($K_c$): **{kc_value}**")
    
    field_area = st.number_input("Field Area (Hectares)", min_value=0.1, max_value=1000.0, value=1.0, step=0.1)
    
    st.divider()
    
    st.header("2. Weather Conditions")
    t_max = st.slider("Max Temp (°C)", 10.0, 50.0, 32.0)
    t_min = st.slider("Min Temp (°C)", 0.0, 35.0, 18.0)
    rh = st.slider("Humidity (%)", 10.0, 100.0, 45.0)
    ws = st.slider("Wind Speed (m/s)", 0.0, 10.0, 2.5)
    solar = st.slider("Solar Rad (MJ/m²)", 5.0, 35.0, 22.0)

# ==========================================
# MAIN CALCULATIONS
# ==========================================

# 1. Reference ET (ET0) - Baseline for grass
et_0 = calculate_reference_et(t_max, t_min, rh, ws, solar)

# 2. Crop ET (ETc) - Specific to the selected crop
et_c = et_0 * kc_value

# 3. Total Water Volume
# Formula: Depth (mm) * Area (m2) = Liters
# 1 Hectare = 10,000 m2
area_m2 = field_area * 10000
total_liters_day = et_c * area_m2
total_m3_day = total_liters_day / 1000  # Convert to cubic meters

# ==========================================
# RESULTS DISPLAY
# ==========================================

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(f"""
    <div class='css-card'>
        <h3>🌱 Crop Water Need (ETc)</h3>
        <div style='font-size: 3.5rem; font-weight: 800; color: #16a34a;'>
            {et_c:.2f}
        </div>
        <div style='font-size: 1.2rem; color: #4b5563; font-weight: 600;'>mm / day</div>
        <div style='margin-top: 15px; padding: 10px; background: #f0fdf4; border-left: 4px solid #16a34a; color: #15803d;'>
            <strong>Reference ET0:</strong> {et_0:.2f} mm/day<br>
            <strong>Crop Factor (Kc):</strong> {kc_value} ({crop_type})
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='css-card'>
        <h3>💧 Total Irrigation Volume</h3>
        <div style='font-size: 3.5rem; font-weight: 800; color: #2563eb;'>
            {total_liters_day:,.0f}
        </div>
        <div style='font-size: 1.2rem; color: #4b5563; font-weight: 600;'>Liters / day</div>
        <div style='margin-top: 15px; padding: 10px; background: #eff6ff; border-left: 4px solid #2563eb; color: #1d4ed8;'>
            <strong>For Area:</strong> {field_area} Hectares<br>
            <strong>In Cubic Meters:</strong> {total_m3_day:.1f} m³
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# VISUALIZATION & PLANNING
# ==========================================

st.markdown("<h3 style='color: white; text-shadow: 1px 1px 2px black;'>📅 7-Day Planning Forecast</h3>", unsafe_allow_html=True)

forecast_days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']
# Simulating slight variation for forecast visual
forecast_values = [et_c * (1 + np.random.uniform(-0.1, 0.1)) for _ in range(7)]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=forecast_days,
    y=forecast_values,
    marker_color='#3b82f6',
    text=[f"{v:.1f} mm" for v in forecast_values],
    textposition='auto',
    name='Daily Water Need'
))

fig.add_hline(y=et_c, line_dash="dash", line_color="red", annotation_text="Today's Need")

fig.update_layout(
    height=300,
    paper_bgcolor='rgba(255,255,255,0.9)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=20, r=20, t=30, b=20),
    yaxis_title="Water Requirement (mm)"
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# DOWNLOAD REPORT
# ==========================================
if st.button("📥 Download Irrigation Plan", use_container_width=True):
    report = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "farm_details": {
            "crop": crop_type,
            "area_ha": field_area,
            "kc_coefficient": kc_value
        },
        "weather_inputs": {
            "temp_max": t_max,
            "humidity": rh,
            "wind": ws
        },
        "results": {
            "reference_et0_mm": round(et_0, 2),
            "crop_etc_mm": round(et_c, 2),
            "total_daily_liters": round(total_liters_day, 0),
            "total_weekly_liters": round(total_liters_day * 7, 0)
        }
    }
    st.download_button(
        label="Confirm Download",
        data=json.dumps(report, indent=4),
        file_name=f"Irrigation_Plan_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json"
    )
