import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import base64
from datetime import datetime
import joblib
import os

# ==========================================
# 1. SETUP & UTILITIES
# ==========================================

st.set_page_config(
    page_title="Smart Irrigation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_img_as_base64(file_path):
    """
    Reads a local image file and converts it to base64 so CSS can use it.
    """
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return None

# ATTEMPT TO LOAD BACKGROUND IMAGE
# Make sure "China Photo.jpg" is in the exact same folder as this script
img_path = "China Photo.jpg"
img_base64 = get_img_as_base64(img_path)

# Fallback image if local file is missing
fallback_url = "https://images.unsplash.com/photo-1625246333195-5848b42814b3?q=80&w=2074"

if img_base64:
    bg_image_css = f"url('data:image/jpeg;base64,{img_base64}')"
else:
    bg_image_css = f"url('{fallback_url}')"

# ==========================================
# 2. CUSTOM CSS (Dark Sidebar & Colorful Graph)
# ==========================================
st.markdown(f"""
<style>
    /* Main Background */
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)), 
                    {bg_image_css};
        background-size: cover;
        background-position: center; 
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* SIDEBAR STYLING - Dark BG for White Text */
    [data-testid="stSidebar"] {{
        background-color: rgba(15, 23, 42, 0.85); /* Dark Slate Blue */
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }}
    
    /* Force White Text in Sidebar */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {{
        color: #ffffff !important;
    }}
    
    [data-testid="stSidebar"] label {{
        color: #e2e8f0 !important; /* Light Grey/White for inputs */
        font-weight: 600;
    }}
    
    [data-testid="stSidebar"] .stMarkdown p {{
        color: #cbd5e1 !important;
    }}
    
    /* TRANSLUCENT CARD STYLE */
    .css-card {{
        background: rgba(255, 255, 255, 0.65); /* 65% opacity white */
        backdrop-filter: blur(12px);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }}
    
    /* Metrics Text */
    [data-testid="stMetricValue"] {{
        font-size: 2.2rem;
        font-weight: 800;
        color: #f0f9ff; 
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    }}
    
    [data-testid="stMetricLabel"] {{
        color: #e0f2fe;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. DATA & MODELS
# ==========================================
CROP_DATA = {
    "Wheat": 1.15, "Rice": 1.20, "Maize (Corn)": 1.20,
    "Tomato": 1.15, "Potato": 1.15, "Cotton": 1.15,
    "Sugarcane": 1.25, "Soybean": 1.15, "Onion": 1.05,
    "Grapes": 0.85, "Citrus": 0.70, "Reference Crop (Grass)": 1.00
}

@st.cache_resource
def load_trained_models():
    try:
        if os.path.exists('best_et_model.pkl') and os.path.exists('scaler.pkl'):
            svm = joblib.load('best_et_model.pkl')
            scaler = joblib.load('scaler.pkl')
            return svm, scaler, True, "SVM AI Model"
        else:
            return None, None, False, "FAO-56 Equation"
    except Exception:
        return None, None, False, "FAO-56 Equation"

svm_model, scaler_model, models_loaded, method = load_trained_models()

def calculate_reference_et(t_max, t_min, rh, ws, solar):
    if models_loaded and svm_model is not None:
        input_data = np.array([[t_max, t_min, rh, ws, solar]])
        input_scaled = scaler_model.transform(input_data)
        et = svm_model.predict(input_scaled)[0]
        return max(0, min(15, et))
    else:
        # Fallback Equation
        T_mean = (t_max + t_min) / 2
        gamma = 0.067
        delta = 4098 * (0.6108 * np.exp((17.27 * T_mean) / (T_mean + 237.3))) / ((T_mean + 237.3) ** 2)
        es = (0.6108 * np.exp((17.27 * t_max)/(t_max + 237.3)) + 0.6108 * np.exp((17.27 * t_min)/(t_min + 237.3))) / 2
        ea = es * (rh / 100)
        Rn = 0.77 * solar
        num = (0.408 * delta * Rn) + (gamma * (900 / (T_mean + 273)) * ws * (es - ea))
        den = delta + (gamma * (1 + 0.34 * ws))
        return max(0, min(15, num / den))

# ==========================================
# 4. APP LAYOUT
# ==========================================

st.markdown(f"""
<div class='css-card' style='text-align: center; background: rgba(255,255,255,0.85);'>
    <h1 style='color: #2563eb; margin: 0;'>🌾 Smart Crop Water Planner</h1>
    <p style='color: #475569;'><b>Method:</b> {method} • <b>Status:</b> {'✅ Ready' if models_loaded else '⚠️ Running Physics Fallback'}</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("1. Farm Details")
    crop_type = st.selectbox("Select Crop Type", list(CROP_DATA.keys()), index=0)
    kc_value = CROP_DATA[crop_type]
    field_area = st.number_input("Field Area (Hectares)", 0.1, 1000.0, 1.0, 0.1)
    
    st.divider()
    st.header("2. Weather")
    t_max = st.slider("Max Temp (°C)", 10.0, 50.0, 32.0)
    t_min = st.slider("Min Temp (°C)", 0.0, 35.0, 18.0)
    rh = st.slider("Humidity (%)", 10.0, 100.0, 45.0)
    ws = st.slider("Wind Speed (m/s)", 0.0, 10.0, 2.5)
    solar = st.slider("Solar Rad (MJ/m²)", 5.0, 35.0, 22.0)

# Calculations
et_0 = calculate_reference_et(t_max, t_min, rh, ws, solar)
et_c = et_0 * kc_value
total_liters_day = (et_c * (field_area * 10000))
total_m3_day = total_liters_day / 1000

# Results Display
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class='css-card'>
        <h3 style='margin:0; color: #166534;'>🌱 Crop Need (ETc)</h3>
        <div style='font-size: 3.5rem; font-weight: 800; color: #15803d;'>{et_c:.2f}</div>
        <div style='color: #4b5563; font-weight: bold;'>mm / day</div>
        <hr style='border-color: rgba(0,0,0,0.1);'>
        <div style='font-size: 0.9rem; color: #374151;'>
            Reference ET0: <b>{et_0:.2f}</b> mm<br>
            Crop Factor (Kc): <b>{kc_value}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='css-card'>
        <h3 style='margin:0; color: #1e40af;'>💧 Irrigation Volume</h3>
        <div style='font-size: 3.5rem; font-weight: 800; color: #2563eb;'>{total_liters_day:,.0f}</div>
        <div style='color: #4b5563; font-weight: bold;'>Liters / day</div>
        <hr style='border-color: rgba(0,0,0,0.1);'>
        <div style='font-size: 0.9rem; color: #374151;'>
            Area: <b>{field_area}</b> Hectares<br>
            Volume in m³: <b>{total_m3_day:.1f}</b> m³
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 5. COLORFUL PLOTLY GRAPH
# ==========================================
st.markdown("<h3 style='color: white; text-shadow: 1px 1px 2px black;'>📅 7-Day Planning Forecast</h3>", unsafe_allow_html=True)

# Generate synthetic forecast data
forecast_vals = [et_c * (1 + np.random.uniform(-0.15, 0.15)) for _ in range(7)]

fig = go.Figure(go.Bar(
    x=[f'Day {i+1}' for i in range(7)], 
    y=forecast_vals,
    # COLOR CONFIGURATION
    marker=dict(
        color=forecast_vals,    # Color varies by value
        colorscale='Tealgrn',   # Beautiful Green-Teal scale
        showscale=False
    ),
    text=[f"{v:.1f}" for v in forecast_vals], 
    textposition='auto',
    textfont=dict(color='white', size=14)
))

fig.update_layout(
    height=300, 
    paper_bgcolor='rgba(0,0,0,0)', 
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    yaxis_title="Water (mm)",
    margin=dict(t=20, b=20, l=20, r=20),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.2)')
)
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. CSV DOWNLOAD
# ==========================================
if st.button("📥 Download Report (CSV)", use_container_width=True):
    data = {
        "Date": [datetime.now().strftime("%Y-%m-%d %H:%M")],
        "Crop Type": [crop_type],
        "Field Area (Ha)": [field_area],
        "Crop Coeff (Kc)": [kc_value],
        "Max Temp (C)": [t_max],
        "Min Temp (C)": [t_min],
        "Humidity (%)": [rh],
        "Wind Speed (m/s)": [ws],
        "Solar Rad (MJ/m2)": [solar],
        "Reference ET0 (mm)": [round(et_0, 2)],
        "Crop ETc (mm)": [round(et_c, 2)],
        "Daily Water (Liters)": [round(total_liters_day, 0)],
        "Daily Water (m3)": [round(total_m3_day, 2)]
    }
    
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Click to Save CSV",
        data=csv,
        file_name=f"Irrigation_Plan_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key='download-csv'
    )
