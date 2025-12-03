import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="ET Prediction System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS (Background Image & Styling)
# ==========================================
st.markdown("""
<style>
    /* Main Background with Image and Dark Overlay */
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)), 
                    url('https://images.unsplash.com/photo-1625246333195-5848b42814b3?q=80&w=2074&auto=format&fit=crop');
        background-size: cover;
        background-position: center; 
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    /* Make metrics and headers readable against the image */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: bold;
        color: white; 
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    [data-testid="stMetricLabel"] {
        color: #f0fdf4;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    h1, h2, h3 {
        color: #1f2937;
        font-weight: 700;
    }
    
    /* Container styling for readability */
    .css-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    /* Transparent Sidebar to blend better */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.95);
        border-right: 1px solid rgba(0,0,0,0.1);
    }
    
    /* Styled Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(16, 185, 129, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD TRAINED MODELS
# ==========================================

@st.cache_resource
def load_trained_models():
    """
    Load the trained SVM model and scaler
    Returns: (svm_model, scaler, success_flag, method_used)
    """
    try:
        if os.path.exists('best_et_model.pkl') and os.path.exists('scaler.pkl'):
            svm = joblib.load('best_et_model.pkl')
            scaler = joblib.load('scaler.pkl')
            return svm, scaler, True, "SVM Model"
        else:
            return None, None, False, "Physics Equation"
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, False, "Physics Equation"

svm_model, scaler_model, models_loaded, method = load_trained_models()

# ==========================================
# PREDICTION FUNCTIONS
# ==========================================

def predict_with_svm(t_max, t_min, rh, ws, solar):
    try:
        # Input array [T_MAX, T_MIN, RH, WS, SOLAR]
        input_data = np.array([[t_max, t_min, rh, ws, solar]])
        input_scaled = scaler_model.transform(input_data)
        et = svm_model.predict(input_scaled)[0]
        return max(0, min(15, et))
    except Exception as e:
        st.error(f"SVM prediction error: {e}")
        return None

def predict_with_equation(t_max, t_min, rh, ws, solar):
    try:
        # FAO-56 Penman-Monteith Calculation
        T_mean = (t_max + t_min) / 2
        gamma = 0.067
        delta = 4098 * (0.6108 * np.exp((17.27 * T_mean) / (T_mean + 237.3))) / ((T_mean + 237.3) ** 2)
        
        es_max = 0.6108 * np.exp((17.27 * t_max) / (t_max + 237.3))
        es_min = 0.6108 * np.exp((17.27 * t_min) / (t_min + 237.3))
        es = (es_max + es_min) / 2
        
        ea = es * (rh / 100)
        Rn = 0.77 * solar
        
        numerator = (0.408 * delta * Rn) + (gamma * (900 / (T_mean + 273)) * ws * (es - ea))
        denominator = delta + (gamma * (1 + 0.34 * ws))
        
        et = numerator / denominator
        return max(0, min(15, et))
    except Exception as e:
        st.error(f"Equation calculation error: {e}")
        return None

def calculate_et(t_max, t_min, rh, ws, solar):
    if models_loaded and svm_model is not None:
        return predict_with_svm(t_max, t_min, rh, ws, solar)
    else:
        return predict_with_equation(t_max, t_min, rh, ws, solar)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_et_interpretation(et):
    if et < 2:
        return {'level': 'Very Low', 'color': '#3b82f6', 'emoji': '❄️', 'desc': 'Minimal water demand.', 'action': 'Monitor soil moisture.'}
    elif et < 4:
        return {'level': 'Low', 'color': '#10b981', 'emoji': '🌱', 'desc': 'Low water requirement.', 'action': 'Maintain regular schedule.'}
    elif et < 6:
        return {'level': 'Moderate', 'color': '#f59e0b', 'emoji': '☀️', 'desc': 'Average water needs.', 'action': 'Standard irrigation.'}
    elif et < 8:
        return {'level': 'High', 'color': '#f97316', 'emoji': '🔥', 'desc': 'High water demand.', 'action': 'Increase frequency.'}
    else:
        return {'level': 'Very High', 'color': '#ef4444', 'emoji': '🚨', 'desc': 'Critical demand.', 'action': 'URGENT: Maximize irrigation.'}

def get_confidence(t_max, t_min, rh, ws, solar):
    issues = []
    if t_max > 45 or t_max < 10: issues.append("Temp max extreme")
    if t_min < 0 or t_min > 35: issues.append("Temp min extreme")
    if rh > 95 or rh < 15: issues.append("Humidity extreme")
    if ws > 8: issues.append("High wind")
    
    if len(issues) == 0: return "HIGH", "🟢", issues
    elif len(issues) <= 2: return "MEDIUM", "🟡", issues
    else: return "LOW", "🔴", issues

# ==========================================
# UI: HEADER
# ==========================================

if models_loaded:
    status_color = "#10b981"
    status_text = "✅ Using Trained SVM Model"
else:
    status_color = "#f59e0b"
    status_text = "⚠️ Using Physics-Based Equation"

st.markdown(f"""
<div class='css-card' style='text-align: center;'>
    <h1 style='color: #667eea; margin: 0;'>💧 Evapotranspiration Prediction</h1>
    <p style='color: #6b7280; margin: 5px 0;'>Real-time Agricultural Water Management Tool</p>
    <div style='margin-top: 10px; padding: 5px; background: {status_color}20; border-radius: 8px; border: 1px solid {status_color}; display: inline-block;'>
        <strong style='color: {status_color};'>{status_text}</strong>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# UI: SIDEBAR INPUTS
# ==========================================

with st.sidebar:
    st.markdown("### 🌤️ Weather Parameters")
    
    t_max = st.slider("Max Temp (°C)", 10.0, 50.0, 30.0, 0.5)
    t_min = st.slider("Min Temp (°C)", 0.0, 35.0, 15.0, 0.5)
    rh = st.slider("Humidity (%)", 10.0, 100.0, 50.0, 1.0)
    ws = st.slider("Wind Speed (m/s)", 0.0, 10.0, 2.0, 0.1)
    solar = st.slider("Solar Rad (MJ/m²)", 5.0, 35.0, 18.0, 0.5)
    
    st.markdown("---")
    st.markdown("### 🎯 Presets")
    col1, col2 = st.columns(2)
    if col1.button("☀️ Summer"):
        t_max, t_min, rh, ws, solar = 42.0, 28.0, 35.0, 3.5, 28.0
        st.rerun()
    if col2.button("❄️ Winter"):
        t_max, t_min, rh, ws, solar = 25.0, 12.0, 60.0, 1.5, 15.0
        st.rerun()

# ==========================================
# CALCULATIONS & MAIN DISPLAY
# ==========================================

et_prediction = calculate_et(t_max, t_min, rh, ws, solar)
interpretation = get_et_interpretation(et_prediction)
confidence, conf_emoji, conf_issues = get_confidence(t_max, t_min, rh, ws, solar)

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.markdown(f"""
    <div class='css-card' style='background: linear-gradient(135deg, {interpretation['color']}15 0%, {interpretation['color']}30 100%); border: 2px solid {interpretation['color']}; text-align: center;'>
        <h3 style='margin:0;'>Predicted ET</h3>
        <div style='font-size: 3.5rem; font-weight: bold; color: {interpretation['color']};'>{et_prediction:.2f}</div>
        <div style='color: #6b7280;'>mm/day</div>
        <hr>
        <div style='font-size: 1.5rem;'>{interpretation['emoji']} {interpretation['level']}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='css-card' style='height: 100%;'>
        <h3 style='margin:0;'>📊 Confidence</h3>
        <div style='text-align: center; margin: 15px 0;'>
            <div style='font-size: 2.5rem;'>{conf_emoji}</div>
            <div style='font-weight: bold;'>{confidence}</div>
        </div>
        <small style='color: #6b7280;'>
            Method: {method}<br>
            Issues: {', '.join(conf_issues) if conf_issues else 'None'}
        </small>
    </div>
    """, unsafe_allow_html=True)

with col3:
    weekly_et = et_prediction * 7
    monthly_et = et_prediction * 30
    st.metric("Weekly Need", f"{weekly_et:.1f} mm")
    st.metric("Monthly Need", f"{monthly_et:.0f} mm")

# ==========================================
# RECOMMENDATIONS & CHART
# ==========================================

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class='css-card'>
        <h3>💡 Recommendations</h3>
        <div style='padding: 10px; background: {interpretation['color']}20; border-radius: 5px; border-left: 4px solid {interpretation['color']};'>
            <strong>Action:</strong> {interpretation['action']}
        </div>
        <ul style='margin-top: 10px; color: #4b5563;'>
            <li>Daily Water: {et_prediction:.2f} mm</li>
            <li>Weekly Water: {weekly_et:.1f} mm (~{weekly_et*10:.0f} L/100m²)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("<div class='css-card'><h3>📈 Classification</h3>", unsafe_allow_html=True)
    ranges = pd.DataFrame({
        'Level': ['Very Low', 'Low', 'Moderate', 'High', 'Very High'],
        'Min': [0, 2, 4, 6, 8],
        'Max': [2, 4, 6, 8, 12],
        'Color': ['#3b82f6', '#10b981', '#f59e0b', '#f97316', '#ef4444']
    })
    
    fig = go.Figure()
    for _, row in ranges.iterrows():
        fig.add_trace(go.Bar(
            x=[row['Max'] - row['Min']], y=[row['Level']], orientation='h',
            marker=dict(color=row['Color']), name=row['Level'],
            text=f"{row['Min']}-{row['Max']}", textposition='inside'
        ))
    
    fig.add_vline(x=et_prediction, line_dash="dash", line_color="black", line_width=3)
    fig.update_layout(barmode='stack', showlegend=False, height=200, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# EXPORT BUTTON
# ==========================================
if st.button("💾 Download Prediction JSON"):
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'method': method,
        'inputs': {'t_max': t_max, 't_min': t_min, 'rh': rh, 'ws': ws, 'solar': solar},
        'prediction': {'et': et_prediction, 'level': interpretation['level']}
    }
    st.download_button("Click to Save", data=json.dumps(export_data, indent=2), file_name="et_prediction.json", mime="application/json")
