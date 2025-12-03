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

# Custom CSS for background and styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    h1, h2, h3 {
        color: #1f2937;
        font-weight: 700;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
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
        # Try to load the saved models
        if os.path.exists('best_et_model.pkl') and os.path.exists('scaler.pkl'):
            svm = joblib.load('best_et_model.pkl')
            scaler = joblib.load('scaler.pkl')
            return svm, scaler, True, "SVM Model"
        else:
            return None, None, False, "Physics Equation"
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, False, "Physics Equation"

# Load models at startup
svm_model, scaler_model, models_loaded, method = load_trained_models()

# ==========================================
# PREDICTION FUNCTIONS
# ==========================================

def predict_with_svm(t_max, t_min, rh, ws, solar):
    """
    Predict ET using the trained SVM model
    
    This is the actual trained model you created!
    Steps:
    1. Organize inputs into array
    2. Scale using the saved scaler (CRITICAL!)
    3. Use SVM to predict
    """
    try:
        # Step 1: Create input array [T_MAX, T_MIN, RH, WS, SOLAR]
        input_data = np.array([[t_max, t_min, rh, ws, solar]])
        
        # Step 2: Scale the inputs (using same scaler from training)
        # This transforms to mean=0, std=1
        input_scaled = scaler_model.transform(input_data)
        
        # Step 3: SVM prediction
        et = svm_model.predict(input_scaled)[0]
        
        # Clamp to realistic range
        return max(0, min(15, et))
        
    except Exception as e:
        st.error(f"SVM prediction error: {e}")
        return None

def predict_with_equation(t_max, t_min, rh, ws, solar):
    """
    Fallback: Calculate ET using FAO-56 Penman-Monteith equation
    """
    try:
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
    """
    Main prediction function - uses SVM if available, else equation
    """
    if models_loaded and svm_model is not None:
        return predict_with_svm(t_max, t_min, rh, ws, solar)
    else:
        return predict_with_equation(t_max, t_min, rh, ws, solar)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_et_interpretation(et):
    if et < 2:
        return {
            'level': 'Very Low', 'color': '#3b82f6', 'emoji': '❄️',
            'desc': 'Minimal water demand. Light irrigation sufficient.',
            'action': 'Monitor soil moisture. Reduce irrigation frequency.'
        }
    elif et < 4:
        return {
            'level': 'Low', 'color': '#10b981', 'emoji': '🌱',
            'desc': 'Low water requirement. Standard irrigation adequate.',
            'action': 'Maintain regular irrigation schedule.'
        }
    elif et < 6:
        return {
            'level': 'Moderate', 'color': '#f59e0b', 'emoji': '☀️',
            'desc': 'Average water needs. Monitor conditions.',
            'action': 'Standard irrigation. Check soil moisture regularly.'
        }
    elif et < 8:
        return {
            'level': 'High', 'color': '#f97316', 'emoji': '🔥',
            'desc': 'High water demand. Increase irrigation.',
            'action': 'Increase irrigation frequency. Monitor crops closely.'
        }
    else:
        return {
            'level': 'Very High', 'color': '#ef4444', 'emoji': '🚨',
            'desc': 'Critical irrigation needed. Risk of water stress.',
            'action': 'URGENT: Maximize irrigation. Consider multiple daily cycles.'
        }

def get_confidence(t_max, t_min, rh, ws, solar):
    issues = []
    if t_max > 45 or t_max < 10:
        issues.append("Temperature max out of typical range")
    if t_min < 0 or t_min > 35:
        issues.append("Temperature min out of typical range")
    if rh > 95 or rh < 15:
        issues.append("Humidity extreme")
    if ws > 8:
        issues.append("Very high wind speed")
    if solar > 30 or solar < 8:
        issues.append("Solar radiation unusual")
    
    if len(issues) == 0:
        return "HIGH", "🟢", issues
    elif len(issues) <= 2:
        return "MEDIUM", "🟡", issues
    else:
        return "LOW", "🔴", issues

# ==========================================
# UI: HEADER
# ==========================================

# Model status indicator
if models_loaded:
    status_color = "#10b981"
    status_text = "✅ Using Trained SVM Model (R² = 0.9996)"
    status_detail = "Predictions from your trained machine learning model"
else:
    status_color = "#f59e0b"
    status_text = "⚠️ Using Physics-Based FAO-56 Equation"
    status_detail = "Model files not found. Place 'best_et_model.pkl' and 'scaler.pkl' in app folder"

st.markdown(f"""
<div style='text-align: center; padding: 20px; background: rgba(255,255,255,0.95); border-radius: 15px; margin-bottom: 20px;'>
    <h1 style='color: #667eea; margin: 0;'>💧 Evapotranspiration Prediction System</h1>
    <p style='color: #6b7280; margin: 10px 0 0 0; font-size: 1.1rem;'>
        Real-time Agricultural Water Management Tool
    </p>
    <div style='margin-top: 15px; padding: 10px; background: {status_color}20; border-radius: 8px; border: 2px solid {status_color};'>
        <strong style='color: {status_color};'>{status_text}</strong><br>
        <small style='color: #6b7280;'>{status_detail}</small>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# UI: SIDEBAR INPUTS
# ==========================================

with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='color: white; margin: 0;'>🌤️ Weather Parameters</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 🌡️ Temperature")
    t_max = st.slider("Maximum Temperature (°C)", 10.0, 50.0, 30.0, 0.5)
    t_min = st.slider("Minimum Temperature (°C)", 0.0, 35.0, 15.0, 0.5)
    
    st.markdown("---")
    st.markdown("### 💧 Humidity")
    rh = st.slider("Relative Humidity (%)", 10.0, 100.0, 50.0, 1.0)
    
    st.markdown("---")
    st.markdown("### 💨 Wind")
    ws = st.slider("Wind Speed (m/s)", 0.0, 10.0, 2.0, 0.1)
    
    st.markdown("---")
    st.markdown("### ☀️ Solar Radiation")
    solar = st.slider("Solar Radiation (MJ/m²/day)", 5.0, 35.0, 18.0, 0.5)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown("### 🎯 Quick Presets")
    col1, col2 = st.columns(2)
    
    if col1.button("☀️ Hot Summer"):
        t_max, t_min, rh, ws, solar = 42.0, 28.0, 35.0, 3.5, 28.0
        st.rerun()
    if col2.button("❄️ Cool Winter"):
        t_max, t_min, rh, ws, solar = 25.0, 12.0, 60.0, 1.5, 15.0
        st.rerun()
    if col1.button("🌧️ Monsoon"):
        t_max, t_min, rh, ws, solar = 28.0, 22.0, 85.0, 2.5, 12.0
        st.rerun()
    if col2.button("🍂 Spring"):
        t_max, t_min, rh, ws, solar = 32.0, 18.0, 45.0, 2.0, 22.0
        st.rerun()

# ==========================================
# CALCULATE PREDICTION
# ==========================================

et_prediction = calculate_et(t_max, t_min, rh, ws, solar)
interpretation = get_et_interpretation(et_prediction)
confidence, conf_emoji, conf_issues = get_confidence(t_max, t_min, rh, ws, solar)

# ==========================================
# UI: MAIN DISPLAY
# ==========================================

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {interpretation['color']}15 0%, {interpretation['color']}30 100%); 
                padding: 30px; border-radius: 15px; text-align: center; border: 3px solid {interpretation['color']};'>
        <h2 style='margin: 0; color: #1f2937;'>Predicted ET</h2>
        <div style='font-size: 4rem; font-weight: bold; color: {interpretation['color']}; margin: 20px 0;'>
            {et_prediction:.2f}
        </div>
        <div style='font-size: 1.5rem; color: #6b7280;'>mm/day</div>
        <div style='margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.8); border-radius: 10px;'>
            <div style='font-size: 1.8rem;'>{interpretation['emoji']}</div>
            <div style='font-size: 1.3rem; font-weight: bold; color: {interpretation['color']};'>
                {interpretation['level']}
            </div>
            <div style='color: #6b7280; margin-top: 10px;'>{interpretation['desc']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    model_info = f"""
    • Model Type: <strong>{method}</strong><br>
    • Training Data: 7,671 days (2004-2024)<br>
    """ + ("""
    • Algorithm: Support Vector Regression<br>
    • Accuracy: R² = 0.9996<br>
    • Error: RMSE = 0.044 mm/day
    """ if models_loaded else """
    • Method: FAO-56 Penman-Monteith<br>
    • Location: Kamshet, Maharashtra<br>
    • Elevation: ~560m
    """)
    
    st.markdown(f"""
    <div style='background: rgba(255,255,255,0.95); padding: 30px; border-radius: 15px; height: 100%;'>
        <h3 style='margin: 0 0 20px 0; color: #1f2937;'>📊 Model Confidence</h3>
        <div style='text-align: center; margin: 20px 0;'>
            <div style='font-size: 3rem;'>{conf_emoji}</div>
            <div style='font-size: 1.8rem; font-weight: bold; margin: 10px 0;'>{confidence}</div>
        </div>
        <div style='background: #f3f4f6; padding: 15px; border-radius: 10px; margin-top: 20px;'>
            <strong>Model Information:</strong><br>
            <small style='color: #6b7280;'>{model_info}</small>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    t_mean = (t_max + t_min) / 2
    t_range = t_max - t_min
    weekly_et = et_prediction * 7
    monthly_et = et_prediction * 30
    
    st.metric("Mean Temp", f"{t_mean:.1f}°C", f"±{t_range/2:.1f}°C")
    st.metric("Weekly ET", f"{weekly_et:.1f} mm", f"{et_prediction:.2f}/day")
    st.metric("Monthly ET", f"{monthly_et:.0f} mm", f"~{monthly_et/25.4:.1f} inches")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# RECOMMENDATIONS
# ==========================================

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div style='background: rgba(255,255,255,0.95); padding: 20px; border-radius: 15px;'>
        <h3 style='color: #1f2937; margin-bottom: 15px;'>💡 Irrigation Recommendations</h3>
        <div style='background: {interpretation['color']}15; padding: 15px; border-radius: 10px; 
                    border-left: 4px solid {interpretation['color']};'>
            <strong style='color: {interpretation['color']};'>ACTION REQUIRED:</strong><br>
            <span style='color: #1f2937;'>{interpretation['action']}</span>
        </div>
        <div style='margin-top: 15px; padding: 15px; background: #f3f4f6; border-radius: 10px;'>
            <strong>Water Application:</strong><br>
            • Daily: <strong>{et_prediction:.2f} mm</strong><br>
            • Weekly: <strong>{weekly_et:.1f} mm</strong> ({weekly_et * 10:.0f} L per 100m²)<br>
            • Monthly: <strong>{monthly_et:.0f} mm</strong> ({monthly_et * 10:.0f} L per 100m²)
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background: rgba(255,255,255,0.95); padding: 20px; border-radius: 15px;'>
        <h3 style='color: #1f2937; margin-bottom: 15px;'>📈 ET Classification</h3>
    """, unsafe_allow_html=True)
    
    ranges = pd.DataFrame({
        'Level': ['Very Low', 'Low', 'Moderate', 'High', 'Very High'],
        'Min': [0, 2, 4, 6, 8],
        'Max': [2, 4, 6, 8, 12],
        'Color': ['#3b82f6', '#10b981', '#f59e0b', '#f97316', '#ef4444']
    })
    
    fig = go.Figure()
    for _, row in ranges.iterrows():
        fig.add_trace(go.Bar(
            x=[row['Max'] - row['Min']],
            y=[row['Level']],
            orientation='h',
            marker=dict(color=row['Color']),
            name=row['Level'],
            text=f"{row['Min']}-{row['Max']} mm/day",
            textposition='inside'
        ))
    
    fig.add_vline(x=et_prediction, line_dash="dash", line_color="black", line_width=3)
    fig.update_layout(
        barmode='stack', showlegend=False, height=250,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(title="ET (mm/day)", range=[0, 12]),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# EXPORT
# ==========================================

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    if st.button("💾 Export Prediction Data", use_container_width=True):
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'location': 'Kamshet, Maharashtra',
            'prediction_method': method,
            'inputs': {
                'max_temperature_C': t_max,
                'min_temperature_C': t_min,
                'relative_humidity_%': rh,
                'wind_speed_ms': ws,
                'solar_radiation_MJ': solar
            },
            'prediction': {
                'et_mm_per_day': round(et_prediction, 2),
                'weekly_mm': round(weekly_et, 1),
                'monthly_mm': round(monthly_et, 0),
                'level': interpretation['level'],
                'confidence': confidence
            },
            'recommendations': interpretation['action']
        }
        
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"ET_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; padding: 20px; background: rgba(255,255,255,0.95); border-radius: 15px;'>
    <p style='color: #6b7280; margin: 0;'>
        <strong>🌾 Agricultural Water Management System</strong><br>
        Prediction Method: <strong>{method}</strong> • NASA POWER Data (2004-2024)<br>
        <small>Kamshet, Maharashtra • {"SVM: R² = 0.9996, RMSE = 0.044 mm/day" if models_loaded else "FAO-56 Penman-Monteith Methodology"}</small>
    </p>
</div>
""", unsafe_allow_html=True)