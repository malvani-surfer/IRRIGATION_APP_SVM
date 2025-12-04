import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return None

# LOAD BACKGROUND
img_path = "China Photo.jpg"
img_base64 = get_img_as_base64(img_path)
fallback_url = "https://images.unsplash.com/photo-1625246333195-5848b42814b3?q=80&w=2074"

bg_image_css = f"url('data:image/jpeg;base64,{img_base64}')" if img_base64 else f"url('{fallback_url}')"

# ==========================================
# 2. CUSTOM CSS
# ==========================================
st.markdown(f"""
<style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), 
                    {bg_image_css};
        background-size: cover;
        background-position: center; 
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* SIDEBAR */
    [data-testid="stSidebar"] {{
        background-color: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }}
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: #ffffff !important;
    }}
    
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p {{
        color: #e2e8f0 !important;
    }}
    
    /* CARDS */
    .css-card {{
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(12px);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }}
    
    .stMetricValue {{ font-weight: 800 !important; }}
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
            
            # Extract Training Data Count if available
            train_count = "Unknown"
            if hasattr(svm, 'shape_fit_'):
                train_count = svm.shape_fit_[0]
            elif hasattr(scaler, 'n_samples_seen_'):
                train_count = scaler.n_samples_seen_
                
            return svm, scaler, True, "SVM AI Model", train_count
        else:
            return None, None, False, "FAO-56 Equation", 0
    except Exception:
        return None, None, False, "FAO-56 Equation", 0

svm_model, scaler_model, models_loaded, method, train_size = load_trained_models()

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
# 4. UI LAYOUT
# ==========================================

st.markdown(f"""
<div class='css-card' style='text-align: center; background: rgba(255,255,255,0.9);'>
    <h1 style='color: #2563eb; margin: 0;'>🌾 Smart Crop Water Planner</h1>
    <p style='color: #475569;'><b>Method:</b> {method} • <b>Status:</b> {'✅ Ready' if models_loaded else '⚠️ Physics Fallback'}</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("1. Farm Details")
    crop_type = st.selectbox("Select Crop Type", list(CROP_DATA.keys()), index=0)
    kc_value = CROP_DATA[crop_type]
    field_area = st.number_input("Field Area (Hectares)", 0.1, 1000.0, 1.0, 0.1)
    
    st.divider()
    st.header("2. Pump Settings")
    pump_rate = st.number_input("Pump Flow Rate (Q) [L/hr]", min_value=100, value=2000, step=100)
    efficiency = st.slider("Irrigation Efficiency (ω)", 0.5, 1.0, 0.90, 0.05)

    st.divider()
    st.header("3. Weather")
    t_max = st.slider("Max Temp (°C)", 10.0, 50.0, 32.0)
    t_min = st.slider("Min Temp (°C)", 0.0, 35.0, 18.0)
    rh = st.slider("Humidity (%)", 10.0, 100.0, 45.0)
    ws = st.slider("Wind Speed (m/s)", 0.0, 10.0, 2.5)
    solar = st.slider("Solar Rad (MJ/m²)", 5.0, 35.0, 22.0)

# ----------------- MAIN CALCULATIONS -----------------
et_0 = calculate_reference_et(t_max, t_min, rh, ws, solar)
et_c = et_0 * kc_value
total_liters_day = (et_c * (field_area * 10000))
total_m3_day = total_liters_day / 1000

# NEW: Pump Runtime Calculation
# T = V / (Q * efficiency) -> Result is in Hours
# Multiply by 60 for Minutes
runtime_hours = total_liters_day / (pump_rate * efficiency)
runtime_minutes = runtime_hours * 60

# ----------------- RESULTS -----------------
# Row 1: Water Volume
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

# Row 2: Pump Schedule (NEW)
st.subheader("🚜 Smart Pump Schedule")
c1, c2 = st.columns([2, 1])

with c1:
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #ea580c 0%, #f97316 100%); padding: 30px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(234, 88, 12, 0.3);'>
        <h2 style='color: white; margin-top: 0; margin-bottom: 10px;'>⏱️ Required Pump Runtime</h2>
        <div style='display: flex; align-items: baseline; gap: 15px;'>
            <span style='font-size: 4rem; font-weight: 800; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>{int(runtime_minutes)}</span>
            <span style='font-size: 1.5rem; opacity: 0.9; font-weight: 600;'>Minutes</span>
        </div>
        <div style='background: rgba(255,255,255,0.2); padding: 12px; border-radius: 8px; margin-top: 20px; font-size: 0.9rem;'>
            <strong>Formula:</strong> T = V / (Q × ω)<br>
            Flow Rate (Q): {pump_rate} L/hr • Efficiency (ω): {int(efficiency*100)}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    # Logic for advice card
    if runtime_minutes > 180:
        status_color = "#fee2e2"
        status_border = "#ef4444"
        icon = "⚠️"
        msg = "High Load: Split into morning & evening shifts."
    elif runtime_minutes < 45:
        status_color = "#dcfce7"
        status_border = "#22c55e"
        icon = "✅"
        msg = "Standard Cycle: Single shift sufficient."
    else:
        status_color = "#e0f2fe"
        status_border = "#3b82f6"
        icon = "ℹ️"
        msg = "Moderate Load: Monitor soil moisture."

    st.markdown(f"""
    <div style='background: {status_color}; border: 2px solid {status_border}; padding: 20px; border-radius: 15px; height: 100%;'>
        <h3 style='color: #1f2937; margin-top:0;'>Total Hours</h3>
        <div style='font-size: 2.5rem; font-weight: bold; color: #374151;'>{runtime_hours:.1f} <span style='font-size: 1rem;'>hrs</span></div>
        <hr style='border-color: rgba(0,0,0,0.1); margin: 15px 0;'>
        <div style='font-size: 1.2rem;'>{icon} <b>Advice:</b></div>
        <div style='color: #4b5563; margin-top: 5px;'>{msg}</div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- PLOTS -----------------
st.markdown("<br><h3 style='color: white; text-shadow: 1px 1px 2px black;'>📅 7-Day Planning Forecast</h3>", unsafe_allow_html=True)
forecast_vals = [et_c * (1 + np.random.uniform(-0.15, 0.15)) for _ in range(7)]

fig = go.Figure(go.Bar(
    x=[f'Day {i+1}' for i in range(7)], y=forecast_vals,
    marker=dict(color=forecast_vals, colorscale='Tealgrn', showscale=False),
    text=[f"{v:.1f}" for v in forecast_vals], textposition='auto',
    textfont=dict(color='white', size=14)
))
fig.update_layout(
    height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'), margin=dict(t=10, b=10, l=10, r=10),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.2)'), xaxis=dict(showgrid=False)
)
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. NEW: INSIDE THE AI BRAIN (SENSITIVITY ANALYSIS)
# ==========================================
if models_loaded:
    with st.expander("🧠 Inside the AI Brain: How the SVM Model Works", expanded=True):
        st.markdown(f"""
        <div style='padding: 10px; background: rgba(255,255,255,0.9); border-radius: 10px; color: #333; margin-bottom: 20px;'>
            This section visualizes the <strong>Support Vector Machine (SVM)</strong> logic. 
            The model was trained on <strong>{train_size:,.0f}</strong> historical weather data points.
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📈 Sensitivity Analysis", "⚖️ Impact Factors"])
        
        # --- TAB 1: SENSITIVITY PLOT ---
        with tab1:
            col_a, col_b = st.columns([1, 3])
            
            with col_a:
                st.markdown("**Test a Variable:**")
                var_options = {
                    'Max Temp': ('T2M_MAX', 0, 10.0, 50.0),
                    'Min Temp': ('T2M_MIN', 1, 0.0, 40.0),
                    'Humidity': ('RH2M', 2, 0.0, 100.0),
                    'Wind Speed': ('WS2M', 3, 0.0, 15.0),
                    'Solar Rad': ('ALLSKY_SFC_SW_DWN', 4, 0.0, 35.0)
                }
                selected_var_label = st.radio("Select Input:", list(var_options.keys()))
                sel_feat_name, sel_feat_idx, sel_min, sel_max = var_options[selected_var_label]

            with col_b:
                base_inputs = np.array([t_max, t_min, rh, ws, solar])
                x_range = np.linspace(sel_min, sel_max, 50)
                input_matrix = np.tile(base_inputs, (50, 1))
                input_matrix[:, sel_feat_idx] = x_range
                
                scaled_matrix = scaler_model.transform(input_matrix)
                y_pred = svm_model.predict(scaled_matrix)
                
                fig_sense = px.line(
                    x=x_range, y=y_pred, 
                    labels={'x': selected_var_label, 'y': 'Predicted ET0'},
                    title=f"Effect of {selected_var_label} on ET0"
                )
                
                # DARK MODE PLOT STYLING
                fig_sense.update_traces(line_color='#22d3ee', line_width=4) # CYAN Line
                fig_sense.add_vline(x=base_inputs[sel_feat_idx], line_dash="dash", line_color="#fbbf24", annotation_text="Current")
                
                fig_sense.update_layout(
                    paper_bgcolor='rgba(15, 23, 42, 0.9)', # Dark Slate Background
                    plot_bgcolor='rgba(15, 23, 42, 1)',    # Dark Plot Area
                    font=dict(color='#e2e8f0'),            # Light Text
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                    height=350
                )
                st.plotly_chart(fig_sense, use_container_width=True)

        # --- TAB 2: LOCAL IMPORTANCE ---
        with tab2:
            st.markdown("**What is driving TODAY's prediction?** (Perturbation Analysis)")
            impacts = []
            feature_names = ['Max Temp', 'Min Temp', 'Humidity', 'Wind Speed', 'Solar Rad']
            base_pred = et_0
            
            for i in range(5):
                tweaked_inputs = np.array([t_max, t_min, rh, ws, solar])
                perturbation = tweaked_inputs[i] * 0.05 if tweaked_inputs[i] != 0 else 1.0
                tweaked_inputs[i] += perturbation
                tw_scaled = scaler_model.transform(tweaked_inputs.reshape(1, -1))
                tw_pred = svm_model.predict(tw_scaled)[0]
                impacts.append(abs(tw_pred - base_pred))

            total_impact = sum(impacts) if sum(impacts) > 0 else 1
            impact_pct = [(x / total_impact) * 100 for x in impacts]
            
            df_imp = pd.DataFrame({'Factor': feature_names, 'Influence': impact_pct})
            df_imp = df_imp.sort_values('Influence', ascending=True)
            
            fig_imp = px.bar(
                df_imp, x='Influence', y='Factor', orientation='h',
                title="Relative Influence (%)", text_auto='.1f'
            )
            fig_imp.update_traces(marker_color='#38bdf8', textfont_color='black') # Light Blue bars
            fig_imp.update_layout(
                paper_bgcolor='rgba(15, 23, 42, 0.9)', 
                plot_bgcolor='rgba(15, 23, 42, 1)',
                font=dict(color='#e2e8f0'),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
            if hasattr(svm_model, 'support_vectors_'):
                st.caption(f"ℹ️ Support Vectors: {len(svm_model.support_vectors_)}")

# ==========================================
# 6. DOWNLOAD
# ==========================================
if st.button("📥 Download Report (CSV)", use_container_width=True):
    data = {
        "Date": [datetime.now().strftime("%Y-%m-%d %H:%M")],
        "Crop Type": [crop_type],
        "Field Area (Ha)": [field_area],
        "Pump Rate (L/hr)": [pump_rate],
        "Efficiency": [efficiency],
        "Reference ET0 (mm)": [round(et_0, 2)],
        "Crop ETc (mm)": [round(et_c, 2)],
        "Daily Water (Liters)": [round(total_liters_day, 0)],
        "Runtime (Minutes)": [round(runtime_minutes, 0)]
    }
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Save CSV", csv, "Irrigation_Plan.csv", "text/csv", key='download-csv')
