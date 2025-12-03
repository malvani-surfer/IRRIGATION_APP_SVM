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
    st.header("2. Weather")
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

# ----------------- RESULTS -----------------
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

# ----------------- PLOTS -----------------
st.markdown("<h3 style='color: white; text-shadow: 1px 1px 2px black;'>📅 7-Day Planning Forecast</h3>", unsafe_allow_html=True)
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
        st.markdown("""
        <div style='padding: 10px; background: rgba(255,255,255,0.9); border-radius: 10px; color: #333;'>
            This section visualizes the <strong>Support Vector Machine (SVM)</strong> logic. 
            The charts below calculate how the model reacts to changes in weather, 
            revealing the non-linear relationships it learned during training.
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📈 Sensitivity Analysis", "⚖️ Impact Factors"])
        
        # --- TAB 1: SENSITIVITY PLOT ---
        with tab1:
            col_a, col_b = st.columns([1, 3])
            
            with col_a:
                st.markdown("**Test a Variable:**")
                # Dropdown to select which variable to flex
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
                # 1. Create a base input array with current user sliders
                # [t_max, t_min, rh, ws, solar]
                base_inputs = np.array([t_max, t_min, rh, ws, solar])
                
                # 2. Generate range for the selected variable
                x_range = np.linspace(sel_min, sel_max, 50)
                
                # 3. Create a matrix of inputs (50 rows, 5 cols)
                # Repeat the base row 50 times
                input_matrix = np.tile(base_inputs, (50, 1))
                
                # 4. Overwrite the column of the selected variable with the range
                input_matrix[:, sel_feat_idx] = x_range
                
                # 5. Scale and Predict
                scaled_matrix = scaler_model.transform(input_matrix)
                y_pred = svm_model.predict(scaled_matrix)
                
                # 6. Plot
                fig_sense = px.line(
                    x=x_range, y=y_pred, 
                    labels={'x': selected_var_label, 'y': 'Predicted ET0 (mm/day)'},
                    title=f"How {selected_var_label} affects ET (while other factors stay constant)"
                )
                fig_sense.update_traces(line_color='#2563eb', line_width=4)
                fig_sense.add_vline(x=base_inputs[sel_feat_idx], line_dash="dash", line_color="red", annotation_text="Current Setting")
                fig_sense.update_layout(
                    paper_bgcolor='rgba(255,255,255,0.9)', 
                    plot_bgcolor='rgba(240,240,240,0.5)',
                    height=350
                )
                st.plotly_chart(fig_sense, use_container_width=True)

        # --- TAB 2: LOCAL IMPORTANCE ---
        with tab2:
            st.markdown("**What is driving TODAY's prediction?** (Perturbation Analysis)")
            # Calculate local importance by tweaking each input by +5% and seeing effect
            impacts = []
            feature_names = ['Max Temp', 'Min Temp', 'Humidity', 'Wind Speed', 'Solar Rad']
            base_pred = et_0
            
            for i in range(5):
                # Create a tweaked input
                tweaked_inputs = np.array([t_max, t_min, rh, ws, solar])
                
                # Perturb by a small amount (e.g., +1 unit or +5%)
                perturbation = tweaked_inputs[i] * 0.05 if tweaked_inputs[i] != 0 else 1.0
                tweaked_inputs[i] += perturbation
                
                # Predict
                tw_scaled = scaler_model.transform(tweaked_inputs.reshape(1, -1))
                tw_pred = svm_model.predict(tw_scaled)[0]
                
                # Calculate absolute change impact
                impact = abs(tw_pred - base_pred)
                impacts.append(impact)

            # Normalize to percentage
            total_impact = sum(impacts) if sum(impacts) > 0 else 1
            impact_pct = [(x / total_impact) * 100 for x in impacts]
            
            df_imp = pd.DataFrame({'Factor': feature_names, 'Influence': impact_pct})
            df_imp = df_imp.sort_values('Influence', ascending=True)
            
            fig_imp = px.bar(
                df_imp, x='Influence', y='Factor', orientation='h',
                title="Relative Influence of Factors on Current Prediction",
                text_auto='.1f'
            )
            fig_imp.update_traces(marker_color='#0f172a', textfont_color='white')
            fig_imp.update_layout(
                paper_bgcolor='rgba(255,255,255,0.9)', 
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Relative Influence (%)"
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
            if hasattr(svm_model, 'support_vectors_'):
                st.caption(f"ℹ️ Model Complexity: This prediction uses weighted distance from {len(svm_model.support_vectors_)} historical 'Support Vector' days.")

# ==========================================
# 6. DOWNLOAD
# ==========================================
if st.button("📥 Download Report (CSV)", use_container_width=True):
    data = {
        "Date": [datetime.now().strftime("%Y-%m-%d %H:%M")],
        "Crop Type": [crop_type],
        "Field Area (Ha)": [field_area],
        "Crop Coeff (Kc)": [kc_value],
        "Reference ET0 (mm)": [round(et_0, 2)],
        "Crop ETc (mm)": [round(et_c, 2)],
        "Daily Water (Liters)": [round(total_liters_day, 0)]
    }
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Save CSV", csv, "Irrigation_Plan.csv", "text/csv", key='download-csv')
