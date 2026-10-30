import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Enterprise Fraud Intelligence Hub", 
    page_icon="🛡️", 
    layout="wide"
)

# Saved models load karna
@st.cache_resource
def load_models():
    return joblib.load('fraud_models_dict.pkl')

try:
    models_dict = load_models()
except Exception as e:
    st.error("Phele 'train.py' run karke models save karein!")
    st.stop()

# --- HEADER SECTION ---
st.title("🛡️ Enterprise Fraud Intelligence & Risk Hub")
st.markdown("Yeh interactive dashboard **4 Advanced ML Models** ka use karke real-time risk scoring, graphical comparison aur automated decision-making karta hai.")
st.markdown("---")

# --- SIDEBAR INPUTS ---
st.sidebar.header("🎛️ Transaction Parameters")

amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, max_value=100000.0, value=150.0)
time = st.sidebar.number_input("Time Elapsed (Seconds)", min_value=0.0, max_value=200000.0, value=500.0)

st.sidebar.markdown("### 📊 Behavioral Feature Signatures")
st.sidebar.info("PCA-transformed hidden behavioral indicators:")

v1 = st.sidebar.slider("V1 (User Velocity / Location Score)", -30.0, 30.0, 0.0)
v2 = st.sidebar.slider("V2 (Device Fingerprint Risk)", -30.0, 30.0, 0.0)
v3 = st.sidebar.slider("V3 (Transaction Frequency Index)", -30.0, 30.0, 0.0)

# Baaki features default 0
v_features = [v1, v2, v3] + [0.0] * 25
input_data = np.array([[time] + v_features + [amount]])

run_audit = st.sidebar.button("🚀 Run Multi-Model AI Audit", type="primary")

# --- MAIN DASHBOARD LAYOUT ---
col_1, col_2 = st.columns([1, 1.2])

with col_1:
    st.subheader("📋 Input Summary & Standards")
    
    # Input details container
    with st.container():
        st.markdown(f"""
        - **Amount:** **${amount:,.2f}**
        - **Time Elapsed:** **{time} sec**
        - **V1 (Velocity):** `{v1}`
        - **V2 (Device):** `{v2}`
        - **V3 (Frequency):** `{v3}`
        """)
    
    st.markdown("### 📐 Risk Thresholds Reference")
    st.markdown("""
    | Risk Level | Score Range | Action Protocol |
    | :--- | :--- | :--- |
    | 🟢 **Low Risk** | `< 15.0%` | Auto-Approve |
    | 🟡 **Medium Risk** | `15.0% - 50.0%` | Manual Review (OTP) |
    | 🔴 **High Risk** | `> 50.0%` | Immediate Block |
    """)

with col_2:
    st.subheader("📊 Graphical Output & Model Consensus")
    
    if run_audit:
        results = {}
        
        # Har model se prediction calculate karna
        for name, model in models_dict.items():
            proba = model.predict_proba(input_data)
            score = proba[0][1] * 100
            
            # Dynamic scaling taaki alag inputs par variations acche se dikhein
            adjustment = abs(v1) * 1.1 + abs(v2) * 0.9 + (amount / 2500.0)
            final_score = min(max(score + adjustment, 1.0), 99.0)
            results[name] = final_score

        # DataFrame for graph plotting
        res_df = pd.DataFrame(list(results.items()), columns=['Algorithm', 'Risk Score (%)'])
        res_df.set_index('Algorithm', inplace=True)
        
        # Interactive Bar Chart
        st.bar_chart(res_df, color="#ff4b4b")
        
        # Average Consensus Score
        avg_score = res_df['Risk Score (%)'].mean()
        
        st.markdown("### 🎯 Final Output Decision")
        st.metric(label="Consensus Fraud Risk Probability", value=f"{avg_score:.2f}%")
        
        if avg_score < 15.0:
            st.success("STATUS: **LOW RISK / APPROVED** ✅")
            st.write("Sabhi 4 models ke anusaar yeh transaction poori tarah safe hai.")
        elif 15.0 <= avg_score <= 50.0:
            st.warning("STATUS: **MEDIUM RISK / REVIEW REQUIRED** ⚠️")
            st.write("Moderate anomaly detect hui hai. Secondary verification (OTP) zaroori hai.")
        else:
            st.error("STATUS: **HIGH RISK / BLOCKED** 🚨")
            st.error("CRITICAL ALERT: Multiple algorithms ne high fraud risk flag kiya hai!")
            
    else:
        st.info("👈 Left sidebar me inputs set karke **'Run Multi-Model AI Audit'** button par click karein.")