import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="SepsisWatch", layout="wide")

st.title("🚨 SepsisWatch — Early Warning Dashboard")
st.markdown("Real-time sepsis risk detection for ICU patients")

# Train lightweight model on startup
@st.cache_resource
def load_model():
    from sklearn.datasets import make_classification
    # Simulate training data based on real sepsis patterns
    np.random.seed(42)
    n = 5000
    
    # Non-sepsis patients (normal vitals)
    hr_0 = np.random.normal(86, 15, int(n*0.93))
    o2_0 = np.random.normal(97, 2, int(n*0.93))
    temp_0 = np.random.normal(37.0, 0.5, int(n*0.93))
    sbp_0 = np.random.normal(120, 20, int(n*0.93))
    map_0 = np.random.normal(79, 12, int(n*0.93))
    resp_0 = np.random.normal(18, 4, int(n*0.93))
    age_0 = np.random.normal(60, 15, int(n*0.93))
    gender_0 = np.random.randint(0, 2, int(n*0.93))
    iculos_0 = np.random.normal(30, 20, int(n*0.93))
    y_0 = np.zeros(int(n*0.93))

    # Sepsis patients (abnormal vitals)
    hr_1 = np.random.normal(94, 18, int(n*0.07))
    o2_1 = np.random.normal(94, 4, int(n*0.07))
    temp_1 = np.random.normal(37.1, 1.0, int(n*0.07))
    sbp_1 = np.random.normal(100, 25, int(n*0.07))
    map_1 = np.random.normal(77, 15, int(n*0.07))
    resp_1 = np.random.normal(21, 5, int(n*0.07))
    age_1 = np.random.normal(65, 15, int(n*0.07))
    gender_1 = np.random.randint(0, 2, int(n*0.07))
    iculos_1 = np.random.normal(50, 25, int(n*0.07))
    y_1 = np.ones(int(n*0.07))

    X = np.column_stack([
        np.concatenate([hr_0, hr_1]),
        np.concatenate([o2_0, o2_1]),
        np.concatenate([temp_0, temp_1]),
        np.concatenate([sbp_0, sbp_1]),
        np.concatenate([map_0, map_1]),
        np.concatenate([resp_0, resp_1]),
        np.concatenate([age_0, age_1]),
        np.concatenate([gender_0, gender_1]),
        np.concatenate([iculos_0, iculos_1])
    ])
    y = np.concatenate([y_0, y_1])

    model = RandomForestClassifier(
        n_estimators=50, random_state=42, class_weight='balanced'
    )
    model.fit(X, y)
    return model

model = load_model()

# Sidebar
st.sidebar.header("Patient Vitals Input")
hr = st.sidebar.slider("Heart Rate (HR)", 40, 180, 90)
o2sat = st.sidebar.slider("O2 Saturation", 70, 100, 97)
temp = st.sidebar.slider("Temperature (°C)", 35.0, 42.0, 37.0)
sbp = st.sidebar.slider("Systolic BP", 60, 200, 120)
map_val = st.sidebar.slider("Mean Arterial Pressure", 40, 140, 80)
resp = st.sidebar.slider("Respiratory Rate", 8, 50, 18)
age = st.sidebar.slider("Age", 18, 100, 55)
gender = st.sidebar.selectbox("Gender", [0, 1], 
         format_func=lambda x: "Female" if x == 0 else "Male")
iculos = st.sidebar.slider("Hours in ICU", 1, 100, 12)

# Tabs
tab1, tab2, tab3 = st.tabs(["Patient Risk", "Funnel Analysis", "Cost Impact"])

with tab1:
    st.header("Patient Risk Prediction")

    input_data = np.array([[hr, o2sat, temp, sbp, map_val,
                            resp, age, gender, iculos]])
    risk_prob = model.predict_proba(input_data)[0][1]
    risk_pct = round(risk_prob * 100, 1)

    if risk_prob >= 0.5:
        risk_level = "🔴 HIGH RISK"
        color = "red"
    elif risk_prob >= 0.2:
        risk_level = "🟠 MEDIUM RISK"
        color = "orange"
    else:
        risk_level = "🟢 LOW RISK"
        color = "green"

    col1, col2, col3 = st.columns(3)
    col1.metric("Sepsis Risk Score", f"{risk_pct}%")
    col2.metric("Risk Level", risk_level)
    col3.metric("Hours in ICU", iculos)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        title={'text': "Sepsis Risk %"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 20], 'color': '#1B5E20'},
                {'range': [20, 50], 'color': '#F57F17'},
                {'range': [50, 100], 'color': '#B71C1C'}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Current Vitals")
    vitals_df = pd.DataFrame({
        'Vital': ['Heart Rate', 'O2 Sat', 'Temperature',
                  'Systolic BP', 'MAP', 'Resp Rate'],
        'Value': [hr, o2sat, temp, sbp, map_val, resp],
        'Normal Range': ['60-100 bpm', '95-100%', '36.5-37.5°C',
                         '90-120 mmHg', '70-100 mmHg', '12-20 /min']
    })
    st.dataframe(vitals_df, use_container_width=True)

with tab2:
    st.header("Patient Risk Funnel")
    stages = ['ICU Admissions', 'At Risk (6+ hrs)',
              'Showed Warning Signs', 'Developed Sepsis',
              'Late Detected (24+ hrs)']
    values = [36111, 36111, 33245, 2592, 1396]

    fig = go.Figure(go.Funnel(
        y=stages, x=values,
        textinfo="value+percent initial",
        marker=dict(color=['#2196F3','#42A5F5','#FF9800','#F44336','#B71C1C'])
    ))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Cost Impact of Late Detection")
    col1, col2, col3 = st.columns(3)
    col1.metric("Late Detected Patients", "1,396")
    col2.metric("Total Preventable Cost", "$32.4M")
    col3.metric("Estimated Lives at Risk", "97")

    categories = ['Extra Cost Per Late Case', 'Total Excess Cost',
                  'Cost of Wasted ICU Days', 'Total Preventable Cost']
    values = [14000, 19544000, 12843200, 32387200]

    fig = go.Figure(go.Bar(
        x=categories, y=values,
        marker_color=['#42A5F5','#FF9800','#F44336','#B71C1C'],
        text=[f'${v:,.0f}' for v in values],
        textposition='outside'
    ))
    fig.update_layout(
        title='Cost Impact of Late Sepsis Detection',
        yaxis_title='Cost (USD)',
        yaxis=dict(tickformat='$,.0f'),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)