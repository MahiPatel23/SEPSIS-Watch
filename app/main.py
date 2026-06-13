import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="SepsisWatch", layout="wide")

st.title("🚨 SepsisWatch — Early Warning Dashboard")
st.markdown("Real-time sepsis risk detection for ICU patients")

# Sidebar
st.sidebar.header("Filters")
ageRange = st.sidebar.slider("Age Range", 0, 100, (20, 80))
icuHours = st.sidebar.slider("Hours in ICU", 0, 100, (0, 50))

# Tabs
tab1, tab2, tab3 = st.tabs(["Patient Risk", "Funnel Analysis", "Cost Impact"])

with tab1:
    st.header("Patient Risk Overview")
    st.info("Model predictions will appear here once connected")

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