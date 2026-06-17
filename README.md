# 🚨 SepsisWatch — Early Warning & Cost Intelligence Dashboard

> Real-time sepsis risk detection for ICU patients using machine learning

## 🚀 Live Dashboard
[Open SepsisWatch Dashboard](https://sepsis-watch.streamlit.app/)

## 📊 Analysis Notebooks
| Notebook | Description |
|---|---|
| [🔬 EDA Analysis](https://mahipatel23.github.io/SEPSIS-Watch/) | Exploratory data analysis on 36,000+ ICU patients |
| [🔽 Funnel Analysis](https://mahipatel23.github.io/SEPSIS-Watch/funnel_analysis.html) | Patient risk funnel from admission to late detection |
| [💰 Cost Impact](https://mahipatel23.github.io/SEPSIS-Watch/cost_analysis.html) | $32M in preventable costs from late sepsis detection |

## 🧠 What is SepsisWatch?
Sepsis kills 270,000 Americans every year and costs hospitals $24B annually. Current ICU alert systems miss early warning signs — SepsisWatch uses machine learning to flag high-risk patients hours before sepsis becomes life-threatening.

## 📁 Project Structure
sepsiswatch/
|--- app/ # Streamlit Dashboard
|--- data/ # Patient data (not uploaded - PhysioNet credentialed)
|--- docs/ # GitHub Pages HTML notebooks
|--- model/ # Trained ML model
|--- notebooks/ # EDA, Funnel, Cost analysis notebooks
|___ requirements.txt

## 🛠 Tech Stack
Python · Pandas · Scikit-learn · Streamlit · Plotly · Joblib

## 📋 Dataset
PhysioNet Sepsis Challenge 2019 — 36,111 real ICU patients
MIMIC-IV (credentialed access pending) — 300,000+ ICU patients

## 👥 Team
- Mahi Patel — EDA, Funnel Analysis, Cost Impact, Dashboard, Deployment
- Pihu — Clinical Feature Engineering, ML Model, SHAP Explainability
- Krish — Frontend Development, Deployment
