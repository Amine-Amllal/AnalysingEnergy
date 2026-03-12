# Renewable Energy Forecasting System

> Time-Series Based Energy Prediction for Optimal Resource Management

## 📋 Project Overview

This project is a time-series-based energy forecasting system applied to renewable energy stations.  
Its objective is to predict days when consumption will exceed production, in order to recommend actions such as overclocking machines or adjusting work schedules. This system ensures optimal energy balance and avoids service interruptions.

## 👥 Team

| Role | Name |
|------|------|
| Developer | AMLLAL Amine |
| Developer | HAJJI Mohamed |
| Supervisor | M. MASROUR Tawfik |

📅 **Academic Year:** 2024-2025

## ✨ Key Features

- **LSTM Neural Networks** — Advanced time series forecasting using optimized LSTM models
- **Multi-variable Prediction** — Predicts multiple energy-related parameters:
  - Maximum energy generation (MW)
  - Total energy demand (MW)
  - Temperature variations
  - Wind speed metrics
  - Surface pressure
  - Precipitation correlation
- **Interactive Web Interface** — User-friendly Streamlit application for visualization
- **Hyperparameter Optimization** — Automated model tuning using Optuna
- **365-Day Forecasting** — Extended prediction capabilities for annual planning

## 🔧 Technical Approach

- Deep learning models specifically designed for time-series analysis
- Integration of meteorological data (temperature, wind speed, precipitation) with historical energy data
- Advanced algorithms that identify potential energy imbalances before they occur
- Automated system suggestions for operational adjustments based on predicted energy scenarios

## 🚀 Quick Start

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <project-folder>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit interface:

```bash
streamlit run app.py
```

## 📚 Documentation

- User Guide
- API Reference
- Tutorials
- Notebooks

## 📁 Project Structure

```
project/
├── app.py               # Streamlit web interface
├── models/              # LSTM model definitions
├── data/                # Energy and meteorological datasets
├── notebooks/           # Jupyter notebooks
├── docs/                # Documentation
└── requirements.txt     # Python dependencies
```