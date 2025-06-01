AnalysingEnergy Documentation
==============================

Welcome to the **AnalysingEnergy** project documentation! This project provides advanced energy prediction and analysis using LSTM neural networks and machine learning techniques.

Overview
--------

AnalysingEnergy is a comprehensive machine learning solution for predicting energy generation and consumption patterns. The project combines meteorological data with energy demand forecasting to provide accurate predictions for the next 365 days.

Key Features
~~~~~~~~~~~~

- **LSTM Neural Networks**: Advanced time series forecasting using optimized LSTM models
- **Multi-variable Prediction**: Predicts multiple energy-related parameters including:

  - Maximum energy generation (MW)
  - Total energy demand (MW)
  - Temperature variations
  - Wind speed metrics
  - Surface pressure
  - Precipitation correlation

- **Interactive Web Interface**: User-friendly Streamlit application for visualization
- **Hyperparameter Optimization**: Automated model tuning using Optuna
- **365-Day Forecasting**: Extended prediction capabilities for annual planning

Project Structure
~~~~~~~~~~~~~~~~~

::

    AnalysingEnergy/
    ├── Data/                    # Dataset files
    │   ├── data.csv            # Main dataset
    │   ├── train_data.csv      # Training data
    │   └── test_data.csv       # Testing data
    ├── Notebooks/              # Jupyter notebooks
    │   ├── models/             # Trained LSTM models
    │   ├── scalers/            # Data preprocessing scalers
    │   └── *.ipynb            # Analysis and modeling notebooks
    ├── interface/              # Web application
    │   └── app.py             # Streamlit interface
    ├── docs/                   # Documentation
    └── requirements.txt        # Dependencies

Quick Start
-----------

Installation
~~~~~~~~~~~~

1. Clone the repository::

    git clone https://github.com/yourusername/AnalysingEnergy.git
    cd AnalysingEnergy

2. Install dependencies::

    pip install -r requirements.txt

3. Run the Streamlit interface::

    streamlit run interface/app.py

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   data_overview
   model_architecture

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/interface
   api/models

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/data_preprocessing
   tutorials/model_training
   tutorials/making_predictions

.. toctree::
   :maxdepth: 2
   :caption: Notebooks

   notebooks/data_preprocessing
   notebooks/lstm_generation
   notebooks/lstm_complete
   notebooks/predicting_365_days

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   changelog
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* {ref}`search`
