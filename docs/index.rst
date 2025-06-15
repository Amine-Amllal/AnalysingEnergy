AnalysingEnergy - Time Series Energy Forecasting System
========================================================

.. raw:: html

   <div style="text-align: center; margin: 30px 0;">
       <h2 style="color: #2E8B57; font-size: 2.5em; margin-bottom: 20px;">
           ⚡ AnalysingEnergy ⚡
       </h2>
       <p style="font-size: 1.3em; color: #555; margin-bottom: 30px;">
           Time-Series-Based Energy Forecasting System for Renewable Energy Stations
       </p>
   </div>

**Project Team**
----------------

.. raw:: html

   <div style="margin: 20px 0;">
       <p><strong>Realized by:</strong></p>
       <ul style="list-style-type: none; padding-left: 0;">
           <li style="margin: 10px 0;">
               🎓 <strong>AMLLAL Amine</strong> - 
               <a href="https://www.linkedin.com/in/amine-amllal/" target="_blank" style="color: #0077B5; text-decoration: none;">
                   <i class="fab fa-linkedin"></i> LinkedIn Profile
               </a>
           </li>
           <li style="margin: 10px 0;">
               🎓 <strong>HAJJI Mohamed</strong> - 
               <a href="https://www.linkedin.com/in/mohamed-hajji-697473364/" target="_blank" style="color: #0077B5; text-decoration: none;">
                   <i class="fab fa-linkedin"></i> LinkedIn Profile
               </a>
           </li>
       </ul>
       <p style="margin-top: 20px;"><strong>Supervised by:</strong></p>
       <ul style="list-style-type: none; padding-left: 0;">
           <li style="margin: 10px 0;">
               👨‍🏫 <strong>M. MASROR Taoufik</strong> - 
               <a href="https://www.linkedin.com/in/tawfik-masrour-43163b85/" target="_blank" style="color: #0077B5; text-decoration: none;">
                   <i class="fab fa-linkedin"></i> LinkedIn Profile
               </a>
           </li>
       </ul>
   </div>

**Project Mission**
-------------------

Our project consists of a time-series-based energy forecasting system applied to renewable energy stations. Its objective is to predict days when consumption will exceed production, in order to recommend actions such as overclocking machines or adjusting work schedules. This system will ensure optimal energy balance and avoid service interruptions.

**Key Objectives:**
~~~~~~~~~~~~~~~~~~~

- 🔮 **Predictive Analytics**: Forecast energy consumption vs. production imbalances
- ⚖️ **Energy Balance Optimization**: Maintain optimal energy equilibrium
- 🚀 **Proactive Recommendations**: Suggest operational adjustments before shortages occur
- 🔄 **Continuous Operation**: Prevent service interruptions through intelligent planning

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
