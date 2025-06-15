Renewable Energy Forecasting System
====================================

.. raw:: html

   <div style="text-align: center; margin: 30px 0;">
     <h1 style="color: #2E8B57; font-size: 2.5em; margin-bottom: 20px;">
       Renewable Energy Forecasting System
     </h1>
     <p style="font-size: 1.2em; color: #555; margin-bottom: 30px;">
       Time-Series Based Energy Prediction for Optimal Resource Management
     </p>
   </div>

Project Overview
----------------

Our project consists of a time-series-based energy forecasting system applied to renewable energy stations. Its objective is to predict days when consumption will exceed production, in order to recommend actions such as overclocking machines or adjusting work schedules. This system will ensure optimal energy balance and avoid service interruptions.

.. raw:: html

   <div style="background-color: #f8f9fa; padding: 20px; border-left: 4px solid #2E8B57; margin: 20px 0;">
     <h3 style="color: #2E8B57; margin-top: 0;">Key Objectives</h3>
     <ul style="list-style-type: none; padding-left: 0;">
       <li>🔮 <strong>Predict Energy Imbalances:</strong> Forecast days when consumption exceeds production</li>
       <li>⚡ <strong>Optimize Operations:</strong> Recommend overclocking and scheduling adjustments</li>
       <li>🔄 <strong>Ensure Continuity:</strong> Prevent service interruptions through proactive planning</li>
       <li>📊 <strong>Data-Driven Insights:</strong> Leverage machine learning for accurate predictions</li>
     </ul>
   </div>

Team
----

.. raw:: html

   <div style="margin: 30px 0;">
     <h3 style="color: #2E8B57; text-align: center; margin-bottom: 30px;">Project Team</h3>
     
     <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 20px;">
       
       <!-- Team Members -->
       <div style="text-align: center; margin: 15px; padding: 20px; background-color: #f8f9fa; border-radius: 10px; min-width: 200px;">
         <h4 style="color: #333; margin-bottom: 10px;">Réalisé par:</h4>
         
         <div style="margin: 15px 0;">
           <p style="font-weight: bold; margin: 5px 0; color: #2E8B57;">AMLLAL Amine</p>
           <a href="https://www.linkedin.com/in/amine-amllal/" target="_blank" style="color: #0077B5; text-decoration: none;">
             <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white" alt="LinkedIn" style="margin-top: 5px;">
           </a>
         </div>
         
         <div style="margin: 15px 0;">
           <p style="font-weight: bold; margin: 5px 0; color: #2E8B57;">HAJJI Mohamed</p>
           <a href="https://www.linkedin.com/in/mohamed-hajji-697473364/" target="_blank" style="color: #0077B5; text-decoration: none;">
             <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white" alt="LinkedIn" style="margin-top: 5px;">
           </a>
         </div>
       </div>
       
       <!-- Supervisor -->
       <div style="text-align: center; margin: 15px; padding: 20px; background-color: #f0f8f0; border-radius: 10px; min-width: 200px;">
         <h4 style="color: #333; margin-bottom: 10px;">Encadré par:</h4>
         
         <div style="margin: 15px 0;">
           <p style="font-weight: bold; margin: 5px 0; color: #2E8B57;">M. MASROR Taoufik</p>
           <a href="https://www.linkedin.com/in/tawfik-masrour-43163b85/" target="_blank" style="color: #0077B5; text-decoration: none;">
             <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white" alt="LinkedIn" style="margin-top: 5px;">
           </a>
         </div>
       </div>
     </div>
   </div>

Technical Approach
------------------

Our system employs advanced machine learning techniques to deliver accurate energy forecasting:

**LSTM Neural Networks**
  Deep learning models specifically designed for time-series analysis, capable of learning complex temporal patterns in energy data.

**Multi-Variable Analysis**
  Integration of meteorological data (temperature, wind speed, precipitation) with historical energy consumption and production patterns.

**Predictive Analytics**
  Advanced algorithms that identify potential energy imbalances before they occur, enabling proactive management decisions.

**Real-Time Recommendations**
  Automated system suggestions for operational adjustments based on predicted energy scenarios.

.. raw:: html

   <div style="background-color: #e8f5e8; padding: 20px; border-radius: 10px; margin: 30px 0; text-align: center;">
     <h3 style="color: #2E8B57; margin-top: 0;">Ready to Get Started?</h3>
     <p style="margin: 10px 0;">Explore our comprehensive documentation to learn how to implement and use our energy forecasting system.</p>
     <a href="getting_started.html" style="display: inline-block; background-color: #2E8B57; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin-top: 10px;">
       Get Started →
     </a>
   </div>

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

Documentation Contents
======================

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
