# Getting Started

This guide will help you set up and run the AnalysingEnergy project on your local machine.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package installer)
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/AnalysingEnergy.git
cd AnalysingEnergy
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The project requires the following main dependencies:

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning library
- **tensorflow**: Deep learning framework
- **streamlit**: Web application framework
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **plotly**: Interactive plotting
- **optuna**: Hyperparameter optimization
- **joblib**: Model serialization

## Running the Application

### Web Interface

Launch the Streamlit web interface:

```bash
streamlit run interface/app.py
```

This will start a local web server, typically at `http://localhost:8501`, where you can interact with the energy analysis interface.

### Jupyter Notebooks

To explore the data analysis and model training notebooks:

```bash
jupyter notebook
```

Navigate to the `Notebooks/` directory to access:

- `Data preprocessing.ipynb`: Data cleaning and preparation
- `LSTM Generation.ipynb`: Model architecture and training
- `LSTM complet.ipynb`: Complete LSTM implementation
- `Predicting_next_365days.ipynb`: Long-term forecasting

## Quick Tour

### 1. Data Overview

The project uses meteorological and energy data with the following features:

- **Temperature**: Maximum, minimum, and average temperatures (°C)
- **Surface Pressure**: Atmospheric pressure (Pa)
- **Wind Speed**: Maximum, minimum, and average wind speeds (m/s)
- **Precipitation**: Precipitation correlation values
- **Energy Demand**: Total energy demand (MW)
- **Energy Generation**: Maximum energy generation (MW)

### 2. Model Architecture

The core prediction system uses LSTM (Long Short-Term Memory) neural networks optimized for time series forecasting:

- **Multi-layered LSTM**: Two LSTM layers with dropout for regularization
- **Hyperparameter Optimization**: Automated tuning using Optuna
- **Feature Scaling**: MinMax scaling for improved model performance
- **Time Series Generation**: Proper sequence preparation for LSTM input

### 3. Prediction Capabilities

The system can predict:

- Next 365 days of energy generation
- Multiple meteorological parameters
- Energy demand patterns
- Comparative analysis between production and consumption

## Configuration

### Environment Variables

You may need to set up the following environment variables:

```bash
# Optional: Set TensorFlow logging level
export TF_CPP_MIN_LOG_LEVEL=2

# Optional: CUDA configuration for GPU usage
export CUDA_VISIBLE_DEVICES=0
```

### Model Paths

The pre-trained models are located in:

- `Notebooks/models/`: LSTM model files (.h5 format)
- `Notebooks/scalers/`: Data preprocessing scalers (.pkl format)

## Troubleshooting

### Common Issues

1. **TensorFlow Installation Issues**:
   ```bash
   pip install tensorflow==2.13.0
   ```

2. **Memory Issues with Large Datasets**:
   - Reduce batch size in model training
   - Use data generators for large files

3. **Streamlit Port Conflicts**:
   ```bash
   streamlit run interface/app.py --server.port 8502
   ```

4. **Missing Dependencies**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

### Performance Optimization

- **GPU Usage**: Install tensorflow-gpu for CUDA support
- **Memory Management**: Monitor RAM usage during model training
- **Parallel Processing**: Utilize multiprocessing for data preprocessing

## Next Steps

1. Explore the {doc}`data_overview` to understand the dataset structure
2. Learn about the {doc}`model_architecture` for technical details
3. Follow the {doc}`tutorials/model_training` tutorial for hands-on experience
4. Check out the {doc}`notebooks/lstm_generation` for implementation details

## Support

If you encounter any issues:

1. Check the troubleshooting section above
2. Review the project's GitHub issues
3. Consult the API documentation for specific function usage
4. Refer to the Jupyter notebooks for implementation examples
