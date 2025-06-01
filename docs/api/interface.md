# Interface API Reference

This section documents the Streamlit web interface for the AnalysingEnergy project.

## Overview

The interface provides an interactive web application for:
- Uploading and visualizing energy data
- Running LSTM predictions
- Displaying forecast results
- Comparing multiple prediction models

## Main Application (`app.py`)

### Core Functions

#### `load_data()`
Loads the energy dataset from CSV files.

**Returns:**
- `pandas.DataFrame`: Loaded dataset with parsed datetime index

**Example:**
```python
data = load_data()
print(data.head())
```

#### `preprocess_data(data)`
Preprocesses the input data for LSTM model consumption.

**Parameters:**
- `data` (pandas.DataFrame): Raw energy dataset

**Returns:**
- `tuple`: Scaled features and target variables
- `dict`: Fitted scalers for inverse transformation

#### `load_models()`
Loads all trained LSTM models and scalers from the models directory.

**Returns:**
- `dict`: Dictionary containing loaded models for each target variable

**Models loaded:**
- `temp2_max(c)_LSTM.h5`: Maximum temperature prediction
- `temp2_min(c)_LSTM.h5`: Minimum temperature prediction
- `temp2_ave(c)_LSTM.h5`: Average temperature prediction
- `wind_speed50_max(ms)_LSTM.h5`: Maximum wind speed prediction
- `wind_speed50_min(ms)_LSTM.h5`: Minimum wind speed prediction
- `wind_speed50_ave(ms)_LSTM.h5`: Average wind speed prediction
- `suface_pressure(pa)_LSTM.h5`: Surface pressure prediction
- `prectotcorr_LSTM.h5`: Precipitation correlation prediction
- `total_demand(mw)_LSTM.h5`: Total energy demand prediction

#### `make_predictions(models, data, days=365)`
Generates predictions for the specified number of days.

**Parameters:**
- `models` (dict): Dictionary of loaded LSTM models
- `data` (pandas.DataFrame): Input data for prediction
- `days` (int, optional): Number of days to predict (default: 365)

**Returns:**
- `pandas.DataFrame`: Predictions for all target variables

#### `plot_predictions(predictions, historical_data=None)`
Creates interactive plots for prediction visualization.

**Parameters:**
- `predictions` (pandas.DataFrame): Model predictions
- `historical_data` (pandas.DataFrame, optional): Historical data for comparison

**Returns:**
- `plotly.graph_objects.Figure`: Interactive plot object

## Streamlit Components

### Sidebar Controls
- **Data Upload**: File uploader for custom datasets
- **Prediction Period**: Slider for selecting forecast duration
- **Model Selection**: Dropdown for choosing specific models
- **Visualization Options**: Checkboxes for plot customization

### Main Display Area
- **Data Overview**: Summary statistics and data preview
- **Prediction Results**: Tabular display of forecast values
- **Interactive Plots**: Time series visualizations
- **Model Performance**: Accuracy metrics and validation scores

## Configuration

### Session State Variables
```python
# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
```

### Layout Configuration
```python
# Page configuration
st.set_page_config(
    page_title="AnalysingEnergy",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

## Error Handling

The interface includes comprehensive error handling for:
- Invalid data formats
- Missing model files
- Prediction failures
- Visualization errors

## Usage Examples

### Basic Usage
```python
# Run the Streamlit application
streamlit run interface/app.py
```

### Accessing via Browser
Navigate to `http://localhost:8501` after starting the application.

## Dependencies

The interface requires:
- `streamlit`: Web application framework
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `plotly`: Interactive visualizations
- `tensorflow`: Model loading and prediction
- `scikit-learn`: Data preprocessing

## Performance Considerations

- **Model Caching**: Models are cached in session state for faster subsequent predictions
- **Data Processing**: Large datasets are processed in chunks to maintain responsiveness
- **Memory Management**: Automatic cleanup of temporary variables

## Future Enhancements

- Real-time data streaming
- Model comparison dashboard
- Export functionality for predictions
- Advanced visualization options
