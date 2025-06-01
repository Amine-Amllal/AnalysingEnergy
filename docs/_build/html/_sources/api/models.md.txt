# Models API Reference

This section provides detailed documentation for the LSTM models and related utilities used in the AnalysingEnergy project.

## Model Architecture Overview

The project uses individual LSTM models for each target variable, allowing for specialized prediction of different energy and meteorological parameters.

## Model Classes and Functions

### LSTM Model Structure

Each model follows a consistent architecture pattern:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_lstm_model(input_shape, units=50, dropout_rate=0.2):
    """
    Creates an LSTM model for time series prediction.
    
    Args:
        input_shape (tuple): Shape of input data (timesteps, features)
        units (int): Number of LSTM units in each layer
        dropout_rate (float): Dropout rate for regularization
    
    Returns:
        tensorflow.keras.Model: Compiled LSTM model
    """
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model
```

## Individual Model Specifications

### Temperature Models

#### Maximum Temperature Model
- **File**: `temp2_max(c)_LSTM.h5`
- **Target**: Maximum daily temperature (°C)
- **Input Features**: All meteorological variables
- **Architecture**: 2-layer LSTM with 50 units each
- **Optimization**: Optuna-tuned hyperparameters

#### Minimum Temperature Model
- **File**: `temp2_min(c)_LSTM.h5`
- **Target**: Minimum daily temperature (°C)
- **Seasonality**: Captures winter/summer variations
- **Performance**: High accuracy for extreme temperature events

#### Average Temperature Model
- **File**: `temp2_ave(c)_LSTM.h5`
- **Target**: Average daily temperature (°C)
- **Use Case**: General temperature trend prediction

### Wind Speed Models

#### Maximum Wind Speed Model
- **File**: `wind_speed50_max(ms)_LSTM.h5`
- **Target**: Maximum wind speed at 50m height (m/s)
- **Critical for**: Peak energy generation capacity

#### Minimum Wind Speed Model
- **File**: `wind_speed50_min(ms)_LSTM.h5`
- **Target**: Minimum wind speed at 50m height (m/s)
- **Use Case**: Energy generation baseline estimation

#### Average Wind Speed Model
- **File**: `wind_speed50_ave(ms)_LSTM.h5`
- **Target**: Average wind speed at 50m height (m/s)
- **Primary use**: Daily energy production forecasting

### Pressure and Precipitation Models

#### Surface Pressure Model
- **File**: `suface_pressure(pa)_LSTM.h5`
- **Target**: Surface atmospheric pressure (Pa)
- **Correlation**: Weather pattern prediction

#### Precipitation Correlation Model
- **File**: `prectotcorr_LSTM.h5`
- **Target**: Precipitation correlation values
- **Impact**: Solar energy generation prediction

### Energy Demand Model

#### Total Demand Model
- **File**: `total_demand(mw)_LSTM.h5`
- **Target**: Total energy demand (MW)
- **Critical**: Grid planning and load balancing
- **Features**: Temperature-dependent demand patterns

## Model Utilities

### Data Preprocessing

```python
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesPreprocessor:
    """Handles data preprocessing for LSTM models."""
    
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.scalers = {}
    
    def create_sequences(self, data, target_column):
        """
        Creates sequences for LSTM input.
        
        Args:
            data (pandas.DataFrame): Input data
            target_column (str): Target variable column name
        
        Returns:
            tuple: (X_sequences, y_values, scaler)
        """
        # Implementation details...
        pass
    
    def inverse_transform(self, predictions, scaler):
        """
        Converts scaled predictions back to original scale.
        
        Args:
            predictions (numpy.array): Scaled predictions
            scaler (MinMaxScaler): Fitted scaler object
        
        Returns:
            numpy.array: Original scale predictions
        """
        return scaler.inverse_transform(predictions)
```

### Model Loading and Saving

```python
import joblib
from tensorflow.keras.models import load_model

class ModelManager:
    """Manages model and scaler loading/saving operations."""
    
    def __init__(self, models_path="Notebooks/models/", scalers_path="Notebooks/scalers/"):
        self.models_path = models_path
        self.scalers_path = scalers_path
    
    def load_model(self, model_name):
        """
        Loads a trained LSTM model.
        
        Args:
            model_name (str): Name of the model file
        
        Returns:
            tensorflow.keras.Model: Loaded model
        """
        model_path = f"{self.models_path}{model_name}"
        return load_model(model_path)
    
    def load_scaler(self, scaler_name):
        """
        Loads a fitted scaler.
        
        Args:
            scaler_name (str): Name of the scaler file
        
        Returns:
            sklearn.preprocessing.MinMaxScaler: Fitted scaler
        """
        scaler_path = f"{self.scalers_path}{scaler_name}"
        return joblib.load(scaler_path)
    
    def load_all_models(self):
        """
        Loads all available models and scalers.
        
        Returns:
            dict: Dictionary containing models and scalers
        """
        models = {}
        scalers = {}
        
        model_files = [
            "temp2_max(c)_LSTM.h5",
            "temp2_min(c)_LSTM.h5",
            "temp2_ave(c)_LSTM.h5",
            "wind_speed50_max(ms)_LSTM.h5",
            "wind_speed50_min(ms)_LSTM.h5",
            "wind_speed50_ave(ms)_LSTM.h5",
            "suface_pressure(pa)_LSTM.h5",
            "prectotcorr_LSTM.h5",
            "total_demand(mw)_LSTM.h5"
        ]
        
        for model_file in model_files:
            target_name = model_file.replace("_LSTM.h5", "")
            models[target_name] = self.load_model(model_file)
            scalers[target_name] = self.load_scaler(f"{target_name}_scaler.pkl")
        
        return {"models": models, "scalers": scalers}
```

## Prediction Pipeline

### Single Model Prediction

```python
def predict_single_variable(model, scaler, input_data, days_ahead=365):
    """
    Makes predictions for a single target variable.
    
    Args:
        model: Trained LSTM model
        scaler: Fitted MinMaxScaler
        input_data: Preprocessed input sequences
        days_ahead: Number of days to predict
    
    Returns:
        numpy.array: Predictions in original scale
    """
    predictions = []
    current_sequence = input_data[-1:].copy()
    
    for _ in range(days_ahead):
        # Predict next value
        pred = model.predict(current_sequence, verbose=0)
        predictions.append(pred[0, 0])
        
        # Update sequence for next prediction
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, :] = pred[0, 0]
    
    # Inverse transform to original scale
    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions)
```

### Multi-Model Ensemble

```python
def ensemble_predictions(models, scalers, input_data, target_variables):
    """
    Generates predictions using multiple models.
    
    Args:
        models (dict): Dictionary of trained models
        scalers (dict): Dictionary of fitted scalers
        input_data: Preprocessed input data
        target_variables (list): List of target variable names
    
    Returns:
        pandas.DataFrame: Combined predictions for all variables
    """
    predictions = {}
    
    for target in target_variables:
        if target in models and target in scalers:
            pred = predict_single_variable(
                models[target], 
                scalers[target], 
                input_data
            )
            predictions[target] = pred.flatten()
    
    return pd.DataFrame(predictions)
```

## Model Performance Metrics

### Evaluation Functions

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(y_true, y_pred):
    """
    Calculates comprehensive model performance metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        dict: Performance metrics
    """
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
```

## Hyperparameter Optimization

The models use Optuna for automated hyperparameter tuning:

- **Learning Rate**: 0.001 - 0.1
- **LSTM Units**: 20 - 100
- **Dropout Rate**: 0.1 - 0.5
- **Batch Size**: 16 - 128
- **Sequence Length**: 10 - 60 days

## Model Validation

- **Time Series Split**: Chronological train/validation/test splits
- **Cross-Validation**: Time series cross-validation with rolling windows
- **Performance Tracking**: MLflow integration for experiment tracking

## Future Enhancements

- **Attention Mechanisms**: Transformer-based architectures
- **Multi-variate Models**: Single model predicting multiple variables
- **Uncertainty Quantification**: Bayesian LSTM implementations
- **Real-time Updates**: Online learning capabilities
