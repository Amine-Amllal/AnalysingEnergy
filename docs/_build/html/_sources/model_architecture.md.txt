# Model Architecture

This section provides a detailed overview of the machine learning architecture used in the AnalysingEnergy project, focusing on LSTM neural networks and optimization techniques.

## Overview

The AnalysingEnergy project employs a sophisticated deep learning architecture based on Long Short-Term Memory (LSTM) neural networks, specifically designed for time series forecasting of energy-related parameters.

## Core Architecture

### LSTM Neural Network Design

The model uses a multi-layered LSTM architecture optimized for energy prediction:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_model(units_1=74, units_2=69, dropout_rate=0.194, activation='relu'):
    model = Sequential([
        # First LSTM layer with return sequences
        LSTM(units_1, activation=activation, return_sequences=True, 
             input_shape=(n_input, n_features)),
        
        # Second LSTM layer
        LSTM(units_2, activation=activation, return_sequences=False),
        
        # Dropout for regularization
        Dropout(dropout_rate),
        
        # Dense hidden layer
        Dense(50, activation='relu'),
        
        # Additional dropout
        Dropout(dropout_rate),
        
        # Output layer for regression
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model
```

### Architecture Components

#### 1. Input Layer
- **Input Shape**: `(n_input, n_features)` where:
  - `n_input = 1`: Time lag window
  - `n_features = 9`: Number of input features

#### 2. LSTM Layers
- **First LSTM Layer**: 
  - Units: 74 (optimized via Optuna)
  - Activation: ReLU
  - Return sequences: True (for stacking)
  
- **Second LSTM Layer**:
  - Units: 69 (optimized via Optuna)
  - Activation: ReLU
  - Return sequences: False (final LSTM output)

#### 3. Regularization
- **Dropout Rate**: 0.194 (optimized)
- **Applied After**: Each LSTM and Dense layer
- **Purpose**: Prevent overfitting

#### 4. Dense Layers
- **Hidden Layer**: 50 units with ReLU activation
- **Output Layer**: 1 unit for regression output

#### 5. Compilation
- **Optimizer**: Adam (adaptive learning rate)
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: RMSE for evaluation

## Hyperparameter Optimization

### Optuna Integration

The project uses Optuna for automated hyperparameter optimization:

```python
import optuna

def objective(trial):
    # Define hyperparameter search space
    units_1 = trial.suggest_int('units_1', 50, 150)
    units_2 = trial.suggest_int('units_2', 50, 100)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    activation = trial.suggest_categorical('activation', ['tanh', 'relu'])
    
    # Build and train model
    model = build_model(units_1, units_2, dropout_rate, activation)
    model.fit(generator, epochs=20, batch_size=32, verbose=0)
    
    # Return validation loss
    loss = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
    return loss

# Optimization study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

### Optimized Parameters

Best parameters found through optimization:

- **units_1**: 74
- **units_2**: 69  
- **dropout_rate**: 0.1938213639314652
- **activation**: 'relu'

## Data Preprocessing Pipeline

### Feature Scaling

```python
from sklearn.preprocessing import MinMaxScaler

# Separate scalers for features and targets
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Fit and transform training data
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))

# Transform test data using fitted scalers
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
```

### Time Series Generation

```python
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Create sequences for LSTM input
generator = TimeseriesGenerator(
    X_train_scaled, 
    y_train_scaled, 
    length=n_input,
    batch_size=1
)
```

## Multi-Model Architecture

### Individual Feature Models

The system includes specialized models for each feature:

1. **Temperature Models**:
   - `temp2_max(c)_LSTM.h5`
   - `temp2_min(c)_LSTM.h5`
   - `temp2_ave(c)_LSTM.h5`

2. **Wind Speed Models**:
   - `wind_speed50_max(ms)_LSTM.h5`
   - `wind_speed50_min(ms)_LSTM.h5`
   - `wind_speed50_ave(ms)_LSTM.h5`

3. **Atmospheric Models**:
   - `suface_pressure(pa)_LSTM.h5`
   - `prectotcorr_LSTM.h5`

4. **Energy Models**:
   - `total_demand(mw)_LSTM.h5`
   - `final_model 291.19.h5` (main generation model)

### Model Ensemble Strategy

```python
# Load multiple models for comprehensive prediction
models = {}
scalers = {}

for feature in features:
    # Load individual model and scaler
    model_path = f"models/{feature.replace('/', '')}_LSTM.h5"
    scaler_path = f"scalers/{feature.replace('/', '')}_scaler.pkl"
    
    models[feature] = load_model(model_path)
    scalers[feature] = joblib.load(scaler_path)
```

## Training Configuration

### Training Parameters

```python
training_config = {
    'epochs': 50,
    'batch_size': 32,
    'validation_split': 0.2,
    'early_stopping': {
        'monitor': 'val_loss',
        'patience': 10,
        'restore_best_weights': True
    }
}
```

### Performance Metrics

- **Primary Metric**: Root Mean Square Error (RMSE)
- **Final Model RMSE**: 291.19 MW
- **Training Loss**: Mean Squared Error (MSE)
- **Validation**: Time-based split to prevent data leakage

## Prediction Pipeline

### Single Prediction

```python
def predict_single_step(model, last_values, scaler_X, scaler_y):
    # Scale input
    scaled_input = scaler_X.transform(last_values)
    reshaped_input = scaled_input.reshape((1, n_input, n_features))
    
    # Predict
    scaled_prediction = model.predict(reshaped_input, verbose=0)
    
    # Inverse transform
    prediction = scaler_y.inverse_transform(scaled_prediction)
    return prediction[0, 0]
```

### Multi-step Forecasting

```python
def predict_future_365_days(model, data, scaler_X, scaler_y, n_days=365):
    predictions = []
    last_values = data.iloc[-1:].values
    
    for _ in range(n_days):
        # Predict next value
        next_pred = predict_single_step(model, last_values, scaler_X, scaler_y)
        predictions.append(next_pred)
        
        # Update input for next prediction
        last_values = np.roll(last_values, -1)
        last_values[0, -1] = next_pred
    
    return predictions
```

## Model Persistence

### Saving Models

```python
# Save trained model
model.save("models/final_model.h5")

# Save scalers
joblib.dump(scaler_X, "scalers/X_scaler.pkl")
joblib.dump(scaler_y, "scalers/y_scaler.pkl")
```

### Loading Models

```python
from tensorflow.keras.models import load_model
import joblib

# Load model with custom objects
model = load_model("models/final_model.h5", 
                  custom_objects={'mse': mean_squared_error})

# Load scalers
scaler_X = joblib.load("scalers/X_scaler.pkl")
scaler_y = joblib.load("scalers/y_scaler.pkl")
```

## Performance Optimization

### Memory Management
- **Batch Processing**: Efficient batch size selection
- **Memory Monitoring**: RAM usage optimization during training
- **Gradient Accumulation**: For large dataset handling

### Computational Efficiency
- **GPU Utilization**: CUDA support for TensorFlow
- **Parallel Processing**: Multi-threaded data loading
- **Model Compilation**: Optimized computation graphs

## Validation Strategy

### Time Series Cross-Validation
- **Walk-Forward Validation**: Chronological validation approach
- **Train-Test Split**: Temporal split (2018-2019 train, 2020+ test)
- **No Data Leakage**: Strict temporal ordering maintained

### Model Evaluation
- **Quantitative Metrics**: RMSE, MAE, MAPE
- **Qualitative Assessment**: Visual prediction plots
- **Residual Analysis**: Error pattern examination

This architecture provides robust and accurate energy forecasting capabilities while maintaining interpretability and scalability for future enhancements.
