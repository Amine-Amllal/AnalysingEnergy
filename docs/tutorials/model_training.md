# Model Training Tutorial

This comprehensive tutorial guides you through training LSTM models for energy prediction in the AnalysingEnergy project.

## Overview

This tutorial covers:

1. Setting up the training environment
2. Building the LSTM architecture
3. Hyperparameter optimization with Optuna
4. Training the model
5. Model evaluation and validation
6. Saving and loading models

## Prerequisites

Ensure you have completed the {doc}`data_preprocessing` tutorial and have the following installed:

```bash
pip install tensorflow optuna scikit-learn matplotlib seaborn joblib
```

## Step 1: Environment Setup

### Import Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Optimization
import optuna

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
```

### Load Preprocessed Data

```python
# Load the preprocessed datasets
train_data = pd.read_csv("Data/train_data.csv", index_col="date", parse_dates=True)
test_data = pd.read_csv("Data/test_data.csv", index_col="date", parse_dates=True)

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Define features and target
features = [
    "temp2_max(c)", "temp2_min(c)", "temp2_ave(c)", 
    "suface_pressure(pa)", "wind_speed50_max(m/s)", "wind_speed50_min(m/s)", 
    "wind_speed50_ave(m/s)", "prectotcorr", "total_demand(mw)"
]

target = "max_generation(mw)"
```

## Step 2: Data Preparation

### Prepare Features and Target

```python
# Extract features and target
X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

print(f"Feature matrix shapes - Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Target vector shapes - Train: {y_train.shape}, Test: {y_test.shape}")
```

### Scale the Data

```python
# Initialize scalers
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Fit on training data
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

# Transform test data
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# Save scalers
joblib.dump(scaler_X, "scalers/X_train_scaler.pkl")
joblib.dump(scaler_y, "scalers/y_train_scaler.pkl")

print("Data scaling completed and scalers saved")
```

### Create Time Series Sequences

```python
# Define sequence parameters
n_input = 1  # Number of time steps to look back
n_features = X_train.shape[1]  # Number of features

# Create time series generator for training
train_generator = TimeseriesGenerator(
    X_train_scaled, 
    y_train_scaled, 
    length=n_input,
    batch_size=1
)

# Prepare test data
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], n_input, n_features))

print(f"Training sequences: {len(train_generator)}")
print(f"Test data shape: {X_test_reshaped.shape}")
```

## Step 3: Model Architecture

### Define Model Building Function

```python
def build_lstm_model(units_1=100, units_2=50, dropout_rate=0.2, activation='tanh'):
    """
    Build LSTM model with specified parameters
    
    Parameters:
    -----------
    units_1 : int
        Number of units in first LSTM layer
    units_2 : int
        Number of units in second LSTM layer
    dropout_rate : float
        Dropout rate for regularization
    activation : str
        Activation function for LSTM layers
    
    Returns:
    --------
    model : tensorflow.keras.Model
        Compiled LSTM model
    """
    model = Sequential([
        # First LSTM layer with return sequences
        LSTM(units_1, 
             activation=activation, 
             return_sequences=True, 
             input_shape=(n_input, n_features)),
        
        # Second LSTM layer
        LSTM(units_2, 
             activation=activation, 
             return_sequences=False),
        
        # Dropout for regularization
        Dropout(dropout_rate),
        
        # Dense hidden layer
        Dense(50, activation='relu'),
        
        # Additional dropout
        Dropout(dropout_rate),
        
        # Output layer for regression
        Dense(1)
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Test model building
test_model = build_lstm_model()
print(test_model.summary())
```

## Step 4: Hyperparameter Optimization

### Define Optuna Objective Function

```python
def objective(trial):
    """
    Optuna objective function for hyperparameter optimization
    
    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object
    
    Returns:
    --------
    float
        Validation loss to minimize
    """
    # Suggest hyperparameters
    units_1 = trial.suggest_int('units_1', 50, 150)
    units_2 = trial.suggest_int('units_2', 50, 100)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    activation = trial.suggest_categorical('activation', ['tanh', 'relu'])
    
    # Build model with suggested parameters
    model = build_lstm_model(
        units_1=units_1,
        units_2=units_2,
        dropout_rate=dropout_rate,
        activation=activation
    )
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=(X_test_reshaped, y_test_scaled),
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Return best validation loss
    best_loss = min(history.history['val_loss'])
    return best_loss
```

### Run Hyperparameter Optimization

```python
# Create study
study = optuna.create_study(direction='minimize')

# Run optimization
print("Starting hyperparameter optimization...")
study.optimize(objective, n_trials=20, timeout=3600)  # 1 hour timeout

# Get best parameters
best_params = study.best_params
print(f"Best parameters: {best_params}")
print(f"Best validation loss: {study.best_value}")
```

### Analyze Optimization Results

```python
# Plot optimization history
optuna.visualization.plot_optimization_history(study).show()

# Plot parameter importance
optuna.visualization.plot_param_importances(study).show()

# Plot parameter relationships
optuna.visualization.plot_parallel_coordinate(study).show()
```

## Step 5: Train Final Model

### Build Final Model with Best Parameters

```python
# Use optimized parameters (or use the found best parameters)
best_params = {
    'units_1': 74,
    'units_2': 69,
    'dropout_rate': 0.1938213639314652,
    'activation': 'relu'
}

# Build final model
final_model = build_lstm_model(**best_params)

print("Final model architecture:")
print(final_model.summary())
```

### Define Training Callbacks

```python
# Define callbacks for training
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='models/best_model_checkpoint.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]
```

### Train the Model

```python
# Train the final model
print("Training final model...")
history = final_model.fit(
    train_generator,
    epochs=50,
    validation_data=(X_test_reshaped, y_test_scaled),
    callbacks=callbacks,
    verbose=1,
    batch_size=32
)

print("Training completed!")
```

### Visualize Training History

```python
# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_title('Model Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# MAE plot
ax2.plot(history.history['mae'], label='Training MAE')
ax2.plot(history.history['val_mae'], label='Validation MAE')
ax2.set_title('Model MAE')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Mean Absolute Error')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

## Step 6: Model Evaluation

### Make Predictions

```python
# Make predictions on test data
y_pred_scaled = final_model.predict(X_test_reshaped)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Convert to 1D arrays for evaluation
y_test_actual = y_test.values
y_pred_final = y_pred.flatten()

print(f"Predictions shape: {y_pred_final.shape}")
print(f"Actual values shape: {y_test_actual.shape}")
```

### Calculate Performance Metrics

```python
# Calculate metrics
mse = mean_squared_error(y_test_actual, y_pred_final)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, y_pred_final)
mape = np.mean(np.abs((y_test_actual - y_pred_final) / y_test_actual)) * 100

print("Model Performance Metrics:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")

# R-squared
r2 = 1 - (np.sum((y_test_actual - y_pred_final) ** 2) / 
          np.sum((y_test_actual - np.mean(y_test_actual)) ** 2))
print(f"R²: {r2:.4f}")
```

### Visualize Predictions

```python
# Create prediction plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# Time series plot
ax1.plot(test_data.index, y_test_actual, label='Actual', alpha=0.8)
ax1.plot(test_data.index, y_pred_final, label='Predicted', alpha=0.8)
ax1.set_title('Actual vs Predicted Energy Generation')
ax1.set_xlabel('Date')
ax1.set_ylabel('Max Generation (MW)')
ax1.legend()
ax1.grid(True)

# Scatter plot
ax2.scatter(y_test_actual, y_pred_final, alpha=0.6)
ax2.plot([y_test_actual.min(), y_test_actual.max()], 
         [y_test_actual.min(), y_test_actual.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Generation (MW)')
ax2.set_ylabel('Predicted Generation (MW)')
ax2.set_title('Prediction Scatter Plot')
ax2.grid(True)

# Add R² to scatter plot
ax2.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax2.transAxes, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()
```

### Residual Analysis

```python
# Calculate residuals
residuals = y_test_actual - y_pred_final

# Residual plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Residuals vs predictions
ax1.scatter(y_pred_final, residuals, alpha=0.6)
ax1.axhline(y=0, color='r', linestyle='--')
ax1.set_xlabel('Predicted Values')
ax1.set_ylabel('Residuals')
ax1.set_title('Residuals vs Predicted')
ax1.grid(True)

# Residuals histogram
ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Residuals')
ax2.set_ylabel('Frequency')
ax2.set_title('Residuals Distribution')
ax2.grid(True)

# Residuals over time
ax3.plot(test_data.index, residuals, alpha=0.7)
ax3.axhline(y=0, color='r', linestyle='--')
ax3.set_xlabel('Date')
ax3.set_ylabel('Residuals')
ax3.set_title('Residuals Over Time')
ax3.grid(True)

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=ax4)
ax4.set_title('Q-Q Plot of Residuals')
ax4.grid(True)

plt.tight_layout()
plt.show()
```

## Step 7: Model Persistence

### Save the Final Model

```python
# Save the trained model
model_filename = f"models/final_model_{rmse:.2f}.h5"
final_model.save(model_filename)

print(f"Model saved as: {model_filename}")

# Save training history
import pickle
with open('models/training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

print("Training history saved")
```

### Save Model Metadata

```python
# Create model metadata
model_metadata = {
    'model_file': model_filename,
    'training_date': pd.Timestamp.now().isoformat(),
    'architecture': best_params,
    'performance': {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    },
    'training_params': {
        'epochs': len(history.history['loss']),
        'n_input': n_input,
        'n_features': n_features,
        'features': features,
        'target': target
    },
    'data_info': {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_period': [str(train_data.index.min()), str(train_data.index.max())],
        'test_period': [str(test_data.index.min()), str(test_data.index.max())]
    }
}

# Save metadata
with open('models/model_metadata.json', 'w') as f:
    import json
    json.dump(model_metadata, f, indent=2)

print("Model metadata saved")
```

## Step 8: Model Loading and Verification

### Load and Test Saved Model

```python
from tensorflow.keras.models import load_model

# Load the saved model
loaded_model = load_model(model_filename, custom_objects={'mse': mean_squared_error})

# Verify model by making predictions
test_predictions = loaded_model.predict(X_test_reshaped[:5])
original_predictions = final_model.predict(X_test_reshaped[:5])

# Check if predictions match
predictions_match = np.allclose(test_predictions, original_predictions)
print(f"Model loading verification: {'✓ Passed' if predictions_match else '✗ Failed'}")
```

## Step 9: Model Performance Summary

### Create Performance Report

```python
def create_performance_report(model_metadata):
    """Create a formatted performance report"""
    
    print("="*60)
    print("MODEL PERFORMANCE REPORT")
    print("="*60)
    print(f"Model File: {model_metadata['model_file']}")
    print(f"Training Date: {model_metadata['training_date']}")
    print()
    
    print("ARCHITECTURE:")
    for param, value in model_metadata['architecture'].items():
        print(f"  {param}: {value}")
    print()
    
    print("PERFORMANCE METRICS:")
    perf = model_metadata['performance']
    print(f"  RMSE: {perf['rmse']:.2f} MW")
    print(f"  MAE: {perf['mae']:.2f} MW")
    print(f"  MAPE: {perf['mape']:.2f}%")
    print(f"  R²: {perf['r2']:.4f}")
    print()
    
    print("TRAINING DETAILS:")
    train = model_metadata['training_params']
    print(f"  Epochs: {train['epochs']}")
    print(f"  Features: {train['n_features']}")
    print(f"  Sequence Length: {train['n_input']}")
    print()
    
    print("DATA INFORMATION:")
    data = model_metadata['data_info']
    print(f"  Training Samples: {data['train_samples']}")
    print(f"  Test Samples: {data['test_samples']}")
    print(f"  Training Period: {data['train_period'][0]} to {data['train_period'][1]}")
    print(f"  Test Period: {data['test_period'][0]} to {data['test_period'][1]}")
    print("="*60)

# Generate report
create_performance_report(model_metadata)
```

## Best Practices and Tips

### Training Best Practices

1. **Data Quality**: Ensure clean, properly preprocessed data
2. **Temporal Splitting**: Always use temporal splits for time series
3. **Scaling**: Fit scalers only on training data
4. **Early Stopping**: Prevent overfitting with early stopping
5. **Model Checkpoints**: Save best models during training
6. **Reproducibility**: Set random seeds for consistent results

### Hyperparameter Optimization Tips

1. **Search Space**: Define reasonable parameter ranges
2. **Budget**: Balance optimization time vs. improvement
3. **Cross-Validation**: Use proper validation for time series
4. **Multiple Runs**: Run optimization multiple times for robustness

### Performance Evaluation

1. **Multiple Metrics**: Use RMSE, MAE, MAPE, and R²
2. **Residual Analysis**: Check for patterns in residuals
3. **Visual Inspection**: Always plot predictions vs. actual values
4. **Domain Knowledge**: Validate results against domain expertise

## Troubleshooting

### Common Issues and Solutions

1. **Memory Errors**: Reduce batch size or use data generators
2. **Slow Training**: Check GPU usage, reduce model complexity
3. **Poor Performance**: Check data quality, feature engineering
4. **Overfitting**: Increase dropout, add regularization
5. **Underfitting**: Increase model complexity, more epochs

This completes the model training tutorial. The trained model is now ready for making predictions on new data. Next, proceed to the {doc}`making_predictions` tutorial to learn how to use the model for forecasting.
