# Making Predictions Tutorial

This tutorial demonstrates how to use the trained LSTM models to make predictions for energy generation and related parameters in the AnalysingEnergy project.

## Overview

This tutorial covers:

1. Loading trained models and scalers
2. Single-step prediction
3. Multi-step forecasting (365 days)
4. Ensemble predictions using multiple models
5. Prediction confidence intervals
6. Visualization and interpretation
7. Exporting results

## Prerequisites

Complete the {doc}`model_training` tutorial and ensure you have:

- Trained LSTM models saved in `models/` directory
- Fitted scalers saved in `scalers/` directory
- Test data available for validation

## Step 1: Environment Setup

### Import Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print(f"TensorFlow version: {tf.__version__}")
```

### Load Data and Configuration

```python
# Load the test dataset
test_data = pd.read_csv("Data/test_data.csv", index_col="date", parse_dates=True)
full_data = pd.read_csv("Data/data.csv", index_col="date", parse_dates=True)

# Define features and target
features = [
    "temp2_max(c)", "temp2_min(c)", "temp2_ave(c)", 
    "suface_pressure(pa)", "wind_speed50_max(m/s)", "wind_speed50_min(m/s)", 
    "wind_speed50_ave(m/s)", "prectotcorr", "total_demand(mw)"
]

target = "max_generation(mw)"

print(f"Test data shape: {test_data.shape}")
print(f"Full data shape: {full_data.shape}")
print(f"Test period: {test_data.index.min()} to {test_data.index.max()}")
```

## Step 2: Load Trained Models and Scalers

### Load Main Model

```python
# Load the main energy generation model
model_path = "models/final_model_291.19.h5"
main_model = load_model(model_path, custom_objects={'mse': mean_squared_error})

print("Main model loaded successfully")
print(main_model.summary())
```

### Load Scalers

```python
# Load the fitted scalers
scaler_X = joblib.load("scalers/X_train_scaler.pkl")
scaler_y = joblib.load("scalers/y_train_scaler.pkl")

print("Scalers loaded successfully")
print(f"Feature scaler range: {scaler_X.data_min_} to {scaler_X.data_max_}")
```

### Load Individual Feature Models

```python
import os
from pathlib import Path

def load_feature_models(models_dir="models", scalers_dir="scalers"):
    """Load all individual feature models and their scalers"""
    
    models = {}
    scalers = {}
    
    # List of individual features with models
    feature_files = [
        "temp2_max(c)", "temp2_min(c)", "temp2_ave(c)",
        "suface_pressure(pa)", "wind_speed50_max(ms)", "wind_speed50_min(ms)",
        "wind_speed50_ave(ms)", "prectotcorr", "total_demand(mw)"
    ]
    
    for feature in feature_files:
        try:
            # Load model
            model_file = f"{feature}_LSTM.h5"
            model_path = os.path.join(models_dir, model_file)
            if os.path.exists(model_path):
                models[feature] = load_model(model_path, custom_objects={'mse': mean_squared_error})
                
            # Load scaler
            scaler_file = f"{feature}_scaler.pkl"
            scaler_path = os.path.join(scalers_dir, scaler_file)
            if os.path.exists(scaler_path):
                scalers[feature] = joblib.load(scaler_path)
                
        except Exception as e:
            print(f"Warning: Could not load {feature}: {e}")
    
    print(f"Loaded {len(models)} individual models")
    print(f"Loaded {len(scalers)} individual scalers")
    
    return models, scalers

# Load individual models and scalers
individual_models, individual_scalers = load_feature_models()
```

## Step 3: Single-Step Prediction

### Prepare Input Data

```python
def prepare_input_data(data, features, scaler_X, n_input=1):
    """
    Prepare input data for LSTM prediction
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    features : list
        List of feature column names
    scaler_X : sklearn scaler
        Fitted feature scaler
    n_input : int
        Sequence length for LSTM
    
    Returns:
    --------
    np.array
        Scaled and reshaped data ready for LSTM input
    """
    # Extract features
    X = data[features].values
    
    # Scale features
    X_scaled = scaler_X.transform(X)
    
    # Reshape for LSTM input
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], n_input, X_scaled.shape[1]))
    
    return X_reshaped, X_scaled

# Prepare test data
X_test_reshaped, X_test_scaled = prepare_input_data(test_data, features, scaler_X)
print(f"Prepared input shape: {X_test_reshaped.shape}")
```

### Make Single Predictions

```python
def make_single_predictions(model, X_input, scaler_y):
    """
    Make single-step predictions
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Trained LSTM model
    X_input : np.array
        Prepared input data
    scaler_y : sklearn scaler
        Fitted target scaler
    
    Returns:
    --------
    np.array
        Predictions in original scale
    """
    # Make predictions
    y_pred_scaled = model.predict(X_input, verbose=0)
    
    # Inverse transform to original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    return y_pred.flatten()

# Make predictions on test data
test_predictions = make_single_predictions(main_model, X_test_reshaped, scaler_y)

print(f"Generated {len(test_predictions)} predictions")
print(f"Prediction range: {test_predictions.min():.2f} to {test_predictions.max():.2f} MW")
```

### Evaluate Single Predictions

```python
# Calculate performance metrics
y_test_actual = test_data[target].values

mse = mean_squared_error(y_test_actual, test_predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, test_predictions)
mape = np.mean(np.abs((y_test_actual - test_predictions) / y_test_actual)) * 100

print("Single-Step Prediction Performance:")
print(f"RMSE: {rmse:.2f} MW")
print(f"MAE: {mae:.2f} MW")
print(f"MAPE: {mape:.2f}%")

# R-squared
r2 = 1 - (np.sum((y_test_actual - test_predictions) ** 2) / 
          np.sum((y_test_actual - np.mean(y_test_actual)) ** 2))
print(f"R²: {r2:.4f}")
```

## Step 4: Multi-Step Forecasting (365 Days)

### Implement Future Prediction Function

```python
def predict_future_days(model, last_data, scaler_X, scaler_y, features, n_days=365, n_input=1):
    """
    Predict future values for specified number of days
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Trained LSTM model
    last_data : pd.DataFrame
        Last known data points
    scaler_X : sklearn scaler
        Feature scaler
    scaler_y : sklearn scaler
        Target scaler
    features : list
        Feature column names
    n_days : int
        Number of days to predict
    n_input : int
        Sequence length
    
    Returns:
    --------
    list
        Future predictions
    """
    predictions = []
    
    # Get last values for features
    last_values = last_data[features].iloc[-n_input:].values
    
    for day in range(n_days):
        # Scale the input
        scaled_input = scaler_X.transform(last_values)
        reshaped_input = scaled_input.reshape((1, n_input, len(features)))
        
        # Make prediction
        scaled_prediction = model.predict(reshaped_input, verbose=0)
        prediction = scaler_y.inverse_transform(scaled_prediction)[0, 0]
        
        predictions.append(prediction)
        
        # Update input for next prediction (simple approach - use last known values)
        # In practice, you might want to predict features as well
        last_values = np.roll(last_values, -1, axis=0)
        if len(last_values) > 0:
            last_values[-1] = last_values[-2]  # Use previous values as approximation
    
    return predictions

# Make 365-day predictions
print("Generating 365-day forecast...")
future_predictions = predict_future_days(
    main_model, 
    full_data, 
    scaler_X, 
    scaler_y, 
    features, 
    n_days=365
)

print(f"Generated {len(future_predictions)} future predictions")
print(f"Future prediction range: {min(future_predictions):.2f} to {max(future_predictions):.2f} MW")
```

### Create Future Dates

```python
# Create future date index
last_date = full_data.index[-1]
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1), 
    periods=365, 
    freq='D'
)

# Create future predictions DataFrame
future_df = pd.DataFrame({
    'predicted_generation': future_predictions
}, index=future_dates)

print(f"Future predictions period: {future_df.index.min()} to {future_df.index.max()}")
```

## Step 5: Ensemble Predictions with Individual Models

### Predict Individual Features for Future

```python
def predict_individual_features(individual_models, individual_scalers, last_data, features, n_days=365):
    """
    Predict individual features using their specific models
    
    Parameters:
    -----------
    individual_models : dict
        Dictionary of individual feature models
    individual_scalers : dict
        Dictionary of individual feature scalers
    last_data : pd.DataFrame
        Last known data
    features : list
        Feature names
    n_days : int
        Number of days to predict
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with predicted features
    """
    feature_predictions = {}
    
    for feature in features:
        # Map feature names to file names
        feature_key = feature.replace('/', '')
        if '(m/s)' in feature:
            feature_key = feature_key.replace('(m/s)', '(ms)')
        
        if feature_key in individual_models and feature_key in individual_scalers:
            try:
                model = individual_models[feature_key]
                scaler = individual_scalers[feature_key]
                
                # Get last values for this feature
                last_values = last_data[[feature]].iloc[-1:].values
                
                predictions = []
                for day in range(n_days):
                    # Scale input
                    scaled_input = scaler.transform(last_values)
                    reshaped_input = scaled_input.reshape((1, 1, 1))
                    
                    # Predict
                    scaled_pred = model.predict(reshaped_input, verbose=0)
                    pred = scaler.inverse_transform(scaled_pred)[0, 0]
                    
                    predictions.append(pred)
                    
                    # Update for next prediction
                    last_values = np.array([[pred]])
                
                feature_predictions[feature] = predictions
                print(f"Predicted {feature}: {len(predictions)} values")
                
            except Exception as e:
                print(f"Warning: Could not predict {feature}: {e}")
                # Use simple extrapolation as fallback
                last_value = last_data[feature].iloc[-1]
                feature_predictions[feature] = [last_value] * n_days
        else:
            # Use simple extrapolation for missing models
            last_value = last_data[feature].iloc[-1]
            feature_predictions[feature] = [last_value] * n_days
    
    return pd.DataFrame(feature_predictions, index=future_dates)

# Predict individual features
print("Predicting individual features...")
future_features = predict_individual_features(
    individual_models, 
    individual_scalers, 
    full_data, 
    features
)

print(f"Predicted features shape: {future_features.shape}")
```

### Ensemble Prediction

```python
def ensemble_prediction(main_model, future_features, scaler_X, scaler_y):
    """
    Make ensemble predictions using predicted features
    
    Parameters:
    -----------
    main_model : tensorflow.keras.Model
        Main LSTM model
    future_features : pd.DataFrame
        Predicted feature values
    scaler_X : sklearn scaler
        Feature scaler
    scaler_y : sklearn scaler
        Target scaler
    
    Returns:
    --------
    np.array
        Ensemble predictions
    """
    # Prepare features
    X_future, _ = prepare_input_data(future_features, features, scaler_X)
    
    # Make predictions
    ensemble_preds = make_single_predictions(main_model, X_future, scaler_y)
    
    return ensemble_preds

# Make ensemble predictions
ensemble_predictions = ensemble_prediction(main_model, future_features, scaler_X, scaler_y)

# Add to future DataFrame
future_df['ensemble_prediction'] = ensemble_predictions
future_df['simple_prediction'] = future_predictions

print("Ensemble predictions completed")
```

## Step 6: Prediction Confidence Intervals

### Calculate Prediction Uncertainty

```python
def calculate_prediction_intervals(predictions, confidence=0.95):
    """
    Calculate prediction intervals based on historical prediction errors
    
    Parameters:
    -----------
    predictions : np.array
        Point predictions
    confidence : float
        Confidence level (e.g., 0.95 for 95% confidence)
    
    Returns:
    --------
    tuple
        (lower_bound, upper_bound)
    """
    # Calculate prediction errors from test set
    test_errors = y_test_actual - test_predictions
    
    # Calculate standard deviation of errors
    error_std = np.std(test_errors)
    
    # Calculate confidence intervals
    alpha = 1 - confidence
    z_score = 1.96  # For 95% confidence interval
    
    margin_of_error = z_score * error_std
    
    lower_bound = predictions - margin_of_error
    upper_bound = predictions + margin_of_error
    
    return lower_bound, upper_bound

# Calculate confidence intervals
lower_bound, upper_bound = calculate_prediction_intervals(ensemble_predictions)

future_df['lower_bound'] = lower_bound
future_df['upper_bound'] = upper_bound

print(f"Confidence intervals calculated with ±{upper_bound[0] - ensemble_predictions[0]:.2f} MW margin")
```

## Step 7: Visualization and Analysis

### Plot Predictions

```python
def plot_predictions(historical_data, future_predictions, test_data=None, test_preds=None):
    """
    Create comprehensive prediction visualization
    
    Parameters:
    -----------
    historical_data : pd.DataFrame
        Historical data
    future_predictions : pd.DataFrame
        Future predictions with confidence intervals
    test_data : pd.DataFrame, optional
        Test data for validation
    test_preds : np.array, optional
        Test predictions
    """
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Historical + Future predictions
    ax1 = axes[0]
    
    # Historical data (last 2 years)
    recent_data = historical_data.last('730D')
    ax1.plot(recent_data.index, recent_data[target], 
             label='Historical', color='blue', alpha=0.8, linewidth=1.5)
    
    # Future predictions
    ax1.plot(future_predictions.index, future_predictions['ensemble_prediction'], 
             label='Future Prediction', color='red', linewidth=2)
    
    # Confidence intervals
    ax1.fill_between(future_predictions.index, 
                     future_predictions['lower_bound'],
                     future_predictions['upper_bound'],
                     alpha=0.3, color='red', label='95% Confidence Interval')
    
    ax1.set_title('Energy Generation: Historical Data and Future Predictions', fontsize=14)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Max Generation (MW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Test data validation (if provided)
    if test_data is not None and test_preds is not None:
        ax2 = axes[1]
        
        ax2.plot(test_data.index, test_data[target], 
                 label='Actual', color='blue', alpha=0.8)
        ax2.plot(test_data.index, test_preds, 
                 label='Predicted', color='red', alpha=0.8)
        
        ax2.set_title('Model Validation on Test Data', fontsize=14)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Max Generation (MW)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Create visualization
plot_predictions(full_data, future_df, test_data, test_predictions)
```

### Seasonal Analysis

```python
def analyze_seasonal_patterns(future_predictions):
    """
    Analyze seasonal patterns in predictions
    
    Parameters:
    -----------
    future_predictions : pd.DataFrame
        Future predictions with date index
    """
    
    # Add temporal features
    df_analysis = future_predictions.copy()
    df_analysis['month'] = df_analysis.index.month
    df_analysis['quarter'] = df_analysis.index.quarter
    df_analysis['day_of_year'] = df_analysis.index.dayofyear
    
    # Monthly averages
    monthly_avg = df_analysis.groupby('month')['ensemble_prediction'].mean()
    
    # Quarterly averages
    quarterly_avg = df_analysis.groupby('quarter')['ensemble_prediction'].mean()
    
    # Plot seasonal patterns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Monthly pattern
    monthly_avg.plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Predicted Average Generation by Month')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Average Generation (MW)')
    ax1.grid(True, alpha=0.3)
    
    # Quarterly pattern
    quarterly_avg.plot(kind='bar', ax=ax2, color='lightcoral')
    ax2.set_title('Predicted Average Generation by Quarter')
    ax2.set_xlabel('Quarter')
    ax2.set_ylabel('Average Generation (MW)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return monthly_avg, quarterly_avg

# Analyze seasonal patterns
monthly_patterns, quarterly_patterns = analyze_seasonal_patterns(future_df)

print("Seasonal Analysis:")
print("\nMonthly Averages:")
for month, avg in monthly_patterns.items():
    print(f"Month {month}: {avg:.2f} MW")

print("\nQuarterly Averages:")
for quarter, avg in quarterly_patterns.items():
    print(f"Q{quarter}: {avg:.2f} MW")
```

### Feature Importance Analysis

```python
def analyze_predicted_features(future_features):
    """
    Analyze the predicted features
    
    Parameters:
    -----------
    future_features : pd.DataFrame
        Predicted features
    """
    
    # Calculate correlations between predicted features and generation
    correlations = future_features.corrwith(future_df['ensemble_prediction'])
    
    # Plot correlations
    plt.figure(figsize=(10, 6))
    correlations.abs().sort_values(ascending=True).plot(kind='barh')
    plt.title('Feature Correlations with Predicted Generation')
    plt.xlabel('Absolute Correlation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return correlations

# Analyze feature correlations
feature_correlations = analyze_predicted_features(future_features)
print("\nFeature Correlations with Predicted Generation:")
for feature, corr in feature_correlations.abs().sort_values(ascending=False).items():
    print(f"{feature}: {corr:.3f}")
```

## Step 8: Export Results

### Save Predictions

```python
def export_predictions(future_df, future_features, filename_prefix="predictions"):
    """
    Export predictions to various formats
    
    Parameters:
    -----------
    future_df : pd.DataFrame
        Future predictions
    future_features : pd.DataFrame
        Predicted features
    filename_prefix : str
        Prefix for output files
    """
    
    # Create results directory
    import os
    os.makedirs("results", exist_ok=True)
    
    # Export main predictions
    future_df.to_csv(f"results/{filename_prefix}_generation.csv")
    print(f"Generation predictions saved to results/{filename_prefix}_generation.csv")
    
    # Export predicted features
    future_features.to_csv(f"results/{filename_prefix}_features.csv")
    print(f"Feature predictions saved to results/{filename_prefix}_features.csv")
    
    # Export combined results
    combined_results = pd.concat([future_features, future_df], axis=1)
    combined_results.to_csv(f"results/{filename_prefix}_combined.csv")
    print(f"Combined results saved to results/{filename_prefix}_combined.csv")
    
    # Export summary statistics
    summary_stats = {
        'prediction_period': [str(future_df.index.min()), str(future_df.index.max())],
        'total_predictions': len(future_df),
        'average_generation': future_df['ensemble_prediction'].mean(),
        'min_generation': future_df['ensemble_prediction'].min(),
        'max_generation': future_df['ensemble_prediction'].max(),
        'std_generation': future_df['ensemble_prediction'].std(),
        'confidence_interval_width': (future_df['upper_bound'] - future_df['lower_bound']).mean()
    }
    
    import json
    with open(f"results/{filename_prefix}_summary.json", 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    print(f"Summary statistics saved to results/{filename_prefix}_summary.json")
    
    return combined_results

# Export all results
combined_results = export_predictions(future_df, future_features)
```

### Create Prediction Report

```python
def create_prediction_report(future_df, test_metrics):
    """
    Create a comprehensive prediction report
    
    Parameters:
    -----------
    future_df : pd.DataFrame
        Future predictions
    test_metrics : dict
        Test performance metrics
    """
    
    report = f"""
ENERGY GENERATION PREDICTION REPORT
=====================================

PREDICTION OVERVIEW:
- Prediction Period: {future_df.index.min().strftime('%Y-%m-%d')} to {future_df.index.max().strftime('%Y-%m-%d')}
- Total Days Predicted: {len(future_df)}
- Model Type: LSTM Neural Network

PREDICTION STATISTICS:
- Average Generation: {future_df['ensemble_prediction'].mean():.2f} MW
- Minimum Generation: {future_df['ensemble_prediction'].min():.2f} MW
- Maximum Generation: {future_df['ensemble_prediction'].max():.2f} MW
- Standard Deviation: {future_df['ensemble_prediction'].std():.2f} MW

CONFIDENCE INTERVALS:
- Confidence Level: 95%
- Average Interval Width: {(future_df['upper_bound'] - future_df['lower_bound']).mean():.2f} MW

MODEL VALIDATION PERFORMANCE:
- RMSE: {test_metrics['rmse']:.2f} MW
- MAE: {test_metrics['mae']:.2f} MW
- MAPE: {test_metrics['mape']:.2f}%
- R²: {test_metrics['r2']:.4f}

SEASONAL PATTERNS:
- Highest Predicted Month: Month {monthly_patterns.idxmax()} ({monthly_patterns.max():.2f} MW)
- Lowest Predicted Month: Month {monthly_patterns.idxmin()} ({monthly_patterns.min():.2f} MW)
- Seasonal Variation: {(monthly_patterns.max() - monthly_patterns.min()):.2f} MW

RECOMMENDATIONS:
1. Monitor actual vs predicted values for model recalibration
2. Consider weather forecast integration for improved accuracy
3. Update model quarterly with new data
4. Validate predictions against domain expertise

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Save report
    with open("results/prediction_report.txt", 'w') as f:
        f.write(report)
    
    print("Prediction report saved to results/prediction_report.txt")
    print(report)

# Create prediction report
test_metrics = {
    'rmse': rmse,
    'mae': mae,
    'mape': mape,
    'r2': r2
}

create_prediction_report(future_df, test_metrics)
```

## Best Practices for Predictions

### Model Maintenance

1. **Regular Updates**: Retrain models with new data quarterly
2. **Performance Monitoring**: Track prediction accuracy over time
3. **Feature Drift**: Monitor for changes in feature distributions
4. **Domain Validation**: Validate predictions with energy experts

### Uncertainty Quantification

1. **Confidence Intervals**: Always provide uncertainty estimates
2. **Ensemble Methods**: Use multiple models for robust predictions
3. **Scenario Analysis**: Consider different prediction scenarios
4. **Error Analysis**: Understand prediction limitations

### Practical Applications

1. **Energy Planning**: Use predictions for capacity planning
2. **Market Operations**: Support energy trading decisions
3. **Grid Management**: Assist in grid stability planning
4. **Investment Decisions**: Inform renewable energy investments

This tutorial provides a comprehensive framework for making reliable energy predictions using the trained LSTM models. The predictions can be used for strategic planning, operational decisions, and long-term energy management.
