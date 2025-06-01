# Data Preprocessing Tutorial

This tutorial walks you through the data preprocessing steps used in the AnalysingEnergy project, from raw data loading to model-ready datasets.

## Overview

Data preprocessing is a crucial step in the AnalysingEnergy pipeline. This tutorial covers:

1. Data loading and exploration
2. Data cleaning and validation
3. Feature engineering
4. Scaling and normalization
5. Time series preparation
6. Train-test splitting

## Prerequisites

Before starting, ensure you have the required libraries installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Step 1: Data Loading

### Loading the Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the main dataset with proper date parsing
data = pd.read_csv(
    "Data/data.csv", 
    index_col="date", 
    parse_dates=True
)

print(f"Dataset shape: {data.shape}")
print(f"Date range: {data.index.min()} to {data.index.max()}")
data.head()
```

### Initial Data Exploration

```python
# Basic information about the dataset
print("Dataset Info:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

print("\nMissing Values:")
print(data.isnull().sum())
```

## Step 2: Data Quality Assessment

### Check for Missing Values

```python
# Visualize missing values
plt.figure(figsize=(12, 6))
sns.heatmap(data.isnull(), yticklabels=False, cbar=True, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Calculate missing value percentages
missing_percentages = (data.isnull().sum() / len(data)) * 100
print("Missing value percentages:")
for col, pct in missing_percentages.items():
    if pct > 0:
        print(f"{col}: {pct:.2f}%")
```

### Data Validation

```python
# Check for duplicate dates
duplicates = data.index.duplicated().sum()
print(f"Duplicate dates: {duplicates}")

# Check for unrealistic values
def validate_ranges(data):
    validation_rules = {
        'temp2_max(c)': (-50, 60),
        'temp2_min(c)': (-50, 60),
        'temp2_ave(c)': (-50, 60),
        'suface_pressure(pa)': (80, 120),
        'wind_speed50_max(m/s)': (0, 50),
        'wind_speed50_min(m/s)': (0, 50),
        'wind_speed50_ave(m/s)': (0, 50),
        'prectotcorr': (0, 1),
        'total_demand(mw)': (0, 20000),
        'max_generation(mw)': (0, 20000)
    }
    
    for column, (min_val, max_val) in validation_rules.items():
        if column in data.columns:
            outliers = ((data[column] < min_val) | (data[column] > max_val)).sum()
            print(f"{column}: {outliers} values outside range [{min_val}, {max_val}]")

validate_ranges(data)
```

## Step 3: Data Cleaning

### Handle Missing Values

```python
# Forward fill for small gaps (up to 3 days)
data_cleaned = data.copy()

for column in data_cleaned.columns:
    # Forward fill small gaps
    data_cleaned[column] = data_cleaned[column].fillna(method='ffill', limit=3)
    
    # Backward fill remaining gaps
    data_cleaned[column] = data_cleaned[column].fillna(method='bfill', limit=3)

print(f"Remaining missing values: {data_cleaned.isnull().sum().sum()}")
```

### Handle Outliers

```python
from scipy import stats

def remove_outliers_zscore(data, threshold=3):
    """Remove outliers using Z-score method"""
    data_no_outliers = data.copy()
    
    for column in data.select_dtypes(include=[np.number]).columns:
        z_scores = np.abs(stats.zscore(data[column].dropna()))
        outlier_mask = z_scores > threshold
        
        # Replace outliers with median
        median_value = data[column].median()
        data_no_outliers.loc[data[column].index[outlier_mask], column] = median_value
        
        print(f"{column}: {outlier_mask.sum()} outliers replaced")
    
    return data_no_outliers

# Apply outlier removal
data_cleaned = remove_outliers_zscore(data_cleaned)
```

## Step 4: Feature Engineering

### Create Additional Features

```python
# Temperature-based features
data_cleaned['temp_range'] = data_cleaned['temp2_max(c)'] - data_cleaned['temp2_min(c)']
data_cleaned['temp_volatility'] = data_cleaned['temp2_ave(c)'].rolling(7).std()

# Wind-based features
data_cleaned['wind_range'] = data_cleaned['wind_speed50_max(m/s)'] - data_cleaned['wind_speed50_min(m/s)']
data_cleaned['wind_volatility'] = data_cleaned['wind_speed50_ave(m/s)'].rolling(7).std()

# Energy efficiency ratio
data_cleaned['energy_efficiency'] = data_cleaned['max_generation(mw)'] / data_cleaned['total_demand(mw)']

# Temporal features
data_cleaned['month'] = data_cleaned.index.month
data_cleaned['day_of_year'] = data_cleaned.index.dayofyear
data_cleaned['quarter'] = data_cleaned.index.quarter

print(f"New dataset shape with engineered features: {data_cleaned.shape}")
```

### Lag Features

```python
def create_lag_features(data, columns, lags=[1, 7, 30]):
    """Create lag features for specified columns"""
    data_with_lags = data.copy()
    
    for column in columns:
        for lag in lags:
            lag_column = f"{column}_lag_{lag}"
            data_with_lags[lag_column] = data[column].shift(lag)
    
    return data_with_lags

# Create lag features for key variables
lag_columns = ['temp2_ave(c)', 'wind_speed50_ave(m/s)', 'total_demand(mw)']
data_with_lags = create_lag_features(data_cleaned, lag_columns)

print(f"Dataset with lag features: {data_with_lags.shape}")
```

## Step 5: Feature Selection

### Define Core Features

```python
# Core features for LSTM model
core_features = [
    "temp2_max(c)", "temp2_min(c)", "temp2_ave(c)", 
    "suface_pressure(pa)", "wind_speed50_max(m/s)", "wind_speed50_min(m/s)", 
    "wind_speed50_ave(m/s)", "prectotcorr", "total_demand(mw)"
]

# Target variable
target = "max_generation(mw)"

# Prepare feature matrix and target vector
X = data_cleaned[core_features].copy()
y = data_cleaned[target].copy()

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
```

### Correlation Analysis

```python
# Correlation matrix
correlation_matrix = data_cleaned[core_features + [target]].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Feature importance based on correlation with target
target_correlation = correlation_matrix[target].abs().sort_values(ascending=False)
print("Feature importance (correlation with target):")
print(target_correlation[1:])  # Exclude self-correlation
```

## Step 6: Data Scaling

### MinMax Scaling

```python
from sklearn.preprocessing import MinMaxScaler
import joblib

# Initialize scalers
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Fit scalers on training data (we'll define train/test split next)
# For now, fit on entire dataset
X_scaled = scaler_X.fit_transform(X.dropna())
y_scaled = scaler_y.fit_transform(y.dropna().values.reshape(-1, 1))

# Save scalers for later use
joblib.dump(scaler_X, "scalers/X_scaler.pkl")
joblib.dump(scaler_y, "scalers/y_scaler.pkl")

print("Scaling completed and scalers saved")
```

### Verify Scaling

```python
# Check scaling results
print("Original data ranges:")
print(X.describe())

print("\nScaled data ranges:")
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print(X_scaled_df.describe())
```

## Step 7: Train-Test Split

### Temporal Split

```python
# Define split date
split_date = '2020-01-01'

# Create temporal split
train_mask = data_cleaned.index < split_date
test_mask = data_cleaned.index >= split_date

# Split features and target
X_train = X[train_mask].dropna()
X_test = X[test_mask].dropna()
y_train = y[train_mask].dropna()
y_test = y[test_mask].dropna()

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training period: {X_train.index.min()} to {X_train.index.max()}")
print(f"Test period: {X_test.index.min()} to {X_test.index.max()}")
```

### Scale Split Data

```python
# Re-fit scalers on training data only
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

# Transform test data using training scalers
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# Save final scalers
joblib.dump(scaler_X, "scalers/X_train_scaler.pkl")
joblib.dump(scaler_y, "scalers/y_train_scaler.pkl")

print("Final scaling completed")
```

## Step 8: Time Series Preparation

### Create LSTM Sequences

```python
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Parameters for LSTM
n_input = 1  # Sequence length
n_features = X_train.shape[1]  # Number of features

# Create time series generator
generator = TimeseriesGenerator(
    X_train_scaled, 
    y_train_scaled, 
    length=n_input,
    batch_size=1
)

print(f"Generated {len(generator)} sequences")
print(f"Input shape per sequence: {generator[0][0].shape}")
print(f"Output shape per sequence: {generator[0][1].shape}")
```

### Prepare Test Data

```python
# Reshape test data for LSTM input
X_test_scaled_reshaped = X_test_scaled.reshape(
    (X_test_scaled.shape[0], n_input, n_features)
)

print(f"Test data reshaped: {X_test_scaled_reshaped.shape}")
```

## Step 9: Save Processed Data

### Export Processed Datasets

```python
# Save processed training and test sets
train_processed = pd.DataFrame(X_train_scaled, 
                              columns=core_features, 
                              index=X_train.index)
train_processed[target] = scaler_y.inverse_transform(y_train_scaled).flatten()

test_processed = pd.DataFrame(X_test_scaled, 
                             columns=core_features, 
                             index=X_test.index)
test_processed[target] = scaler_y.inverse_transform(y_test_scaled).flatten()

# Save to CSV
train_processed.to_csv("Data/train_data_processed.csv")
test_processed.to_csv("Data/test_data_processed.csv")

print("Processed datasets saved")
```

## Step 10: Data Visualization

### Plot Processed Data

```python
# Plot original vs processed data comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Temperature
axes[0,0].plot(data_cleaned.index, data_cleaned['temp2_ave(c)'], alpha=0.7)
axes[0,0].set_title('Average Temperature')
axes[0,0].set_ylabel('Temperature (°C)')

# Wind Speed
axes[0,1].plot(data_cleaned.index, data_cleaned['wind_speed50_ave(m/s)'], alpha=0.7)
axes[0,1].set_title('Average Wind Speed')
axes[0,1].set_ylabel('Wind Speed (m/s)')

# Energy Demand
axes[1,0].plot(data_cleaned.index, data_cleaned['total_demand(mw)'], alpha=0.7)
axes[1,0].set_title('Energy Demand')
axes[1,0].set_ylabel('Demand (MW)')

# Energy Generation
axes[1,1].plot(data_cleaned.index, data_cleaned['max_generation(mw)'], alpha=0.7)
axes[1,1].set_title('Energy Generation')
axes[1,1].set_ylabel('Generation (MW)')

plt.tight_layout()
plt.show()
```

## Summary

This preprocessing pipeline:

1. **Loaded and validated** the raw energy dataset
2. **Cleaned** missing values and outliers
3. **Engineered** additional features for better prediction
4. **Selected** core features for model training
5. **Scaled** data using MinMax normalization
6. **Split** data temporally for proper validation
7. **Prepared** sequences for LSTM input
8. **Saved** processed data and scalers

The processed data is now ready for LSTM model training. The next step would be to follow the {doc}`model_training` tutorial to build and train the neural network.

## Best Practices

- **Always** use temporal splitting for time series data
- **Save scalers** fitted on training data only
- **Validate** data ranges and check for anomalies
- **Engineer features** based on domain knowledge
- **Visualize** data at each preprocessing step
- **Document** all preprocessing decisions for reproducibility
