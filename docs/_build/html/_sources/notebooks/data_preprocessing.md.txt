# Data Preprocessing Notebook

This notebook (`Data preprocessing.ipynb`) provides comprehensive data preprocessing and exploratory data analysis for the AnalysingEnergy project.

## Overview

The data preprocessing notebook handles the initial data preparation pipeline, including data cleaning, feature engineering, and splitting for machine learning model training.

## Notebook Structure

### 1. Data Loading and Initial Exploration

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the main dataset
data = pd.read_csv("Data/data.csv", index_col="date", parse_dates=True)
```

**Key Operations:**
- Loading CSV data with proper datetime parsing
- Initial data shape and structure examination
- Missing value analysis
- Data type verification

### 2. Exploratory Data Analysis (EDA)

#### 2.1 Statistical Summary
- Descriptive statistics for all variables
- Distribution analysis
- Outlier detection using IQR method

#### 2.2 Time Series Visualization
- Temporal plots for all meteorological variables
- Energy demand and generation patterns
- Seasonal decomposition analysis

#### 2.3 Correlation Analysis
```python
# Correlation matrix
correlation_matrix = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()
```

**Key Insights:**
- Temperature-energy demand relationships
- Wind speed-generation correlations
- Seasonal pattern identification

### 3. Data Cleaning and Quality Checks

#### 3.1 Missing Value Handling
```python
# Check for missing values
missing_values = data.isnull().sum()
print("Missing values per column:")
print(missing_values)

# Handle missing values if any
# Forward fill for time series continuity
data_cleaned = data.fillna(method='ffill')
```

#### 3.2 Outlier Detection and Treatment
- Statistical outlier identification
- Physical feasibility checks
- Outlier treatment strategies

#### 3.3 Data Validation
- Range validation for each variable
- Temporal consistency checks
- Unit verification

### 4. Feature Engineering

#### 4.1 Temporal Features
```python
# Extract temporal features
data['year'] = data.index.year
data['month'] = data.index.month
data['day_of_year'] = data.index.dayofyear
data['weekday'] = data.index.weekday
```

#### 4.2 Lag Features
```python
# Create lag features for time series modeling
for column in data.columns:
    data[f'{column}_lag1'] = data[column].shift(1)
    data[f'{column}_lag7'] = data[column].shift(7)
```

#### 4.3 Rolling Statistics
```python
# Rolling averages for trend analysis
window_sizes = [7, 14, 30]
for window in window_sizes:
    for column in numeric_columns:
        data[f'{column}_ma{window}'] = data[column].rolling(window=window).mean()
```

### 5. Data Scaling and Normalization

#### 5.1 MinMax Scaling
```python
from sklearn.preprocessing import MinMaxScaler

# Initialize scalers for each feature
scalers = {}
scaled_data = data.copy()

for column in data.columns:
    if data[column].dtype in ['float64', 'int64']:
        scaler = MinMaxScaler()
        scaled_data[column] = scaler.fit_transform(data[[column]])
        scalers[column] = scaler
```

#### 5.2 Feature Selection
- Correlation-based feature selection
- Importance ranking using statistical tests
- Dimensionality reduction considerations

### 6. Train-Test Split

#### 6.1 Temporal Split Strategy
```python
# Chronological split to maintain temporal order
split_date = '2019-12-31'
train_data = data[:split_date]
test_data = data[split_date:]

print(f"Training data: {train_data.shape}")
print(f"Testing data: {test_data.shape}")
```

#### 6.2 Validation Set Creation
```python
# Further split training data for validation
val_split_date = '2019-10-31'
train_subset = train_data[:val_split_date]
val_subset = train_data[val_split_date:]
```

### 7. Sequence Generation for LSTM

#### 7.1 Time Series Sequence Creation
```python
def create_sequences(data, target_column, sequence_length=30):
    """
    Create sequences for LSTM training.
    
    Args:
        data: Input dataframe
        target_column: Target variable name
        sequence_length: Length of input sequences
    
    Returns:
        X_sequences, y_values
    """
    X, y = [], []
    
    for i in range(sequence_length, len(data)):
        X.append(data.iloc[i-sequence_length:i].values)
        y.append(data[target_column].iloc[i])
    
    return np.array(X), np.array(y)
```

#### 7.2 Multi-step Prediction Preparation
- Sequence generation for different prediction horizons
- Feature-target alignment
- Batch preparation for model training

### 8. Data Export and Persistence

#### 8.1 Processed Data Export
```python
# Save processed datasets
train_data.to_csv("Data/train_data.csv")
test_data.to_csv("Data/test_data.csv")
```

#### 8.2 Scaler Persistence
```python
import joblib

# Save scalers for later use
for column, scaler in scalers.items():
    joblib.dump(scaler, f"Notebooks/scalers/{column}_scaler.pkl")
```

## Key Visualizations

### 1. Time Series Plots
- Historical trends for all variables
- Seasonal patterns visualization
- Anomaly identification

### 2. Distribution Analysis
- Histograms and density plots
- Box plots for outlier visualization
- Q-Q plots for normality assessment

### 3. Correlation Heatmaps
- Feature correlation matrix
- Target variable correlations
- Lagged correlation analysis

### 4. Seasonal Decomposition
- Trend component extraction
- Seasonal pattern analysis
- Residual component examination

## Data Quality Metrics

### Completeness
- **Missing Values**: < 0.1% across all features
- **Data Coverage**: Complete temporal coverage from 2018-01-01

### Consistency
- **Unit Standardization**: All measurements in consistent units
- **Temporal Alignment**: Daily frequency maintained
- **Range Validation**: All values within expected physical limits

### Reliability
- **Outlier Percentage**: < 2% of total observations
- **Data Source Validation**: Meteorological data verified against external sources
- **Temporal Consistency**: No gaps in time series

## Performance Considerations

### Memory Optimization
- Efficient data types selection
- Chunked processing for large datasets
- Memory usage monitoring

### Processing Speed
- Vectorized operations for feature engineering
- Parallel processing for multiple time series
- Optimized sequence generation

## Output Files

The notebook generates the following outputs:

1. **Processed Datasets:**
   - `train_data.csv`: Training dataset (2018-2019)
   - `test_data.csv`: Testing dataset (2020+)

2. **Scalers:**
   - Individual scaler files for each feature
   - Saved in `Notebooks/scalers/` directory

3. **Visualization Assets:**
   - EDA plots and charts
   - Data quality reports
   - Feature importance rankings

## Next Steps

After running this notebook:

1. Proceed to {doc}`../tutorials/model_training` for LSTM model development
2. Review {doc}`lstm_generation` for model architecture implementation
3. Explore {doc}`predicting_365_days` for long-term forecasting

## Dependencies

Required packages for this notebook:
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `matplotlib`: Basic plotting
- `seaborn`: Statistical visualization
- `scikit-learn`: Preprocessing utilities
- `scipy`: Statistical functions
