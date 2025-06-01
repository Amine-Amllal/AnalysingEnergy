# Data Overview

This section provides a comprehensive overview of the datasets used in the AnalysingEnergy project, including data structure, features, and preprocessing steps.

## Dataset Structure

The project uses three main datasets located in the `Data/` directory:

### Core Datasets

1. **`data.csv`**: Complete dataset containing all historical records
2. **`train_data.csv`**: Training subset for model development
3. **`test_data.csv`**: Testing subset for model evaluation

## Data Features

The datasets contain the following features with temporal indexing:

### Temporal Index
- **`date`**: Date index in YYYY-MM-DD format, used as the primary time series index

### Meteorological Features

#### Temperature Measurements (°C)
- **`temp2_max(c)`**: Daily maximum temperature
- **`temp2_min(c)`**: Daily minimum temperature  
- **`temp2_ave(c)`**: Daily average temperature

#### Atmospheric Conditions
- **`suface_pressure(pa)`**: Surface atmospheric pressure in Pascals
- **`prectotcorr`**: Precipitation correlation coefficient

#### Wind Measurements (m/s)
- **`wind_speed50_max(m/s)`**: Daily maximum wind speed at 50m height
- **`wind_speed50_min(m/s)`**: Daily minimum wind speed at 50m height
- **`wind_speed50_ave(m/s)`**: Daily average wind speed at 50m height

### Energy Features

#### Energy Demand
- **`total_demand(mw)`**: Total energy demand in Megawatts

#### Energy Generation
- **`max_generation(mw)`**: Maximum energy generation capacity in Megawatts

## Data Characteristics

### Temporal Coverage
- **Time Period**: Starting from 2018-01-01
- **Frequency**: Daily measurements
- **Duration**: Multi-year historical data for comprehensive analysis

### Data Quality
- **Completeness**: High data completeness with minimal missing values
- **Consistency**: Standardized units and measurement intervals
- **Validation**: Data quality checks implemented in preprocessing notebooks

## Sample Data Structure

```python
import pandas as pd

# Example data structure
data.head()
```

| date       | temp2_max(c) | temp2_min(c) | temp2_ave(c) | suface_pressure(pa) | wind_speed50_max(m/s) | wind_speed50_min(m/s) | wind_speed50_ave(m/s) | prectotcorr | total_demand(mw) | max_generation(mw) |
|------------|--------------|--------------|--------------|---------------------|----------------------|----------------------|----------------------|-------------|------------------|-------------------|
| 2018-01-01 | 24.48        | 13.78        | 19.130       | 101.08              | 5.05                | 0.23                | 2.640               | 0.00        | 8000.0          | 7651.0           |
| 2018-01-02 | 23.16        | 15.28        | 19.220       | 100.94              | 6.20                | 1.59                | 3.895               | 0.01        | 7900.0          | 7782.0           |
| 2018-01-03 | 22.65        | 11.52        | 17.085       | 101.12              | 6.96                | 3.64                | 5.300               | 0.17        | 7900.0          | 7707.0           |

## Data Preprocessing

### Scaling and Normalization

The project employs MinMax scaling for all numerical features:

```python
from sklearn.preprocessing import MinMaxScaler

# Individual scalers for each feature
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Scaling applied to maintain feature relationships
X_scaled = scaler_X.fit_transform(X_features)
y_scaled = scaler_y.fit_transform(y_target)
```

### Feature Engineering

#### Lag Features
- **Time lag**: 1-day lag for sequence generation
- **Sequence length**: Optimized for LSTM input requirements

#### Target Variable Processing
- **Primary target**: `max_generation(mw)` for energy generation prediction
- **Secondary targets**: Individual meteorological parameters for comprehensive forecasting

### Data Splitting Strategy

```python
# Temporal split to maintain chronological order
train_data = data[:'2019-12-31']  # Training period
test_data = data['2020-01-01':]   # Testing period
```

## Data Relationships

### Correlation Analysis

Key correlations observed in the dataset:

1. **Temperature-Energy Relationship**:
   - Higher temperatures correlate with increased energy demand
   - Seasonal patterns in both temperature and energy consumption

2. **Wind-Generation Relationship**:
   - Wind speed variations impact renewable energy generation
   - Strong correlation between wind patterns and generation capacity

3. **Pressure-Weather Patterns**:
   - Surface pressure influences overall weather conditions
   - Indirect impact on energy demand through weather patterns

### Seasonal Patterns

- **Daily Cycles**: Clear daily patterns in energy demand
- **Seasonal Variations**: Temperature and wind patterns show seasonal trends
- **Weather Dependencies**: Energy generation correlates with meteorological conditions

## Data Quality Metrics

### Completeness
- **Missing Values**: < 1% across all features
- **Data Gaps**: No significant temporal gaps in the time series

### Consistency
- **Unit Standardization**: All measurements in consistent units
- **Temporal Alignment**: All features aligned to daily timestamps

### Validation Rules
- **Range Checks**: Values within expected physical limits
- **Outlier Detection**: Statistical outlier identification and handling
- **Temporal Consistency**: Chronological order validation

## Usage in Machine Learning Pipeline

### Feature Selection
```python
# Features used for prediction
features = [
    "temp2_max(c)", "temp2_min(c)", "temp2_ave(c)", 
    "suface_pressure(pa)", "wind_speed50_max(m/s)", "wind_speed50_min(m/s)", 
    "wind_speed50_ave(m/s)", "prectotcorr", "total_demand(mw)"
]

# Target variable
target = "max_generation(mw)"
```

### Time Series Preparation
```python
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Sequence generation for LSTM input
generator = TimeseriesGenerator(
    X_scaled, y_scaled, 
    length=n_input, 
    batch_size=batch_size
)
```

## Data Visualization

The project includes comprehensive data visualization for:

- **Time series plots**: Temporal patterns in all features
- **Correlation matrices**: Feature relationships
- **Distribution plots**: Statistical distributions of variables
- **Seasonal decomposition**: Trend, seasonal, and residual components

## Storage and Access

### File Formats
- **CSV format**: Human-readable and widely compatible
- **Pandas integration**: Direct loading with date parsing
- **Memory efficiency**: Optimized data types for processing

### Loading Data
```python
import pandas as pd

# Load with proper date indexing
data = pd.read_csv(
    "Data/data.csv", 
    index_col="date", 
    parse_dates=True
)
```

This data structure provides the foundation for all machine learning models and analysis in the AnalysingEnergy project.
