# Contributing to AnalysingEnergy

Thank you for your interest in contributing to the AnalysingEnergy project! This guide will help you understand how to contribute effectively to this energy prediction and analysis system.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Contributing Guidelines](#contributing-guidelines)
4. [Code Standards](#code-standards)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Submitting Contributions](#submitting-contributions)
8. [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.8 or higher
- Git for version control
- Basic understanding of machine learning and time series analysis
- Familiarity with TensorFlow/Keras and data science libraries

### Project Overview

AnalysingEnergy is a machine learning project that:
- Predicts energy generation and consumption using LSTM neural networks
- Analyzes meteorological data for energy forecasting
- Provides a Streamlit web interface for interactive predictions
- Supports 365-day long-term forecasting

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/yourusername/AnalysingEnergy.git
cd AnalysingEnergy

# Add the original repository as upstream
git remote add upstream https://github.com/originaluser/AnalysingEnergy.git
```

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt  # If available
```

### 3. Development Dependencies

Install additional tools for development:

```bash
pip install pytest pytest-cov black flake8 isort jupyter-lab
```

### 4. Verify Installation

```bash
# Test basic functionality
python -c "import tensorflow; import pandas; import streamlit; print('Setup successful!')"

# Run tests (if available)
pytest tests/
```

## Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

1. **Bug Reports**: Found an issue? Report it!
2. **Feature Requests**: Have an idea for improvement?
3. **Code Contributions**: Bug fixes, new features, optimizations
4. **Documentation**: Improve existing docs or add new ones
5. **Examples**: Jupyter notebooks, tutorials, use cases
6. **Performance Improvements**: Model optimization, code efficiency

### Before You Start

1. **Check existing issues**: Look for similar bug reports or feature requests
2. **Create an issue**: For significant changes, create an issue first to discuss
3. **Follow project standards**: Review this guide and existing code style

## Code Standards

### Python Code Style

We follow PEP 8 with some specific guidelines:

#### Formatting
```bash
# Format code with Black
black --line-length 88 your_file.py

# Sort imports with isort
isort your_file.py

# Check style with flake8
flake8 your_file.py
```

#### Naming Conventions
```python
# Variables and functions: snake_case
def load_energy_data():
    file_path = "data/energy_data.csv"
    return data

# Classes: PascalCase
class LSTMPredictor:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_SEQUENCE_LENGTH = 30
DEFAULT_BATCH_SIZE = 32
```

#### Documentation Strings
```python
def train_lstm_model(data, target_variable, epochs=100):
    """
    Train LSTM model for energy prediction.
    
    Args:
        data (pandas.DataFrame): Input training data
        target_variable (str): Name of target column
        epochs (int, optional): Number of training epochs. Defaults to 100.
    
    Returns:
        tensorflow.keras.Model: Trained LSTM model
        
    Raises:
        ValueError: If target_variable not in data columns
        
    Example:
        >>> model = train_lstm_model(data, 'total_demand(mw)')
        >>> predictions = model.predict(test_data)
    """
    pass
```

### Machine Learning Code Guidelines

#### Model Training
```python
# Good: Clear, reproducible model training
def create_lstm_model(input_shape, units=50, dropout_rate=0.2, random_seed=42):
    """Create reproducible LSTM model."""
    tf.random.set_seed(random_seed)
    
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(1, activation='linear')
    ])
    
    return model

# Good: Comprehensive evaluation
def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate model with multiple metrics."""
    predictions = model.predict(X_test)
    predictions_scaled = scaler.inverse_transform(predictions)
    
    metrics = {
        'mae': mean_absolute_error(y_test, predictions_scaled),
        'rmse': np.sqrt(mean_squared_error(y_test, predictions_scaled)),
        'r2': r2_score(y_test, predictions_scaled)
    }
    
    return metrics
```

#### Data Processing
```python
# Good: Robust data preprocessing
def preprocess_energy_data(data, handle_outliers=True, fill_missing=True):
    """
    Preprocess energy data with comprehensive cleaning.
    
    Args:
        data: Raw energy dataset
        handle_outliers: Whether to remove statistical outliers
        fill_missing: Whether to fill missing values
    
    Returns:
        Cleaned and preprocessed dataset
    """
    data_clean = data.copy()
    
    if fill_missing:
        data_clean = data_clean.fillna(method='ffill')
    
    if handle_outliers:
        data_clean = remove_outliers(data_clean)
    
    return data_clean
```

### File Organization

```
AnalysingEnergy/
├── Data/                    # Datasets
├── Notebooks/              # Jupyter notebooks
│   ├── models/             # Trained models
│   ├── scalers/           # Data scalers
│   └── *.ipynb           # Analysis notebooks
├── interface/              # Web application
├── src/                   # Source code modules (if adding)
│   ├── models/           # Model definitions
│   ├── data/             # Data processing
│   └── utils/            # Utility functions
├── tests/                 # Test files
├── docs/                  # Documentation
└── requirements.txt       # Dependencies
```

## Testing

### Writing Tests

Create tests for new functionality:

```python
# tests/test_models.py
import pytest
import numpy as np
import pandas as pd
from src.models.lstm_model import create_lstm_model, preprocess_data

def test_lstm_model_creation():
    """Test LSTM model creation with valid parameters."""
    input_shape = (30, 9)
    model = create_lstm_model(input_shape)
    
    assert model is not None
    assert len(model.layers) == 5  # 2 LSTM + 2 Dropout + 1 Dense
    assert model.input_shape == (None, 30, 9)

def test_data_preprocessing():
    """Test data preprocessing pipeline."""
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'temp_max': np.random.normal(20, 5, 100),
        'wind_speed': np.random.normal(10, 3, 100),
        'energy_demand': np.random.normal(1000, 200, 100)
    }, index=dates)
    
    # Add some missing values
    data.iloc[10:15, 0] = np.nan
    
    processed_data = preprocess_data(data)
    
    assert processed_data.isnull().sum().sum() == 0
    assert len(processed_data) == len(data)

@pytest.fixture
def sample_energy_data():
    """Fixture providing sample energy data for tests."""
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    return pd.DataFrame({
        'temp2_max(c)': np.random.normal(20, 10, 365),
        'wind_speed50_ave(ms)': np.random.normal(8, 3, 365),
        'total_demand(mw)': np.random.normal(1500, 300, 365)
    }, index=dates)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v
```

## Documentation

### Adding Documentation

1. **Code Documentation**: Use comprehensive docstrings
2. **README Updates**: Update README.md for significant changes
3. **Notebook Documentation**: Add markdown cells explaining code
4. **API Documentation**: Document new functions and classes

### Documentation Style

```python
def predict_energy_demand(model, features, scaler, days_ahead=30):
    """
    Predict energy demand for specified number of days.
    
    This function uses a trained LSTM model to generate energy demand
    predictions based on meteorological features.
    
    Args:
        model (tensorflow.keras.Model): Trained LSTM model
        features (numpy.ndarray): Input features for prediction
        scaler (sklearn.preprocessing.MinMaxScaler): Fitted scaler for inverse transform
        days_ahead (int, optional): Number of days to predict. Defaults to 30.
    
    Returns:
        numpy.ndarray: Predicted energy demand values
        
    Raises:
        ValueError: If input features have incorrect shape
        TypeError: If model is not a Keras model
        
    Examples:
        >>> model = load_model('energy_model.h5')
        >>> features = np.random.rand(1, 30, 9)
        >>> predictions = predict_energy_demand(model, features, scaler)
        >>> print(f"Predicted demand: {predictions[0]:.2f} MW")
        
    Note:
        Ensure that input features are scaled using the same scaler
        that was used during model training.
    """
    pass
```

### Jupyter Notebook Guidelines

```python
# Cell 1: Title and Overview
"""
# Energy Demand Prediction Analysis

This notebook demonstrates the implementation of LSTM models for 
predicting energy demand based on meteorological data.

## Objectives:
1. Load and preprocess energy data
2. Train LSTM model
3. Evaluate model performance
4. Generate predictions
"""

# Cell 2: Imports with explanations
"""
## Library Imports

We use the following libraries:
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- tensorflow: Deep learning framework
- matplotlib: Data visualization
"""
import pandas as pd
import numpy as np
# ... other imports

# Cell 3: Data loading with description
"""
## Data Loading

Loading the energy dataset which contains:
- Meteorological features (temperature, wind speed, pressure)
- Energy demand values
- Temporal information (daily frequency)
"""
```

## Submitting Contributions

### Creating Pull Requests

1. **Create a branch**: 
```bash
git checkout -b feature/your-feature-name
```

2. **Make changes**: Implement your feature or fix
3. **Test thoroughly**: Ensure all tests pass
4. **Commit with clear messages**:
```bash
git commit -m "Add energy demand forecasting module

- Implement LSTM-based demand prediction
- Add comprehensive error handling
- Include unit tests and documentation
- Optimize for 365-day forecasting"
```

4. **Push to your fork**:
```bash
git push origin feature/your-feature-name
```

5. **Create Pull Request**: Use GitHub interface

### Pull Request Guidelines

#### Title and Description
- **Clear title**: Summarize the change in one line
- **Detailed description**: Explain what, why, and how
- **Link issues**: Reference related issues with `Fixes #123`

#### Checklist
Before submitting, ensure:
- [ ] Code follows project style guidelines
- [ ] Tests added for new functionality
- [ ] All tests pass
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
- [ ] Performance impact considered

### Review Process

1. **Automated checks**: CI/CD pipeline runs tests
2. **Code review**: Maintainers review your code
3. **Feedback incorporation**: Address review comments
4. **Merge**: After approval, code is merged

## Community Guidelines

### Communication

- **Be respectful**: Treat all community members with respect
- **Be constructive**: Provide helpful feedback and suggestions
- **Be patient**: Remember that this is a volunteer-driven project
- **Ask questions**: Don't hesitate to ask for help or clarification

### Code of Conduct

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Getting Help

If you need help:

1. **Check documentation**: Review existing docs and examples
2. **Search issues**: Look for similar questions or problems
3. **Create an issue**: For bugs or questions
4. **Join discussions**: Participate in issue discussions

## Development Roadmap

### Current Priorities

1. **Model Improvements**:
   - Implement attention mechanisms
   - Add ensemble methods
   - Optimize hyperparameters

2. **Interface Enhancements**:
   - Real-time data integration
   - Advanced visualization options
   - Export functionality

3. **Documentation**:
   - Complete API documentation
   - Tutorial videos
   - Best practices guide

### Future Goals

- Multi-region energy predictions
- Integration with external weather APIs
- Deployment automation
- Mobile application

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- Project documentation

Thank you for contributing to AnalysingEnergy!
