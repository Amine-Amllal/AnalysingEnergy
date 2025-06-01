# Changelog

All notable changes to the AnalysingEnergy project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- ReadTheDocs documentation structure
- Comprehensive API documentation
- Jupyter notebook documentation
- Contributing guidelines
- Changelog file

### Changed
- Improved code organization
- Enhanced documentation structure

### Fixed
- Documentation formatting issues

## [2.1.0] - 2025-06-01

### Added
- 365-day long-term energy forecasting capability
- Advanced ensemble prediction methods
- Uncertainty quantification with confidence intervals
- Seasonal pattern analysis in forecasts
- Interactive Plotly visualizations for forecasts
- Bootstrap sampling for prediction intervals
- Multiple export formats (CSV, Excel, JSON)
- Comprehensive forecast reporting system

### Changed
- Enhanced LSTM model architecture with deeper networks
- Improved prediction accuracy through ensemble methods
- Optimized memory usage for long-term predictions
- Updated Streamlit interface with new visualization options

### Fixed
- Memory leak issues in long-term prediction loops
- Scaling inconsistencies in multi-step predictions
- Date handling in future prediction generation

## [2.0.0] - 2025-05-15

### Added
- Optuna hyperparameter optimization integration
- Multi-target LSTM model training pipeline
- Advanced data preprocessing with outlier detection
- Model performance monitoring and logging
- Automated model selection based on validation metrics
- Cross-validation for time series models
- Enhanced error handling and logging system

### Changed
- **BREAKING**: Refactored model training pipeline
- **BREAKING**: Updated data preprocessing interface
- Improved model architecture with configurable layers
- Enhanced scalability for multiple target variables
- Updated requirements with version specifications

### Removed
- **BREAKING**: Deprecated old manual hyperparameter tuning methods
- Legacy data loading functions

### Fixed
- Training instability with large datasets
- Incorrect scaling in certain edge cases
- Model convergence issues with specific parameter combinations

## [1.3.0] - 2025-04-20

### Added
- Streamlit web interface for interactive predictions
- Real-time model inference capabilities
- Interactive data visualization dashboard
- Model comparison tools in web interface
- Export functionality for predictions
- User-friendly file upload system

### Changed
- Enhanced user experience with improved UI/UX
- Optimized model loading for faster inference
- Improved error messages and user feedback

### Fixed
- Interface responsiveness issues
- File upload validation problems
- Visualization rendering bugs on different screen sizes

## [1.2.0] - 2025-03-10

### Added
- Individual LSTM models for each target variable:
  - Temperature (max, min, average) prediction models
  - Wind speed (max, min, average) prediction models
  - Surface pressure prediction model
  - Precipitation correlation prediction model
  - Total energy demand prediction model
- Model persistence and loading system
- Comprehensive model evaluation metrics
- Data scaler persistence for consistent preprocessing

### Changed
- Improved model training efficiency
- Enhanced data preprocessing pipeline
- Better separation of concerns in model architecture

### Fixed
- Model overfitting issues through improved regularization
- Data leakage in time series splitting
- Inconsistent scaling across different model training sessions

## [1.1.0] - 2025-02-15

### Added
- Advanced LSTM neural network architecture
- Time series sequence generation for LSTM input
- Comprehensive data preprocessing pipeline
- Model training with early stopping and checkpointing
- Performance evaluation with multiple metrics (MAE, RMSE, R², MAPE)
- Data visualization capabilities

### Changed
- Upgraded from basic statistical models to deep learning approach
- Improved prediction accuracy significantly
- Enhanced data handling for time series analysis

### Fixed
- Data preprocessing inconsistencies
- Memory usage optimization
- Training stability improvements

## [1.0.0] - 2025-01-01

### Added
- Initial project structure
- Basic data loading and exploration capabilities
- Core dataset with meteorological and energy features:
  - Temperature measurements (max, min, average)
  - Wind speed data at 50m height
  - Surface pressure measurements
  - Precipitation correlation values
  - Energy demand and generation data
- Basic statistical analysis tools
- Data quality assessment utilities
- Initial documentation and README

### Project Features at v1.0.0
- **Data Management**: CSV-based dataset handling
- **Analysis Tools**: Basic statistical analysis and visualization
- **Data Quality**: Missing value detection and basic cleaning
- **Documentation**: Initial project documentation

---

## Version History Summary

| Version | Release Date | Key Features |
|---------|--------------|--------------|
| [Unreleased] | TBD | ReadTheDocs documentation |
| 2.1.0 | 2025-06-01 | 365-day forecasting, ensemble methods |
| 2.0.0 | 2025-05-15 | Optuna optimization, multi-target training |
| 1.3.0 | 2025-04-20 | Streamlit web interface |
| 1.2.0 | 2025-03-10 | Individual LSTM models |
| 1.1.0 | 2025-02-15 | LSTM neural networks |
| 1.0.0 | 2025-01-01 | Initial release |

## Breaking Changes

### v2.0.0
- **Model Training Interface**: The `train_model()` function signature has changed
  ```python
  # Old (v1.x)
  model = train_model(data, target, epochs)
  
  # New (v2.x)
  model = train_model(data, target, config=training_config)
  ```

- **Data Preprocessing**: New preprocessing pipeline requires different imports
  ```python
  # Old (v1.x)
  from src.preprocessing import preprocess_data
  
  # New (v2.x)
  from src.data.preprocessing import DataPreprocessor
  preprocessor = DataPreprocessor()
  processed_data = preprocessor.fit_transform(data)
  ```

### v1.2.0
- **Model File Structure**: Models are now saved individually per target variable
- **Scaler Persistence**: Scalers must be saved and loaded separately

## Migration Guides

### Migrating from v1.x to v2.0.0

1. **Update Training Code**:
   ```python
   # Create training configuration
   training_config = {
       'epochs': 100,
       'batch_size': 32,
       'optimization': True,
       'validation_split': 0.2
   }
   
   # Use new training interface
   from src.models.multi_target_lstm import MultiTargetLSTM
   trainer = MultiTargetLSTM(target_variables)
   models = trainer.train_all_models(train_data, val_data, config=training_config)
   ```

2. **Update Preprocessing**:
   ```python
   from src.data.preprocessing import DataPreprocessor
   
   preprocessor = DataPreprocessor()
   train_data, test_data = preprocessor.preprocess_data(raw_data)
   ```

### Migrating from v2.0.x to v2.1.0

1. **Long-term Predictions**:
   ```python
   from src.models.long_term_predictor import LongTermPredictor
   
   predictor = LongTermPredictor()
   predictor.load_models_and_scalers()
   predictions_365 = predictor.predict_future(data, days_ahead=365)
   ```

## Known Issues

### Current Issues (as of latest version)
- Large memory usage during 365-day prediction generation
- Occasional convergence issues with very sparse datasets
- Web interface may be slow with very large datasets

### Fixed Issues
- ✅ Model overfitting (fixed in v1.2.0)
- ✅ Data leakage in time series splits (fixed in v1.2.0)
- ✅ Training instability (fixed in v2.0.0)
- ✅ Memory leaks in prediction loops (fixed in v2.1.0)

## Deprecation Notices

### Scheduled for Removal

#### v3.0.0 (Planned)
- Legacy statistical models (deprecated since v1.1.0)
- Old preprocessing utilities (deprecated since v2.0.0)
- Manual hyperparameter tuning methods (deprecated since v2.0.0)

### Recently Removed
- Basic statistical prediction methods (removed in v1.1.0)
- Legacy data loading functions (removed in v2.0.0)

## Contributors

We thank all contributors who have helped improve AnalysingEnergy:

### Core Team
- Project maintainers and lead developers

### Contributors by Version

#### v2.1.0
- Enhanced long-term forecasting capabilities
- Improved documentation and examples
- Performance optimizations

#### v2.0.0
- Hyperparameter optimization integration
- Multi-target model architecture
- Advanced preprocessing pipeline

#### v1.3.0
- Streamlit web interface development
- User experience improvements
- Interactive visualization features

#### v1.2.0 - v1.0.0
- Initial LSTM implementation
- Core project structure
- Basic functionality and features

## Acknowledgments

Special thanks to:
- TensorFlow team for the deep learning framework
- Streamlit team for the web application framework
- Optuna team for hyperparameter optimization tools
- The open-source community for various supporting libraries

---

For more detailed information about any release, please check the corresponding GitHub release notes and documentation.
