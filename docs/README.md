# AnalysingEnergy Documentation

This directory contains the documentation source files for the AnalysingEnergy project, built using Sphinx and designed for ReadTheDocs.

## Structure

```
docs/
├── conf.py                 # Sphinx configuration
├── index.rst               # Main documentation index
├── requirements.txt        # Documentation dependencies
├── _static/                # Static files (CSS, images)
│   └── custom.css         # Custom styling
├── api/                   # API reference documentation
│   ├── interface.md       # Streamlit interface API
│   └── models.md          # LSTM models API
├── notebooks/             # Notebook documentation
│   ├── data_preprocessing.md
│   ├── lstm_generation.md
│   ├── lstm_complete.md
│   └── predicting_365_days.md
├── tutorials/             # Step-by-step tutorials
│   ├── data_preprocessing.md
│   ├── model_training.md
│   └── making_predictions.md
├── getting_started.md     # Installation and setup
├── data_overview.md       # Dataset documentation
├── model_architecture.md  # Model design
├── contributing.md        # Contributing guidelines
├── changelog.md           # Version history
└── license.md            # License information
```

## Building Documentation

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Build HTML documentation:**
   ```bash
   # Using the batch file (Windows)
   make.bat html
   
   # Or directly with Sphinx
   sphinx-build -b html . _build/html
   ```

3. **Clean build directory:**
   ```bash
   make.bat clean
   ```

4. **Check external links:**
   ```bash
   make.bat linkcheck
   ```

### ReadTheDocs Deployment

The documentation is configured for automatic deployment on ReadTheDocs using:
- `.readthedocs.yaml` configuration file
- `requirements.txt` for dependencies
- Sphinx RTD theme

## Features

- **Responsive Design:** Works on desktop and mobile devices
- **Search Functionality:** Full-text search across all documentation
- **API Documentation:** Comprehensive API reference with examples
- **Notebook Integration:** Embedded Jupyter notebook documentation
- **Multiple Formats:** Available in HTML, PDF, and ePub formats
- **Cross-References:** Extensive internal linking between sections

## Contributing to Documentation

1. Edit the appropriate `.md` or `.rst` files
2. Build locally to test changes
3. Submit a pull request

For detailed contribution guidelines, see [contributing.md](contributing.md).

## Troubleshooting

### Common Issues

1. **Build fails with extension errors:**
   - Check that all dependencies in `requirements.txt` are installed
   - Verify Python version compatibility (3.8+ recommended)

2. **Missing API documentation:**
   - Ensure the main project is installed (`pip install -e .`)
   - Check that module imports work correctly

3. **Styling issues:**
   - Clear the `_build` directory and rebuild
   - Check `custom.css` for any syntax errors

### Getting Help

- Check the [Sphinx documentation](https://www.sphinx-doc.org/)
- Review [ReadTheDocs guides](https://docs.readthedocs.io/)
- Open an issue in the project repository
