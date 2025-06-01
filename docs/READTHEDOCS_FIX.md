# ReadTheDocs Build Troubleshooting Guide

## ✅ Fix Applied: Package Installation Error

**Problem:** Build failed at `python -m pip install --upgrade --upgrade-strategy only-if-needed --no-cache-dir .`

**Root Cause:** ReadTheDocs was trying to install the project as a Python package, but the project lacked proper package configuration.

**Solution Applied:**

### 1. Updated ReadTheDocs Configuration
- **File:** `.readthedocs.yaml`
- **Change:** Removed the `- method: pip path: .` line that was causing the package installation attempt
- **Result:** ReadTheDocs now only installs documentation dependencies from `docs/requirements.txt`

### 2. Minimized Documentation Dependencies
- **File:** `docs/requirements.txt`
- **Change:** Removed heavy dependencies (TensorFlow, Streamlit, etc.) that aren't needed for building docs
- **Result:** Faster, more reliable builds

### 3. Added Mock Imports
- **File:** `docs/conf.py`
- **Change:** Added `autodoc_mock_imports` list to mock missing packages
- **Result:** Documentation can reference code without actually importing heavy dependencies

### 4. Created Package Structure (Optional Backup)
- **File:** `setup.py` - Makes project installable as Python package if needed
- **File:** `interface/__init__.py` - Proper package structure

## Current ReadTheDocs Configuration

```yaml
# .readthedocs.yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.11"

sphinx:
   configuration: docs/conf.py
   builder: html
   fail_on_warning: false

python:
   install:
   - requirements: docs/requirements.txt  # Only install docs dependencies

formats:
  - pdf
  - epub
```

## Minimal Documentation Dependencies

```text
# docs/requirements.txt
sphinx>=5.0.0,<7.0.0
sphinx-rtd-theme>=1.2.0
myst-parser>=0.18.0
nbsphinx>=0.8.0
sphinx-autodoc-typehints>=1.19.0
sphinx-copybutton>=0.5.0
ipython>=8.0.0
numpy>=1.21.0
pandas>=1.5.0
```

## Mock Imports Configuration

```python
# docs/conf.py
autodoc_mock_imports = [
    'tensorflow',
    'streamlit', 
    'sklearn',
    'scikit-learn',
    'plotly',
    'seaborn',
    'matplotlib',
    'optuna',
    'statsmodels',
    'joblib',
    'tqdm',
]
```

## Testing Your Fix

### Local Build Test
```powershell
cd "c:\Users\Idea\Documents\GitHub\AnalysingEnergy"
sphinx-build -b html docs docs\_build\html
```

### ReadTheDocs Deployment
1. Push the updated files to GitHub
2. ReadTheDocs will automatically detect the changes
3. The build should now succeed

## Common ReadTheDocs Build Issues

### Issue: Package Installation Failures
**Symptom:** `pip install .` fails
**Fix:** Remove package installation from `.readthedocs.yaml` or add proper `setup.py`

### Issue: Missing Dependencies
**Symptom:** Import errors during build
**Fix:** Add missing packages to `docs/requirements.txt` or mock them

### Issue: Heavy Dependencies
**Symptom:** Build timeouts or memory errors
**Fix:** Use mock imports instead of installing heavy packages

### Issue: Path Problems
**Symptom:** Files not found
**Fix:** Check paths in `conf.py` and ensure all referenced files exist

## Build Status Verification

✅ **Local Build:** Successful  
✅ **Dependencies:** Minimized  
✅ **Mock Imports:** Configured  
✅ **ReadTheDocs Config:** Fixed  

## Next Steps

1. **Commit Changes:** Push all updated files to GitHub
2. **Monitor Build:** Check ReadTheDocs build logs for success
3. **Verify Output:** Review generated documentation
4. **Optimize Further:** Add more content or styling as needed

The documentation should now build successfully on ReadTheDocs! 🎉
