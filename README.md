# 3D Lung Lesion Detection - Medical Image Analysis Project

**Course:** ED6001 Medical Image Analysis  
**Institution:** IIT Madras  
**Submission Deadline:** November 14, 2025
**Submitted By:** Karan Singh (ED22B052)

**GitHub Repository:** [https://github.com/KaranSingh52ED/lung_lesion_detection_medical_imaging](https://github.com/KaranSingh52ED/lung_lesion_detection_medical_imaging)  
**Full Report:** [PDF Report Link](https://drive.google.com/file/d/13x_CBIZaviY_HbjtB2xE_FgRP2tiNvOz/view?usp=drive_link)

---

## Overview

This project implements an automated system for classifying lung lesions as benign or malignant using texture and morphological features extracted from CT scan images. The system employs image segmentation, feature extraction (GLCM, LBP, Wavelet, and Morphological features), and an SVM classifier to achieve accurate classification.

The codebase is designed to be easy to use, well-documented, and ready to run out of the box. Simply run the Python script to execute the complete pipeline.

---

## Project Structure

```
lung_lesion_detection_medical_imaging/
│
├── lung_lesion_detection.py       # Main Python script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── datasets/                       # Dataset directory
│   ├── train/
│   │   ├── benign/                # Training benign images
│   │   └── malignant/             # Training malignant images
│   ├── valid/
│   │   ├── benign/                # Validation benign images
│   │   └── malignant/             # Validation malignant images
│   └── test/
│       ├── benign/                # Test benign images
│       └── malignant/             # Test malignant images
│
└── outputs/                        # Generated outputs
    ├── sample_images.png
    ├── segmentation_ref.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── cross_validation.png
    ├── metrics_comparison.png
    ├── feature_importance.png
    ├── results.json               # All metrics in JSON format
    └── results_summary.txt        # Human-readable summary
```

---

## Features

- **Image Segmentation**: Automatic lesion segmentation using Otsu thresholding and morphological operations
- **Feature Extraction**: Multi-dimensional feature extraction including:
  - **GLCM**: Gray-Level Co-occurrence Matrix for texture analysis
  - **LBP**: Local Binary Patterns for local texture patterns
  - **Wavelet Transform**: Multi-scale frequency analysis
  - **Morphological Features**: Shape and size characteristics
- **Machine Learning**: SVM classifier with RBF kernel for classification
- **Comprehensive Evaluation**: Metrics including accuracy, sensitivity, specificity, AUC, and ROC curves
- **Visualization**: Automatic generation of all required figures and plots

---

## Quick Start

### Prerequisites

- Python 3.7 or higher (Python 3.8+ recommended)
- pip (Python package installer)
- 8GB+ RAM recommended for optimal performance

### Installation

1. **Clone or download this repository**

2. **Install dependencies using requirements.txt:**

```bash
pip install -r requirements.txt
```

For Windows users with system Python, you may need:

```bash
pip install -r requirements.txt --break-system-packages
```

For Linux/Mac users experiencing permission issues:

```bash
pip install -r requirements.txt --user
```

3. **Verify installation:**

```bash
python -c "import numpy, pandas, sklearn, skimage, cv2, pywt, matplotlib, seaborn; print('All packages installed successfully!')"
```

### Setting Up Virtual Environment (Recommended)

Using a virtual environment helps keep project dependencies isolated:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# When done working, deactivate:
deactivate
```

---

## Dataset Preparation

### Downloading the Dataset

The dataset is available on Google Drive. You can download it using the following link:

**[📥 Download Dataset from Google Drive](https://drive.google.com/drive/folders/19PiTDKIhAzweuPxiU9IiYxxiL7iQ1cWr?usp=drive_link)**

The dataset folder contains three subfolders:

- `train/` - Training images
- `valid/` - Validation images
- `test/` - Test images

Each of these folders contains:

- `benign/` - Benign lung lesion images
- `malignant/` - Malignant lung lesion images

### Organizing the Dataset

After downloading, organize the dataset in the following structure within your project directory:

```
datasets/
├── train/
│   ├── benign/        # Training benign images (.png or .jpg)
│   └── malignant/     # Training malignant images (.png or .jpg)
├── valid/
│   ├── benign/        # Validation benign images
│   └── malignant/     # Validation malignant images
└── test/
    ├── benign/        # Test benign images
    └── malignant/     # Test malignant images
```

**Important Notes:**

- Folder names must be exactly `benign` and `malignant` (lowercase, case-sensitive)
- Supported image formats: `.png`, `.jpg`, `.jpeg`
- Images will be automatically resized to 128×128 pixels during processing
- For best results, use at least 50+ images per class in the training set

---

## Running the Code

1. **Navigate to the project directory:**

```bash
cd path/to/lung_lesion_detection_medical_imaging
```

2. **Run the script:**

```bash
python lung_lesion_detection.py
```

On some systems, you may need to use:

```bash
python3 lung_lesion_detection.py
```

3. **Monitor progress:**

The script will print progress messages to the console, showing:

- Number of images loaded from each folder
- Segmentation progress
- Feature extraction progress
- Training progress
- Final evaluation metrics

**What happens:**

The script executes the following steps in order:

- **Setup & Imports**: Loads all required libraries
- **Data Loading**: Loads images from the datasets folder
- **Preprocessing**: Normalizes and resizes images
- **Segmentation**: Applies Otsu thresholding and morphological operations
- **Feature Extraction**: Extracts GLCM, LBP, Wavelet, and morphological features
- **Model Training**: Trains SVM classifier on training data
- **Evaluation**: Evaluates model on validation and test sets
- **Visualization**: Generates all required figures
- **Results Export**: Saves metrics to JSON and text files

**Expected execution time:** 3-5 minutes depending on dataset size and system specifications.

---

## Output Files

After running the code, all outputs will be saved in the `outputs/` folder:

### Visualization Figures

- **`sample_images.png`**: Sample images from your dataset
- **`segmentation_ref.png`**: Examples showing original images and segmentation results
- **`confusion_matrices.png`**: Confusion matrices for train, validation, and test sets
- **`roc_curves.png`**: ROC curves showing classifier performance across all datasets
- **`cross_validation.png`**: Cross-validation results and stability analysis
- **`metrics_comparison.png`**: Comparison of different evaluation metrics
- **`feature_importance.png`**: Analysis of feature importance and contribution

### Results Files

- **`results.json`**: Complete results in structured JSON format

  - Contains all metrics (accuracy, sensitivity, specificity, AUC, etc.)
  - Includes dataset statistics
  - Use this file to extract exact values for your report

- **`results_summary.txt`**: Human-readable summary of results
  - Quick overview of performance metrics
  - Suitable for quick reference

### Understanding the Results

The `results.json` file contains structured data that you can use in your report. Key sections include:

- **`dataset_info`**: Statistics about your dataset (number of images per class, etc.)
- **`train_metrics`**: Performance metrics on training set
- **`valid_metrics`**: Performance metrics on validation set
- **`test_metrics`**: Performance metrics on test set (most important)
- **`cross_validation`**: Cross-validation scores

---

## Expected Performance

Based on the reference paper (Gao et al., 2019) and typical performance on the LIDC-IDRI dataset, you should expect the following performance ranges:

| Metric          | Expected Range | Interpretation        |
| --------------- | -------------- | --------------------- |
| **AUC**         | 0.80 - 0.95    | Good to Excellent     |
| **Sensitivity** | 0.75 - 0.95    | Good recall rate      |
| **Specificity** | 0.70 - 0.90    | Good specificity      |
| **Accuracy**    | 0.80 - 0.93    | Good overall accuracy |

**What these ranges mean:**

- ✅ **Results within range**: Your model is performing well and as expected
- ⚠️ **Below 0.70**: Check data quality, ensure correct labeling, verify images are CT scans
- 🚨 **Above 0.98**: May indicate overfitting or data leakage - review your setup

---

## Methodology Overview

The system follows this workflow:

```
CT Images
    ↓
Preprocessing (Normalization, Resizing to 128×128)
    ↓
Segmentation (Otsu Thresholding + Morphological Operations)
    ↓
Feature Extraction
    ├── GLCM Features (Texture)
    ├── LBP Features (Local Patterns)
    ├── Wavelet Features (Multi-scale)
    └── Morphological Features (Shape & Size)
    ↓
Feature Concatenation & Normalization
    ↓
SVM Classifier (RBF Kernel)
    ↓
Evaluation & Results
```

### Key Components

**Segmentation**: Uses Otsu's method for automatic thresholding, followed by morphological operations (opening and closing) to clean up the segmented regions.

**Feature Extraction**:

- **GLCM**: Computes texture features at multiple distances and angles
- **LBP**: Captures local texture patterns using 24 points on radius 3
- **Wavelet**: Uses Daubechies wavelets for multi-resolution analysis
- **Morphological**: Extracts shape features like area, perimeter, and eccentricity

**Classification**: Support Vector Machine with RBF kernel, chosen for its effectiveness with high-dimensional feature spaces and small to medium datasets.

---

## Troubleshooting

### Python Version Issues

**Problem:** Code won't run or gives version-related errors.

**Solution:**

```bash
# Check your Python version
python --version

# Need Python 3.7+. If yours is too old:
# Download Python 3.8+ from python.org, or use conda:
conda create -n lung_lesion python=3.8
conda activate lung_lesion
```

### Package Installation Fails

**Problem:** `pip install` fails or packages won't install.

**Solution:**

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Try with user flag (Linux/Mac)
pip install -r requirements.txt --user

# Or use conda for problematic packages
conda install scikit-learn scikit-image opencv -c conda-forge
pip install pywavelets
```

### Import Errors

**Problem:** `ModuleNotFoundError` or `ImportError` when running code.

**Solution:**

```bash
# Verify all packages are installed
python -c "import numpy, pandas, sklearn, skimage, cv2, pywt, matplotlib, seaborn, scipy; print('All packages OK!')"

# If any import fails, install that package:
pip install <package-name>

# Check which Python you're using
which python   # Linux/Mac
where python   # Windows
```

### Dataset Not Found

**Problem:** Error message: `⚠️ Warning: datasets/train/benign does not exist!`

**Solution:**

1. Verify your folder structure exactly matches the required structure
2. Ensure you're running the script from the project root directory
3. Check that folder names are exactly `benign` and `malignant` (lowercase, case-sensitive)
4. Verify image files are in `.png` or `.jpg` format

### No Images Found

**Problem:** `⚠️ No images found in datasets/train/benign`

**Solution:**

- Ensure images are in `.png`, `.jpg`, or `.jpeg` format
- Verify images aren't corrupted (try opening them in an image viewer)
- Check that files are actual image files, not text files with image extensions
- Verify images are in the correct folders

### Memory Errors or Slow Execution

**Problem:** Code runs very slowly or crashes with memory errors.

**Solution:**

- Close other applications to free up RAM
- Reduce the number of images for testing
- The code already resizes images to 128×128, but you could reduce this further if needed
- Use a machine with at least 8GB RAM for best results

### Poor Accuracy (< 70%)

**Problem:** Model accuracy is lower than expected.

**Possible Causes:**

- Mislabeled images (benign images in malignant folder or vice versa)
- Too few images (need at least 50+ per class for good results)
- Poor image quality or corrupted images
- Severe class imbalance

**Solutions:**

- Double-check your labels carefully
- Use more data if available (100+ per class is recommended)
- Verify images are actually CT scans of lung nodules
- Check the data distribution in `results.json` to see if classes are balanced

### Results Files Not Generated

**Problem:** `results.json` or other output files are missing.

**Solution:**

- Verify the script ran to completion without errors
- Check that the `outputs/` folder exists and is writable
- Look for error messages in the console output

### Permission Denied Errors (Linux/Mac)

**Problem:** Can't install packages due to permissions.

**Solution:**

```bash
# Use --user flag (recommended)
pip install -r requirements.txt --user

# Or use sudo (not recommended, but works)
sudo pip install -r requirements.txt
```

### Windows-Specific Issues

**Problem:** Various Windows-related errors.

**Solution:**

- If using system Python, add `--break-system-packages` flag:
  ```bash
  pip install -r requirements.txt --break-system-packages
  ```
- If PowerShell blocks scripts:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
- Try using Command Prompt instead of PowerShell if issues persist

---

## Reference

This project is based on research by:

**Gao et al. (2019)** - "3D texture features for lung nodule classification"

- **Method:** 3D texture features with SVM classifier
- **Dataset:** 285 LIDC-IDRI patients
- **Results:**
  - AUC: 0.907
  - Sensitivity: 0.98 (voting method)
  - Specificity: 0.78 (voting method)
- **Key finding:** 3D features outperformed 2D by 23-28%

**How this implementation compares:**

This project uses a similar approach but adapted for 2D slices:

- Uses GLCM + LBP + Wavelet features (instead of Contourlets)
- Works with 2D slices (computationally simpler than full 3D)
- Employs the same SVM classifier approach
- Expected performance: 80-90% of the reference paper's performance

---

## Getting Help

If you encounter issues:

1. **Check the troubleshooting section** - Most common problems are covered there
2. **Verify prerequisites** - Ensure Python and all packages are installed correctly
3. **Review error messages** - They usually tell you exactly what's wrong
4. **Check folder structure** - Ensure your dataset is organized exactly as specified
5. **Verify working directory** - Make sure you're running from the project root

---

## License

This project is provided for educational purposes as part of ED6001 Medical Image Analysis course at IIT Madras.

---

## Acknowledgments

- LIDC-IDRI dataset for providing the lung CT scan images
- Reference paper by Gao et al. (2019) for methodology inspiration
- Scikit-learn, scikit-image, and OpenCV communities for excellent libraries

---
