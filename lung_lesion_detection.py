"""
3D Lung Lesion Detection using Texture and Morphological Features

ED6001 Medical Image Analysis - IIT Madras
Author: Karan Singh (ED22B052)

Trying to classify lung lesions as benign or malignant from CT scans.
Using 5-fold cross-validation for evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import os

# ignore warnings, they're annoying
warnings.filterwarnings("ignore")
import logging

logging.getLogger("matplotlib").setLevel(logging.ERROR)

# create outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# image processing imports
import cv2
from skimage import io, measure, morphology
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import threshold_otsu
from scipy import ndimage
import pywt

# ML stuff
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    make_scorer,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

# set random seed so results are reproducible
np.random.seed(42)
print("Libraries loaded")


# =============================================================================
# Data Loading Functions
# =============================================================================


def load_images_from_folder(folder_path, label):
    """Loads images from a folder and assigns the label"""
    images = []
    labels = []

    if not folder_path.exists():
        print(f"Warning: {folder_path} doesn't exist")
        return images, labels

    image_files = list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpg"))

    if len(image_files) == 0:
        print(f"No images found in {folder_path}")
        return images, labels

    print(f"  Loading {folder_path.name}: ", end="")

    for img_file in image_files:
        try:
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                images.append(img)
                labels.append(label)
        except Exception as e:
            print(f"\n  Error loading {img_file.name}: {e}")
            continue

    print(f"{len(images)} images")
    return images, labels


def load_dataset(base_dir):
    """Loads benign and malignant images"""
    print(f"\nLoading from: {base_dir}")

    benign_images, benign_labels = load_images_from_folder(base_dir / "benign", 0)
    malignant_images, malignant_labels = load_images_from_folder(
        base_dir / "malignant", 1
    )

    all_images = benign_images + malignant_images
    all_labels = benign_labels + malignant_labels

    return np.array(all_images), np.array(all_labels)


print("Data loading functions defined")


# =============================================================================
# Load Datasets
# =============================================================================

BASE_DIR = Path("datasets")
TRAIN_DIR = BASE_DIR / "train"
VALID_DIR = BASE_DIR / "valid"
TEST_DIR = BASE_DIR / "test"

print("\n" + "=" * 60)
print("LOADING DATASETS")
print("=" * 60)

X_train, y_train = load_dataset(TRAIN_DIR)
X_valid, y_valid = load_dataset(VALID_DIR)
X_test, y_test = load_dataset(TEST_DIR)

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(
    f"Training:   {len(X_train):4d} images (Benign: {np.sum(y_train==0):3d}, Malignant: {np.sum(y_train==1):3d})"
)
print(
    f"Validation: {len(X_valid):4d} images (Benign: {np.sum(y_valid==0):3d}, Malignant: {np.sum(y_valid==1):3d})"
)
print(
    f"Test:       {len(X_test):4d} images (Benign: {np.sum(y_test==0):3d}, Malignant: {np.sum(y_test==1):3d})"
)
print(f"TOTAL:      {len(X_train)+len(X_valid)+len(X_test):4d} images")
print("=" * 60)

if len(X_train) == 0:
    raise ValueError("No training data found!")

print("\nData loaded")


# =============================================================================
# Visualize Sample Images
# =============================================================================


def visualize_samples(images, labels, n_samples=6):
    """Shows some sample images"""
    if len(images) == 0:
        print("Nothing to visualize")
        return

    benign_idx = np.where(labels == 0)[0]
    malignant_idx = np.where(labels == 1)[0]

    n_benign = min(len(benign_idx), 3)
    n_malignant = min(len(malignant_idx), 3)

    if n_benign + n_malignant == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    for ax in axes:
        ax.axis("off")

    plot_idx = 0

    for i in range(n_benign):
        if plot_idx < 6:
            axes[plot_idx].imshow(images[benign_idx[i]], cmap="gray")
            axes[plot_idx].set_title(
                "Benign Nodule", fontsize=12, fontweight="bold", color="green"
            )
            plot_idx += 1

    for i in range(n_malignant):
        if plot_idx < 6:
            axes[plot_idx].imshow(images[malignant_idx[i]], cmap="gray")
            axes[plot_idx].set_title(
                "Malignant Nodule", fontsize=12, fontweight="bold", color="red"
            )
            plot_idx += 1

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig("outputs/sample_images.png", dpi=300)
    plt.close()

    print("Sample images saved")


if len(X_train) > 0:
    visualize_samples(X_train, y_train)


# =============================================================================
# Segmentation
# =============================================================================


def segment_nodule(image):
    """Segments nodule using Otsu thresholding and some morphological ops"""
    try:
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        thresh = threshold_otsu(blurred)
        binary = blurred > thresh
        cleaned = morphology.remove_small_objects(binary, min_size=50)
        cleaned = ndimage.binary_fill_holes(cleaned)
        kernel = morphology.disk(3)
        cleaned = morphology.closing(cleaned, kernel)
        return binary.astype(np.uint8), cleaned.astype(np.uint8)
    except:
        return np.zeros_like(image, dtype=np.uint8), np.zeros_like(
            image, dtype=np.uint8
        )


# show segmentation example
if len(X_train) > 0:
    sample_img = X_train[0]
    original_seg, cleaned_seg = segment_nodule(sample_img)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(sample_img, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(original_seg, cmap="gray")
    axes[1].set_title("Otsu Threshold")
    axes[1].axis("off")

    axes[2].imshow(cleaned_seg, cmap="gray")
    axes[2].set_title("Cleaned")
    axes[2].axis("off")

    overlay = cv2.cvtColor(sample_img, cv2.COLOR_GRAY2RGB)
    overlay[cleaned_seg > 0] = [255, 0, 0]
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    plt.subplots_adjust(wspace=0.3)
    plt.savefig("outputs/segmentation_ref.png", dpi=300)
    plt.close()

    print("Segmentation example saved")


# =============================================================================
# Feature Extraction Functions
# =============================================================================


def extract_glcm_features(image):
    """Gets texture features using GLCM"""
    try:
        img_norm = image.astype(np.float64)
        img_min, img_max = img_norm.min(), img_norm.max()

        if img_max - img_min < 1e-8:
            return np.zeros(12)

        img_norm = ((img_norm - img_min) / (img_max - img_min) * 255).astype(np.uint8)

        glcm = graycomatrix(
            img_norm,
            distances=[1, 2, 3],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256,
            symmetric=True,
            normed=True,
        )

        features = []
        props = [
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
            "correlation",
            "ASM",
        ]
        for prop in props:
            values = graycoprops(glcm, prop)
            mean_val = np.nan_to_num(values.mean(), nan=0.0, posinf=0.0, neginf=0.0)
            std_val = np.nan_to_num(values.std(), nan=0.0, posinf=0.0, neginf=0.0)
            features.extend([mean_val, std_val])

        return np.array(features)
    except:
        return np.zeros(12)


def extract_lbp_features(image):
    """Extract Local Binary Pattern features"""
    try:
        lbp = local_binary_pattern(image, 24, 3, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)
        features = list(hist)

        lbp_mean = np.nan_to_num(lbp.mean(), nan=0.0)
        lbp_std = np.nan_to_num(lbp.std(), nan=0.0)
        lbp_var = np.nan_to_num(lbp.var(), nan=0.0)
        features.extend([lbp_mean, lbp_std, lbp_var])

        return np.array(features)
    except:
        return np.zeros(29)


def extract_wavelet_features(image):
    """Gets wavelet features using db4"""
    try:
        coeffs = pywt.wavedec2(image, "db4", level=3)
        features = []

        cA = coeffs[0]
        features.extend(
            [
                np.nan_to_num(cA.mean(), nan=0.0),
                np.nan_to_num(cA.std(), nan=0.0),
                np.nan_to_num(cA.var(), nan=0.0),
            ]
        )

        for detail_level in coeffs[1:]:
            cH, cV, cD = detail_level
            for coeff in [cH, cV, cD]:
                features.extend(
                    [
                        np.nan_to_num(coeff.mean(), nan=0.0),
                        np.nan_to_num(coeff.std(), nan=0.0),
                        np.nan_to_num(coeff.var(), nan=0.0),
                        np.nan_to_num(np.percentile(coeff, 25), nan=0.0),
                        np.nan_to_num(np.percentile(coeff, 75), nan=0.0),
                    ]
                )

        return np.array(features)
    except:
        return np.zeros(48)


def extract_morphological_features(image, mask):
    """Extract shape/morphological features"""
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return np.zeros(10)

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        
        if area < 1:
            return np.zeros(10)

        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter**2 + 1e-8)

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / (h + 1e-8)
        extent = area / (w * h + 1e-8)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-8)

        equiv_diameter = np.sqrt(4 * area / (np.pi + 1e-8))

        features = [
            area,
            perimeter,
            circularity,
            aspect_ratio,
            extent,
            solidity,
            equiv_diameter,
            w,
            h,
            w * h,
        ]

        features = [np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0) for f in features]

        return np.array(features)
    except:
        return np.zeros(10)


def extract_all_features(image):
    """Extracts all features for an image"""
    try:
        _, mask = segment_nodule(image)
        glcm = extract_glcm_features(image)
        lbp = extract_lbp_features(image)
        wavelet = extract_wavelet_features(image)
        morph = extract_morphological_features(image, mask)

        all_features = np.concatenate([glcm, lbp, wavelet, morph])
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)

        return all_features
    except:
        return np.zeros(99)


if len(X_train) > 0:
    test_features = extract_all_features(X_train[0])
    print(f"\nFeature extraction ready - {len(test_features)} features total")


# =============================================================================
# Extract Features from All Images
# =============================================================================


def extract_features_from_dataset(images, labels, name=""):
    """Extracts features from all images in the dataset"""
    if len(images) == 0:
        return np.array([]), np.array([])

    features_list = []
    valid_labels = []

    print(f"\nExtracting features from {name}...")

    for i, (img, label) in enumerate(zip(images, labels)):
        try:
            features = extract_all_features(img)
            features_list.append(features)
            valid_labels.append(label)

            if (i + 1) % 100 == 0 or (i + 1) == len(images):
                print(f"  Progress: {i+1}/{len(images)}", end="\r")
        except Exception as e:
            print(f"\n  Error at image {i}: {e}")
            continue

    print(f"\nDone: {len(features_list)} images processed")
    return np.array(features_list), np.array(valid_labels)


print("\n" + "=" * 60)
print("FEATURE EXTRACTION PHASE")
print("=" * 60)

X_train_features, y_train_features = extract_features_from_dataset(
    X_train, y_train, "Training"
)
X_valid_features, y_valid_features = extract_features_from_dataset(
    X_valid, y_valid, "Validation"
)
X_test_features, y_test_features = extract_features_from_dataset(
    X_test, y_test, "Test"
)

print("\n" + "=" * 60)
print(f"Training:   {X_train_features.shape}")
print(f"Validation: {X_valid_features.shape}")
print(f"Test:       {X_test_features.shape}")
print("=" * 60)


# =============================================================================
# 5-FOLD CROSS-VALIDATION
# =============================================================================

print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION")
print("=" * 60)

# pipeline with scaling and SVM
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=True,
                random_state=42,
                class_weight="balanced",
            ),
        ),
    ]
)

# stratified k-fold for cross-validation
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# metrics to compute during CV
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

print("\nRunning 5-fold cross-validation...")
cv_results = cross_validate(
    pipeline,
    X_train_features,
    y_train_features,
    cv=cv_strategy,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=True
)

# get results for each fold
cv_test_accuracy = cv_results['test_accuracy']
cv_test_precision = cv_results['test_precision']
cv_test_recall = cv_results['test_recall']
cv_test_f1 = cv_results['test_f1']
cv_test_auc = cv_results['test_roc_auc']

print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 60)
print(f"\nAccuracy per fold:  {[f'{s:.4f}' for s in cv_test_accuracy]}")
print(f"  Mean ± Std: {cv_test_accuracy.mean():.4f} ± {cv_test_accuracy.std():.4f}")

print(f"\nPrecision per fold: {[f'{s:.4f}' for s in cv_test_precision]}")
print(f"  Mean ± Std: {cv_test_precision.mean():.4f} ± {cv_test_precision.std():.4f}")

print(f"\nRecall per fold:    {[f'{s:.4f}' for s in cv_test_recall]}")
print(f"  Mean ± Std: {cv_test_recall.mean():.4f} ± {cv_test_recall.std():.4f}")

print(f"\nF1-Score per fold:  {[f'{s:.4f}' for s in cv_test_f1]}")
print(f"  Mean ± Std: {cv_test_f1.mean():.4f} ± {cv_test_f1.std():.4f}")

print(f"\nAUC per fold:       {[f'{s:.4f}' for s in cv_test_auc]}")
print(f"  Mean ± Std: {cv_test_auc.mean():.4f} ± {cv_test_auc.std():.4f}")
print("=" * 60)


# =============================================================================
# Train Model on Full Training Set
# =============================================================================

print("\n" + "=" * 60)
print("TRAINING MODEL ON FULL TRAINING SET")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_valid_scaled = scaler.transform(X_valid_features)
X_test_scaled = scaler.transform(X_test_features)

clf = SVC(
    kernel="rbf",
    C=10.0,
    gamma="scale",
    probability=True,
    random_state=42,
    class_weight="balanced",
)
clf.fit(X_train_scaled, y_train_features)

print("Model trained on full training set")


# =============================================================================
# Evaluation on Train/Valid/Test Sets
# =============================================================================


def calculate_metrics(y_true, y_pred, y_prob):
    """Computes evaluation metrics"""
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc_score = roc_auc_score(y_true, y_prob)
    except:
        auc_score = 0.0

    return {
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "auc": auc_score,
        "confusion_matrix": cm,
    }


print("\nEvaluating final model...")

# Training set
y_train_pred = clf.predict(X_train_scaled)
y_train_prob = clf.predict_proba(X_train_scaled)[:, 1]
train_metrics = calculate_metrics(y_train_features, y_train_pred, y_train_prob)

print(f"\nTraining Set: Accuracy={train_metrics['accuracy']:.4f}, AUC={train_metrics['auc']:.4f}")

# Validation set
valid_metrics = None
if len(X_valid_features) > 0:
    y_valid_pred = clf.predict(X_valid_scaled)
    y_valid_prob = clf.predict_proba(X_valid_scaled)[:, 1]
    valid_metrics = calculate_metrics(y_valid_features, y_valid_pred, y_valid_prob)
    print(f"Validation Set: Accuracy={valid_metrics['accuracy']:.4f}, AUC={valid_metrics['auc']:.4f}")

# Test set
test_metrics = None
if len(X_test_features) > 0:
    y_test_pred = clf.predict(X_test_scaled)
    y_test_prob = clf.predict_proba(X_test_scaled)[:, 1]
    test_metrics = calculate_metrics(y_test_features, y_test_pred, y_test_prob)
    print(f"Test Set: Accuracy={test_metrics['accuracy']:.4f}, AUC={test_metrics['auc']:.4f}")


# =============================================================================
# Visualizations
# =============================================================================

print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# confusion matrices
print("\nPlotting confusion matrices...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.heatmap(
    train_metrics["confusion_matrix"],
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[0],
    cbar=False,
    square=True,
)
axes[0].set_title("Training Set", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Actual", fontsize=12)
axes[0].set_xlabel("Predicted", fontsize=12)
axes[0].set_xticklabels(["Benign", "Malignant"])
axes[0].set_yticklabels(["Benign", "Malignant"])

if valid_metrics:
    sns.heatmap(
        valid_metrics["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=axes[1],
        cbar=False,
        square=True,
    )
    axes[1].set_title("Validation Set", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Actual", fontsize=12)
    axes[1].set_xlabel("Predicted", fontsize=12)
    axes[1].set_xticklabels(["Benign", "Malignant"])
    axes[1].set_yticklabels(["Benign", "Malignant"])

if test_metrics:
    sns.heatmap(
        test_metrics["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Oranges",
        ax=axes[2],
        cbar=False,
        square=True,
    )
    axes[2].set_title("Test Set", fontsize=14, fontweight="bold")
    axes[2].set_ylabel("Actual", fontsize=12)
    axes[2].set_xlabel("Predicted", fontsize=12)
    axes[2].set_xticklabels(["Benign", "Malignant"])
    axes[2].set_yticklabels(["Benign", "Malignant"])

plt.tight_layout()
plt.savefig("outputs/confusion_matrices.png", dpi=300, bbox_inches="tight")
plt.close()
print("Confusion matrices saved")

# 5-fold CV results
print("Plotting 5-fold CV results...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

metrics_names = ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1-Score', 'AUC']
metrics_data = [cv_test_accuracy, cv_test_precision, cv_test_recall, cv_test_f1, cv_test_auc]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

for idx, (metric_name, metric_data, color) in enumerate(zip(metrics_names, metrics_data, colors)):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    ax.bar(range(1, 6), metric_data, color=color, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.axhline(
        metric_data.mean(),
        color='r',
        linestyle='--',
        lw=2,
        label=f'Mean: {metric_data.mean():.4f}'
    )
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'5-Fold CV: {metric_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(metric_data):
        ax.text(i + 1, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

# Hide the last subplot (bottom right)
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig("outputs/cross_validation.png", dpi=300, bbox_inches="tight")
plt.close()
print("Cross-validation plots saved")

# ROC curves
print("Plotting ROC curves...")

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

fpr_train, tpr_train, _ = roc_curve(y_train_features, y_train_prob)
ax.plot(
    fpr_train,
    tpr_train,
    lw=2,
    label=f"Training (AUC={train_metrics['auc']:.4f})",
    color="#3498db",
)

if valid_metrics:
    fpr_valid, tpr_valid, _ = roc_curve(y_valid_features, y_valid_prob)
    ax.plot(
        fpr_valid,
        tpr_valid,
        lw=2,
        label=f"Validation (AUC={valid_metrics['auc']:.4f})",
        color="#2ecc71",
    )

if test_metrics:
    fpr_test, tpr_test, _ = roc_curve(y_test_features, y_test_prob)
    ax.plot(
        fpr_test,
        tpr_test,
        lw=2,
        label=f"Test (AUC={test_metrics['auc']:.4f})",
        color="#e74c3c",
    )

ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random")

ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)

plt.savefig("outputs/roc_curves.png", dpi=300, bbox_inches="tight")
plt.close()
print("ROC curves saved")

# metrics comparison
print("Plotting metrics comparison...")

metrics_names = ["Accuracy", "Precision", "Sensitivity", "Specificity", "F1", "AUC"]
train_values = [
    train_metrics["accuracy"],
    train_metrics["precision"],
    train_metrics["sensitivity"],
    train_metrics["specificity"],
    train_metrics["f1_score"],
    train_metrics["auc"],
]

fig, ax = plt.subplots(1, 1, figsize=(14, 6))

x = np.arange(len(metrics_names))
width = 0.25

ax.bar(x - width, train_values, width, label="Training", color="#3498db", alpha=0.8)

if valid_metrics:
    valid_values = [
        valid_metrics["accuracy"],
        valid_metrics["precision"],
        valid_metrics["sensitivity"],
        valid_metrics["specificity"],
        valid_metrics["f1_score"],
        valid_metrics["auc"],
    ]
    ax.bar(x, valid_values, width, label="Validation", color="#2ecc71", alpha=0.8)

if test_metrics:
    test_values = [
        test_metrics["accuracy"],
        test_metrics["precision"],
        test_metrics["sensitivity"],
        test_metrics["specificity"],
        test_metrics["f1_score"],
        test_metrics["auc"],
    ]
    ax.bar(x + width, test_values, width, label="Test", color="#e74c3c", alpha=0.8)

# Add value labels
for i, v in enumerate(train_values):
    ax.text(i - width, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
if valid_metrics:
    for i, v in enumerate(valid_values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
if test_metrics:
    for i, v in enumerate(test_values):
        ax.text(i + width, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

ax.set_xlabel("Metrics", fontsize=12, fontweight="bold")
ax.set_ylabel("Score", fontsize=12, fontweight="bold")
ax.set_title("Performance Comparison", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.set_ylim([0, 1.1])
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/metrics_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("Metrics comparison saved")

# feature importance
print("Analyzing feature importance...")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train_features)

importances = rf.feature_importances_
top_20_idx = np.argsort(importances)[::-1][:20]

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.bar(
    range(20), importances[top_20_idx], color="#9b59b6", alpha=0.8, edgecolor="black"
)
ax.set_xlabel("Feature Index", fontsize=12, fontweight="bold")
ax.set_ylabel("Importance", fontsize=12, fontweight="bold")
ax.set_title("Top 20 Most Important Features", fontsize=14, fontweight="bold")
ax.set_xticks(range(20))
ax.set_xticklabels(top_20_idx, rotation=45)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()
print("Feature importance saved")


# =============================================================================
# Save Results
# =============================================================================

import json

print("\nSaving results...")

results = {
    "dataset": {
        "train_size": int(len(X_train)),
        "valid_size": int(len(X_valid)),
        "test_size": int(len(X_test)),
        "train_benign": int(np.sum(y_train == 0)),
        "train_malignant": int(np.sum(y_train == 1)),
        "test_benign": int(np.sum(y_test == 0)),
        "test_malignant": int(np.sum(y_test == 1)),
    },
    "features": {"total": int(X_train_features.shape[1])},
    "train_metrics": {
        "accuracy": float(train_metrics["accuracy"]),
        "sensitivity": float(train_metrics["sensitivity"]),
        "specificity": float(train_metrics["specificity"]),
        "precision": float(train_metrics["precision"]),
        "f1_score": float(train_metrics["f1_score"]),
        "auc": float(train_metrics["auc"]),
    },
    "cross_validation": {
        "mean_accuracy": float(cv_test_accuracy.mean()),
        "std_accuracy": float(cv_test_accuracy.std()),
        "mean_auc": float(cv_test_auc.mean()),
        "std_auc": float(cv_test_auc.std()),
        "mean_precision": float(cv_test_precision.mean()),
        "std_precision": float(cv_test_precision.std()),
        "mean_recall": float(cv_test_recall.mean()),
        "std_recall": float(cv_test_recall.std()),
        "mean_f1": float(cv_test_f1.mean()),
        "std_f1": float(cv_test_f1.std()),
        "fold_accuracies": [float(s) for s in cv_test_accuracy],
        "fold_precisions": [float(s) for s in cv_test_precision],
        "fold_recalls": [float(s) for s in cv_test_recall],
        "fold_f1_scores": [float(s) for s in cv_test_f1],
        "fold_aucs": [float(s) for s in cv_test_auc],
    },
}

if test_metrics:
    results["test_metrics"] = {
        "accuracy": float(test_metrics["accuracy"]),
        "sensitivity": float(test_metrics["sensitivity"]),
        "specificity": float(test_metrics["specificity"]),
        "precision": float(test_metrics["precision"]),
        "f1_score": float(test_metrics["f1_score"]),
        "auc": float(test_metrics["auc"]),
    }

with open("outputs/results.json", "w") as f:
    json.dump(results, f, indent=4)

# save text summary
summary = f"""
{'='*70}
3D LUNG LESION DETECTION - RESULTS (5-FOLD CV)
{'='*70}

DATASET:
--------
Training:   {len(X_train)} images (Benign: {np.sum(y_train==0)}, Malignant: {np.sum(y_train==1)})
Validation: {len(X_valid)} images (Benign: {np.sum(y_valid==0)}, Malignant: {np.sum(y_valid==1)})
Test:       {len(X_test)} images (Benign: {np.sum(y_test==0)}, Malignant: {np.sum(y_test==1)})

FEATURES: {X_train_features.shape[1]}

5-FOLD CROSS-VALIDATION RESULTS:
---------------------------------
Accuracy:  {cv_test_accuracy.mean():.4f} ± {cv_test_accuracy.std():.4f}
Precision: {cv_test_precision.mean():.4f} ± {cv_test_precision.std():.4f}
Recall:    {cv_test_recall.mean():.4f} ± {cv_test_recall.std():.4f}
F1-Score:  {cv_test_f1.mean():.4f} ± {cv_test_f1.std():.4f}
AUC:       {cv_test_auc.mean():.4f} ± {cv_test_auc.std():.4f}

TRAINING SET (Final Model):
----------------------------
Accuracy:    {train_metrics['accuracy']:.4f}
Sensitivity: {train_metrics['sensitivity']:.4f}
Specificity: {train_metrics['specificity']:.4f}
AUC:         {train_metrics['auc']:.4f}
"""

if test_metrics:
    summary += f"""
TEST SET (Final Model):
-----------------------
Accuracy:    {test_metrics['accuracy']:.4f}
Sensitivity: {test_metrics['sensitivity']:.4f}
Specificity: {test_metrics['specificity']:.4f}
AUC:         {test_metrics['auc']:.4f}
"""

summary += f"""
{'='*70}
"""

with open("outputs/results_summary.txt", "w") as f:
    f.write(summary)

print(summary)
print("\nResults saved to outputs/")


# =============================================================================
# Final Summary
# =============================================================================

print("\n" + "=" * 70)
print("PIPELINE COMPLETED")
print("=" * 70)
print("\nGenerated files:")
print("  1. sample_images.png")
print("  2. segmentation_ref.png")
print("  3. confusion_matrices.png")
print("  4. cross_validation.png")
print("  5. roc_curves.png")
print("  6. metrics_comparison.png")
print("  7. feature_importance.png")
print("  8. results_summary.txt")
print("  9. results.json")
print("\nPrimary evaluation: 5-Fold Cross-Validation")
print(f"   CV Accuracy: {cv_test_accuracy.mean():.4f} ± {cv_test_accuracy.std():.4f}")
print(f"   CV AUC:      {cv_test_auc.mean():.4f} ± {cv_test_auc.std():.4f}")
print("=" * 70)