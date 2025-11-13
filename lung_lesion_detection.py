"""
3D Lung Lesion Detection using Texture and Morphological Features

ED6001 Medical Image Analysis - IIT Madras
Author: Karan Singh (ED22B052)

This script implements a machine learning pipeline for classifying lung lesions
as benign or malignant using CT scan images.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import os

# suppress warnings - they get annoying during training
warnings.filterwarnings("ignore")
import logging

logging.getLogger("matplotlib").setLevel(logging.ERROR)

# make outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# image processing libraries
import cv2
from skimage import io, measure, morphology
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import threshold_otsu
from scipy import ndimage
import pywt

# ML stuff
from sklearn.model_selection import train_test_split, cross_val_score
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
)
from sklearn.ensemble import RandomForestClassifier

# set seed for reproducibility
np.random.seed(42)
print("✅ Libraries loaded successfully!")


# =============================================================================
# Data Loading Functions
# =============================================================================


def load_images_from_folder(folder_path, label):
    """
    Load all images from a folder and assign them the given label
    Returns lists of images and labels
    """
    images = []
    labels = []

    if not folder_path.exists():
        print(f"⚠️  Warning: {folder_path} does not exist!")
        return images, labels

    # get all png and jpg files
    image_files = list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpg"))

    if len(image_files) == 0:
        print(f"⚠️  No images found in {folder_path}")
        return images, labels

    print(f"  Loading {folder_path.name}: ", end="")

    for img_file in image_files:
        try:
            # read as grayscale and resize to 128x128
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
    """Load both benign and malignant images from a directory"""
    print(f"\nLoading from: {base_dir}")

    # load benign (label = 0) and malignant (label = 1)
    benign_images, benign_labels = load_images_from_folder(base_dir / "benign", 0)
    malignant_images, malignant_labels = load_images_from_folder(
        base_dir / "malignant", 1
    )

    # combine them
    all_images = benign_images + malignant_images
    all_labels = benign_labels + malignant_labels

    return np.array(all_images), np.array(all_labels)


print("Data loading functions ready")


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

# print dataset summary
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
    raise ValueError("No training data found! Check your dataset folder structure.")

print("\n✅ Data loaded successfully!")


# =============================================================================
# Visualize Sample Images
# =============================================================================


def visualize_samples(images, labels, n_samples=6):
    """Visualize some sample images"""
    if len(images) == 0:
        print("No images to visualize")
        return

    # get indices for each class
    benign_idx = np.where(labels == 0)[0]
    malignant_idx = np.where(labels == 1)[0]

    # take 3 of each
    n_benign = min(len(benign_idx), 3)
    n_malignant = min(len(malignant_idx), 3)

    if n_benign + n_malignant == 0:
        print("Not enough samples")
        return

    # create figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    # turn off all axes
    for ax in axes:
        ax.axis("off")

    plot_idx = 0

    # plot benign samples
    for i in range(n_benign):
        if plot_idx < 6:
            axes[plot_idx].imshow(images[benign_idx[i]], cmap="gray")
            axes[plot_idx].set_title(
                "Benign Nodule", fontsize=12, fontweight="bold", color="green"
            )
            plot_idx += 1

    # plot malignant samples
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

    print("✅ Sample images saved")


# run visualization
if len(X_train) > 0:
    visualize_samples(X_train, y_train)


# =============================================================================
# Segmentation
# =============================================================================


def segment_nodule(image):
    """
    Segment nodule using Otsu thresholding + morphological operations
    Returns: (original_binary, cleaned_binary)
    """
    try:
        # apply gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # otsu thresholding
        thresh = threshold_otsu(blurred)
        binary = blurred > thresh

        # clean up the mask
        cleaned = morphology.remove_small_objects(binary, min_size=50)
        cleaned = ndimage.binary_fill_holes(cleaned)

        # closing operation to fill small holes
        kernel = morphology.disk(3)
        cleaned = morphology.closing(cleaned, kernel)

        return binary.astype(np.uint8), cleaned.astype(np.uint8)
    except:
        # if something goes wrong, return empty masks
        return np.zeros_like(image, dtype=np.uint8), np.zeros_like(
            image, dtype=np.uint8
        )


# visualize segmentation on a sample image
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

    # create overlay
    overlay = cv2.cvtColor(sample_img, cv2.COLOR_GRAY2RGB)
    overlay[cleaned_seg > 0] = [255, 0, 0]  # red overlay
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    plt.subplots_adjust(wspace=0.3)
    plt.savefig("outputs/segmentation_ref.png", dpi=300)
    plt.close()

    print("✅ Segmentation example saved")


# =============================================================================
# Feature Extraction Functions
# =============================================================================


def extract_glcm_features(image):
    """Extract texture features using GLCM"""
    try:
        # normalize to 0-255
        img_norm = (
            (image - image.min()) / (image.max() - image.min() + 1e-8) * 255
        ).astype(np.uint8)

        # compute GLCM at different distances and angles
        glcm = graycomatrix(
            img_norm,
            distances=[1, 2, 3],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256,
            symmetric=True,
            normed=True,
        )

        # extract properties
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
            features.extend([values.mean(), values.std()])

        return np.array(features)
    except:
        return np.zeros(12)


def extract_lbp_features(image):
    """Extract Local Binary Pattern features"""
    try:
        # compute LBP
        lbp = local_binary_pattern(image, 24, 3, method="uniform")

        # get histogram
        hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)
        features = list(hist)

        # add some stats
        features.extend([lbp.mean(), lbp.std(), lbp.var()])

        return np.array(features)
    except:
        return np.zeros(29)


def extract_wavelet_features(image):
    """Extract wavelet features using db4 wavelet"""
    try:
        # 3-level wavelet decomposition
        coeffs = pywt.wavedec2(image, "db4", level=3)
        features = []

        # approximation coefficients
        cA = coeffs[0]
        features.extend([cA.mean(), cA.std(), cA.var()])

        # detail coefficients
        for detail_level in coeffs[1:]:
            cH, cV, cD = detail_level
            for coeff in [cH, cV, cD]:
                features.extend(
                    [
                        coeff.mean(),
                        coeff.std(),
                        coeff.var(),
                        np.percentile(coeff, 25),
                        np.percentile(coeff, 75),
                    ]
                )

        return np.array(features)
    except:
        return np.zeros(48)


def extract_morphological_features(image, mask):
    """Extract shape/morphological features from the segmented region"""
    try:
        # find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return np.zeros(10)

        # get largest contour
        cnt = max(contours, key=cv2.contourArea)

        # basic measurements
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter**2 + 1e-8)

        # bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / (h + 1e-8)
        extent = area / (w * h + 1e-8)

        # convex hull
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-8)

        equiv_diameter = np.sqrt(4 * area / (np.pi + 1e-8))

        return np.array(
            [
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
        )
    except:
        return np.zeros(10)


def extract_all_features(image):
    """Extract all features from an image"""
    try:
        # first segment the nodule
        _, mask = segment_nodule(image)

        # extract different types of features
        glcm = extract_glcm_features(image)
        lbp = extract_lbp_features(image)
        wavelet = extract_wavelet_features(image)
        morph = extract_morphological_features(image, mask)

        # concatenate all features
        return np.concatenate([glcm, lbp, wavelet, morph])
    except:
        return np.zeros(99)


# test feature extraction
if len(X_train) > 0:
    test_features = extract_all_features(X_train[0])
    print(f"\n✅ Feature extraction ready - Total features: {len(test_features)}")


# =============================================================================
# Extract Features from All Images
# =============================================================================


def extract_features_from_dataset(images, labels, name=""):
    """Extract features from all images in a dataset"""
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

            # progress indicator
            if (i + 1) % 10 == 0 or (i + 1) == len(images):
                print(f"  Progress: {i+1}/{len(images)}", end="\r")
        except Exception as e:
            print(f"\n  Error at image {i}: {e}")
            continue

    print(f"\n✅ Completed: {len(features_list)} images processed")
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
X_test_features, y_test_features = extract_features_from_dataset(X_test, y_test, "Test")

print("\n" + "=" * 60)
print(f"Training:   {X_train_features.shape}")
print(f"Validation: {X_valid_features.shape}")
print(f"Test:       {X_test_features.shape}")
print("=" * 60)


# =============================================================================
# Normalize Features
# =============================================================================

print("\nNormalizing features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_valid_scaled = scaler.transform(X_valid_features)
X_test_scaled = scaler.transform(X_test_features)

print("✅ Features normalized")


# =============================================================================
# Train SVM Classifier
# =============================================================================

print("\n" + "=" * 60)
print("TRAINING SVM CLASSIFIER")
print("=" * 60)

# using RBF kernel SVM
clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
clf.fit(X_train_scaled, y_train_features)

print("✅ SVM training complete")


# =============================================================================
# Evaluation Functions
# =============================================================================


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate all evaluation metrics"""
    acc = accuracy_score(y_true, y_pred)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # AUC
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


# evaluate on all datasets
print("\nEvaluating model...")

# training set
y_train_pred = clf.predict(X_train_scaled)
y_train_prob = clf.predict_proba(X_train_scaled)[:, 1]
train_metrics = calculate_metrics(y_train_features, y_train_pred, y_train_prob)

print(f"\nTraining Set:")
print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
print(f"  AUC: {train_metrics['auc']:.4f}")

# validation set
if len(X_valid_features) > 0:
    y_valid_pred = clf.predict(X_valid_scaled)
    y_valid_prob = clf.predict_proba(X_valid_scaled)[:, 1]
    valid_metrics = calculate_metrics(y_valid_features, y_valid_pred, y_valid_prob)

    print(f"\nValidation Set:")
    print(f"  Accuracy: {valid_metrics['accuracy']:.4f}")
    print(f"  AUC: {valid_metrics['auc']:.4f}")

# test set
test_metrics = None
if len(X_test_features) > 0:
    y_test_pred = clf.predict(X_test_scaled)
    y_test_prob = clf.predict_proba(X_test_scaled)[:, 1]
    test_metrics = calculate_metrics(y_test_features, y_test_pred, y_test_prob)

    print(f"\nTest Set:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")


# =============================================================================
# Visualizations
# =============================================================================

print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# 1. Confusion Matrices
print("\nPlotting confusion matrices...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# training confusion matrix
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

# validation confusion matrix
if len(X_valid_features) > 0:
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

# test confusion matrix
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

print("✅ Confusion matrices saved")

# 2. ROC Curves
print("Plotting ROC curves...")

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# training ROC
fpr_train, tpr_train, _ = roc_curve(y_train_features, y_train_prob)
ax.plot(
    fpr_train,
    tpr_train,
    lw=2,
    label=f"Training (AUC = {train_metrics['auc']:.4f})",
    color="#3498db",
)

# validation ROC
if len(X_valid_features) > 0:
    fpr_valid, tpr_valid, _ = roc_curve(y_valid_features, y_valid_prob)
    ax.plot(
        fpr_valid,
        tpr_valid,
        lw=2,
        label=f"Validation (AUC = {valid_metrics['auc']:.4f})",
        color="#2ecc71",
    )

# test ROC
if test_metrics:
    fpr_test, tpr_test, _ = roc_curve(y_test_features, y_test_prob)
    ax.plot(
        fpr_test,
        tpr_test,
        lw=2,
        label=f"Test (AUC = {test_metrics['auc']:.4f})",
        color="#e74c3c",
    )

# diagonal line
ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier")

ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)

plt.savefig("outputs/roc_curves.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ ROC curves saved")

# 3. Metrics Comparison
print("Plotting metrics comparison...")

metrics_names = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1-Score"]
train_values = [
    train_metrics["accuracy"],
    train_metrics["sensitivity"],
    train_metrics["specificity"],
    train_metrics["precision"],
    train_metrics["f1_score"],
]

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

x = np.arange(len(metrics_names))
width = 0.25

ax.bar(x - width, train_values, width, label="Training", color="#3498db", alpha=0.8)

if len(X_valid_features) > 0:
    valid_values = [
        valid_metrics["accuracy"],
        valid_metrics["sensitivity"],
        valid_metrics["specificity"],
        valid_metrics["precision"],
        valid_metrics["f1_score"],
    ]
    ax.bar(x, valid_values, width, label="Validation", color="#2ecc71", alpha=0.8)

if test_metrics:
    test_values = [
        test_metrics["accuracy"],
        test_metrics["sensitivity"],
        test_metrics["specificity"],
        test_metrics["precision"],
        test_metrics["f1_score"],
    ]
    ax.bar(x + width, test_values, width, label="Test", color="#e74c3c", alpha=0.8)

ax.set_xlabel("Metrics", fontsize=12, fontweight="bold")
ax.set_ylabel("Score", fontsize=12, fontweight="bold")
ax.set_title("Performance Metrics Comparison", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.set_ylim([0, 1.1])
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/metrics_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ Metrics comparison saved")


# =============================================================================
# Cross-Validation
# =============================================================================

print("\nPerforming cross-validation...")

cv_scores = cross_val_score(
    clf, X_train_scaled, y_train_features, cv=5, scoring="accuracy", n_jobs=-1
)
cv_auc = cross_val_score(
    clf, X_train_scaled, y_train_features, cv=5, scoring="roc_auc", n_jobs=-1
)

print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"CV AUC: {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")

# plot cross-validation results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(1, 6), cv_scores, color="#3498db", alpha=0.8)
axes[0].axhline(
    cv_scores.mean(),
    color="r",
    linestyle="--",
    lw=2,
    label=f"Mean: {cv_scores.mean():.4f}",
)
axes[0].set_xlabel("Fold", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Accuracy", fontsize=12, fontweight="bold")
axes[0].set_title("5-Fold Cross-Validation Accuracy", fontsize=14, fontweight="bold")
axes[0].legend()
axes[0].set_ylim([0, 1.1])
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(range(1, 6), cv_auc, color="#2ecc71", alpha=0.8)
axes[1].axhline(
    cv_auc.mean(), color="r", linestyle="--", lw=2, label=f"Mean: {cv_auc.mean():.4f}"
)
axes[1].set_xlabel("Fold", fontsize=12, fontweight="bold")
axes[1].set_ylabel("AUC", fontsize=12, fontweight="bold")
axes[1].set_title("5-Fold Cross-Validation AUC", fontsize=14, fontweight="bold")
axes[1].legend()
axes[1].set_ylim([0, 1.1])
axes[1].grid(axis="y", alpha=0.3)

plt.subplots_adjust(wspace=0.3)
plt.savefig("outputs/cross_validation.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ Cross-validation results saved")


# =============================================================================
# Feature Importance Analysis
# =============================================================================

print("\nAnalyzing feature importance...")

# use random forest to get feature importances
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train_features)

importances = rf.feature_importances_
top_20_idx = np.argsort(importances)[::-1][:20]

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.bar(range(20), importances[top_20_idx], color="#9b59b6", alpha=0.8)
ax.set_xlabel("Feature Index", fontsize=12, fontweight="bold")
ax.set_ylabel("Importance", fontsize=12, fontweight="bold")
ax.set_title("Top 20 Most Important Features", fontsize=14, fontweight="bold")
ax.set_xticks(range(20))
ax.set_xticklabels(top_20_idx, rotation=45)
ax.grid(axis="y", alpha=0.3)

plt.subplots_adjust(bottom=0.15)
plt.savefig("outputs/feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ Feature importance saved")


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
        "mean_accuracy": float(cv_scores.mean()),
        "std_accuracy": float(cv_scores.std()),
        "mean_auc": float(cv_auc.mean()),
        "std_auc": float(cv_auc.std()),
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

# save as JSON
with open("outputs/results.json", "w") as f:
    json.dump(results, f, indent=4)

# save text summary
summary = f"""
{'='*70}
LUNG LESION DETECTION - RESULTS SUMMARY
{'='*70}

DATASET:
--------
Training:   {len(X_train)} images (Benign: {np.sum(y_train==0)}, Malignant: {np.sum(y_train==1)})
Validation: {len(X_valid)} images (Benign: {np.sum(y_valid==0)}, Malignant: {np.sum(y_valid==1)})
Test:       {len(X_test)} images (Benign: {np.sum(y_test==0)}, Malignant: {np.sum(y_test==1)})

FEATURES: {X_train_features.shape[1]}

TRAINING PERFORMANCE:
---------------------
Accuracy:    {train_metrics['accuracy']:.4f} ({train_metrics['accuracy']*100:.2f}%)
Sensitivity: {train_metrics['sensitivity']:.4f}
Specificity: {train_metrics['specificity']:.4f}
AUC:         {train_metrics['auc']:.4f}
"""

if test_metrics:
    summary += f"""
TEST PERFORMANCE:
-----------------
Accuracy:    {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)
Sensitivity: {test_metrics['sensitivity']:.4f}
Specificity: {test_metrics['specificity']:.4f}
AUC:         {test_metrics['auc']:.4f}
"""

summary += f"""
CROSS-VALIDATION (5-Fold):
--------------------------
Mean Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
Mean AUC:      {cv_auc.mean():.4f} ± {cv_auc.std():.4f}

{'='*70}
"""

with open("outputs/results_summary.txt", "w") as f:
    f.write(summary)

print(summary)
print("\n✅ Results saved to outputs/")


# =============================================================================
# Final Summary
# =============================================================================

print("\n" + "=" * 70)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nGenerated files in 'outputs/' folder:")
print("  1. sample_images.png")
print("  2. segmentation_ref.png")
print("  3. confusion_matrices.png")
print("  4. roc_curves.png")
print("  5. metrics_comparison.png")
print("  6. cross_validation.png")
print("  7. feature_importance.png")
print("  8. results_summary.txt")
print("  9. results.json")
print("\nNext steps:")
print("  - Check results.json for your metric values")
print("  - Use the generated figures in your report")
print("  - Submit before deadline!")
print("=" * 70)
print("\n✅ All done! Good luck with your submission! 🚀")
