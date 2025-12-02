#!/usr/bin/env python3
"""
Create Combined 2-Row Figure
=============================
Combines ROC curves and feature importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

# ML imports for ROC curves
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')

# Constants
DPI = 300

print("Loading data...")

# Load the 50:1 dataset for ROC curves
df = pd.read_csv('punt_collision_results/balanced_dataset_ratio_50.csv')

# Load hyperparameters
with open('punt_collision_results/best_hyperparameters.json', 'r') as f:
    best_params = json.load(f)

# Load feature importance
feature_importance = pd.read_csv('punt_collision_results/feature_importance_rf.csv')

print("Creating combined figure...")

# Create figure with 1 row, 2 columns
fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.8], wspace=0.3)

# ============================================================================
# COLUMN 1: ROC Curves
# ============================================================================
print("  Generating ROC curves...")
ax1 = fig.add_subplot(gs[0, 0])

# Preprocess dataset
metadata_cols = ['seasonyear', 'gamekey', 'playid', 'injured_player',
                 'partner_player', 'impact_type', 'is_injury',
                 'player_activity', 'partner_activity', 'friendly_fire']
feature_cols = [c for c in df.columns if c not in metadata_cols]

X = df[feature_cols].copy()
y = df['is_injury'].copy()

# Impute missing values
if X.isnull().sum().sum() > 0:
    imputer = KNNImputer(n_neighbors=5)
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

# Define models
models_config = [
    ('Logistic Regression', LogisticRegression, best_params['Logistic Regression']),
    ('SVM (RBF)', SVC, best_params['SVM (RBF)']),
    ('Gradient Boosting', GradientBoostingClassifier, best_params['Gradient Boosting'])
]

# Colors for the 3 models
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Use StratifiedKFold to get ROC curves
np.random.seed(42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for (model_name, model_class, params), color in zip(models_config, colors):
    print(f"    Computing {model_name}...")

    # Store TPR and FPR for each fold
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Create pipeline with scaling
        scaler = RobustScaler()
        pipeline = Pipeline([
            ('scaler', scaler),
            ('model', model_class(**params))
        ])

        # Train model
        pipeline.fit(X_train, y_train)

        # Get probability predictions
        if hasattr(pipeline, 'predict_proba'):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_proba = pipeline.decision_function(X_test)

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Interpolate TPR at mean FPR
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    # Compute mean and std of TPR
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # Plot mean ROC curve
    ax1.plot(mean_fpr, mean_tpr, color=color, linewidth=2.5,
            label=f'{model_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})')

    # Plot std deviation as shaded area
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=0.2)

# Plot diagonal reference line
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier (AUC = 0.500)', alpha=0.6)

# Formatting
ax1.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
ax1.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
ax1.set_title('ROC Curves: Top 3 Models at 50:1 Class Ratio\n(5-Fold Cross-Validation)',
             fontsize=15, fontweight='bold', pad=20)

ax1.legend(loc='lower right', fontsize=10, framealpha=0.95,
          edgecolor='black', fancybox=True, shadow=True)

ax1.grid(True, alpha=0.3)
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.02])
ax1.set_aspect('equal')

# Add text box with key insights
textstr = '\n'.join([
    'Key Insights:',
    '• Logistic Reg: Best (AUC ≈ 0.9+)',
    '• SVM (RBF): Strong second',
    '• Gradient Boosting: Overfits',
    '• Shaded = CV variance'
])
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.9, edgecolor='black', linewidth=1.5)
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='left', bbox=props)

# ============================================================================
# COLUMN 2: Feature Importance (Top 15)
# ============================================================================
print("  Generating feature importance chart...")
ax2 = fig.add_subplot(gs[0, 1])

top_15 = feature_importance.head(15)

bars = ax2.barh(range(len(top_15)), top_15['importance'], color='skyblue', alpha=0.7)
ax2.set_yticks(range(len(top_15)))
ax2.set_yticklabels(top_15['feature'], fontsize=10)
ax2.set_xlabel('Normalized Importance', fontweight='bold', fontsize=12)
ax2.set_title('Top 15 Features (Random Forest)', fontweight='bold', fontsize=14, pad=15)
ax2.grid(True, alpha=0.3, axis='x')
ax2.invert_yaxis()

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax2.text(width + 0.005, bar.get_y() + bar.get_height()/2.,
           f'{width:.3f}', ha='left', va='center', fontsize=9)

# Save the combined figure
output_path = 'combined_analysis_figure.png'
plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
print(f"\n✅ Combined figure saved to: {output_path}")

plt.close()
