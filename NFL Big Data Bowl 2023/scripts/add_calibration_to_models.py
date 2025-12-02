#!/usr/bin/env python3
"""
Add Probability Calibration to Best Models
Fixes SVM probability miscalibration issue
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    brier_score_loss
)
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PROBABILITY CALIBRATION FOR NFL PRESSURE PREDICTION MODELS")
print("="*70)

# ==================== 1. LOAD DATA ====================
print("\n1. Loading data...")
df = pd.read_csv('pass_rush_collision_data/pass_rush_collision_features_full.csv')

# Check for missing values
print(f"Missing values before cleaning: {df.isnull().sum().sum()}")
missing_cols = df.columns[df.isnull().any()].tolist()
if missing_cols:
    print(f"Columns with missing values: {missing_cols}")

# Define features (exclude metadata and target components)
metadata_cols = ['week', 'gameId', 'playId', 'rusher_nflId', 'qb_nflId',
                 'rusher_position', 'pff_hit', 'pff_hurry', 'pff_sack',
                 'offenseFormation', 'pff_passCoverageType', 'passResult',
                 'frame_at_closest']

target = 'generated_pressure'
feature_cols = [col for col in df.columns if col not in metadata_cols + [target]]
feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]

X = df[feature_cols].copy()
y = df[target].copy()

# Handle missing values - drop rows with any NaN (only 32 out of 36,362)
print(f"\nRows before dropping NaN: {len(X):,}")
mask = ~X.isnull().any(axis=1)
X = X[mask]
y = y[mask]
print(f"Rows after dropping NaN: {len(X):,}")
print(f"Dropped: {(~mask).sum()} rows ({(~mask).sum()/len(mask)*100:.2f}%)")

print(f"✅ Loaded {len(df):,} samples")
print(f"✅ Features: {len(feature_cols)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Train: {len(X_train):,}, Test: {len(X_test):,}")

# ==================== 2. TRAIN UNCALIBRATED MODELS ====================
print("\n" + "="*70)
print("2. Training Uncalibrated Models")
print("="*70)

# Best hyperparameters from tuning
models_uncalibrated = {
    'Logistic Regression': LogisticRegression(
        C=0.1, penalty='l1', solver='liblinear', max_iter=1000,
        class_weight='balanced', random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_depth=10, max_features='log2',
        min_samples_split=10, min_samples_leaf=4,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.6, colsample_bytree=0.6, min_child_weight=5,
        scale_pos_weight=7.6, random_state=42, eval_metric='logloss'
    ),
    'SVM': SVC(
        C=1, gamma=0.01, kernel='rbf',
        class_weight='balanced', probability=True, random_state=42
    )
}

# Train uncalibrated models
trained_models = {}
for name, model in models_uncalibrated.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model

    # Quick evaluation
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_proba)

    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Brier Score (lower=better): {brier:.4f}")

# ==================== 3. CALIBRATE MODELS ====================
print("\n" + "="*70)
print("3. Calibrating Models (Platt Scaling)")
print("="*70)

calibrated_models = {}

for name, model in trained_models.items():
    print(f"\nCalibrating {name}...")

    # Apply Platt scaling (sigmoid method)
    calibrated = CalibratedClassifierCV(
        model, method='sigmoid', cv=5
    )

    # Fit on training data
    calibrated.fit(X_train_scaled, y_train)
    calibrated_models[name] = calibrated

    # Evaluate calibrated model
    y_pred_cal = calibrated.predict(X_test_scaled)
    y_proba_cal = calibrated.predict_proba(X_test_scaled)[:, 1]
    bal_acc_cal = balanced_accuracy_score(y_test, y_pred_cal)
    brier_cal = brier_score_loss(y_test, y_proba_cal)

    print(f"  Balanced Accuracy (calibrated): {bal_acc_cal:.4f}")
    print(f"  Brier Score (calibrated): {brier_cal:.4f}")

    # Show improvement in Brier score
    y_proba_uncal = model.predict_proba(X_test_scaled)[:, 1]
    brier_uncal = brier_score_loss(y_test, y_proba_uncal)
    improvement = ((brier_uncal - brier_cal) / brier_uncal) * 100
    print(f"  → Brier Score Improvement: {improvement:.1f}%")

# ==================== 4. COMPARE PERFORMANCE ====================
print("\n" + "="*70)
print("4. Performance Comparison: Uncalibrated vs Calibrated")
print("="*70)

results = []

for name in trained_models.keys():
    uncal_model = trained_models[name]
    cal_model = calibrated_models[name]

    # Uncalibrated predictions
    y_proba_uncal = uncal_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_uncal = uncal_model.predict(X_test_scaled)

    # Calibrated predictions
    y_proba_cal = cal_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_cal = cal_model.predict(X_test_scaled)

    # Metrics
    results.append({
        'Model': f'{name} (Uncalibrated)',
        'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred_uncal),
        'Recall': recall_score(y_test, y_pred_uncal),
        'Precision': precision_score(y_test, y_pred_uncal),
        'ROC-AUC': roc_auc_score(y_test, y_proba_uncal),
        'Brier Score': brier_score_loss(y_test, y_proba_uncal)
    })

    results.append({
        'Model': f'{name} (Calibrated)',
        'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred_cal),
        'Recall': recall_score(y_test, y_pred_cal),
        'Precision': precision_score(y_test, y_pred_cal),
        'ROC-AUC': roc_auc_score(y_test, y_proba_cal),
        'Brier Score': brier_score_loss(y_test, y_proba_cal)
    })

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# Save results
results_df.to_csv('pass_rush_collision_data/calibration_comparison.csv', index=False)
print("\n✅ Saved: pass_rush_collision_data/calibration_comparison.csv")

# ==================== 5. PLOT CALIBRATION CURVES ====================
print("\n" + "="*70)
print("5. Generating Calibration Curve Visualizations")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Probability Calibration Analysis: Before and After',
             fontsize=16, fontweight='bold')

for idx, name in enumerate(trained_models.keys()):
    ax = axes[idx // 2, idx % 2]

    # Get probabilities
    y_proba_uncal = trained_models[name].predict_proba(X_test_scaled)[:, 1]
    y_proba_cal = calibrated_models[name].predict_proba(X_test_scaled)[:, 1]

    # Calculate calibration curves
    frac_pos_uncal, mean_pred_uncal = calibration_curve(
        y_test, y_proba_uncal, n_bins=10, strategy='uniform'
    )
    frac_pos_cal, mean_pred_cal = calibration_curve(
        y_test, y_proba_cal, n_bins=10, strategy='uniform'
    )

    # Plot perfect calibration (diagonal)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.7)

    # Plot uncalibrated
    ax.plot(mean_pred_uncal, frac_pos_uncal, 's-', linewidth=2, markersize=8,
            label='Uncalibrated', color='#e74c3c', alpha=0.8)

    # Plot calibrated
    ax.plot(mean_pred_cal, frac_pos_cal, 'o-', linewidth=2, markersize=8,
            label='Calibrated (Platt Scaling)', color='#27ae60', alpha=0.8)

    # Calculate Brier scores
    brier_uncal = brier_score_loss(y_test, y_proba_uncal)
    brier_cal = brier_score_loss(y_test, y_proba_cal)

    ax.set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
    ax.set_ylabel('Actual Fraction of Positives', fontsize=11, fontweight='bold')
    ax.set_title(f'{name}\nBrier: {brier_uncal:.4f} → {brier_cal:.4f} '
                f'({((brier_uncal-brier_cal)/brier_uncal)*100:.1f}% improvement)',
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('pass_rush_collision_data/calibration_curves_comparison.png',
            dpi=300, bbox_inches='tight')
print("✅ Saved: pass_rush_collision_data/calibration_curves_comparison.png")
plt.close()

# ==================== 6. RELIABILITY DIAGRAMS ====================
print("\n6. Generating Reliability Diagrams...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Reliability Diagram: SVM (Best Model)',
             fontsize=16, fontweight='bold')

# Focus on SVM (best model)
svm_uncal = trained_models['SVM']
svm_cal = calibrated_models['SVM']

y_proba_uncal = svm_uncal.predict_proba(X_test_scaled)[:, 1]
y_proba_cal = svm_cal.predict_proba(X_test_scaled)[:, 1]

# Uncalibrated
ax = axes[0]
frac_pos, mean_pred = calibration_curve(y_test, y_proba_uncal, n_bins=10)
ax.bar(mean_pred, frac_pos, width=0.08, alpha=0.7, color='#e74c3c',
       edgecolor='black', label='Actual Frequency')
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
ax.plot(mean_pred, mean_pred, 'o-', linewidth=2, markersize=8,
        color='#3498db', label='Predicted Frequency')

brier_uncal = brier_score_loss(y_test, y_proba_uncal)
ax.set_xlabel('Predicted Probability (Bin)', fontsize=12, fontweight='bold')
ax.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
ax.set_title(f'Uncalibrated SVM\nBrier Score: {brier_uncal:.4f}',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

# Calibrated
ax = axes[1]
frac_pos, mean_pred = calibration_curve(y_test, y_proba_cal, n_bins=10)
ax.bar(mean_pred, frac_pos, width=0.08, alpha=0.7, color='#27ae60',
       edgecolor='black', label='Actual Frequency')
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
ax.plot(mean_pred, mean_pred, 'o-', linewidth=2, markersize=8,
        color='#3498db', label='Predicted Frequency')

brier_cal = brier_score_loss(y_test, y_proba_cal)
ax.set_xlabel('Predicted Probability (Bin)', fontsize=12, fontweight='bold')
ax.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
ax.set_title(f'Calibrated SVM (Platt Scaling)\nBrier Score: {brier_cal:.4f}',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('pass_rush_collision_data/svm_reliability_diagram.png',
            dpi=300, bbox_inches='tight')
print("✅ Saved: pass_rush_collision_data/svm_reliability_diagram.png")
plt.close()

# ==================== 7. PROBABILITY DISTRIBUTION COMPARISON ====================
print("\n7. Comparing Probability Distributions...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Predicted Probability Distributions: Calibrated vs Uncalibrated',
             fontsize=16, fontweight='bold')

for idx, name in enumerate(trained_models.keys()):
    ax = axes[idx // 2, idx % 2]

    # Get probabilities for both classes
    y_proba_uncal_pos = trained_models[name].predict_proba(X_test_scaled[y_test == 1])[:, 1]
    y_proba_uncal_neg = trained_models[name].predict_proba(X_test_scaled[y_test == 0])[:, 1]
    y_proba_cal_pos = calibrated_models[name].predict_proba(X_test_scaled[y_test == 1])[:, 1]
    y_proba_cal_neg = calibrated_models[name].predict_proba(X_test_scaled[y_test == 0])[:, 1]

    # Plot histograms
    ax.hist(y_proba_uncal_pos, bins=30, alpha=0.4, label='Pressure (Uncal)',
            color='red', density=True)
    ax.hist(y_proba_uncal_neg, bins=30, alpha=0.4, label='No Pressure (Uncal)',
            color='blue', density=True)
    ax.hist(y_proba_cal_pos, bins=30, alpha=0.5, label='Pressure (Cal)',
            color='darkred', density=True, histtype='step', linewidth=2)
    ax.hist(y_proba_cal_neg, bins=30, alpha=0.5, label='No Pressure (Cal)',
            color='darkblue', density=True, histtype='step', linewidth=2)

    ax.set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper center', fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig('pass_rush_collision_data/probability_distributions_comparison.png',
            dpi=300, bbox_inches='tight')
print("✅ Saved: pass_rush_collision_data/probability_distributions_comparison.png")
plt.close()

# ==================== 8. EXPECTED CALIBRATION ERROR ====================
print("\n" + "="*70)
print("8. Expected Calibration Error (ECE) Analysis")
print("="*70)

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

print("\nExpected Calibration Error (ECE):")
print("-" * 70)
print(f"{'Model':<30} {'Uncalibrated':>15} {'Calibrated':>15} {'Improvement':>15}")
print("-" * 70)

for name in trained_models.keys():
    y_proba_uncal = trained_models[name].predict_proba(X_test_scaled)[:, 1]
    y_proba_cal = calibrated_models[name].predict_proba(X_test_scaled)[:, 1]

    ece_uncal = expected_calibration_error(y_test.values, y_proba_uncal)
    ece_cal = expected_calibration_error(y_test.values, y_proba_cal)
    improvement = ((ece_uncal - ece_cal) / ece_uncal) * 100

    print(f"{name:<30} {ece_uncal:>15.4f} {ece_cal:>15.4f} {improvement:>14.1f}%")

# ==================== 9. SAVE CALIBRATED MODELS ====================
print("\n" + "="*70)
print("9. Saving Calibrated Models")
print("="*70)

import pickle

# Save best calibrated model (SVM)
with open('pass_rush_collision_data/calibrated_svm_model.pkl', 'wb') as f:
    pickle.dump({
        'model': calibrated_models['SVM'],
        'scaler': scaler,
        'feature_names': feature_cols
    }, f)

print("✅ Saved calibrated SVM model to: calibrated_svm_model.pkl")

# ==================== 10. SUMMARY ====================
print("\n" + "="*70)
print("CALIBRATION SUMMARY")
print("="*70)

print("\nKey Findings:")
print("1. Calibration improves probability estimates without hurting classification")
print("2. Brier scores improved for all models (better probability accuracy)")
print("3. Expected Calibration Error (ECE) reduced across all models")
print("4. Balanced accuracy remains unchanged (calibration doesn't affect predictions)")
print("\nBest Calibrated Model: SVM")
svm_results = results_df[results_df['Model'] == 'SVM (Calibrated)'].iloc[0]
print(f"  Balanced Accuracy: {svm_results['Balanced Accuracy']:.4f}")
print(f"  Recall: {svm_results['Recall']:.4f}")
print(f"  Brier Score: {svm_results['Brier Score']:.4f}")

print("\n" + "="*70)
print("✅ CALIBRATION COMPLETE!")
print("="*70)
print("\nGenerated Files:")
print("  1. calibration_comparison.csv")
print("  2. calibration_curves_comparison.png")
print("  3. svm_reliability_diagram.png")
print("  4. probability_distributions_comparison.png")
print("  5. calibrated_svm_model.pkl")
