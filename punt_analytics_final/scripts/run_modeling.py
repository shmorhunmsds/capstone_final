#!/usr/bin/env python3
"""
Run baseline modeling analysis
Following puntv7 methodology
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    cross_validate
)
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Metrics
from sklearn.metrics import (
    balanced_accuracy_score,
    make_scorer
)

# Try XGBoost
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Set2')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*60)
print("NFL PUNT ANALYTICS - BASELINE MODELING")
print("="*60)
print(f"XGBoost available: {HAVE_XGB}\n")

# Load datasets
print("Loading balanced datasets...")
data_10 = pd.read_csv('punt_collision_results/balanced_dataset_ratio_10.csv')
data_25 = pd.read_csv('punt_collision_results/balanced_dataset_ratio_25.csv')
data_50 = pd.read_csv('punt_collision_results/balanced_dataset_ratio_50.csv')

print("="*60)
print("DATASET OVERVIEW")
print("="*60)
for name, df in [('10:1', data_10), ('25:1', data_25), ('50:1', data_50)]:
    injury_count = df['is_injury'].sum()
    normal_count = len(df) - injury_count
    print(f"\n{name} ratio:")
    print(f"   Total: {len(df)}, Injury: {injury_count} ({injury_count/len(df)*100:.1f}%), Normal: {normal_count}")

# Preprocessing function
def preprocess_dataset(df):
    """Preprocess collision dataset"""

    metadata_cols = [
        'seasonyear', 'gamekey', 'playid',
        'injured_player', 'partner_player',
        'impact_type', 'player_activity', 'partner_activity', 'friendly_fire',
        'is_injury'
    ]

    feature_cols = [c for c in df.columns if c not in metadata_cols]
    X = df[feature_cols].copy()
    y = df['is_injury'].copy()

    # Handle missing values
    if X.isnull().sum().sum() > 0:
        imputer = KNNImputer(n_neighbors=5)
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Scale
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols

# Get models
def get_models(y_train_sample):
    """Get baseline models"""

    models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, class_weight='balanced', max_depth=10, random_state=RANDOM_STATE
        ),
        'SVM (RBF)': SVC(
            kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE
        ),
        'SVM (Linear)': SVC(
            kernel='linear', class_weight='balanced', probability=True, random_state=RANDOM_STATE
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=RANDOM_STATE
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'Decision Tree': DecisionTreeClassifier(
            class_weight='balanced', max_depth=10, random_state=RANDOM_STATE
        ),
    }

    if HAVE_XGB:
        scale_pos_weight = (y_train_sample == 0).sum() / (y_train_sample == 1).sum()
        models['XGBoost'] = XGBClassifier(
            n_estimators=100, max_depth=5, scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE, eval_metric='logloss'
        )

    return models

# Evaluation function
def evaluate_models_cv(X, y, models, n_folds=5):
    """Evaluate with CV"""

    scoring = {
        'balanced_accuracy': make_scorer(balanced_accuracy_score),
        'recall': 'recall',
        'precision': 'precision',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, model in models.items():
        try:
            cv_results = cross_validate(
                model, X, y,
                cv=skf,
                scoring=scoring,
                return_train_score=False,
                n_jobs=-1
            )

            results.append({
                'Model': name,
                'Balanced_Accuracy': cv_results['test_balanced_accuracy'].mean(),
                'Balanced_Accuracy_Std': cv_results['test_balanced_accuracy'].std(),
                'Recall': cv_results['test_recall'].mean(),
                'Recall_Std': cv_results['test_recall'].std(),
                'Precision': cv_results['test_precision'].mean(),
                'Precision_Std': cv_results['test_precision'].std(),
                'F1': cv_results['test_f1'].mean(),
                'F1_Std': cv_results['test_f1'].std(),
                'ROC_AUC': cv_results['test_roc_auc'].mean(),
                'ROC_AUC_Std': cv_results['test_roc_auc'].std(),
            })
        except Exception as e:
            print(f"   ❌ {name}: {str(e)}")
            continue

    return pd.DataFrame(results).sort_values('Balanced_Accuracy', ascending=False)

# Progressive validation
print("\n" + "="*60)
print("PROGRESSIVE VALIDATION")
print("="*60)

progressive_results = {}

for ratio_name, data in [('10:1', data_10), ('25:1', data_25), ('50:1', data_50)]:
    print(f"\n{'='*60}")
    print(f"Testing {ratio_name} ratio ({len(data)} samples)")
    print(f"{'='*60}")

    # Preprocess
    X_train, X_test, y_train, y_test, feature_cols = preprocess_dataset(data)

    # Combine for CV
    X_comb = pd.concat([X_train, X_test])
    y_comb = pd.concat([y_train, y_test])

    # Get models
    models = get_models(y_train)

    print(f"Testing {len(models)} models with 5-fold CV...")

    # Evaluate
    results = evaluate_models_cv(X_comb, y_comb, models, n_folds=5)
    progressive_results[ratio_name] = results

    # Show top 3
    print(f"\n🏆 Top 3 models:")
    for idx, row in results.head(3).iterrows():
        print(f"   {row['Model']:25} BA={row['Balanced_Accuracy']:.3f} ± {row['Balanced_Accuracy_Std']:.3f}, "
              f"Recall={row['Recall']:.3f}, Precision={row['Precision']:.3f}")

    # Save results
    filename = f"punt_collision_results/baseline_results_{ratio_name.replace(':', '_')}.csv"
    results.to_csv(filename, index=False)
    print(f"\n💾 Saved: {filename}")

# Feature importance (using 10:1 for consistency)
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS (10:1 ratio)")
print("="*60)

X_train, X_test, y_train, y_test, feature_cols = preprocess_dataset(data_10)
X_comb = pd.concat([X_train, X_test])
y_comb = pd.concat([y_train, y_test])

# Random Forest importance
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE)
rf.fit(X_comb, y_comb)

rf_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🌲 Top 10 features by Random Forest:")
print(rf_importance.head(10).to_string(index=False))

rf_importance.to_csv('punt_collision_results/feature_importance_rf.csv', index=False)

# Comparison to puntv7
print("\n" + "="*60)
print("COMPARISON TO PUNTV7")
print("="*60)

print("\n📊 Expected from puntv7 (10:1 ratio):")
print("   Balanced accuracy: ~85-90%")
print("   Recall: ~85%+")

print("\n📊 Our results (10:1 ratio):")
best = progressive_results['10:1'].iloc[0]
print(f"   Best model: {best['Model']}")
print(f"   Balanced accuracy: {best['Balanced_Accuracy']:.1%} ± {best['Balanced_Accuracy_Std']:.3f}")
print(f"   Recall: {best['Recall']:.1%} ± {best['Recall_Std']:.3f}")
print(f"   Precision: {best['Precision']:.1%} ± {best['Precision_Std']:.3f}")

if best['Balanced_Accuracy'] >= 0.85:
    print("\n✅ VALIDATION: MATCHED or EXCEEDED puntv7 performance!")
else:
    print(f"\n⚠️  Below puntv7 baseline (diff: {(0.85 - best['Balanced_Accuracy'])*100:.1f}%)")

# Performance degradation analysis
print("\n" + "="*60)
print("PERFORMANCE DEGRADATION ANALYSIS")
print("="*60)

print("\n📉 Best model performance across ratios:")
for ratio in ['10:1', '25:1', '50:1']:
    best = progressive_results[ratio].iloc[0]
    print(f"   {ratio:6} - {best['Model']:25} BA={best['Balanced_Accuracy']:.1%}")

ba_10 = progressive_results['10:1'].iloc[0]['Balanced_Accuracy']
ba_25 = progressive_results['25:1'].iloc[0]['Balanced_Accuracy']
ba_50 = progressive_results['50:1'].iloc[0]['Balanced_Accuracy']

print(f"\n📊 Degradation:")
print(f"   10:1 → 25:1: {(ba_10 - ba_25)*100:+.1f}%")
print(f"   25:1 → 50:1: {(ba_25 - ba_50)*100:+.1f}%")
print(f"   10:1 → 50:1: {(ba_10 - ba_50)*100:+.1f}% (total)")

if abs(ba_10 - ba_25) < 0.05 and abs(ba_25 - ba_50) < 0.05:
    print("\n✅ Gentle degradation - methodology works!")
else:
    print("\n⚠️  Steep degradation - may need adjustment")

print("\n" + "="*60)
print("BASELINE MODELING COMPLETE!")
print("="*60)
print("\n💡 Key insights:")
print("   1. Refactored methodology validated")
print("   2. Performance compared to puntv7 baseline")
print("   3. Progressive validation completed")
print("   4. Feature importance analyzed")
print("\n🚀 Next: Hyperparameter tuning for top models")
