#!/usr/bin/env python3
"""
Hyperparameter Tuning for Top Models
Focus on Logistic Regression, Gradient Boosting, and SVM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import make_scorer, balanced_accuracy_score

# Models to tune
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*70)
print("HYPERPARAMETER TUNING - TOP 3 MODELS")
print("="*70)
print("\nTarget: Optimize for Balanced Accuracy (handles class imbalance)")
print("Strategy: GridSearchCV with 5-fold Stratified CV")
print("\nModels to tune:")
print("  1. Logistic Regression (current best)")
print("  2. Gradient Boosting (2nd best)")
print("  3. SVM (RBF) (non-linear option)")

# Load and preprocess function
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

    # Scale
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    return X_scaled, y, feature_cols

# Define parameter grids
def get_param_grids():
    """Define hyperparameter grids for tuning"""

    param_grids = {
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'class_weight': ['balanced'],
            'max_iter': [1000],
            'random_state': [RANDOM_STATE]
        },

        'Gradient Boosting': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10],
            'random_state': [RANDOM_STATE]
        },

        'SVM (RBF)': {
            'C': [0.1, 1.0, 10.0, 100.0],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf'],
            'class_weight': ['balanced'],
            'probability': [True],
            'random_state': [RANDOM_STATE]
        }
    }

    return param_grids

# Tune a single model
def tune_model(model_name, model, param_grid, X, y, cv_folds=5):
    """Tune hyperparameters for a model"""

    print(f"\n{'='*70}")
    print(f"TUNING: {model_name}")
    print(f"{'='*70}")
    print(f"Parameter grid size: {np.prod([len(v) for v in param_grid.values()]):,} combinations")

    # Setup cross-validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    # Setup grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=make_scorer(balanced_accuracy_score),
        cv=skf,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    # Fit
    print(f"\nStarting grid search with {cv_folds}-fold CV...")
    start_time = time()
    grid_search.fit(X, y)
    elapsed_time = time() - start_time

    print(f"\n✅ Grid search complete! Time: {elapsed_time/60:.1f} minutes")

    # Results
    print(f"\n🏆 Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")

    print(f"\n📊 Performance:")
    print(f"   Best CV Score (BA): {grid_search.best_score_:.4f}")
    print(f"   Mean train score: {grid_search.cv_results_['mean_train_score'][grid_search.best_index_]:.4f}")

    # Check overfitting
    train_score = grid_search.cv_results_['mean_train_score'][grid_search.best_index_]
    cv_score = grid_search.best_score_
    overfit_gap = train_score - cv_score

    if overfit_gap > 0.1:
        print(f"   ⚠️  Overfitting detected: {overfit_gap:.3f} gap between train/CV")
    else:
        print(f"   ✅ No significant overfitting: {overfit_gap:.3f} gap")

    # Top 5 parameter combinations
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('mean_test_score', ascending=False)

    print(f"\n📋 Top 5 parameter combinations:")
    for i, (idx, row) in enumerate(results_df.head(5).iterrows(), 1):
        print(f"   {i}. Score: {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")
        params_str = ', '.join([f"{k.replace('param_', '')}={v}"
                                for k, v in row.items()
                                if k.startswith('param_')])
        print(f"      {params_str}")

    return {
        'model_name': model_name,
        'best_estimator': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_,
        'grid_search': grid_search
    }

# Run tuning for all models
def run_tuning_pipeline(data, ratio_name):
    """Run complete tuning pipeline"""

    print(f"\n{'='*70}")
    print(f"TUNING PIPELINE - {ratio_name} RATIO")
    print(f"{'='*70}")

    # Preprocess
    print(f"\nPreprocessing {len(data)} samples...")
    X, y, feature_cols = preprocess_dataset(data)
    print(f"   Features: {len(feature_cols)}")
    print(f"   Injury rate: {y.mean()*100:.1f}%")

    # Get parameter grids
    param_grids = get_param_grids()

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'SVM (RBF)': SVC()
    }

    # Tune each model
    tuning_results = {}

    for model_name in models.keys():
        result = tune_model(
            model_name,
            models[model_name],
            param_grids[model_name],
            X, y,
            cv_folds=5
        )
        tuning_results[model_name] = result

    return tuning_results, X, y

# Compare baseline vs tuned
def compare_baseline_vs_tuned(baseline_results, tuning_results):
    """Compare baseline and tuned performance"""

    print(f"\n{'='*70}")
    print("BASELINE vs TUNED COMPARISON")
    print(f"{'='*70}")

    print(f"\n{'Model':<25} {'Baseline BA':<15} {'Tuned BA':<15} {'Improvement':<15}")
    print("-"*70)

    improvements = []

    for model_name in tuning_results.keys():
        # Get baseline score
        baseline_row = baseline_results[baseline_results['Model'] == model_name]
        if len(baseline_row) > 0:
            baseline_ba = baseline_row['Balanced_Accuracy'].values[0]
        else:
            baseline_ba = 0

        # Get tuned score
        tuned_ba = tuning_results[model_name]['best_score']

        # Calculate improvement
        improvement = tuned_ba - baseline_ba
        improvements.append({
            'Model': model_name,
            'Baseline_BA': baseline_ba,
            'Tuned_BA': tuned_ba,
            'Improvement': improvement,
            'Improvement_Pct': improvement / baseline_ba * 100 if baseline_ba > 0 else 0
        })

        print(f"{model_name:<25} {baseline_ba:<15.4f} {tuned_ba:<15.4f} {improvement:>+14.4f}")

    improvements_df = pd.DataFrame(improvements)

    print(f"\n💡 Summary:")
    avg_improvement = improvements_df['Improvement'].mean()
    print(f"   Average improvement: {avg_improvement:+.4f} ({avg_improvement/improvements_df['Baseline_BA'].mean()*100:+.1f}%)")

    best_improvement = improvements_df.loc[improvements_df['Improvement'].idxmax()]
    print(f"   Best improvement: {best_improvement['Model']} ({best_improvement['Improvement']:+.4f})")

    return improvements_df

# Main execution
print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

# Load datasets
data_10 = pd.read_csv('punt_collision_results/balanced_dataset_ratio_10.csv')
data_25 = pd.read_csv('punt_collision_results/balanced_dataset_ratio_25.csv')
data_50 = pd.read_csv('punt_collision_results/balanced_dataset_ratio_50.csv')

# Load baseline results
baseline_10 = pd.read_csv('punt_collision_results/baseline_results_10_1.csv')
baseline_25 = pd.read_csv('punt_collision_results/baseline_results_25_1.csv')
baseline_50 = pd.read_csv('punt_collision_results/baseline_results_50_1.csv')

print(f"\nDatasets loaded:")
print(f"   10:1 ratio: {len(data_10)} samples")
print(f"   25:1 ratio: {len(data_25)} samples")
print(f"   50:1 ratio: {len(data_50)} samples")

# Run tuning for 10:1 ratio (primary focus)
print("\n" + "="*70)
print("FOCUS: TUNING ON 10:1 RATIO")
print("="*70)
print("\nRationale: Most balanced dataset, best for finding optimal parameters")

tuning_results_10, X_10, y_10 = run_tuning_pipeline(data_10, "10:1")

# Compare to baseline
improvements_10 = compare_baseline_vs_tuned(baseline_10, tuning_results_10)

# Save tuning results
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save improvements
improvements_10.to_csv('punt_collision_results/tuning_improvements_10_1.csv', index=False)
print("\n✅ Saved: punt_collision_results/tuning_improvements_10_1.csv")

# Save best parameters for each model
import json

best_params_all = {}
for model_name, result in tuning_results_10.items():
    best_params_all[model_name] = result['best_params']

with open('punt_collision_results/best_hyperparameters.json', 'w') as f:
    json.dump(best_params_all, f, indent=2, default=str)
print("✅ Saved: punt_collision_results/best_hyperparameters.json")

# Save detailed CV results
for model_name, result in tuning_results_10.items():
    cv_results_df = pd.DataFrame(result['cv_results'])
    filename = f"punt_collision_results/tuning_cv_results_{model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.csv"
    cv_results_df.to_csv(filename, index=False)
    print(f"✅ Saved: {filename}")

# Create comparison visualization
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Baseline vs Tuned comparison
ax = axes[0]
x_pos = np.arange(len(improvements_10))
width = 0.35

bars1 = ax.bar(x_pos - width/2, improvements_10['Baseline_BA'], width,
               label='Baseline', alpha=0.8, color='#3498db')
bars2 = ax.bar(x_pos + width/2, improvements_10['Tuned_BA'], width,
               label='Tuned', alpha=0.8, color='#e74c3c')

ax.set_xlabel('Model', fontweight='bold', fontsize=12)
ax.set_ylabel('Balanced Accuracy', fontweight='bold', fontsize=12)
ax.set_title('Baseline vs Tuned Performance', fontweight='bold', fontsize=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(improvements_10['Model'], rotation=45, ha='right')
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(0, 1.0)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

# Plot 2: Improvement magnitude
ax = axes[1]
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in improvements_10['Improvement']]
bars = ax.barh(range(len(improvements_10)), improvements_10['Improvement'],
               color=colors, alpha=0.8)

ax.set_yticks(range(len(improvements_10)))
ax.set_yticklabels(improvements_10['Model'])
ax.set_xlabel('Improvement (BA)', fontweight='bold', fontsize=12)
ax.set_title('Tuning Improvement by Model', fontweight='bold', fontsize=14)
ax.axvline(0, color='black', linewidth=0.8)
ax.grid(alpha=0.3, axis='x')

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width:+.4f}',
            ha='left' if width > 0 else 'right',
            va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('punt_collision_results/tuning_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: punt_collision_results/tuning_comparison.png")
plt.close()

# Final summary
print("\n" + "="*70)
print("HYPERPARAMETER TUNING COMPLETE!")
print("="*70)

print("\n🏆 Best Model After Tuning:")
best_model = improvements_10.loc[improvements_10['Tuned_BA'].idxmax()]
print(f"   Model: {best_model['Model']}")
print(f"   Tuned BA: {best_model['Tuned_BA']:.4f}")
print(f"   Improvement: {best_model['Improvement']:+.4f}")

print("\n💡 Key Findings:")
if improvements_10['Improvement'].mean() > 0:
    print(f"   ✅ Tuning improved performance by {improvements_10['Improvement'].mean()*100:.1f}% on average")
else:
    print(f"   ⚠️  Baseline parameters were already near-optimal")

print("\n📁 Generated files:")
print("   • tuning_improvements_10_1.csv - Performance comparison")
print("   • best_hyperparameters.json - Optimal parameters for each model")
print("   • tuning_cv_results_*.csv - Detailed CV results for each model")
print("   • tuning_comparison.png - Visualization")

print("\n🚀 Next Steps:")
print("   1. Test tuned models on 25:1 and 50:1 ratios")
print("   2. Create comprehensive visualizations")
print("   3. Perform error analysis")
print("   4. Generate final technical report")

print("\n" + "="*70)
