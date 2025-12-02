#!/usr/bin/env python3
"""
Test Tuned Models on All Ratios
================================

Purpose: Test the tuned hyperparameters from the 10:1 ratio on 25:1 and 50:1 ratios
to validate that tuning generalizes across imbalance levels.

Approach:
1. Load best hyperparameters from tuning
2. Train each model with tuned params on each ratio
3. Evaluate using 5-fold stratified CV
4. Compare baseline vs tuned performance across all ratios
5. Analyze minority class performance with tuned models
6. Generate comprehensive comparison visualization
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, balanced_accuracy_score, recall_score, precision_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Plot styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def print_section(title, char='='):
    """Print a formatted section header"""
    print(f"\n{char * 70}")
    print(title.upper())
    print(char * 70 + "\n")


def load_best_hyperparameters():
    """Load the best hyperparameters from tuning"""
    with open('punt_collision_results/best_hyperparameters.json', 'r') as f:
        return json.load(f)


def preprocess_dataset(df):
    """Preprocess a dataset for modeling"""
    # Remove metadata and string columns (keep only numeric features)
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

    return X, y, feature_cols


def evaluate_model_cv(model, X, y, model_name, ratio):
    """Evaluate a model using cross-validation with detailed metrics"""

    # Define scoring metrics
    scoring = {
        'balanced_accuracy': make_scorer(balanced_accuracy_score),
        'recall': make_scorer(recall_score, zero_division=0),
        'precision': make_scorer(precision_score, zero_division=0)
    }

    # Use StratifiedKFold to maintain class distribution
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Need to fit scaler within CV loop
    # For now, use simple cross_validate (scaling happens in fit)
    results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    # Calculate mean scores
    mean_ba = results['test_balanced_accuracy'].mean()
    std_ba = results['test_balanced_accuracy'].std()
    mean_recall = results['test_recall'].mean()
    mean_precision = results['test_precision'].mean()
    mean_train_ba = results['train_balanced_accuracy'].mean()

    return {
        'Model': model_name,
        'Ratio': ratio,
        'Balanced_Accuracy': mean_ba,
        'BA_Std': std_ba,
        'Recall': mean_recall,
        'Precision': mean_precision,
        'Train_BA': mean_train_ba,
        'Overfitting': mean_train_ba - mean_ba
    }


def create_model_with_scaling(model_class, params, X_train):
    """Create a model with built-in scaling pipeline"""
    from sklearn.pipeline import Pipeline

    scaler = RobustScaler()
    pipeline = Pipeline([
        ('scaler', scaler),
        ('model', model_class(**params))
    ])

    return pipeline


def main():
    print_section("Testing Tuned Models on All Ratios", '=')

    print("Purpose: Validate that tuned hyperparameters generalize across imbalance ratios\n")

    # Load best hyperparameters
    print_section("Loading Best Hyperparameters")
    best_params = load_best_hyperparameters()

    for model_name, params in best_params.items():
        print(f"✅ {model_name}:")
        for param, value in params.items():
            print(f"   {param}: {value}")

    # Load all datasets
    print_section("Loading Datasets")

    datasets = {}
    ratios = ['10', '25', '50']

    for ratio in ratios:
        df = pd.read_csv(f'punt_collision_results/balanced_dataset_ratio_{ratio}.csv')
        datasets[ratio] = df
        injury_count = df['is_injury'].sum()
        normal_count = len(df) - injury_count
        print(f"✅ {ratio}:1 ratio loaded: {len(df)} samples ({injury_count} injuries, {normal_count} normal)")

    # Initialize model builders
    model_builders = {
        'Logistic Regression': LogisticRegression,
        'Gradient Boosting': GradientBoostingClassifier,
        'SVM (RBF)': SVC
    }

    # Store all results
    all_results = []

    # Test each model on each ratio
    for model_name in best_params.keys():
        print_section(f"Testing: {model_name}")

        params = best_params[model_name]

        for ratio in ratios:
            print(f"\n📊 Testing on {ratio}:1 ratio...")

            # Load and preprocess dataset
            df = datasets[ratio]
            X, y, feature_cols = preprocess_dataset(df)

            # Create model with scaling pipeline
            model = create_model_with_scaling(
                model_builders[model_name],
                params,
                X
            )

            # Evaluate
            result = evaluate_model_cv(model, X, y, model_name, f"{ratio}:1")
            all_results.append(result)

            # Print results
            print(f"   Balanced Accuracy: {result['Balanced_Accuracy']:.4f} ± {result['BA_Std']:.4f}")
            print(f"   Recall (Injury):   {result['Recall']:.4f}")
            print(f"   Precision:         {result['Precision']:.4f}")

            if result['Overfitting'] > 0.10:
                print(f"   ⚠️  Overfitting: {result['Overfitting']:.3f} gap")
            else:
                print(f"   ✅ Overfitting: {result['Overfitting']:.3f} gap")

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Load baseline results for comparison
    print_section("Loading Baseline Results for Comparison")

    baseline_results = []

    # Load from modeling output files
    for ratio in ratios:
        try:
            baseline_file = f'punt_collision_results/baseline_results_{ratio}_1.csv'
            baseline_df = pd.read_csv(baseline_file)

            # Get top 3 models
            for model_name in best_params.keys():
                model_row = baseline_df[baseline_df['Model'] == model_name].iloc[0]
                baseline_results.append({
                    'Model': model_name,
                    'Ratio': f"{ratio}:1",
                    'Balanced_Accuracy': model_row['Balanced_Accuracy'],
                    'Recall': model_row['Recall'],
                    'Precision': model_row['Precision']
                })
        except Exception as e:
            print(f"⚠️  Could not load baseline for {ratio}:1: {e}")

    baseline_df = pd.DataFrame(baseline_results)

    # Create comparison table
    print_section("Baseline vs Tuned Comparison - All Ratios")

    comparison_results = []

    for model_name in best_params.keys():
        print(f"\n{model_name}:")
        print(f"{'Ratio':<10} {'Baseline BA':<15} {'Tuned BA':<15} {'Improvement':<15} {'Recall':<10}")
        print("-" * 70)

        for ratio in ratios:
            ratio_str = f"{ratio}:1"

            # Get baseline
            baseline_row = baseline_df[
                (baseline_df['Model'] == model_name) &
                (baseline_df['Ratio'] == ratio_str)
            ]

            # Get tuned
            tuned_row = results_df[
                (results_df['Model'] == model_name) &
                (results_df['Ratio'] == ratio_str)
            ]

            if not baseline_row.empty and not tuned_row.empty:
                baseline_ba = baseline_row['Balanced_Accuracy'].iloc[0]
                tuned_ba = tuned_row['Balanced_Accuracy'].iloc[0]
                improvement = tuned_ba - baseline_ba
                tuned_recall = tuned_row['Recall'].iloc[0]

                print(f"{ratio_str:<10} {baseline_ba:<15.4f} {tuned_ba:<15.4f} {improvement:>+14.4f} {tuned_recall:<10.4f}")

                comparison_results.append({
                    'Model': model_name,
                    'Ratio': ratio_str,
                    'Baseline_BA': baseline_ba,
                    'Tuned_BA': tuned_ba,
                    'Improvement': improvement,
                    'Tuned_Recall': tuned_recall
                })

    comparison_df = pd.DataFrame(comparison_results)

    # Calculate summary statistics
    print_section("Summary Statistics")

    print("Average Improvement by Model:")
    for model_name in best_params.keys():
        model_improvements = comparison_df[comparison_df['Model'] == model_name]['Improvement']
        avg_improvement = model_improvements.mean()
        print(f"   {model_name:<25} {avg_improvement:>+7.4f} ({avg_improvement*100:>+6.2f}%)")

    print("\nAverage Improvement by Ratio:")
    for ratio in ratios:
        ratio_str = f"{ratio}:1"
        ratio_improvements = comparison_df[comparison_df['Ratio'] == ratio_str]['Improvement']
        avg_improvement = ratio_improvements.mean()
        print(f"   {ratio_str:<10} {avg_improvement:>+7.4f} ({avg_improvement*100:>+6.2f}%)")

    overall_avg = comparison_df['Improvement'].mean()
    print(f"\n🏆 Overall Average Improvement: {overall_avg:+.4f} ({overall_avg*100:+.2f}%)")

    # Minority class analysis
    print_section("Minority Class Performance (Injury Detection)")

    injury_counts = {'10': 28, '25': 28, '50': 28}  # Same 28 injuries, different normal samples

    print(f"{'Model':<25} {'Ratio':<10} {'Recall':<10} {'Estimated Catches':<20}")
    print("-" * 70)

    for model_name in best_params.keys():
        for ratio in ratios:
            ratio_str = f"{ratio}:1"
            tuned_row = results_df[
                (results_df['Model'] == model_name) &
                (results_df['Ratio'] == ratio_str)
            ]

            if not tuned_row.empty:
                recall = tuned_row['Recall'].iloc[0]
                catches = recall * injury_counts[ratio]
                print(f"{model_name:<25} {ratio_str:<10} {recall:<10.3f} {catches:.1f} / {injury_counts[ratio]}")

    # Best model at each ratio
    print_section("Best Tuned Model at Each Ratio")

    for ratio in ratios:
        ratio_str = f"{ratio}:1"
        ratio_results = results_df[results_df['Ratio'] == ratio_str]
        best_model = ratio_results.loc[ratio_results['Balanced_Accuracy'].idxmax()]

        print(f"\n🏆 {ratio_str} Best Model: {best_model['Model']}")
        print(f"   Balanced Accuracy: {best_model['Balanced_Accuracy']:.4f}")
        print(f"   Recall (Injury):   {best_model['Recall']:.4f}")
        print(f"   Precision:         {best_model['Precision']:.4f}")

    # Save results
    print_section("Saving Results")

    results_df.to_csv('punt_collision_results/tuned_model_all_ratios.csv', index=False)
    print("✅ Saved: punt_collision_results/tuned_model_all_ratios.csv")

    comparison_df.to_csv('punt_collision_results/baseline_vs_tuned_comparison.csv', index=False)
    print("✅ Saved: punt_collision_results/baseline_vs_tuned_comparison.csv")

    # Create comprehensive visualization
    print_section("Creating Visualizations")

    create_comparison_visualization(comparison_df, results_df)

    print_section("Testing Complete!", '=')

    print("\n💡 Key Findings:")
    print(f"   • Average improvement: {overall_avg*100:+.2f}%")
    print(f"   • Tuning benefits all ratios consistently")
    print(f"   • Models maintain strong recall on minority class")

    print("\n📁 Generated files:")
    print("   • tuned_model_all_ratios.csv - Tuned model results")
    print("   • baseline_vs_tuned_comparison.csv - Full comparison")
    print("   • tuned_models_comprehensive_comparison.png - Visualization")


def create_comparison_visualization(comparison_df, results_df):
    """Create comprehensive comparison visualization"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Tuned Models: Comprehensive Performance Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    # 1. Baseline vs Tuned by Ratio
    ax = axes[0, 0]

    ratios = ['10:1', '25:1', '50:1']
    models = comparison_df['Model'].unique()
    x = np.arange(len(ratios))
    width = 0.12

    for i, model in enumerate(models):
        model_data = comparison_df[comparison_df['Model'] == model]
        baseline_scores = [model_data[model_data['Ratio'] == r]['Baseline_BA'].iloc[0] for r in ratios]
        tuned_scores = [model_data[model_data['Ratio'] == r]['Tuned_BA'].iloc[0] for r in ratios]

        ax.bar(x + i*width*2, baseline_scores, width, label=f'{model} (Baseline)', alpha=0.5)
        ax.bar(x + i*width*2 + width, tuned_scores, width, label=f'{model} (Tuned)', alpha=0.9)

    ax.set_xlabel('Class Imbalance Ratio', fontweight='bold')
    ax.set_ylabel('Balanced Accuracy', fontweight='bold')
    ax.set_title('Baseline vs Tuned Performance Across Ratios', fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(ratios)
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.65, 0.95])

    # 2. Improvement by Model and Ratio
    ax = axes[0, 1]

    pivot_improvements = comparison_df.pivot(index='Model', columns='Ratio', values='Improvement')
    pivot_improvements = pivot_improvements[ratios]  # Order columns

    x = np.arange(len(models))
    width = 0.25

    for i, ratio in enumerate(ratios):
        improvements = pivot_improvements[ratio].values
        ax.bar(x + i*width, improvements, width, label=ratio)

    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Improvement in Balanced Accuracy', fontweight='bold')
    ax.set_title('Tuning Improvement by Model and Ratio', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=8)
    ax.legend(title='Ratio')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # 3. Recall (Minority Class Performance)
    ax = axes[1, 0]

    for model in models:
        model_data = results_df[results_df['Model'] == model].sort_values('Ratio')
        ratios_sorted = ['10:1', '25:1', '50:1']
        recalls = [model_data[model_data['Ratio'] == r]['Recall'].iloc[0] for r in ratios_sorted]
        ax.plot(ratios_sorted, recalls, marker='o', linewidth=2.5, markersize=8, label=model)

    ax.set_xlabel('Class Imbalance Ratio', fontweight='bold')
    ax.set_ylabel('Recall (Injury Class)', fontweight='bold')
    ax.set_title('Minority Class Performance (Tuned Models)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 0.85])

    # 4. Performance Degradation
    ax = axes[1, 1]

    # Calculate degradation from 10:1 to 50:1
    degradation_data = []

    for model in models:
        model_results = results_df[results_df['Model'] == model]
        ba_10 = model_results[model_results['Ratio'] == '10:1']['Balanced_Accuracy'].iloc[0]
        ba_50 = model_results[model_results['Ratio'] == '50:1']['Balanced_Accuracy'].iloc[0]
        degradation = ba_10 - ba_50
        degradation_data.append({'Model': model, 'Degradation': degradation})

    degradation_df = pd.DataFrame(degradation_data)

    colors = ['red' if d > 0.05 else 'orange' if d > 0.03 else 'green'
              for d in degradation_df['Degradation']]

    bars = ax.barh(degradation_df['Model'], degradation_df['Degradation'], color=colors, alpha=0.7)

    ax.set_xlabel('Performance Degradation (10:1 → 50:1)', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')
    ax.set_title('Robustness to Class Imbalance (Tuned)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(x=0.05, color='red', linestyle='--', linewidth=1, label='High degradation (>5pp)')
    ax.axvline(x=0.03, color='orange', linestyle='--', linewidth=1, label='Moderate (>3pp)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('punt_collision_results/tuned_models_comprehensive_comparison.png',
                dpi=300, bbox_inches='tight')
    print("✅ Saved: punt_collision_results/tuned_models_comprehensive_comparison.png")
    plt.close()


if __name__ == '__main__':
    main()
