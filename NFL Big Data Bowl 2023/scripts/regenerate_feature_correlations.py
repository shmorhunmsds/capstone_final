#!/usr/bin/env python3
"""
Regenerate Feature Correlations Visualization
Fixed to exclude target variable components (pff_hit, pff_hurry, pff_sack)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

def main():
    print("Loading feature dataset...")

    # Load dataset
    df = pd.read_csv('pass_rush_collision_data/pass_rush_collision_features_full.csv')

    print(f"✅ Loaded {len(df):,} samples")
    print(f"✅ Total columns: {len(df.columns)}")

    # Define features to EXCLUDE from modeling (data leakage or metadata)
    exclude_cols = [
        'week', 'gameId', 'playId', 'rusher_nflId', 'qb_nflId',
        'rusher_position', 'offenseFormation', 'pff_passCoverageType', 'passResult',
        'generated_pressure',  # Target variable
        'pff_hit', 'pff_hurry', 'pff_sack',  # Target components - DATA LEAKAGE!
        'frame_at_closest'  # Not used in modeling (timestamp identifier)
    ]

    # Select only numeric features used in modeling
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]

    print(f"\n✅ Modeling features: {len(feature_cols)}")
    print(f"✅ Excluded columns: {len(exclude_cols)}")

    # Calculate correlations with target
    numeric_features = df[feature_cols + ['generated_pressure']].copy()
    correlations = numeric_features.corr()['generated_pressure'].abs().sort_values(ascending=False)

    # Remove target itself
    correlations = correlations.drop('generated_pressure')

    print(f"\n📊 Top 10 Features by Correlation:")
    print(correlations.head(10))

    # Create visualization
    print("\nCreating visualization...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Get top 15 features
    top_features = correlations.head(15)

    # Create horizontal bar chart
    colors = ['#e74c3c' if corr > 0.5 else '#3498db' if corr > 0.3 else '#95a5a6'
              for corr in top_features.values]

    bars = ax.barh(range(len(top_features)), top_features.values, color=colors, alpha=0.8)

    # Add value labels
    for i, (feature, corr) in enumerate(top_features.items()):
        ax.text(corr + 0.01, i, f'{corr:.3f}',
                va='center', fontsize=10, fontweight='bold')

    # Format axes
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index, fontsize=11)
    ax.set_xlabel('Absolute Correlation with Pressure', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Features by Correlation with QB Pressure Generation',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, max(top_features.values) * 1.15)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.8, label='Strong (|r| > 0.5)'),
        Patch(facecolor='#3498db', alpha=0.8, label='Moderate (0.3 < |r| ≤ 0.5)'),
        Patch(facecolor='#95a5a6', alpha=0.8, label='Weak (|r| ≤ 0.3)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # Add note about excluded features
    note_text = "Note: Target components (pff_hit, pff_hurry, pff_sack) excluded to prevent data leakage"
    ax.text(0.5, -0.08, note_text, transform=ax.transAxes,
            ha='center', fontsize=9, style='italic', color='red')

    plt.tight_layout()

    # Save
    output_file = 'feature_correlations.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Saved: {output_file}")
    print(f"   Resolution: 300 DPI")
    print(f"   Format: PNG")

    plt.close()

    # Verify no leakage
    print("\n🔍 Data Leakage Check:")
    leakage_cols = ['pff_hit', 'pff_hurry', 'pff_sack']
    for col in leakage_cols:
        if col in top_features.index:
            print(f"   ❌ ERROR: {col} found in top features (DATA LEAKAGE!)")
        else:
            print(f"   ✅ {col} correctly excluded")

    print("\n✅ Feature correlation visualization regenerated successfully!")
    print(f"✅ No data leakage detected")

if __name__ == "__main__":
    main()
