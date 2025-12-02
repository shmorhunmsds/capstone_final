#!/usr/bin/env python3
"""
Create Combined EDA Visualization Panel (2x3)
Combines best charts from individual analyses:
- Panel A: Pressure Event Distribution (from eda_collision_analysis.png)
- Panel B: Feature Correlations (from feature_correlations.png)
- Panel C: Min Distance Distribution
- Panel D: Closing Speed Distribution
- Panel E: Combined Speed Distribution
- Panel F: Collision Intensity Distribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

def load_data():
    """Load all necessary datasets"""
    print("Loading datasets...")

    # Load feature dataset
    features_df = pd.read_csv('pass_rush_collision_data/pass_rush_collision_features_full.csv')

    # Load PFF data for pressure statistics
    pff = pd.read_csv('nfl-big-data-bowl-2023/pffScoutingData.csv')

    print(f"✅ Loaded {len(features_df):,} feature samples")
    print(f"✅ Loaded {len(pff):,} PFF records")

    return features_df, pff


def create_combined_panel(features_df, pff):
    """Create 2x3 panel with best EDA visualizations"""

    print("\nCreating combined 2x3 panel (2 rows x 3 columns)...")

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # ==================== PANEL A: PRESSURE EVENT DISTRIBUTION ====================
    print("  Panel A: Pressure Event Distribution")
    ax_a = fig.add_subplot(gs[0, 0])

    # Get pressure counts from feature dataset (has actual pressure data)
    pressure_counts = pd.DataFrame({
        'Hits': [features_df['pff_hit'].sum()],
        'Hurries': [features_df['pff_hurry'].sum()],
        'Sacks': [features_df['pff_sack'].sum()]
    }).T

    bars = ax_a.bar(pressure_counts.index, pressure_counts[0],
                    color=['#d62728', '#ff7f0e', '#2ca02c'], alpha=0.8)
    ax_a.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax_a.set_title('(A) QB Pressure Events Distribution', fontsize=13, fontweight='bold', pad=10)
    ax_a.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax_a.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height):,}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # ==================== PANEL B: FEATURE CORRELATIONS ====================
    print("  Panel B: Feature Correlations")
    ax_b = fig.add_subplot(gs[0, 1])

    # Define features to EXCLUDE (target components and metadata)
    exclude_cols = [
        'week', 'gameId', 'playId', 'rusher_nflId', 'qb_nflId',
        'rusher_position', 'offenseFormation', 'pff_passCoverageType', 'passResult',
        'generated_pressure', 'pff_hit', 'pff_hurry', 'pff_sack', 'frame_at_closest'
    ]

    # Select only numeric features used in modeling
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    feature_cols = [col for col in feature_cols if features_df[col].dtype in ['int64', 'float64']]

    # Calculate correlations with target
    numeric_features = features_df[feature_cols + ['generated_pressure']].copy()
    correlations = numeric_features.corr()['generated_pressure'].abs().sort_values(ascending=False)
    correlations = correlations.drop('generated_pressure')

    # Get top 10 features
    top_features = correlations.head(10)

    # Create horizontal bar chart
    colors = ['#e74c3c' if corr > 0.5 else '#3498db' if corr > 0.3 else '#95a5a6'
              for corr in top_features.values]

    bars_b = ax_b.barh(range(len(top_features)), top_features.values, color=colors, alpha=0.8)

    # Add value labels
    for i, (feature, corr) in enumerate(top_features.items()):
        ax_b.text(corr + 0.01, i, f'{corr:.3f}',
                 va='center', fontsize=9, fontweight='bold')

    ax_b.set_yticks(range(len(top_features)))
    ax_b.set_yticklabels([f.replace('_', ' ').title()[:25] for f in top_features.index], fontsize=9)
    ax_b.set_xlabel('Absolute Correlation', fontsize=11, fontweight='bold')
    ax_b.set_title('(B) Top 10 Features by Correlation with Pressure',
                   fontsize=13, fontweight='bold', pad=10)
    ax_b.grid(axis='x', alpha=0.3, linestyle='--')
    ax_b.set_xlim(0, max(top_features.values) * 1.2)

    # ==================== PANEL C: MIN DISTANCE DISTRIBUTION ====================
    print("  Panel C: Min Distance Distribution")
    ax_c = fig.add_subplot(gs[0, 2])

    pressure_data = features_df[features_df['generated_pressure'] == 1]['min_distance'].dropna()
    no_pressure_data = features_df[features_df['generated_pressure'] == 0]['min_distance'].dropna()

    ax_c.hist(no_pressure_data, bins=40, alpha=0.6, label='No Pressure',
              color='#3498db', density=True, edgecolor='black', linewidth=0.5)
    ax_c.hist(pressure_data, bins=40, alpha=0.7, label='Pressure',
              color='#e74c3c', density=True, edgecolor='black', linewidth=0.5)

    ax_c.set_xlabel('Minimum Distance (yards)', fontsize=11, fontweight='bold')
    ax_c.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax_c.set_title('(C) Rusher-QB Minimum Distance Distribution',
                   fontsize=13, fontweight='bold', pad=10)
    ax_c.legend(fontsize=10, loc='upper right')
    ax_c.grid(alpha=0.3)

    # Add mean lines
    ax_c.axvline(pressure_data.mean(), color='#e74c3c', linestyle='--',
                linewidth=2, alpha=0.8, label=f'Pressure Mean: {pressure_data.mean():.2f}')
    ax_c.axvline(no_pressure_data.mean(), color='#3498db', linestyle='--',
                linewidth=2, alpha=0.8, label=f'No Pressure Mean: {no_pressure_data.mean():.2f}')

    # ==================== PANEL D: MAX CLOSING SPEED DISTRIBUTION ====================
    print("  Panel D: Max Closing Speed Distribution")
    ax_d = fig.add_subplot(gs[1, 0])

    pressure_data_cs = features_df[features_df['generated_pressure'] == 1]['max_closing_speed'].dropna()
    no_pressure_data_cs = features_df[features_df['generated_pressure'] == 0]['max_closing_speed'].dropna()

    ax_d.hist(no_pressure_data_cs, bins=40, alpha=0.6, label='No Pressure',
              color='#3498db', density=True, edgecolor='black', linewidth=0.5)
    ax_d.hist(pressure_data_cs, bins=40, alpha=0.7, label='Pressure',
              color='#e74c3c', density=True, edgecolor='black', linewidth=0.5)

    ax_d.set_xlabel('Max Closing Speed (yards/sec)', fontsize=11, fontweight='bold')
    ax_d.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax_d.set_title('(D) Rusher-QB Max Closing Speed Distribution',
                   fontsize=13, fontweight='bold', pad=10)
    ax_d.legend(fontsize=10, loc='upper right')
    ax_d.grid(alpha=0.3)

    # Add mean lines
    ax_d.axvline(pressure_data_cs.mean(), color='#e74c3c', linestyle='--',
                linewidth=2, alpha=0.8)
    ax_d.axvline(no_pressure_data_cs.mean(), color='#3498db', linestyle='--',
                linewidth=2, alpha=0.8)

    # ==================== PANEL E: COMBINED SPEED AT CLOSEST ====================
    print("  Panel E: Combined Speed at Closest")
    ax_e = fig.add_subplot(gs[1, 1])

    pressure_data_comb = features_df[features_df['generated_pressure'] == 1]['combined_speed_at_closest'].dropna()
    no_pressure_data_comb = features_df[features_df['generated_pressure'] == 0]['combined_speed_at_closest'].dropna()

    ax_e.hist(no_pressure_data_comb, bins=40, alpha=0.6, label='No Pressure',
              color='#3498db', density=True, edgecolor='black', linewidth=0.5)
    ax_e.hist(pressure_data_comb, bins=40, alpha=0.7, label='Pressure',
              color='#e74c3c', density=True, edgecolor='black', linewidth=0.5)

    ax_e.set_xlabel('Combined Speed (yards/sec)', fontsize=11, fontweight='bold')
    ax_e.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax_e.set_title('(E) Combined Speed at Closest Approach',
                   fontsize=13, fontweight='bold', pad=10)
    ax_e.legend(fontsize=10, loc='upper right')
    ax_e.grid(alpha=0.3)

    # Add mean lines
    ax_e.axvline(pressure_data_comb.mean(), color='#e74c3c', linestyle='--',
                linewidth=2, alpha=0.8)
    ax_e.axvline(no_pressure_data_comb.mean(), color='#3498db', linestyle='--',
                linewidth=2, alpha=0.8)

    # ==================== PANEL F: COLLISION INTENSITY DISTRIBUTION ====================
    print("  Panel F: Collision Intensity Distribution")
    ax_f = fig.add_subplot(gs[1, 2])

    pressure_data_ci = features_df[features_df['generated_pressure'] == 1]['collision_intensity'].dropna()
    no_pressure_data_ci = features_df[features_df['generated_pressure'] == 0]['collision_intensity'].dropna()

    ax_f.hist(no_pressure_data_ci, bins=40, alpha=0.6, label='No Pressure',
              color='#3498db', density=True, edgecolor='black', linewidth=0.5)
    ax_f.hist(pressure_data_ci, bins=40, alpha=0.7, label='Pressure',
              color='#e74c3c', density=True, edgecolor='black', linewidth=0.5)

    ax_f.set_xlabel('Collision Intensity', fontsize=11, fontweight='bold')
    ax_f.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax_f.set_title('(F) Collision Intensity Distribution',
                   fontsize=13, fontweight='bold', pad=10)
    ax_f.legend(fontsize=10, loc='upper right')
    ax_f.grid(alpha=0.3)

    # Add mean lines
    ax_f.axvline(pressure_data_ci.mean(), color='#e74c3c', linestyle='--',
                linewidth=2, alpha=0.8)
    ax_f.axvline(no_pressure_data_ci.mean(), color='#3498db', linestyle='--',
                linewidth=2, alpha=0.8)

    # ==================== OVERALL TITLE ====================
    fig.suptitle('Combined EDA Analysis: QB Pressure Prediction Features',
                 fontsize=16, fontweight='bold', y=0.995)

    return fig


def main():
    """Main workflow"""

    print("\n" + "="*70)
    print("CREATING COMBINED EDA VISUALIZATION PANEL (2 ROWS x 3 COLUMNS)")
    print("="*70)

    # Load data
    features_df, pff = load_data()

    # Create combined panel
    fig = create_combined_panel(features_df, pff)

    # Save
    output_file = 'combined_eda_panel.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')

    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE")
    print("="*70)
    print(f"✅ Saved: {output_file}")
    print(f"   Resolution: 300 DPI")
    print(f"   Dimensions: 18x12 inches (5400x3600 pixels)")
    print(f"   Layout: 2 rows x 3 columns")
    print("\nPanel Contents:")
    print("  (A) QB Pressure Events Distribution")
    print("  (B) Top 10 Features by Correlation with Pressure")
    print("  (C) Rusher-QB Minimum Distance Distribution")
    print("  (D) Rusher-QB Max Closing Speed Distribution")
    print("  (E) Combined Speed at Closest Approach")
    print("  (F) Collision Intensity Distribution")

    plt.close()

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
