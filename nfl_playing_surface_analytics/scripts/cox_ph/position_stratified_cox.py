"""
Position-Stratified Cox Proportional Hazards Analysis
Examines whether field type effect varies by player position
"""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import seaborn as sns
from nfl_playing_surface_analytics.scripts.data_loader import load_and_preprocess_data

# Load data
print("Loading data...")
InjuryRecord, PlayList, PlayerTrackData, corr_term = load_and_preprocess_data()

# Merge with PlayList to get FieldType and PlayerGamePlay
print("Merging datasets...")
selected_data = PlayerTrackData.merge(
    PlayList[['PlayKey', 'PlayerKey', 'PlayerGamePlay', 'FieldType', 'RosterPosition']],
    on=['PlayKey', 'PlayerKey'],
    how='left'
)

# Add injury indicator
injury_dict = InjuryRecord.set_index(['PlayKey', 'PlayerKey'])['DM_M1'].to_dict()
selected_data['DM_M1'] = selected_data.set_index(['PlayKey', 'PlayerKey']).index.map(
    lambda x: injury_dict.get(x, 0)
)

# Engineer features at player-play level
print("Engineering features...")
# Handle column name collision from merge
if 'RosterPosition_x' in selected_data.columns:
    selected_data = selected_data.rename(columns={'RosterPosition_x': 'RosterPosition'})
    selected_data = selected_data.drop(columns=['RosterPosition_y'], errors='ignore')
if 'FieldType_y' in selected_data.columns:
    # Use FieldType from PlayList (FieldType_y), drop FieldType from PlayerTrackData (FieldType_x)
    selected_data = selected_data.rename(columns={'FieldType_y': 'FieldType'})
    selected_data = selected_data.drop(columns=['FieldType_x'], errors='ignore')

agg_dict = {
    's': ['mean', 'std', 'max', 'min'],
    'a': ['mean', 'std', 'max'],
    'sx': ['max', 'min'],
    'sy': ['max', 'min'],
    'a_fwd': ['mean', 'max'],
    'a_sid': ['mean', 'std'],
    'DM_M1': 'first',
    'PlayerGamePlay': 'first',
    'FieldType': 'first',
    'RosterPosition': 'first'
}
features = selected_data.groupby(['PlayKey', 'PlayerKey']).agg(agg_dict).reset_index()

# Flatten column names
features.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in features.columns]

# Add percentile features separately
grouped = selected_data.groupby(['PlayKey', 'PlayerKey'])
features['s_percentile_75'] = grouped['s'].apply(lambda x: np.percentile(x, 75) if len(x) > 0 else 0).values
features['a_percentile_90'] = grouped['a'].apply(lambda x: np.percentile(x, 90) if len(x) > 0 else 0).values

# Calculate derived features
features['speed_range'] = features['s_max'] - features['s_min']
features['speed_cv'] = features['s_std'] / (features['s_mean'] + 1e-7)
features['lateral_dominance'] = np.abs(features['a_sid_mean']) / (np.abs(features['a_fwd_mean']) + 1e-7)

# Create binary field type
features['FieldSynthetic'] = (features['FieldType_first'] == 'Synthetic').astype(int)

# Get top positions by sample size
position_counts = features['RosterPosition_first'].value_counts()
top_positions = position_counts[position_counts >= 100].index.tolist()[:7]

print(f"\nAnalyzing {len(top_positions)} position groups with sufficient sample size:")
for pos in top_positions:
    n_samples = position_counts[pos]
    n_injuries = features[features['RosterPosition_first'] == pos]['DM_M1_first'].sum()
    print(f"  {pos}: {n_samples} samples, {n_injuries} injuries")

# Run Cox PH for each position
results = []

for position in top_positions:
    print(f"\n{'='*60}")
    print(f"Position: {position}")
    print('='*60)

    pos_data = features[features['RosterPosition_first'] == position].copy()
    n_injuries = pos_data['DM_M1_first'].sum()

    if n_injuries < 3:
        print(f"  Skipping - only {n_injuries} injuries")
        continue

    # Simple Cox PH (field type only)
    simple_data = pd.DataFrame({
        'duration': pos_data['PlayerGamePlay_first'],
        'event': pos_data['DM_M1_first'],
        'FieldSynthetic': pos_data['FieldSynthetic']
    })

    cph_simple = CoxPHFitter(penalizer=0.01)
    cph_simple.fit(simple_data, duration_col='duration', event_col='event')

    hr_simple = np.exp(cph_simple.params_['FieldSynthetic'])
    p_simple = cph_simple.summary.loc['FieldSynthetic', 'p']
    ci_lower_simple = np.exp(cph_simple.confidence_intervals_.loc['FieldSynthetic', '95% lower-bound'])
    ci_upper_simple = np.exp(cph_simple.confidence_intervals_.loc['FieldSynthetic', '95% upper-bound'])

    print(f"\nSimple Model (Field Type Only):")
    print(f"  HR = {hr_simple:.3f} (95% CI: {ci_lower_simple:.3f} - {ci_upper_simple:.3f})")
    print(f"  p-value = {p_simple:.4f}")

    # Adjusted Cox PH (field type + movement features)
    adjusted_data = pd.DataFrame({
        'duration': pos_data['PlayerGamePlay_first'],
        'event': pos_data['DM_M1_first'],
        'FieldSynthetic': pos_data['FieldSynthetic'],
        's_std': pos_data['s_std'],
        'a_max': pos_data['a_max'],
        'speed_range': pos_data['speed_range'],
        'a_90th': pos_data['a_percentile_90'],
        'lateral_dominance': pos_data['lateral_dominance']
    })

    cph_adjusted = CoxPHFitter(penalizer=0.01)
    cph_adjusted.fit(adjusted_data, duration_col='duration', event_col='event')

    hr_adjusted = np.exp(cph_adjusted.params_['FieldSynthetic'])
    p_adjusted = cph_adjusted.summary.loc['FieldSynthetic', 'p']
    ci_lower_adjusted = np.exp(cph_adjusted.confidence_intervals_.loc['FieldSynthetic', '95% lower-bound'])
    ci_upper_adjusted = np.exp(cph_adjusted.confidence_intervals_.loc['FieldSynthetic', '95% upper-bound'])

    print(f"\nAdjusted Model (Field Type + Movement):")
    print(f"  HR = {hr_adjusted:.3f} (95% CI: {ci_lower_adjusted:.3f} - {ci_upper_adjusted:.3f})")
    print(f"  p-value = {p_adjusted:.4f}")

    results.append({
        'Position': position,
        'N_Samples': len(pos_data),
        'N_Injuries': int(n_injuries),
        'HR_Simple': hr_simple,
        'HR_Simple_Lower': ci_lower_simple,
        'HR_Simple_Upper': ci_upper_simple,
        'P_Simple': p_simple,
        'HR_Adjusted': hr_adjusted,
        'HR_Adjusted_Lower': ci_lower_adjusted,
        'HR_Adjusted_Upper': ci_upper_adjusted,
        'P_Adjusted': p_adjusted
    })

# Create results dataframe
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('HR_Simple', ascending=False)

print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('/home/pshmo/the_zoo_v1/position_cox_results.csv', index=False)
print("\nResults saved to: position_cox_results.csv")

# Create forest plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

# Plot 1: Simple Model (Field Type Only)
y_pos = np.arange(len(results_df))
colors_simple = ['red' if p < 0.05 else 'gray' for p in results_df['P_Simple']]

ax1.scatter(results_df['HR_Simple'], y_pos, s=100, c=colors_simple, alpha=0.7, zorder=3)
for i, row in results_df.iterrows():
    idx = list(results_df.index).index(i)
    ax1.plot([row['HR_Simple_Lower'], row['HR_Simple_Upper']], [idx, idx],
             c=colors_simple[idx], linewidth=2, alpha=0.7, zorder=2)

ax1.axvline(x=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(results_df['Position'])
ax1.set_xlabel('Hazard Ratio (95% CI)', fontsize=11)
ax1.set_title('Simple Model: Field Type Only\n(Red = p<0.05, Gray = non-significant)', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
ax1.set_xlim(0, max(results_df['HR_Simple_Upper'].max() * 1.1, 2.5))

# Plot 2: Adjusted Model (Field Type + Movement)
colors_adjusted = ['red' if p < 0.05 else 'gray' for p in results_df['P_Adjusted']]

ax2.scatter(results_df['HR_Adjusted'], y_pos, s=100, c=colors_adjusted, alpha=0.7, zorder=3)
for i, row in results_df.iterrows():
    idx = list(results_df.index).index(i)
    ax2.plot([row['HR_Adjusted_Lower'], row['HR_Adjusted_Upper']], [idx, idx],
             c=colors_adjusted[idx], linewidth=2, alpha=0.7, zorder=2)

ax2.axvline(x=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(results_df['Position'])
ax2.set_xlabel('Hazard Ratio (95% CI)', fontsize=11)
ax2.set_title('Adjusted Model: Field Type + Movement\n(Red = p<0.05, Gray = non-significant)', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.set_xlim(0, max(results_df['HR_Adjusted_Upper'].max() * 1.1, 2.5))

plt.tight_layout()
plt.savefig('/home/pshmo/the_zoo_v1/position_cox_forest_plot.png', dpi=300, bbox_inches='tight')
print("Forest plot saved to: position_cox_forest_plot.png")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print("\nKey Findings:")
print("1. Simple Model shows which positions have higher injury risk on synthetic turf")
print("2. Adjusted Model shows if this is due to field type or movement patterns")
print("3. Large difference between Simple and Adjusted = confounding by movement")
print("4. HR > 1.0 = Increased risk on synthetic, HR < 1.0 = Decreased risk")
print("5. Red points = statistically significant (p < 0.05)")
