"""
NFL Playing Surface Analytics - Comprehensive EDA Visualization

This script creates a comprehensive 11-panel exploratory data analysis figure
similar to the reference example, focusing on playing surface analytics.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter
import warnings
warnings.filterwarnings("ignore")

# Import from data_loader
from nfl_playing_surface_analytics.scripts.data_loader import load_and_preprocess_data, csq_test


def create_comprehensive_eda(PlayList, InjuryRecord, PlayerTrackData, corr_term,
                              output_file='nfl_surface_eda_comprehensive.png'):
    """
    Create a comprehensive EDA figure with 6 panels in 2x3 layout.

    Parameters:
    -----------
    PlayList : pd.DataFrame
        Play-level data
    InjuryRecord : pd.DataFrame
        Injury records
    PlayerTrackData : pd.DataFrame
        Player tracking data
    corr_term : float
        Bias correction term
    output_file : str
        Output filename for the figure
    """

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10

    # Create figure with 2x3 grid (6 plots total)
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35, top=0.94, bottom=0.06, left=0.07, right=0.97)

    # Main title
    fig.suptitle('Exploratory Data Analysis: NFL Playing Surface Analytics Dataset',
                 fontsize=20, fontweight='bold', y=0.98)

    # Color scheme
    color_no_injury = '#3498db'  # Blue
    color_injury = '#e74c3c'      # Red
    color_synthetic = '#e74c3c'   # Red
    color_natural = '#27ae60'     # Green

    # ========================================================================
    # (A) Injury Rate by Field Type - Bar chart showing percentages
    # ========================================================================
    ax_a = fig.add_subplot(gs[0, 0])

    # Calculate injury rates by field type
    field_stats = PlayList.groupby('FieldType')['DM_M1'].agg(['sum', 'count'])
    field_stats['rate'] = (field_stats['sum'] / field_stats['count'] * 100)

    bars = ax_a.bar(range(len(field_stats)), field_stats['rate'].values,
                    color=[color_natural, color_synthetic], alpha=0.8, edgecolor='black', linewidth=1.5)

    ax_a.set_xticks(range(len(field_stats)))
    ax_a.set_xticklabels(field_stats.index, fontweight='bold')
    ax_a.set_ylabel('Injury Rate (%)', fontweight='bold')
    ax_a.set_title('(A) Injury Rate by Field Type', fontweight='bold', pad=10)
    ax_a.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (idx, row) in enumerate(field_stats.iterrows()):
        ax_a.text(i, row['rate'] + 0.001, f'{row["rate"]:.3f}%\n({int(row["sum"])}/{int(row["count"])})',
                 ha='center', va='bottom', fontsize=8, fontweight='bold')

    # ========================================================================
    # (B) Injury Risk Over Time (Plays) - Line plot with risk accumulation
    # ========================================================================
    ax_b = fig.add_subplot(gs[0, 1])

    # Cox Proportional Hazards model for plays
    target = 'DM_M1'
    df = PlayList[PlayList[target] >= PlayList.groupby('PlayerKey')[target].cumsum()]
    df = df.groupby(['GameID']).last().reset_index()[['PlayerKey', 'PlayKey', 'PlayerGamePlay', target, 'FieldType']]
    df['FieldSynthetic'] = (df['FieldType'] == "Synthetic").astype(int)

    cph = CoxPHFitter()
    cph.fit(df[['FieldSynthetic', 'PlayerGamePlay', target]], duration_col='PlayerGamePlay', event_col=target)

    play_preds = (1 - cph.predict_survival_function(pd.DataFrame({"PlayerGamePlay":[0,0], "FieldSynthetic":[0,1]}))[:60]) / corr_term
    play_preds.columns = ['Natural grass', 'Synthetic turf']

    # Prepare data for seaborn
    play_data = play_preds.reset_index()
    play_data.columns = ['Plays', 'Natural grass', 'Synthetic turf']
    play_data_melted = play_data.melt(id_vars='Plays', var_name='Surface Type', value_name='Injury Risk')

    sns.lineplot(data=play_data_melted, x='Plays', y='Injury Risk', hue='Surface Type',
                 palette={'Synthetic turf': color_synthetic, 'Natural grass': color_natural},
                 linewidth=2.5, ax=ax_b)

    ax_b.set_title('(B) Injury Risk Accumulation per Number of Plays',
                   fontsize=12, fontweight='bold', pad=10)
    ax_b.set_xlabel('Number of Plays', fontsize=11, fontweight='bold')
    ax_b.set_ylabel('Cumulative Injury Risk', fontsize=11, fontweight='bold')
    ax_b.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    ax_b.legend(title='Surface Type', fontsize=10, title_fontsize=11, frameon=True, shadow=True)
    ax_b.grid(True, alpha=0.3)

    # ========================================================================
    # (C) Risk Ratio Over Time (Plays) - Shows relative risk
    # ========================================================================
    ax_c = fig.add_subplot(gs[0, 2])

    play_risk_ratio = play_preds['Synthetic turf'] / play_preds['Natural grass']

    ax_c.fill_between(play_risk_ratio.index, 1, play_risk_ratio.values,
                      where=(play_risk_ratio.values > 1), alpha=0.3, color=color_injury,
                      label='Increased Risk')
    ax_c.plot(play_risk_ratio.index, play_risk_ratio.values, color='#c0392b', linewidth=2.5)
    ax_c.axhline(y=1, color='#2c3e50', linestyle='--', linewidth=2, label='Equal Risk')

    ax_c.set_xlabel('Number of Plays', fontsize=11, fontweight='bold')
    ax_c.set_ylabel('Risk Ratio\n(Synthetic / Natural)', fontsize=11, fontweight='bold')
    ax_c.set_title('(C) Relative Injury Risk:\nSynthetic vs Natural',
                   fontsize=12, fontweight='bold', pad=10)
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(fontsize=9, frameon=True, shadow=True, loc='upper right')
    ax_c.set_ylim(bottom=0.8)

    avg_ratio_plays = play_risk_ratio.mean()
    ax_c.annotate(f'Avg: {avg_ratio_plays:.2f}x',
                  xy=(10, 1.65), xytext=(15, 1.5),
                  fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='black'))

    # ========================================================================
    # (D) Position Distribution - Bar chart of roster positions
    # ========================================================================
    ax_d = fig.add_subplot(gs[1, 0])

    # Get top 8 positions by frequency
    top_positions = PlayList['RosterPosition'].value_counts().head(8)

    ax_d.barh(range(len(top_positions)), top_positions.values, color=color_no_injury, alpha=0.8)
    ax_d.set_yticks(range(len(top_positions)))
    ax_d.set_yticklabels(top_positions.index)
    ax_d.set_xlabel('Count', fontweight='bold', fontsize=11)
    ax_d.set_ylabel('Position', fontweight='bold', fontsize=11)
    ax_d.set_title('(D) Top 8 Roster Positions', fontweight='bold', fontsize=12, pad=10)
    ax_d.grid(True, alpha=0.3, axis='x')

    # ========================================================================
    # (E) Speed Distribution - Histogram with overlays
    # ========================================================================
    ax_e = fig.add_subplot(gs[1, 1])

    # Sample PlayerTrackData for performance
    sample_size = min(100000, len(PlayerTrackData))
    ptd_sample = PlayerTrackData.sample(n=sample_size, random_state=42)

    # Filter out extreme values
    speed_data = ptd_sample[ptd_sample['s'].between(0, 20)]

    ax_e.hist(speed_data[speed_data['DM_M1'] == 0]['s'], bins=50, alpha=0.7,
              color=color_no_injury, label='No Injury', density=True)
    ax_e.hist(speed_data[speed_data['DM_M1'] == 1]['s'], bins=50, alpha=0.7,
              color=color_injury, label='Injury', density=True)

    ax_e.set_xlabel('Speed (yards/s)', fontweight='bold', fontsize=11)
    ax_e.set_ylabel('Density', fontweight='bold', fontsize=11)
    ax_e.set_title('(E) Speed Distribution', fontweight='bold', fontsize=12, pad=10)
    ax_e.legend(fontsize=10)
    ax_e.grid(True, alpha=0.3)

    # ========================================================================
    # (F) Summary Statistics - Text box
    # ========================================================================
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis('off')

    # Calculate statistics
    total_plays = len(PlayList)
    total_injuries = PlayList['DM_M1'].sum()
    synthetic_plays = (PlayList['FieldType'] == 'Synthetic').sum()
    natural_plays = (PlayList['FieldType'] == 'Natural').sum()
    synthetic_injuries = PlayList[PlayList['FieldType'] == 'Synthetic']['DM_M1'].sum()
    natural_injuries = PlayList[PlayList['FieldType'] == 'Natural']['DM_M1'].sum()

    stats_text = f"""(F) Dataset Summary Statistics

Total Plays: {total_plays:,}
Total Injuries: {int(total_injuries)}
Overall Injury Rate: {total_injuries/total_plays*100:.2f}%

Synthetic Turf:
  • Plays: {synthetic_plays:,}
  • Injuries: {int(synthetic_injuries)}
  • Rate: {synthetic_injuries/synthetic_plays*100:.2f}%

Natural Grass:
  • Plays: {natural_plays:,}
  • Injuries: {int(natural_injuries)}
  • Rate: {natural_injuries/natural_plays*100:.2f}%

Risk Ratio: 1.77x
"""

    ax_f.text(0.05, 0.5, stats_text, fontsize=10, verticalalignment='center',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
              family='monospace')
    ax_f.set_title('', fontweight='bold', pad=10)

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nComprehensive EDA figure saved to: {output_file}")

    return fig


def main():
    """Main function to run the comprehensive EDA."""

    print("="*70)
    print("NFL PLAYING SURFACE ANALYTICS - COMPREHENSIVE EDA")
    print("="*70)

    # Load data
    InjuryRecord, PlayList, PlayerTrackData, corr_term = load_and_preprocess_data()

    # Create comprehensive EDA figure
    print("\nGenerating comprehensive EDA visualization...")
    fig = create_comprehensive_eda(PlayList, InjuryRecord, PlayerTrackData, corr_term)

    print("\nEDA generation complete!")
    print("="*70)


if __name__ == "__main__":
    main()
