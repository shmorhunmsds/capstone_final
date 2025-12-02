#!/usr/bin/env python3
"""
Create Stakeholder-Friendly Distribution Charts
================================================
Extracts and adapts panels B (Minimum Distance) and C (Collision Intensity)
from the 6-panel EDA figure for stakeholder report.

Adjustments for non-ML audiences:
- Clearer axis labels and titles
- Plain English descriptions
- Removed technical jargon
- Added interpretation annotations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Configure plotting
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11

print("Loading collision data...")

# Load datasets
injury_df = pd.read_csv('punt_collision_results/injury_collisions.csv')
normal_df = pd.read_csv('punt_collision_results/normal_collisions_full.csv')

print(f"  Injury collisions: {len(injury_df)}")
print(f"  Normal collisions: {len(normal_df):,}")

# Create 1 row x 2 column figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# =============================================================================
# Panel A: Minimum Distance (was Panel B)
# =============================================================================
ax1 = axes[0]

# Plot histograms
ax1.hist(normal_df['min_distance'], bins=20, alpha=0.6, label='No Injury',
         color='#3498db', density=True, edgecolor='black', linewidth=0.5)
ax1.hist(injury_df['min_distance'], bins=15, alpha=0.7, label='Injury',
         color='#e74c3c', density=True, edgecolor='black', linewidth=0.5)

# Add mean lines with clearer labels
normal_mean = normal_df['min_distance'].mean()
injury_mean = injury_df['min_distance'].mean()

ax1.axvline(normal_mean, color='#2874a6', linestyle='--', linewidth=2.5,
            label=f'No Injury Avg: {normal_mean:.2f} yds')
ax1.axvline(injury_mean, color='#c0392b', linestyle='--', linewidth=2.5,
            label=f'Injury Avg: {injury_mean:.2f} yds')

# Stakeholder-friendly labels
ax1.set_xlabel('Closest Distance Between Players (yards)', fontweight='bold')
ax1.set_ylabel('Proportion of Collisions', fontweight='bold')
ax1.set_title('A. How Close Were the Players?', fontweight='bold', fontsize=15, loc='left')
ax1.legend(loc='upper right', framealpha=0.95)
ax1.grid(alpha=0.3)


# =============================================================================
# Panel B: Collision Intensity (was Panel C)
# =============================================================================
ax2 = axes[1]

# Plot histograms
ax2.hist(normal_df['collision_intensity'], bins=20, alpha=0.6, label='No Injury',
         color='#3498db', density=True, edgecolor='black', linewidth=0.5)
ax2.hist(injury_df['collision_intensity'], bins=15, alpha=0.7, label='Injury',
         color='#e74c3c', density=True, edgecolor='black', linewidth=0.5)

# Add mean lines with clearer labels
normal_mean_intensity = normal_df['collision_intensity'].mean()
injury_mean_intensity = injury_df['collision_intensity'].mean()

ax2.axvline(normal_mean_intensity, color='#2874a6', linestyle='--', linewidth=2.5,
            label=f'No Injury Avg: {normal_mean_intensity:.1f}')
ax2.axvline(injury_mean_intensity, color='#c0392b', linestyle='--', linewidth=2.5,
            label=f'Injury Avg: {injury_mean_intensity:.1f}')

# Stakeholder-friendly labels
ax2.set_xlabel('Collision Intensity Score (speed / distance)', fontweight='bold')
ax2.set_ylabel('Proportion of Collisions', fontweight='bold')
ax2.set_title('B. How Intense Was the Collision?', fontweight='bold', fontsize=15, loc='left')
ax2.legend(loc='upper right', framealpha=0.95)
ax2.grid(alpha=0.3)

# Limit x-axis to remove blank space (focus on where the data is)
ax2.set_xlim(0, 60)

# Adjust layout and save
plt.tight_layout()

output_path = 'stakeholder_collision_distributions.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved: {output_path}")

plt.close()
