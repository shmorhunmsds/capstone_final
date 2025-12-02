#!/usr/bin/env python3
"""
Build Full Pass Rush Collision Dataset
Process all 8 weeks of Big Data Bowl 2023 tracking data and engineer collision features.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# FEATURE CALCULATION FUNCTIONS
# ==============================================================================

def calculate_rusher_qb_features(rusher_tracking, qb_tracking):
    """
    Calculate collision features between a rusher and QB.
    Returns dict of features or None if insufficient data.
    """

    # Need at least 3 frames for both players
    if len(rusher_tracking) < 3 or len(qb_tracking) < 3:
        return None

    # Find common frames
    rusher_tracking = rusher_tracking.sort_values('frameId').copy()
    qb_tracking = qb_tracking.sort_values('frameId').copy()

    common_frames = set(rusher_tracking['frameId']).intersection(set(qb_tracking['frameId']))

    if len(common_frames) < 3:
        return None

    # Filter to common frames
    rusher_common = rusher_tracking[rusher_tracking['frameId'].isin(common_frames)].sort_values('frameId')
    qb_common = qb_tracking[qb_tracking['frameId'].isin(common_frames)].sort_values('frameId')

    # Merge on frameId
    merged = rusher_common.merge(qb_common, on='frameId', suffixes=('_rusher', '_qb'))

    if len(merged) < 3:
        return None

    # Calculate distances
    merged['distance'] = np.sqrt(
        (merged['x_rusher'] - merged['x_qb'])**2 +
        (merged['y_rusher'] - merged['y_qb'])**2
    )

    features = {}

    # ==================== DISTANCE FEATURES ====================
    features['min_distance'] = merged['distance'].min()
    features['avg_distance'] = merged['distance'].mean()
    features['distance_at_start'] = merged['distance'].iloc[0]
    features['distance_at_end'] = merged['distance'].iloc[-1]

    # Find closest approach
    min_dist_idx = merged['distance'].idxmin()
    closest_frame = merged.loc[min_dist_idx]
    features['frame_at_closest'] = closest_frame['frameId']

    # ==================== SPEED FEATURES ====================
    features['rusher_max_speed'] = merged['s_rusher'].max()
    features['rusher_avg_speed'] = merged['s_rusher'].mean()
    features['rusher_speed_at_closest'] = closest_frame['s_rusher']

    features['qb_max_speed'] = merged['s_qb'].max()
    features['qb_avg_speed'] = merged['s_qb'].mean()
    features['qb_speed_at_closest'] = closest_frame['s_qb']

    # ==================== ACCELERATION FEATURES ====================
    features['rusher_max_accel'] = merged['a_rusher'].max()
    features['rusher_avg_accel'] = merged['a_rusher'].mean()
    features['rusher_accel_at_closest'] = closest_frame['a_rusher']

    features['qb_max_accel'] = merged['a_qb'].max()
    features['qb_avg_accel'] = merged['a_qb'].mean()
    features['qb_accel_at_closest'] = closest_frame['a_qb']

    # ==================== RELATIVE MOTION ====================
    features['combined_speed_at_closest'] = (
        closest_frame['s_rusher'] + closest_frame['s_qb']
    )

    # Closing speed (rate of distance decrease)
    distance_change = np.gradient(merged['distance'])
    features['max_closing_speed'] = -np.min(distance_change)
    features['avg_closing_speed'] = -np.mean(distance_change[distance_change < 0]) if any(distance_change < 0) else 0

    # ==================== ORIENTATION/DIRECTION ====================
    dx = closest_frame['x_qb'] - closest_frame['x_rusher']
    dy = closest_frame['y_qb'] - closest_frame['y_rusher']
    approach_angle = np.degrees(np.arctan2(dy, dx))

    features['approach_angle'] = approach_angle
    features['rusher_orientation_at_closest'] = closest_frame['o_rusher']
    features['qb_orientation_at_closest'] = closest_frame['o_qb']

    # Angle alignment
    rusher_angle_diff = abs(closest_frame['o_rusher'] - approach_angle)
    features['rusher_angle_alignment'] = min(rusher_angle_diff, 360 - rusher_angle_diff)

    # ==================== TEMPORAL FEATURES ====================
    features['time_to_closest_approach'] = closest_frame['frameId'] / 10.0
    features['total_frames'] = len(merged)
    features['play_duration'] = features['total_frames'] / 10.0

    # ==================== COLLISION INTENSITY ====================
    min_dist_norm = 1 / (features['min_distance'] + 0.1)
    speed_norm = features['combined_speed_at_closest']

    features['collision_intensity_raw'] = min_dist_norm * speed_norm
    features['weighted_closing_speed'] = features['max_closing_speed'] / (features['min_distance'] + 1.0)

    return features


def process_week_data(week_num, tracking_df, rushers_df, pff_df, plays_df):
    """
    Process all rush attempts for a single week.
    """

    print(f"\n{'='*70}")
    print(f"Processing Week {week_num}")
    print(f"{'='*70}")

    all_features = []
    successful = 0
    failed = 0

    print(f"Total rushes in week {week_num}: {len(rushers_df):,}")

    for idx, rusher_play in tqdm(rushers_df.iterrows(), total=len(rushers_df), desc=f"Week {week_num}"):
        game_id = rusher_play['gameId']
        play_id = rusher_play['playId']
        rusher_id = rusher_play['nflId']

        # Get QB for this play
        qb_play = pff_df[(pff_df['gameId'] == game_id) &
                         (pff_df['playId'] == play_id) &
                         (pff_df['pff_role'] == 'Pass')]

        if len(qb_play) == 0:
            failed += 1
            continue

        qb_id = qb_play.iloc[0]['nflId']

        # Get tracking data
        rusher_tracking = tracking_df[(tracking_df['gameId'] == game_id) &
                                       (tracking_df['playId'] == play_id) &
                                       (tracking_df['nflId'] == rusher_id)]

        qb_tracking = tracking_df[(tracking_df['gameId'] == game_id) &
                                   (tracking_df['playId'] == play_id) &
                                   (tracking_df['nflId'] == qb_id)]

        # Calculate features
        features = calculate_rusher_qb_features(rusher_tracking, qb_tracking)

        if features:
            # Add metadata
            features['week'] = week_num
            features['gameId'] = game_id
            features['playId'] = play_id
            features['rusher_nflId'] = rusher_id
            features['qb_nflId'] = qb_id
            features['rusher_position'] = rusher_play['pff_positionLinedUp']
            features['generated_pressure'] = rusher_play['generated_pressure']
            features['pff_hit'] = rusher_play['pff_hit']
            features['pff_hurry'] = rusher_play['pff_hurry']
            features['pff_sack'] = rusher_play['pff_sack']

            # Add play context
            play_info = plays_df[(plays_df['gameId'] == game_id) &
                                 (plays_df['playId'] == play_id)]

            if len(play_info) > 0:
                play_info = play_info.iloc[0]
                features['down'] = play_info.get('down', np.nan)
                features['yardsToGo'] = play_info.get('yardsToGo', np.nan)
                features['defendersInBox'] = play_info.get('defendersInBox', np.nan)
                features['offenseFormation'] = play_info.get('offenseFormation', 'Unknown')
                features['pff_passCoverageType'] = play_info.get('pff_passCoverageType', 'Unknown')
                features['passResult'] = play_info.get('passResult', 'Unknown')

            all_features.append(features)
            successful += 1
        else:
            failed += 1

    features_df = pd.DataFrame(all_features)

    print(f"\n✅ Week {week_num} complete:")
    print(f"   Successful: {successful:,} ({successful/len(rushers_df)*100:.1f}%)")
    print(f"   Failed: {failed:,}")
    print(f"   Pressure events: {features_df['generated_pressure'].sum():,} ({features_df['generated_pressure'].mean()*100:.2f}%)")

    return features_df


# ==============================================================================
# MAIN PROCESSING PIPELINE
# ==============================================================================

def main():
    """
    Main processing pipeline - load all weeks and build complete dataset.
    """

    start_time = datetime.now()

    print("="*70)
    print("PASS RUSH COLLISION DATASET BUILDER")
    print("Big Data Bowl 2023 - Full 8-Week Processing")
    print("="*70)
    print(f"Started: {start_time}\n")

    # Create output directory
    output_dir = 'pass_rush_collision_data'
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Output directory: {output_dir}/\n")

    # Load metadata (only once)
    print("Loading metadata...")
    plays = pd.read_csv('nfl-big-data-bowl-2023/plays.csv')
    pff = pd.read_csv('nfl-big-data-bowl-2023/pffScoutingData.csv')
    players = pd.read_csv('nfl-big-data-bowl-2023/players.csv')

    print(f"✅ Loaded {len(plays):,} plays")
    print(f"✅ Loaded {len(pff):,} PFF records")
    print(f"✅ Loaded {len(players):,} players")

    # Create target variable for all rushers
    rushers = pff[pff['pff_role'] == 'Pass Rush'].copy()
    rushers['generated_pressure'] = (
        (rushers['pff_hit'] == 1) |
        (rushers['pff_hurry'] == 1) |
        (rushers['pff_sack'] == 1)
    ).astype(int)

    print(f"\n✅ Total rush attempts: {len(rushers):,}")
    print(f"✅ Pressure events: {rushers['generated_pressure'].sum():,} ({rushers['generated_pressure'].mean()*100:.2f}%)")

    # Process each week
    all_weeks_data = []

    for week_num in range(1, 9):  # Weeks 1-8
        week_file = f'nfl-big-data-bowl-2023/week{week_num}.csv'

        if not os.path.exists(week_file):
            print(f"\n⚠️  Warning: {week_file} not found, skipping...")
            continue

        # Load tracking data for this week
        print(f"\nLoading tracking data for week {week_num}...")
        tracking = pd.read_csv(week_file)
        print(f"✅ Loaded {len(tracking):,} tracking records")

        # Get rushers for this week
        week_games = tracking['gameId'].unique()
        week_rushers = rushers[rushers['gameId'].isin(week_games)]

        # Process week
        week_features = process_week_data(
            week_num=week_num,
            tracking_df=tracking,
            rushers_df=week_rushers,
            pff_df=pff,
            plays_df=plays
        )

        if len(week_features) > 0:
            # Save individual week file
            week_output = f'{output_dir}/week{week_num}_features.csv'
            week_features.to_csv(week_output, index=False)
            print(f"✅ Saved: {week_output}")

            all_weeks_data.append(week_features)

        # Clear memory
        del tracking

    # Combine all weeks
    print("\n" + "="*70)
    print("COMBINING ALL WEEKS")
    print("="*70)

    full_dataset = pd.concat(all_weeks_data, ignore_index=True)

    print(f"Total samples: {len(full_dataset):,}")
    print(f"Total features: {len(full_dataset.columns)}")
    print(f"\nClass distribution:")
    print(f"  Pressure: {full_dataset['generated_pressure'].sum():,} ({full_dataset['generated_pressure'].mean()*100:.2f}%)")
    print(f"  No Pressure: {(full_dataset['generated_pressure']==0).sum():,} ({(full_dataset['generated_pressure']==0).mean()*100:.2f}%)")

    # Normalize collision intensity across all weeks
    print("\nNormalizing collision intensity...")
    max_combined_speed = full_dataset['combined_speed_at_closest'].max()

    if max_combined_speed > 0:
        min_dist_norm = 1 / (full_dataset['min_distance'] + 0.1)
        speed_norm = full_dataset['combined_speed_at_closest'] / max_combined_speed
        full_dataset['collision_intensity'] = min_dist_norm * speed_norm

        print(f"✅ Collision intensity range: [{full_dataset['collision_intensity'].min():.3f}, {full_dataset['collision_intensity'].max():.3f}]")

    # Save complete dataset
    full_output = f'{output_dir}/pass_rush_collision_features_full.csv'
    full_dataset.to_csv(full_output, index=False)
    print(f"\n✅ Saved complete dataset: {full_output}")

    # Save summary statistics
    summary_output = f'{output_dir}/dataset_summary.txt'
    with open(summary_output, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PASS RUSH COLLISION DATASET SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Processing time: {datetime.now() - start_time}\n\n")

        f.write(f"Total samples: {len(full_dataset):,}\n")
        f.write(f"Total features: {len(full_dataset.columns)}\n")
        f.write(f"Weeks processed: {full_dataset['week'].nunique()}\n\n")

        f.write("Target Variable Distribution:\n")
        f.write(f"  Pressure events: {full_dataset['generated_pressure'].sum():,} ({full_dataset['generated_pressure'].mean()*100:.2f}%)\n")
        f.write(f"  No pressure: {(full_dataset['generated_pressure']==0).sum():,} ({(full_dataset['generated_pressure']==0).mean()*100:.2f}%)\n\n")

        f.write("Pressure Type Breakdown:\n")
        f.write(f"  Hits: {(full_dataset['pff_hit']==1).sum():,}\n")
        f.write(f"  Hurries: {(full_dataset['pff_hurry']==1).sum():,}\n")
        f.write(f"  Sacks: {(full_dataset['pff_sack']==1).sum():,}\n\n")

        f.write("Feature Categories:\n")
        f.write("  - Distance features: 4\n")
        f.write("  - Speed features: 6\n")
        f.write("  - Acceleration features: 6\n")
        f.write("  - Relative motion: 3\n")
        f.write("  - Orientation: 4\n")
        f.write("  - Temporal: 3\n")
        f.write("  - Collision intensity: 3\n")
        f.write("  - Play context: 6\n")
        f.write("  - Metadata: 10\n\n")

        f.write("Top Features by Correlation with Pressure:\n")
        numeric_features = full_dataset.select_dtypes(include=[np.number]).columns
        correlations = full_dataset[numeric_features].corr()['generated_pressure'].abs().sort_values(ascending=False)
        corr_items = list(correlations.head(11).items())[1:]  # Skip target itself
        for i, (feature, corr) in enumerate(corr_items, 1):
            f.write(f"  {i:2d}. {feature}: {corr:.4f}\n")

    print(f"✅ Saved summary: {summary_output}")

    # Create feature list
    feature_list_output = f'{output_dir}/feature_list.txt'
    with open(feature_list_output, 'w') as f:
        f.write("Complete Feature List\n")
        f.write("="*70 + "\n\n")
        for i, col in enumerate(full_dataset.columns, 1):
            dtype = full_dataset[col].dtype
            f.write(f"{i:3d}. {col:40s} ({dtype})\n")

    print(f"✅ Saved feature list: {feature_list_output}")

    # Final summary
    print("\n" + "="*70)
    print("✅ DATASET BUILD COMPLETE!")
    print("="*70)
    print(f"Total processing time: {datetime.now() - start_time}")
    print(f"\nOutput files in {output_dir}/:")
    print(f"  - pass_rush_collision_features_full.csv (main dataset)")
    print(f"  - week1_features.csv ... week8_features.csv (individual weeks)")
    print(f"  - dataset_summary.txt")
    print(f"  - feature_list.txt")
    print("\n✅ Ready for modeling!")


if __name__ == "__main__":
    main()
