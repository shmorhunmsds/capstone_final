#!/usr/bin/env python3
"""
NFL Punt Analytics - Collision Feature Extraction
==================================================

This script extracts collision features from NFL punt plays with improved
collision detection criteria and proper feature engineering.

Key improvements from original:
1. Tighter collision threshold (2.5 yards vs 5 yards)
2. Collision quality filtering (speed, closing rate)
3. No global normalization of collision_intensity
4. Stratified sampling options for class balance

Author: Patrick Shmorhun
Date: 2025-10-12
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class CollisionFeatureExtractor:
    """
    Extract collision features from player tracking data with quality controls.
    """

    def __init__(self, collision_threshold='moderate'):
        """
        Initialize the collision feature extractor.

        Parameters
        ----------
        collision_threshold : str, default='moderate'
            Collision detection threshold:
            - 'strict': Real physical contact (< 1.5 yards, high speed)
            - 'moderate': High probability contact (< 2.5 yards, moderate speed)
            - 'loose': Original behavior (< 5.0 yards)
        """
        self.collision_threshold = collision_threshold
        self.motion_data = None
        self.video_review = None

        # Threshold parameters
        self.thresholds = {
            'strict': {
                'min_distance': 1.5,
                'max_relative_speed': 8.0,
                'max_closing_speed': 5.0,
                'min_collision_intensity': 5.0
            },
            'moderate': {
                'min_distance': 2.5,
                'max_relative_speed': 6.0,
                'max_closing_speed': 3.0,
                'min_collision_intensity': 2.0
            },
            'loose': {
                'min_distance': 5.0,
                'max_relative_speed': 3.0,
                'max_closing_speed': 1.0,
                'min_collision_intensity': 0.5
            }
        }

        print(f"Initialized CollisionFeatureExtractor with '{collision_threshold}' threshold")
        self._print_threshold_info()

    def _print_threshold_info(self):
        """Print current threshold settings"""
        params = self.thresholds[self.collision_threshold]
        print(f"  Max distance: {params['min_distance']} yards")
        print(f"  Min relative speed: {params['max_relative_speed']} yards/sec")
        print(f"  Min closing speed: {params['max_closing_speed']} yards/sec")
        print(f"  Min collision intensity: {params['min_collision_intensity']}")

    def load_data(self, data_dir='NFL-Punt-Analytics-Competition'):
        """
        Load all necessary datasets.

        Parameters
        ----------
        data_dir : str
            Directory containing NFL punt analytics data
        """
        print(f"\n{'='*60}")
        print("LOADING DATA")
        print(f"{'='*60}")

        # Load video review (collision details and injury records)
        video_path = os.path.join(data_dir, 'video_review.csv')
        self.video_review = pd.read_csv(video_path)
        self.video_review.columns = self.video_review.columns.str.lower().str.replace('_', '')

        # Convert primarypartnergsisid to numeric
        self.video_review['primarypartnergsisid'] = pd.to_numeric(
            self.video_review['primarypartnergsisid'], errors='coerce'
        )

        print(f"✅ Video review records: {len(self.video_review)}")
        print(f"   Records with collision partners: {self.video_review['primarypartnergsisid'].notna().sum()}")

        # Load NGS player tracking data
        print("\n📍 Loading NGS player tracking data...")
        ngs_files = [
            'NGS-2016-pre.csv', 'NGS-2016-post.csv',
            'NGS-2016-reg-wk1-6.csv', 'NGS-2016-reg-wk7-12.csv', 'NGS-2016-reg-wk13-17.csv',
            'NGS-2017-pre.csv', 'NGS-2017-post.csv',
            'NGS-2017-reg-wk1-6.csv', 'NGS-2017-reg-wk7-12.csv', 'NGS-2017-reg-wk13-17.csv'
        ]

        ngs_dfs = []
        for filename in ngs_files:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df.columns = df.columns.str.lower().str.replace('_', '')
                ngs_dfs.append(df)
                print(f"   ✓ {filename}")
            else:
                print(f"   ✗ {filename} not found")

        self.motion_data = pd.concat(ngs_dfs, ignore_index=True)
        print(f"\n✅ Loaded {len(self.motion_data):,} motion records")
        print(f"   Unique players: {self.motion_data['gsisid'].nunique():,}")
        print(f"   Unique plays: {self.motion_data.groupby(['seasonyear', 'gamekey', 'playid']).ngroups:,}")

    def calculate_collision_features(self, player1_motion, player2_motion) -> Optional[Dict]:
        """
        Calculate collision features from two players' movement data.

        Parameters
        ----------
        player1_motion : DataFrame
            Tracking data for player 1
        player2_motion : DataFrame
            Tracking data for player 2

        Returns
        -------
        dict or None
            Dictionary of collision features, or None if calculation fails
        """
        # Copy to avoid modifying original data
        p1_motion = player1_motion.copy()
        p2_motion = player2_motion.copy()

        # Convert time to datetime
        p1_motion['time'] = pd.to_datetime(p1_motion['time'])
        p2_motion['time'] = pd.to_datetime(p2_motion['time'])

        # Sort by time
        p1_motion = p1_motion.sort_values('time')
        p2_motion = p2_motion.sort_values('time')

        # Calculate seconds from start
        p1_motion['seconds'] = (p1_motion['time'] - p1_motion['time'].min()).dt.total_seconds()
        p2_motion['seconds'] = (p2_motion['time'] - p2_motion['time'].min()).dt.total_seconds()

        # Find overlapping time period
        max_start = max(p1_motion['seconds'].min(), p2_motion['seconds'].min())
        min_end = min(p1_motion['seconds'].max(), p2_motion['seconds'].max())

        if max_start >= min_end:
            return None

        # Create common time points (10Hz sampling)
        common_times = np.arange(max_start, min_end, 0.1)

        if len(common_times) < 3:
            return None

        # Interpolate player data to common times
        def interpolate_player_data(motion_data, times):
            interp_data = pd.DataFrame({'time': times})
            for col in ['x', 'y', 'dis', 'o', 'dir']:
                if col in motion_data.columns:
                    interp_data[col] = np.interp(times, motion_data['seconds'], motion_data[col])
            return interp_data

        p1_interp = interpolate_player_data(p1_motion, common_times)
        p2_interp = interpolate_player_data(p2_motion, common_times)

        # ==================================================================
        # CALCULATE COLLISION FEATURES
        # ==================================================================

        features = {}

        # 1. DISTANCE METRICS
        distances = np.sqrt((p1_interp['x'] - p2_interp['x'])**2 +
                           (p1_interp['y'] - p2_interp['y'])**2)

        features['min_distance'] = distances.min()
        features['distance_at_start'] = distances.iloc[0]
        features['distance_at_end'] = distances.iloc[-1]
        features['avg_distance'] = distances.mean()
        features['distance_std'] = distances.std()

        # Find closest approach
        min_dist_idx = distances.idxmin()
        features['time_to_closest_approach'] = common_times[min_dist_idx]

        # 2. VELOCITY CALCULATIONS
        p1_vx = np.gradient(p1_interp['x']) / 0.1
        p1_vy = np.gradient(p1_interp['y']) / 0.1
        p2_vx = np.gradient(p2_interp['x']) / 0.1
        p2_vy = np.gradient(p2_interp['y']) / 0.1

        # Relative velocity (how fast they're approaching)
        rel_vx = p1_vx - p2_vx
        rel_vy = p1_vy - p2_vy
        relative_speed = np.sqrt(rel_vx**2 + rel_vy**2)

        features['max_relative_speed'] = np.nanmax(relative_speed)
        features['avg_relative_speed'] = np.nanmean(relative_speed)
        features['relative_speed_at_closest'] = relative_speed[min_dist_idx]

        # 3. APPROACH ANGLES
        if not pd.isna(min_dist_idx):
            # Vector from player 2 to player 1 at closest approach
            dx = p1_interp['x'].iloc[min_dist_idx] - p2_interp['x'].iloc[min_dist_idx]
            dy = p1_interp['y'].iloc[min_dist_idx] - p2_interp['y'].iloc[min_dist_idx]
            collision_angle = np.degrees(np.arctan2(dy, dx))

            # Player orientations at collision
            p1_orientation = p1_interp['o'].iloc[min_dist_idx]
            p2_orientation = p2_interp['o'].iloc[min_dist_idx]

            features['collision_angle'] = collision_angle
            features['p1_orientation_at_collision'] = p1_orientation
            features['p2_orientation_at_collision'] = p2_orientation

            # Angle differences (head-on vs side collision)
            features['p1_angle_diff'] = abs(p1_orientation - collision_angle)
            features['p2_angle_diff'] = abs(p2_orientation - collision_angle)

            # Speed at collision
            features['p1_speed_at_collision'] = p1_interp['dis'].iloc[min_dist_idx]
            features['p2_speed_at_collision'] = p2_interp['dis'].iloc[min_dist_idx]
        else:
            features['p1_speed_at_collision'] = np.nan
            features['p2_speed_at_collision'] = np.nan

        # 4. SPEED CHARACTERISTICS
        features['p1_max_speed'] = p1_interp['dis'].max()
        features['p2_max_speed'] = p2_interp['dis'].max()
        features['p1_avg_speed'] = p1_interp['dis'].mean()
        features['p2_avg_speed'] = p2_interp['dis'].mean()
        features['p1_speed_std'] = p1_interp['dis'].std()
        features['p2_speed_std'] = p2_interp['dis'].std()

        # 5. ACCELERATION (change in speed)
        p1_acc = np.gradient(p1_interp['dis']) / 0.1
        p2_acc = np.gradient(p2_interp['dis']) / 0.1
        features['p1_max_acc'] = np.nanmax(np.abs(p1_acc))
        features['p2_max_acc'] = np.nanmax(np.abs(p2_acc))
        features['p1_avg_acc'] = np.nanmean(p1_acc)
        features['p2_avg_acc'] = np.nanmean(p2_acc)

        # 6. CLOSING SPEED (rate of distance change)
        distance_gradient = np.gradient(distances) / 0.1
        features['max_closing_speed'] = -np.nanmin(distance_gradient)  # Negative = closing
        features['avg_closing_speed'] = -np.nanmean(distance_gradient[distance_gradient < 0]) if any(distance_gradient < 0) else 0

        # 7. TIMING FEATURES
        play_duration = common_times[-1] - common_times[0]
        features['play_duration'] = play_duration
        features['collision_timing'] = features['time_to_closest_approach'] / play_duration if play_duration > 0 else 0

        # ==================================================================
        # ENGINEERED FEATURES (NO GLOBAL NORMALIZATION!)
        # ==================================================================

        # 8. Speed ratios and differences
        features['speed_ratio'] = features['p1_max_speed'] / (features['p2_max_speed'] + 1e-6)
        features['speed_difference'] = abs(features['p1_max_speed'] - features['p2_max_speed'])
        features['combined_speed'] = features['p1_max_speed'] + features['p2_max_speed']

        # 9. Speed retention at collision (indicates impact)
        if not pd.isna(features.get('p1_speed_at_collision')) and features['p1_max_speed'] > 0:
            features['p1_speed_retention'] = features['p1_speed_at_collision'] / (features['p1_max_speed'] + 1e-6)
        else:
            features['p1_speed_retention'] = np.nan

        if not pd.isna(features.get('p2_speed_at_collision')) and features['p2_max_speed'] > 0:
            features['p2_speed_retention'] = features['p2_speed_at_collision'] / (features['p2_max_speed'] + 1e-6)
        else:
            features['p2_speed_retention'] = np.nan

        # 10. COLLISION INTENSITY (RAW - NO NORMALIZATION!)
        # Physics-based metric: inverse distance * relative speed
        # Higher values = more intense collision
        # DO NOT normalize globally - let the model scaler handle this!
        min_dist_component = 1.0 / (features['min_distance'] + 0.1)
        speed_component = features['max_relative_speed']
        features['collision_intensity'] = min_dist_component * speed_component

        # 11. Collision quality score (0-1 scale, for filtering)
        dist_score = 1.0 / (1.0 + features['min_distance'])
        speed_score = min(features['max_relative_speed'] / 15.0, 1.0)
        closing_score = min(features['max_closing_speed'] / 10.0, 1.0)
        features['collision_quality'] = dist_score * 0.5 + speed_score * 0.3 + closing_score * 0.2

        # 12. Approach efficiency (how directly they approached)
        if features['distance_at_start'] > 0:
            features['approach_efficiency'] = 1.0 - (features['min_distance'] / features['distance_at_start'])
        else:
            features['approach_efficiency'] = 0.0

        return features

    def is_valid_collision(self, features: Dict) -> bool:
        """
        Determine if features represent a valid collision based on threshold.

        Parameters
        ----------
        features : dict
            Collision features dictionary

        Returns
        -------
        bool
            True if collision meets quality criteria
        """
        params = self.thresholds[self.collision_threshold]

        # Check all criteria
        criteria = [
            features['min_distance'] < params['min_distance'],
            features['max_relative_speed'] > params['max_relative_speed'],
            features['max_closing_speed'] > params['max_closing_speed'],
            features['collision_intensity'] > params['min_collision_intensity']
        ]

        return all(criteria)

    def extract_injury_collisions(self) -> pd.DataFrame:
        """
        Extract collision features for all documented injury cases.

        Returns
        -------
        DataFrame
            Injury collision features with metadata
        """
        print(f"\n{'='*60}")
        print("EXTRACTING INJURY COLLISION FEATURES")
        print(f"{'='*60}")

        injury_features = []
        skipped = 0

        for idx, injury in self.video_review.iterrows():
            season = injury['seasonyear']
            gamekey = injury['gamekey']
            playid = injury['playid']
            injured_player = injury['gsisid']

            # Skip if no collision partner identified
            if pd.isna(injury.get('primarypartnergsisid')):
                skipped += 1
                continue

            partner_player = injury['primarypartnergsisid']

            # Get motion data for both players in this play
            play_motion = self.motion_data[
                (self.motion_data['seasonyear'] == season) &
                (self.motion_data['gamekey'] == gamekey) &
                (self.motion_data['playid'] == playid)
            ]

            injured_motion = play_motion[play_motion['gsisid'] == injured_player]
            partner_motion = play_motion[play_motion['gsisid'] == partner_player]

            # Need sufficient tracking data
            if len(injured_motion) < 3 or len(partner_motion) < 3:
                skipped += 1
                continue

            # Calculate collision features
            collision_features = self.calculate_collision_features(injured_motion, partner_motion)

            if collision_features is not None:
                # Add metadata
                collision_features['seasonyear'] = season
                collision_features['gamekey'] = gamekey
                collision_features['playid'] = playid
                collision_features['injured_player'] = injured_player
                collision_features['partner_player'] = partner_player
                collision_features['impact_type'] = injury.get('primaryimpacttype', 'Unknown')
                collision_features['player_activity'] = injury.get('playeractivityderived', 'Unknown')
                collision_features['partner_activity'] = injury.get('primarypartneractivityderived', 'Unknown')
                collision_features['friendly_fire'] = injury.get('friendlyfire', 'Unknown')
                collision_features['is_injury'] = 1

                injury_features.append(collision_features)
            else:
                skipped += 1

        injury_df = pd.DataFrame(injury_features)

        print(f"✅ Successfully extracted {len(injury_df)} injury collisions")
        print(f"⚠️  Skipped {skipped} cases (no partner or insufficient data)")

        if len(injury_df) > 0:
            print(f"\n📊 Injury Collision Statistics:")
            print(f"   Min distance: {injury_df['min_distance'].mean():.2f} ± {injury_df['min_distance'].std():.2f} yards")
            print(f"   Max relative speed: {injury_df['max_relative_speed'].mean():.2f} ± {injury_df['max_relative_speed'].std():.2f} yds/sec")
            print(f"   Collision intensity: {injury_df['collision_intensity'].mean():.2f} ± {injury_df['collision_intensity'].std():.2f}")

        return injury_df

    def extract_normal_collisions(self, max_collisions: int = 10000,
                                  batch_size: int = 1000) -> pd.DataFrame:
        """
        Extract collision features from non-injury plays.

        Parameters
        ----------
        max_collisions : int
            Maximum number of normal collisions to extract
        batch_size : int
            Save to disk every N collisions (memory management)

        Returns
        -------
        DataFrame
            Normal collision features
        """
        print(f"\n{'='*60}")
        print(f"EXTRACTING NORMAL COLLISION FEATURES (max={max_collisions:,})")
        print(f"{'='*60}")

        # Get injury plays to exclude
        injury_plays = set()
        for _, injury in self.video_review.iterrows():
            injury_plays.add((injury['seasonyear'], injury['gamekey'], injury['playid']))

        # Get all non-injury plays
        all_plays = self.motion_data.groupby(['seasonyear', 'gamekey', 'playid']).size().reset_index(name='count')
        non_injury_plays = all_plays[~all_plays.apply(
            lambda x: (x['seasonyear'], x['gamekey'], x['playid']) in injury_plays, axis=1
        )]

        print(f"Total non-injury plays available: {len(non_injury_plays):,}")

        normal_features = []
        plays_processed = 0
        collisions_found = 0
        collisions_passed_filter = 0

        # Create output directory
        output_dir = 'punt_collision_results/batches'
        os.makedirs(output_dir, exist_ok=True)

        # Process plays
        for _, play in non_injury_plays.iterrows():
            if collisions_passed_filter >= max_collisions:
                print(f"\n✅ Reached target: {max_collisions:,} collisions")
                break

            plays_processed += 1

            if plays_processed % 100 == 0:
                print(f"  Processed {plays_processed:,} plays | "
                      f"Found {collisions_found:,} raw | "
                      f"Passed filter {collisions_passed_filter:,}")

            # Get all players in this play
            play_data = self.motion_data[
                (self.motion_data['seasonyear'] == play['seasonyear']) &
                (self.motion_data['gamekey'] == play['gamekey']) &
                (self.motion_data['playid'] == play['playid'])
            ]

            play_players = play_data['gsisid'].unique()

            if len(play_players) < 2:
                continue

            # Process all pairwise combinations
            for i in range(len(play_players)):
                for j in range(i+1, len(play_players)):
                    if collisions_passed_filter >= max_collisions:
                        break

                    player1 = play_players[i]
                    player2 = play_players[j]

                    player1_data = play_data[play_data['gsisid'] == player1]
                    player2_data = play_data[play_data['gsisid'] == player2]

                    if len(player1_data) < 3 or len(player2_data) < 3:
                        continue

                    # Calculate features
                    collision_features = self.calculate_collision_features(player1_data, player2_data)

                    if collision_features is not None:
                        collisions_found += 1

                        # Apply quality filter
                        if self.is_valid_collision(collision_features):
                            collisions_passed_filter += 1

                            # Add metadata
                            collision_features['seasonyear'] = play['seasonyear']
                            collision_features['gamekey'] = play['gamekey']
                            collision_features['playid'] = play['playid']
                            collision_features['injured_player'] = player1
                            collision_features['partner_player'] = player2
                            collision_features['impact_type'] = 'No injury'
                            collision_features['is_injury'] = 0

                            normal_features.append(collision_features)

                            # Save batch periodically
                            if len(normal_features) >= batch_size:
                                batch_df = pd.DataFrame(normal_features)
                                batch_num = collisions_passed_filter // batch_size
                                filename = f'{output_dir}/normal_collisions_batch_{batch_num:04d}.csv'
                                batch_df.to_csv(filename, index=False)
                                print(f"    💾 Saved batch {batch_num}: {filename}")
                                normal_features = []

        # Save remaining collisions
        if normal_features:
            batch_df = pd.DataFrame(normal_features)
            filename = f'{output_dir}/normal_collisions_batch_final.csv'
            batch_df.to_csv(filename, index=False)
            print(f"    💾 Saved final batch: {filename}")

        # Load all batches
        print(f"\n📦 Loading all batches...")
        batch_files = [f for f in os.listdir(output_dir) if f.startswith('normal_collisions_batch_')]
        batch_dfs = [pd.read_csv(f'{output_dir}/{f}') for f in sorted(batch_files)]
        normal_df = pd.concat(batch_dfs, ignore_index=True)

        print(f"\n✅ Extraction complete!")
        print(f"   Plays processed: {plays_processed:,}")
        print(f"   Raw collisions found: {collisions_found:,}")
        print(f"   Passed quality filter: {collisions_passed_filter:,}")
        print(f"   Filter pass rate: {collisions_passed_filter/collisions_found*100:.1f}%")

        if len(normal_df) > 0:
            print(f"\n📊 Normal Collision Statistics:")
            print(f"   Min distance: {normal_df['min_distance'].mean():.2f} ± {normal_df['min_distance'].std():.2f} yards")
            print(f"   Max relative speed: {normal_df['max_relative_speed'].mean():.2f} ± {normal_df['max_relative_speed'].std():.2f} yds/sec")
            print(f"   Collision intensity: {normal_df['collision_intensity'].mean():.2f} ± {normal_df['collision_intensity'].std():.2f}")

        return normal_df

    def create_balanced_dataset(self, injury_df: pd.DataFrame, normal_df: pd.DataFrame,
                               ratio: int = 10, stratified: bool = True) -> pd.DataFrame:
        """
        Create a balanced dataset with specified injury:normal ratio.

        Parameters
        ----------
        injury_df : DataFrame
            Injury collision features
        normal_df : DataFrame
            Normal collision features
        ratio : int, default=10
            Normal collisions per injury (e.g., 10 = 10:1 ratio)
        stratified : bool, default=True
            If True, sample stratified by min_distance bins

        Returns
        -------
        DataFrame
            Combined balanced dataset
        """
        print(f"\n{'='*60}")
        print(f"CREATING BALANCED DATASET (Ratio {ratio}:1)")
        print(f"{'='*60}")

        n_injury = len(injury_df)
        n_normal_target = n_injury * ratio

        if stratified and n_normal_target < len(normal_df):
            print(f"Using stratified sampling by min_distance...")

            # Define distance bins
            bins = [0, 1.0, 1.5, 2.0, 2.5]
            samples_per_bin = n_normal_target // len(bins)

            sampled_normals = []
            for i in range(len(bins)-1):
                bin_low = bins[i]
                bin_high = bins[i+1]

                bin_data = normal_df[
                    (normal_df['min_distance'] >= bin_low) &
                    (normal_df['min_distance'] < bin_high)
                ]

                n_sample = min(samples_per_bin, len(bin_data))
                if n_sample > 0:
                    sampled = bin_data.sample(n=n_sample, random_state=42)
                    sampled_normals.append(sampled)
                    print(f"   Bin [{bin_low:.1f}, {bin_high:.1f}): sampled {n_sample} / {len(bin_data)}")

            # Sample remainder from full distribution
            n_sampled = sum(len(df) for df in sampled_normals)
            if n_sampled < n_normal_target:
                remainder = n_normal_target - n_sampled
                remaining = normal_df.sample(n=remainder, random_state=42)
                sampled_normals.append(remaining)
                print(f"   Remainder: sampled {remainder}")

            normal_sample = pd.concat(sampled_normals, ignore_index=True)
        else:
            # Simple random sampling
            n_sample = min(n_normal_target, len(normal_df))
            normal_sample = normal_df.sample(n=n_sample, random_state=42)
            print(f"Using random sampling: {n_sample} normal collisions")

        # Combine datasets
        full_dataset = pd.concat([injury_df, normal_sample], ignore_index=True)
        full_dataset = full_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"\n✅ Dataset created:")
        print(f"   Injury collisions: {n_injury} ({n_injury/len(full_dataset)*100:.1f}%)")
        print(f"   Normal collisions: {len(normal_sample)} ({len(normal_sample)/len(full_dataset)*100:.1f}%)")
        print(f"   Total samples: {len(full_dataset)}")
        print(f"   Actual ratio: 1:{len(normal_sample)//n_injury}")

        return full_dataset

    def save_dataset(self, df: pd.DataFrame, filepath: str):
        """Save dataset to CSV"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"\n💾 Saved dataset: {filepath}")
        print(f"   Shape: {df.shape}")
        print(f"   Size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")


def main():
    """Main execution function"""

    print("="*60)
    print("NFL PUNT ANALYTICS - COLLISION FEATURE EXTRACTION")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize extractor with moderate threshold
    extractor = CollisionFeatureExtractor(collision_threshold='moderate')

    # Load data
    extractor.load_data()

    # Extract injury collisions
    injury_df = extractor.extract_injury_collisions()
    extractor.save_dataset(injury_df, 'punt_collision_results/injury_collisions.csv')

    # Extract normal collisions (up to 5000)
    normal_df = extractor.extract_normal_collisions(max_collisions=5000)
    extractor.save_dataset(normal_df, 'punt_collision_results/normal_collisions_full.csv')

    # Create balanced datasets at different ratios
    for ratio in [10, 25, 50]:
        balanced_df = extractor.create_balanced_dataset(
            injury_df, normal_df,
            ratio=ratio,
            stratified=True
        )
        extractor.save_dataset(
            balanced_df,
            f'punt_collision_results/balanced_dataset_ratio_{ratio}.csv'
        )

    print(f"\n{'='*60}")
    print("🎉 FEATURE EXTRACTION COMPLETE!")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    return injury_df, normal_df


if __name__ == "__main__":
    injury_df, normal_df = main()
