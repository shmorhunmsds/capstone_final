#!/usr/bin/env python3
"""
Collision Explorer - Interactive Tool
======================================

Explore and visualize collision data interactively.
View collision statistics and create custom animations.

Author: Peter Shmorhun
Date: 2025-10-20
"""

import pandas as pd
import numpy as np
from collision_animation import CollisionAnimator
import os


class CollisionExplorer:
    """Interactive exploration of collision data."""

    def __init__(self):
        """Initialize the collision explorer."""
        self.injury_df = pd.read_csv('/Users/petershmorhun/Desktop/punt_analytics_final/punt_collision_results/injury_collisions.csv')
        self.animator = CollisionAnimator()

        print("="*70)
        print("NFL PUNT ANALYTICS - COLLISION EXPLORER")
        print("="*70)
        print(f"Loaded {len(self.injury_df)} injury collisions")

    def show_statistics(self):
        """Display collision statistics."""
        print("\n" + "="*70)
        print("COLLISION STATISTICS")
        print("="*70)

        df = self.injury_df

        print(f"\nTotal collisions: {len(df)}")
        print(f"\nImpact Types:")
        print(df['impact_type'].value_counts())

        print(f"\nPlayer Activities:")
        print(df['player_activity'].value_counts())

        print(f"\nCollision Intensity:")
        print(f"  Mean: {df['collision_intensity'].mean():.2f}")
        print(f"  Std:  {df['collision_intensity'].std():.2f}")
        print(f"  Min:  {df['collision_intensity'].min():.2f}")
        print(f"  Max:  {df['collision_intensity'].max():.2f}")

        print(f"\nMinimum Distance:")
        print(f"  Mean: {df['min_distance'].mean():.2f} yards")
        print(f"  Std:  {df['min_distance'].std():.2f} yards")
        print(f"  Min:  {df['min_distance'].min():.2f} yards")
        print(f"  Max:  {df['min_distance'].max():.2f} yards")

        print(f"\nRelative Speed:")
        print(f"  Mean: {df['max_relative_speed'].mean():.2f} yds/s")
        print(f"  Std:  {df['max_relative_speed'].std():.2f} yds/s")
        print(f"  Min:  {df['max_relative_speed'].min():.2f} yds/s")
        print(f"  Max:  {df['max_relative_speed'].max():.2f} yds/s")

    def list_top_collisions(self, n=10, sort_by='collision_intensity'):
        """
        List top collisions by specified metric.

        Parameters
        ----------
        n : int
            Number of collisions to show
        sort_by : str
            Metric to sort by
        """
        print(f"\n" + "="*70)
        print(f"TOP {n} COLLISIONS (by {sort_by})")
        print("="*70)

        top = self.injury_df.nlargest(n, sort_by)

        for idx, (i, row) in enumerate(top.iterrows(), 1):
            print(f"\n{idx}. Collision #{i}")
            print(f"   Season: {row['seasonyear']}, Game: {row['gamekey']}, Play: {row['playid']}")
            print(f"   Players: {int(row['injured_player'])} vs {int(row['partner_player'])}")
            print(f"   Impact: {row['impact_type']}")
            print(f"   Activity: {row['player_activity']} vs {row['partner_activity']}")
            print(f"   Intensity: {row['collision_intensity']:.2f}")
            print(f"   Min Distance: {row['min_distance']:.2f} yards")
            print(f"   Max Relative Speed: {row['max_relative_speed']:.2f} yds/s")
            print(f"   Closing Speed: {row['max_closing_speed']:.2f} yds/s")

    def compare_collisions(self, indices):
        """
        Compare multiple collisions side-by-side.

        Parameters
        ----------
        indices : list
            List of collision indices to compare
        """
        print(f"\n" + "="*70)
        print(f"COLLISION COMPARISON")
        print("="*70)

        metrics = [
            'collision_intensity',
            'min_distance',
            'max_relative_speed',
            'max_closing_speed',
            'combined_speed',
            'p1_speed_at_collision',
            'p2_speed_at_collision'
        ]

        comparison = self.injury_df.iloc[indices][metrics]

        print("\nMetric Comparison:")
        print(comparison.T.to_string())

        print("\nMetadata:")
        for idx in indices:
            row = self.injury_df.iloc[idx]
            print(f"\nCollision #{idx}:")
            print(f"  Impact: {row['impact_type']}")
            print(f"  Activity: {row['player_activity']} vs {row['partner_activity']}")

    def create_custom_animation(self, collision_index, output_file=None,
                               show_all_players=True, fps=10, trail_length=15):
        """
        Create animation for a specific collision.

        Parameters
        ----------
        collision_index : int
            Index of collision in the dataframe
        output_file : str, optional
            Output filename (default: auto-generated)
        show_all_players : bool
            Show all players on field
        fps : int
            Frames per second
        trail_length : int
            Trail length in frames
        """
        row = self.injury_df.iloc[collision_index]

        if output_file is None:
            output_file = f'visualizations/collision_{collision_index:03d}.gif'

        print(f"\n" + "="*70)
        print(f"CREATING ANIMATION FOR COLLISION #{collision_index}")
        print("="*70)
        print(f"Impact: {row['impact_type']}")
        print(f"Intensity: {row['collision_intensity']:.2f}")
        print(f"Min Distance: {row['min_distance']:.2f} yards")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        try:
            self.animator.create_animation(
                season=int(row['seasonyear']),
                gamekey=int(row['gamekey']),
                playid=int(row['playid']),
                player1_id=int(row['injured_player']),
                player2_id=int(row['partner_player']),
                output_file=output_file,
                fps=fps,
                show_all_players=show_all_players,
                trail_length=trail_length
            )
        except Exception as e:
            print(f"❌ Error creating animation: {e}")

    def find_collisions_by_criteria(self, **criteria):
        """
        Find collisions matching specific criteria.

        Parameters
        ----------
        **criteria : keyword arguments
            Criteria to filter by (e.g., impact_type='Helmet-to-helmet')

        Returns
        -------
        DataFrame
            Matching collisions
        """
        df = self.injury_df.copy()

        for key, value in criteria.items():
            if key in df.columns:
                if isinstance(value, str):
                    df = df[df[key] == value]
                elif isinstance(value, tuple):  # Range query
                    df = df[(df[key] >= value[0]) & (df[key] <= value[1])]

        print(f"\n" + "="*70)
        print(f"SEARCH RESULTS: {len(df)} collisions found")
        print("="*70)

        if len(df) > 0:
            print("\nMatching collisions:")
            for idx, (i, row) in enumerate(df.iterrows(), 1):
                print(f"{idx}. Index #{i}: {row['impact_type']} | "
                      f"Intensity: {row['collision_intensity']:.2f} | "
                      f"Distance: {row['min_distance']:.2f} yds")

        return df


def interactive_menu():
    """Run interactive menu."""
    explorer = CollisionExplorer()

    while True:
        print("\n" + "="*70)
        print("MENU")
        print("="*70)
        print("1. Show collision statistics")
        print("2. List top collisions by intensity")
        print("3. List top collisions by speed")
        print("4. List top collisions by distance (closest)")
        print("5. Create animation for specific collision")
        print("6. Find collisions by impact type")
        print("7. Exit")
        print("="*70)

        choice = input("\nEnter your choice (1-7): ").strip()

        if choice == '1':
            explorer.show_statistics()

        elif choice == '2':
            explorer.list_top_collisions(n=10, sort_by='collision_intensity')

        elif choice == '3':
            explorer.list_top_collisions(n=10, sort_by='max_relative_speed')

        elif choice == '4':
            # List closest collisions (smallest min_distance)
            print(f"\n" + "="*70)
            print(f"CLOSEST COLLISIONS (by min_distance)")
            print("="*70)
            closest = explorer.injury_df.nsmallest(10, 'min_distance')
            for idx, (i, row) in enumerate(closest.iterrows(), 1):
                print(f"{idx}. Index #{i}: Distance: {row['min_distance']:.3f} yds | "
                      f"{row['impact_type']} | Intensity: {row['collision_intensity']:.2f}")

        elif choice == '5':
            try:
                idx = int(input("Enter collision index: "))
                if 0 <= idx < len(explorer.injury_df):
                    show_all = input("Show all players? (y/n): ").lower() == 'y'
                    explorer.create_custom_animation(idx, show_all_players=show_all)
                else:
                    print("Invalid index!")
            except ValueError:
                print("Invalid input!")

        elif choice == '6':
            print("\nAvailable impact types:")
            print(explorer.injury_df['impact_type'].unique())
            impact = input("Enter impact type: ").strip()
            results = explorer.find_collisions_by_criteria(impact_type=impact)

        elif choice == '7':
            print("\nGoodbye!")
            break

        else:
            print("Invalid choice!")


if __name__ == "__main__":
    # Quick examples
    explorer = CollisionExplorer()

    # Show statistics
    explorer.show_statistics()

    # List top 5 collisions
    explorer.list_top_collisions(n=5, sort_by='collision_intensity')

    # Uncomment to run interactive menu
    interactive_menu()
