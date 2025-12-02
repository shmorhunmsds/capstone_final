#!/usr/bin/env python3
"""
NFL Punt Analytics - Collision Animation Visualization
======================================================

Creates animated visualizations of concussion events on a football field
for video presentation and analysis.

Features:
- Football field rendering with proper dimensions (120 yards x 53.33 yards)
- Player tracking visualization with directional indicators
- Collision highlighting with intensity metrics
- Speed and distance overlays
- Export to MP4 video format

Author: Peter Shmorhun
Date: 2025-10-20
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Circle, Wedge
from matplotlib.collections import LineCollection
import os
from typing import Tuple, Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')


class FootballFieldVisualizer:
    """
    Create football field visualization with proper NFL dimensions.
    """

    def __init__(self, figsize=(16, 9)):
        """
        Initialize the football field visualizer.

        Parameters
        ----------
        figsize : tuple
            Figure size in inches (width, height)
        """
        self.figsize = figsize

        # NFL field dimensions (in yards)
        self.field_length = 120  # Including endzones
        self.field_width = 53.33  # 160 feet = 53.33 yards
        self.endzone_length = 10

        # Colors
        self.field_color = '#2d5c2e'  # Dark green
        self.line_color = 'white'
        self.endzone_color = '#1e3d1f'

    def create_field(self, ax):
        """
        Draw the football field on the given axis.

        Parameters
        ----------
        ax : matplotlib axis
            Axis to draw on
        """
        # Set field background
        ax.set_facecolor(self.field_color)

        # Field boundary
        ax.add_patch(patches.Rectangle(
            (0, 0), self.field_length, self.field_width,
            linewidth=2, edgecolor=self.line_color, facecolor=self.field_color
        ))

        # Endzones
        ax.add_patch(patches.Rectangle(
            (0, 0), self.endzone_length, self.field_width,
            linewidth=1, edgecolor=self.line_color, facecolor=self.endzone_color, alpha=0.3
        ))
        ax.add_patch(patches.Rectangle(
            (110, 0), self.endzone_length, self.field_width,
            linewidth=1, edgecolor=self.line_color, facecolor=self.endzone_color, alpha=0.3
        ))

        # Yard lines
        for yard in range(10, 111, 5):
            linewidth = 2 if yard % 10 == 0 else 1
            alpha = 1.0 if yard % 10 == 0 else 0.5
            ax.plot([yard, yard], [0, self.field_width],
                   color=self.line_color, linewidth=linewidth, alpha=alpha)

        # Hash marks (simplified)
        for yard in range(10, 110):
            ax.plot([yard, yard], [self.field_width/2 - 0.3, self.field_width/2 + 0.3],
                   color=self.line_color, linewidth=1, alpha=0.5)

        # Yard numbers (every 10 yards)
        for yard in range(20, 110, 10):
            if yard <= 50:
                number = yard - 10
            else:
                number = 110 - yard

            if number > 0:
                ax.text(yard, 5, str(number), color=self.line_color,
                       fontsize=16, fontweight='bold', ha='center', rotation=0)
                ax.text(yard, self.field_width - 5, str(number), color=self.line_color,
                       fontsize=16, fontweight='bold', ha='center', rotation=180)

        # Set axis properties
        ax.set_xlim(0, self.field_length)
        ax.set_ylim(0, self.field_width)
        ax.set_aspect('equal')
        ax.axis('off')


class CollisionAnimator:
    """
    Animate player collisions on a football field.
    """

    def __init__(self, data_dir='NFL-Punt-Analytics-Competition'):
        """
        Initialize the collision animator.

        Parameters
        ----------
        data_dir : str
            Directory containing NFL data
        """
        self.data_dir = data_dir
        self.field_viz = FootballFieldVisualizer(figsize=(20, 11))
        self.motion_data = None

    def load_tracking_data(self, season: int, gamekey: int, playid: int) -> pd.DataFrame:
        """
        Load tracking data for a specific play.

        Parameters
        ----------
        season : int
            Season year
        gamekey : int
            Game key
        playid : int
            Play ID

        Returns
        -------
        DataFrame
            Player tracking data for the play
        """
        # Determine which file to load based on season
        if season == 2016:
            files = [
                'NGS-2016-pre.csv', 'NGS-2016-post.csv',
                'NGS-2016-reg-wk1-6.csv', 'NGS-2016-reg-wk7-12.csv',
                'NGS-2016-reg-wk13-17.csv'
            ]
        else:  # 2017
            files = [
                'NGS-2017-pre.csv', 'NGS-2017-post.csv',
                'NGS-2017-reg-wk1-6.csv', 'NGS-2017-reg-wk7-12.csv',
                'NGS-2017-reg-wk13-17.csv'
            ]

        # Try each file until we find the play
        for filename in files:
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                continue

            try:
                df = pd.read_csv(filepath, low_memory=False)
                df.columns = df.columns.str.lower().str.replace('_', '')

                play_data = df[
                    (df['seasonyear'] == season) &
                    (df['gamekey'] == gamekey) &
                    (df['playid'] == playid)
                ]

                if len(play_data) > 0:
                    print(f"✅ Found play in {filename}")
                    print(f"   Total tracking points: {len(play_data):,}")
                    print(f"   Unique players: {play_data['gsisid'].nunique()}")
                    return play_data

            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
                continue

        raise ValueError(f"Play not found: Season {season}, Game {gamekey}, Play {playid}")

    def prepare_animation_data(self, play_data: pd.DataFrame,
                              player1_id: int, player2_id: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for animation.

        Parameters
        ----------
        play_data : DataFrame
            Full play tracking data
        player1_id : int
            Injured player ID
        player2_id : int
            Collision partner ID

        Returns
        -------
        tuple
            (player1_data, player2_data, other_players_data)
        """
        # Convert time to datetime and create frame indices
        play_data = play_data.copy()
        play_data['time'] = pd.to_datetime(play_data['time'])
        play_data = play_data.sort_values('time')

        # Create frame number based on time
        unique_times = sorted(play_data['time'].unique())
        time_to_frame = {t: i for i, t in enumerate(unique_times)}
        play_data['frame'] = play_data['time'].map(time_to_frame)

        # Separate player data
        p1_data = play_data[play_data['gsisid'] == player1_id].copy()
        p2_data = play_data[play_data['gsisid'] == player2_id].copy()
        other_data = play_data[~play_data['gsisid'].isin([player1_id, player2_id])].copy()

        print(f"\n📊 Animation Data:")
        print(f"   Player 1 frames: {len(p1_data)}")
        print(f"   Player 2 frames: {len(p2_data)}")
        print(f"   Other players: {other_data['gsisid'].nunique()}")
        print(f"   Total frames: {play_data['frame'].max() + 1}")

        return p1_data, p2_data, other_data

    def calculate_distance(self, p1_data: pd.DataFrame, p2_data: pd.DataFrame,
                          frame: int) -> Optional[float]:
        """Calculate distance between players at a given frame."""
        p1_frame = p1_data[p1_data['frame'] == frame]
        p2_frame = p2_data[p2_data['frame'] == frame]

        if len(p1_frame) == 0 or len(p2_frame) == 0:
            return None

        dx = p1_frame['x'].values[0] - p2_frame['x'].values[0]
        dy = p1_frame['y'].values[0] - p2_frame['y'].values[0]

        return np.sqrt(dx**2 + dy**2)

    def create_animation(self, season: int, gamekey: int, playid: int,
                        player1_id: int, player2_id: int,
                        output_file: str = 'collision_animation.gif',
                        fps: int = 10, show_all_players: bool = True,
                        trail_length: int = 10):
        """
        Create an animated visualization of the collision.

        Parameters
        ----------
        season : int
            Season year
        gamekey : int
            Game key
        playid : int
            Play ID
        player1_id : int
            Injured player ID
        player2_id : int
            Collision partner ID
        output_file : str
            Output filename for the animation
        fps : int
            Frames per second
        show_all_players : bool
            Whether to show all players or just collision pair
        trail_length : int
            Number of frames to show in player trails
        """
        print(f"\n{'='*70}")
        print(f"CREATING COLLISION ANIMATION")
        print(f"{'='*70}")
        print(f"Play: Season {season}, Game {gamekey}, Play {playid}")
        print(f"Players: {player1_id} vs {player2_id}")

        # Load data
        play_data = self.load_tracking_data(season, gamekey, playid)
        p1_data, p2_data, other_data = self.prepare_animation_data(
            play_data, player1_id, player2_id
        )

        # Find collision moment (minimum distance)
        max_frame = min(p1_data['frame'].max(), p2_data['frame'].max())
        distances = []
        for frame in range(max_frame + 1):
            dist = self.calculate_distance(p1_data, p2_data, frame)
            if dist is not None:
                distances.append((frame, dist))

        if not distances:
            raise ValueError("No overlapping frames found between players")

        collision_frame = min(distances, key=lambda x: x[1])[0]
        min_distance = min(distances, key=lambda x: x[1])[1]

        print(f"\n⚡ Collision detected at frame {collision_frame}")
        print(f"   Minimum distance: {min_distance:.2f} yards")

        # Create figure
        fig, ax = plt.subplots(figsize=self.field_viz.figsize)
        fig.patch.set_facecolor('#1a1a1a')

        # Draw field
        self.field_viz.create_field(ax)

        # Initialize plot elements
        p1_marker, = ax.plot([], [], 'o', color='#ff3333', markersize=16,
                            markeredgecolor='white', markeredgewidth=2,
                            label='Injured Player', zorder=5)
        p2_marker, = ax.plot([], [], 'o', color='#3366ff', markersize=16,
                            markeredgecolor='white', markeredgewidth=2,
                            label='Collision Partner', zorder=5)

        # Player trails
        p1_trail, = ax.plot([], [], '-', color='#ff3333', linewidth=3, alpha=0.5, zorder=3)
        p2_trail, = ax.plot([], [], '-', color='#3366ff', linewidth=3, alpha=0.5, zorder=3)

        # Direction arrows
        p1_arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle='->',
                                  mutation_scale=20, color='#ff3333', linewidth=2, zorder=4)
        p2_arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle='->',
                                  mutation_scale=20, color='#3366ff', linewidth=2, zorder=4)
        ax.add_patch(p1_arrow)
        ax.add_patch(p2_arrow)

        # Distance line
        distance_line, = ax.plot([], [], '--', color='yellow', linewidth=2, alpha=0.7, zorder=4)

        # Other players
        other_markers = []
        if show_all_players:
            for _ in range(other_data['gsisid'].nunique()):
                marker, = ax.plot([], [], 'o', color='white', markersize=8,
                                alpha=0.4, zorder=2)
                other_markers.append(marker)

        # Info text
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                          fontsize=14, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                          color='white', family='monospace')

        # Collision warning
        collision_warning = ax.text(0.5, 0.98, '', transform=ax.transAxes,
                                   fontsize=20, verticalalignment='top',
                                   ha='center', fontweight='bold',
                                   color='red', zorder=10)

        # Legend
        ax.legend(loc='upper right', fontsize=12, framealpha=0.8)

        def init():
            """Initialize animation."""
            p1_marker.set_data([], [])
            p2_marker.set_data([], [])
            p1_trail.set_data([], [])
            p2_trail.set_data([], [])
            distance_line.set_data([], [])
            info_text.set_text('')
            collision_warning.set_text('')
            return [p1_marker, p2_marker, p1_trail, p2_trail,
                   distance_line, info_text, collision_warning] + other_markers

        def update(frame):
            """Update animation for each frame."""
            # Get player positions
            p1_frame = p1_data[p1_data['frame'] == frame]
            p2_frame = p2_data[p2_data['frame'] == frame]

            if len(p1_frame) > 0 and len(p2_frame) > 0:
                p1_x, p1_y = p1_frame['x'].values[0], p1_frame['y'].values[0]
                p2_x, p2_y = p2_frame['x'].values[0], p2_frame['y'].values[0]
                p1_dir = p1_frame['dir'].values[0]
                p2_dir = p2_frame['dir'].values[0]
                p1_speed = p1_frame['dis'].values[0]
                p2_speed = p2_frame['dis'].values[0]

                # Update markers
                p1_marker.set_data([p1_x], [p1_y])
                p2_marker.set_data([p2_x], [p2_y])

                # Update trails
                trail_frames = p1_data[
                    (p1_data['frame'] >= max(0, frame - trail_length)) &
                    (p1_data['frame'] <= frame)
                ]
                if len(trail_frames) > 0:
                    p1_trail.set_data(trail_frames['x'], trail_frames['y'])

                trail_frames = p2_data[
                    (p2_data['frame'] >= max(0, frame - trail_length)) &
                    (p2_data['frame'] <= frame)
                ]
                if len(trail_frames) > 0:
                    p2_trail.set_data(trail_frames['x'], trail_frames['y'])

                # Update direction arrows
                arrow_length = 3
                p1_dx = arrow_length * np.cos(np.radians(p1_dir))
                p1_dy = arrow_length * np.sin(np.radians(p1_dir))
                p2_dx = arrow_length * np.cos(np.radians(p2_dir))
                p2_dy = arrow_length * np.sin(np.radians(p2_dir))

                p1_arrow.set_positions((p1_x, p1_y), (p1_x + p1_dx, p1_y + p1_dy))
                p2_arrow.set_positions((p2_x, p2_y), (p2_x + p2_dx, p2_y + p2_dy))

                # Distance line
                distance_line.set_data([p1_x, p2_x], [p1_y, p2_y])
                dist = np.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2)

                # Info text
                time_sec = frame * 0.1  # 10 Hz sampling
                info_text.set_text(
                    f"Time: {time_sec:.1f}s | Frame: {frame}\n"
                    f"Distance: {dist:.2f} yards\n"
                    f"P1 Speed: {p1_speed:.2f} yds/s\n"
                    f"P2 Speed: {p2_speed:.2f} yds/s"
                )

                # Collision warning
                if abs(frame - collision_frame) <= 2:
                    collision_warning.set_text('⚡ COLLISION ⚡')
                else:
                    collision_warning.set_text('')

                # Update other players
                if show_all_players:
                    other_frame = other_data[other_data['frame'] == frame]
                    unique_players = other_frame['gsisid'].unique()

                    for i, player_id in enumerate(unique_players):
                        if i < len(other_markers):
                            player_frame = other_frame[other_frame['gsisid'] == player_id]
                            if len(player_frame) > 0:
                                other_markers[i].set_data(
                                    [player_frame['x'].values[0]],
                                    [player_frame['y'].values[0]]
                                )

                    # Hide unused markers
                    for i in range(len(unique_players), len(other_markers)):
                        other_markers[i].set_data([], [])

            return [p1_marker, p2_marker, p1_trail, p2_trail,
                   distance_line, info_text, collision_warning] + other_markers

        # Create animation
        anim = animation.FuncAnimation(
            fig, update, init_func=init,
            frames=range(0, max_frame + 1),
            interval=1000/fps, blit=True
        )

        # Save animation
        print(f"\n💾 Saving animation to {output_file}...")

        # Determine file format and writer
        if output_file.endswith('.gif'):
            # Use pillow for GIF
            writer = animation.PillowWriter(fps=fps)
            anim.save(output_file, writer=writer, dpi=100)
        elif output_file.endswith('.mp4'):
            # Try ffmpeg for MP4
            try:
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=fps, metadata=dict(artist='NFL Punt Analytics'),
                               bitrate=3000)
                anim.save(output_file, writer=writer, dpi=150)
            except (KeyError, RuntimeError):
                print("⚠️  ffmpeg not available, saving as GIF instead...")
                output_file = output_file.replace('.mp4', '.gif')
                writer = animation.PillowWriter(fps=fps)
                anim.save(output_file, writer=writer, dpi=100)
        else:
            # Default to HTML
            anim.save(output_file, writer='html', fps=fps, dpi=100)

        print(f"✅ Animation saved successfully!")
        print(f"   Duration: {(max_frame + 1) / fps:.1f} seconds")

        if os.path.exists(output_file):
            print(f"   File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

        plt.close()

        return anim


def main():
    """Example usage: Create animation for most severe collision."""

    # Load injury collision data
    injury_df = pd.read_csv('punt_collision_results/injury_collisions.csv')

    # Get most severe collision
    most_severe = injury_df.nlargest(1, 'collision_intensity').iloc[0]

    print("Creating animation for most severe collision:")
    print(f"  Collision intensity: {most_severe['collision_intensity']:.2f}")
    print(f"  Impact type: {most_severe['impact_type']}")
    print(f"  Min distance: {most_severe['min_distance']:.2f} yards")

    # Create animator
    animator = CollisionAnimator()

    # Create animation (GIF format for compatibility)
    animator.create_animation(
        season=int(most_severe['seasonyear']),
        gamekey=int(most_severe['gamekey']),
        playid=int(most_severe['playid']),
        player1_id=int(most_severe['injured_player']),
        player2_id=int(most_severe['partner_player']),
        output_file='visualizations/most_severe_collision.gif',
        fps=10,
        show_all_players=True,
        trail_length=15
    )


if __name__ == "__main__":
    # Create output directory
    os.makedirs('visualizations', exist_ok=True)
    main()
