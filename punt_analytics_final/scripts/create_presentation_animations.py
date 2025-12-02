#!/usr/bin/env python3
"""
Create Multiple Collision Animations for Presentation
=====================================================

Generate a collection of collision animations showcasing different
types of impacts and collision severities for video presentation.

Author: Peter Shmorhun
Date: 2025-10-20
"""

import pandas as pd
import os
from collision_animation import CollisionAnimator


def create_presentation_animations():
    """
    Create multiple collision animations for presentation.
    """
    print("="*70)
    print("CREATING PRESENTATION COLLISION ANIMATIONS")
    print("="*70)

    # Create output directory
    os.makedirs('visualizations/presentation', exist_ok=True)

    # Load injury collision data
    injury_df = pd.read_csv('punt_collision_results/injury_collisions.csv')

    print(f"\nTotal injury collisions available: {len(injury_df)}")

    # Create animator
    animator = CollisionAnimator()

    # Selection 1: Most severe collision
    print("\n" + "="*70)
    print("1. MOST SEVERE COLLISION (Highest Intensity)")
    print("="*70)

    most_severe = injury_df.nlargest(1, 'collision_intensity').iloc[0]
    print(f"Collision intensity: {most_severe['collision_intensity']:.2f}")
    print(f"Impact type: {most_severe['impact_type']}")
    print(f"Min distance: {most_severe['min_distance']:.2f} yards")
    print(f"Max relative speed: {most_severe['max_relative_speed']:.2f} yds/s")

    try:
        animator.create_animation(
            season=int(most_severe['seasonyear']),
            gamekey=int(most_severe['gamekey']),
            playid=int(most_severe['playid']),
            player1_id=int(most_severe['injured_player']),
            player2_id=int(most_severe['partner_player']),
            output_file='visualizations/presentation/01_most_severe_collision.gif',
            fps=10,
            show_all_players=True,
            trail_length=15
        )
    except Exception as e:
        print(f"❌ Error creating animation: {e}")

    # Selection 2: Helmet-to-helmet collision
    print("\n" + "="*70)
    print("2. HELMET-TO-HELMET COLLISION")
    print("="*70)

    h2h_collisions = injury_df[injury_df['impact_type'] == 'Helmet-to-helmet']
    h2h_severe = h2h_collisions.nlargest(1, 'collision_intensity').iloc[0]

    print(f"Collision intensity: {h2h_severe['collision_intensity']:.2f}")
    print(f"Min distance: {h2h_severe['min_distance']:.2f} yards")
    print(f"Player activity: {h2h_severe['player_activity']}")
    print(f"Partner activity: {h2h_severe['partner_activity']}")

    try:
        animator.create_animation(
            season=int(h2h_severe['seasonyear']),
            gamekey=int(h2h_severe['gamekey']),
            playid=int(h2h_severe['playid']),
            player1_id=int(h2h_severe['injured_player']),
            player2_id=int(h2h_severe['partner_player']),
            output_file='visualizations/presentation/02_helmet_to_helmet.gif',
            fps=10,
            show_all_players=True,
            trail_length=15
        )
    except Exception as e:
        print(f"❌ Error creating animation: {e}")

    # Selection 3: Tackling collision
    print("\n" + "="*70)
    print("3. TACKLING COLLISION")
    print("="*70)

    tackling = injury_df[injury_df['player_activity'] == 'Tackling']
    if len(tackling) > 0:
        tackling_severe = tackling.nlargest(1, 'collision_intensity').iloc[0]

        print(f"Collision intensity: {tackling_severe['collision_intensity']:.2f}")
        print(f"Impact type: {tackling_severe['impact_type']}")
        print(f"Max closing speed: {tackling_severe['max_closing_speed']:.2f} yds/s")

        try:
            animator.create_animation(
                season=int(tackling_severe['seasonyear']),
                gamekey=int(tackling_severe['gamekey']),
                playid=int(tackling_severe['playid']),
                player1_id=int(tackling_severe['injured_player']),
                player2_id=int(tackling_severe['partner_player']),
                output_file='visualizations/presentation/03_tackling_collision.gif',
                fps=10,
                show_all_players=True,
                trail_length=15
            )
        except Exception as e:
            print(f"❌ Error creating animation: {e}")

    # Selection 4: High-speed collision
    print("\n" + "="*70)
    print("4. HIGH-SPEED COLLISION")
    print("="*70)

    high_speed = injury_df.nlargest(1, 'max_relative_speed').iloc[0]
    print(f"Max relative speed: {high_speed['max_relative_speed']:.2f} yds/s")
    print(f"Collision intensity: {high_speed['collision_intensity']:.2f}")
    print(f"Combined speed: {high_speed['combined_speed']:.2f} yds/s")

    try:
        animator.create_animation(
            season=int(high_speed['seasonyear']),
            gamekey=int(high_speed['gamekey']),
            playid=int(high_speed['playid']),
            player1_id=int(high_speed['injured_player']),
            player2_id=int(high_speed['partner_player']),
            output_file='visualizations/presentation/04_high_speed_collision.gif',
            fps=10,
            show_all_players=True,
            trail_length=15
        )
    except Exception as e:
        print(f"❌ Error creating animation: {e}")

    # Selection 5: Close-contact collision (smallest min_distance)
    print("\n" + "="*70)
    print("5. CLOSE-CONTACT COLLISION")
    print("="*70)

    close_contact = injury_df.nsmallest(3, 'min_distance').iloc[1]  # Skip the first (already used)
    print(f"Min distance: {close_contact['min_distance']:.2f} yards")
    print(f"Collision intensity: {close_contact['collision_intensity']:.2f}")
    print(f"Impact type: {close_contact['impact_type']}")

    try:
        animator.create_animation(
            season=int(close_contact['seasonyear']),
            gamekey=int(close_contact['gamekey']),
            playid=int(close_contact['playid']),
            player1_id=int(close_contact['injured_player']),
            player2_id=int(close_contact['partner_player']),
            output_file='visualizations/presentation/05_close_contact_collision.gif',
            fps=10,
            show_all_players=True,
            trail_length=15
        )
    except Exception as e:
        print(f"❌ Error creating animation: {e}")

    print("\n" + "="*70)
    print("✅ PRESENTATION ANIMATIONS COMPLETE!")
    print("="*70)
    print("\nAnimations saved to: visualizations/presentation/")
    print("\nYou can use these GIF files in your video presentation.")
    print("To convert to MP4 format (if you have ffmpeg installed):")
    print("  ffmpeg -i animation.gif -movflags faststart -pix_fmt yuv420p -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' output.mp4")


def create_summary_report():
    """Create a summary report of all animations created."""

    injury_df = pd.read_csv('punt_collision_results/injury_collisions.csv')

    report = []
    report.append("="*70)
    report.append("COLLISION ANIMATION SUMMARY REPORT")
    report.append("="*70)
    report.append("")

    # Most severe
    most_severe = injury_df.nlargest(1, 'collision_intensity').iloc[0]
    report.append("1. Most Severe Collision")
    report.append(f"   - File: 01_most_severe_collision.gif")
    report.append(f"   - Intensity: {most_severe['collision_intensity']:.2f}")
    report.append(f"   - Impact: {most_severe['impact_type']}")
    report.append(f"   - Distance: {most_severe['min_distance']:.2f} yards")
    report.append("")

    # Helmet-to-helmet
    h2h_collisions = injury_df[injury_df['impact_type'] == 'Helmet-to-helmet']
    h2h_severe = h2h_collisions.nlargest(1, 'collision_intensity').iloc[0]
    report.append("2. Helmet-to-Helmet Collision")
    report.append(f"   - File: 02_helmet_to_helmet.gif")
    report.append(f"   - Intensity: {h2h_severe['collision_intensity']:.2f}")
    report.append(f"   - Activities: {h2h_severe['player_activity']} vs {h2h_severe['partner_activity']}")
    report.append("")

    # Tackling
    tackling = injury_df[injury_df['player_activity'] == 'Tackling']
    if len(tackling) > 0:
        tackling_severe = tackling.nlargest(1, 'collision_intensity').iloc[0]
        report.append("3. Tackling Collision")
        report.append(f"   - File: 03_tackling_collision.gif")
        report.append(f"   - Intensity: {tackling_severe['collision_intensity']:.2f}")
        report.append(f"   - Closing speed: {tackling_severe['max_closing_speed']:.2f} yds/s")
        report.append("")

    # High-speed
    high_speed = injury_df.nlargest(1, 'max_relative_speed').iloc[0]
    report.append("4. High-Speed Collision")
    report.append(f"   - File: 04_high_speed_collision.gif")
    report.append(f"   - Relative speed: {high_speed['max_relative_speed']:.2f} yds/s")
    report.append(f"   - Combined speed: {high_speed['combined_speed']:.2f} yds/s")
    report.append("")

    # Close-contact
    close_contact = injury_df.nsmallest(3, 'min_distance').iloc[1]
    report.append("5. Close-Contact Collision")
    report.append(f"   - File: 05_close_contact_collision.gif")
    report.append(f"   - Min distance: {close_contact['min_distance']:.2f} yards")
    report.append(f"   - Impact: {close_contact['impact_type']}")
    report.append("")

    report.append("="*70)
    report.append("Dataset Statistics:")
    report.append(f"  Total injury collisions: {len(injury_df)}")
    report.append(f"  Helmet-to-helmet: {len(injury_df[injury_df['impact_type'] == 'Helmet-to-helmet'])}")
    report.append(f"  Helmet-to-body: {len(injury_df[injury_df['impact_type'] == 'Helmet-to-body'])}")
    report.append(f"  Average intensity: {injury_df['collision_intensity'].mean():.2f}")
    report.append(f"  Average min distance: {injury_df['min_distance'].mean():.2f} yards")
    report.append("="*70)

    report_text = "\n".join(report)
    print(report_text)

    # Save to file
    with open('visualizations/presentation/ANIMATION_SUMMARY.txt', 'w') as f:
        f.write(report_text)

    print("\n📄 Summary report saved to: visualizations/presentation/ANIMATION_SUMMARY.txt")


if __name__ == "__main__":
    create_presentation_animations()
    print("\n")
    create_summary_report()
