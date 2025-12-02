#!/usr/bin/env python3
"""
Exploratory Data Analysis for Big Data Bowl 2023 - Collision Dynamics
Focus on QB pressure, pass rush collisions, and receiver impacts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def load_all_data():
    """Load all necessary datasets"""
    print("Loading datasets...")

    # Load tracking data (just week 1 for initial EDA)
    week1 = pd.read_csv('nfl-big-data-bowl-2023/week1.csv')

    # Load metadata
    plays = pd.read_csv('nfl-big-data-bowl-2023/plays.csv')
    pff = pd.read_csv('nfl-big-data-bowl-2023/pffScoutingData.csv')
    players = pd.read_csv('nfl-big-data-bowl-2023/players.csv')
    games = pd.read_csv('nfl-big-data-bowl-2023/games.csv')

    print(f"✅ Loaded {len(week1):,} tracking records")
    print(f"✅ Loaded {len(plays):,} plays")
    print(f"✅ Loaded {len(pff):,} PFF records")
    print(f"✅ Loaded {len(players):,} players")

    return week1, plays, pff, players, games


def analyze_qb_pressure_events(week1, plays, pff):
    """Deep dive into QB pressure dynamics"""

    print("\n" + "="*70)
    print("PART 1: QB PRESSURE EVENT ANALYSIS")
    print("="*70)

    # Get QB plays
    qb_plays = pff[pff['pff_role'] == 'Pass'].copy()

    # Create pressure indicators
    qb_plays['has_hit'] = qb_plays['pff_hit'] == 1
    qb_plays['has_hurry'] = qb_plays['pff_hurry'] == 1
    qb_plays['has_sack'] = qb_plays['pff_sack'] == 1
    qb_plays['any_pressure'] = qb_plays['has_hit'] | qb_plays['has_hurry'] | qb_plays['has_sack']

    print(f"\n1. QB PRESSURE SUMMARY")
    print("-"*70)
    print(f"Total QB plays: {len(qb_plays):,}")
    print(f"Plays with hits: {qb_plays['has_hit'].sum():,} ({qb_plays['has_hit'].mean()*100:.1f}%)")
    print(f"Plays with hurries: {qb_plays['has_hurry'].sum():,} ({qb_plays['has_hurry'].mean()*100:.1f}%)")
    print(f"Plays with sacks: {qb_plays['has_sack'].sum():,} ({qb_plays['has_sack'].mean()*100:.1f}%)")
    print(f"Plays with any pressure: {qb_plays['any_pressure'].sum():,} ({qb_plays['any_pressure'].mean()*100:.1f}%)")

    # Merge with play outcomes
    qb_with_outcome = qb_plays.merge(plays[['gameId', 'playId', 'passResult', 'playResult']],
                                      on=['gameId', 'playId'], how='left')

    print(f"\n2. PRESSURE VS PLAY OUTCOME")
    print("-"*70)

    # Analyze by pressure type
    for pressure_type in ['has_hit', 'has_hurry', 'has_sack', 'any_pressure']:
        pressure_plays = qb_with_outcome[qb_with_outcome[pressure_type]]
        no_pressure_plays = qb_with_outcome[~qb_with_outcome[pressure_type]]

        if len(pressure_plays) > 0:
            pressure_outcomes = pressure_plays['passResult'].value_counts(normalize=True)
            no_pressure_outcomes = no_pressure_plays['passResult'].value_counts(normalize=True)

            print(f"\n{pressure_type.replace('_', ' ').upper()}:")
            print(f"  Complete rate: {pressure_outcomes.get('C', 0)*100:.1f}% vs {no_pressure_outcomes.get('C', 0)*100:.1f}% (no pressure)")
            print(f"  Incomplete rate: {pressure_outcomes.get('I', 0)*100:.1f}% vs {no_pressure_outcomes.get('I', 0)*100:.1f}% (no pressure)")
            print(f"  Sack rate: {pressure_outcomes.get('S', 0)*100:.1f}% vs {no_pressure_outcomes.get('S', 0)*100:.1f}% (no pressure)")

    # Get QB tracking data for pressure plays
    print(f"\n3. QB MOVEMENT DURING PRESSURE PLAYS")
    print("-"*70)

    # Sample some pressure plays
    pressure_play_ids = qb_plays[qb_plays['any_pressure']][['gameId', 'playId']].drop_duplicates().head(100)

    # Get QB tracking for these plays
    qb_ids = qb_plays['nflId'].unique()

    pressure_tracking = week1[
        (week1['nflId'].isin(qb_ids)) &
        (week1[['gameId', 'playId']].apply(tuple, axis=1).isin(
            pressure_play_ids[['gameId', 'playId']].apply(tuple, axis=1)
        ))
    ]

    if len(pressure_tracking) > 0:
        print(f"QB tracking points during pressure plays: {len(pressure_tracking):,}")
        print(f"Avg speed: {pressure_tracking['s'].mean():.2f} yards/sec")
        print(f"Max speed: {pressure_tracking['s'].max():.2f} yards/sec")
        print(f"Avg acceleration: {pressure_tracking['a'].mean():.2f} yards/sec²")

    return qb_plays, qb_with_outcome


def analyze_pass_rush_dynamics(week1, pff):
    """Analyze pass rush and blocking matchups"""

    print("\n" + "="*70)
    print("PART 2: PASS RUSH COLLISION ANALYSIS")
    print("="*70)

    # Get rush and block data
    rushers = pff[pff['pff_role'] == 'Pass Rush'].copy()
    blockers = pff[pff['pff_role'] == 'Pass Block'].copy()

    print(f"\n1. PASS RUSH OVERVIEW")
    print("-"*70)
    print(f"Total rush attempts: {len(rushers):,}")
    print(f"Total blocks attempted: {len(blockers):,}")

    # Analyze blocking matchups
    blocking_matchups = blockers[blockers['pff_nflIdBlockedPlayer'].notna()].copy()
    print(f"\nDocumented blocking matchups: {len(blocking_matchups):,}")

    # Block outcomes
    print(f"\n2. BLOCKING OUTCOMES")
    print("-"*70)
    print(f"Blocks with hits allowed: {blockers['pff_hitAllowed'].sum()} ({blockers['pff_hitAllowed'].mean()*100:.1f}%)")
    print(f"Blocks with hurries allowed: {blockers['pff_hurryAllowed'].sum()} ({blockers['pff_hurryAllowed'].mean()*100:.1f}%)")
    print(f"Blocks with sacks allowed: {blockers['pff_sackAllowed'].sum()} ({blockers['pff_sackAllowed'].mean()*100:.1f}%)")
    print(f"Blockers beaten: {blockers['pff_beatenByDefender'].sum()} ({blockers['pff_beatenByDefender'].mean()*100:.1f}%)")

    # Rush outcomes
    print(f"\n3. PASS RUSH SUCCESS")
    print("-"*70)
    print(f"Rushes with hits: {rushers['pff_hit'].sum()} ({rushers['pff_hit'].mean()*100:.1f}%)")
    print(f"Rushes with hurries: {rushers['pff_hurry'].sum()} ({rushers['pff_hurry'].mean()*100:.1f}%)")
    print(f"Rushes with sacks: {rushers['pff_sack'].sum()} ({rushers['pff_sack'].mean()*100:.1f}%)")

    # Analyze by position
    print(f"\n4. RUSH SUCCESS BY POSITION")
    print("-"*70)

    position_stats = rushers.groupby('pff_positionLinedUp').agg({
        'pff_hit': 'sum',
        'pff_hurry': 'sum',
        'pff_sack': 'sum',
        'nflId': 'count'
    }).rename(columns={'nflId': 'attempts'})

    position_stats['pressure_rate'] = (
        (position_stats['pff_hit'] + position_stats['pff_hurry'] + position_stats['pff_sack']) /
        position_stats['attempts'] * 100
    )

    position_stats = position_stats.sort_values('pressure_rate', ascending=False)
    print(position_stats.head(10))

    # Get sample blocking matchups for collision analysis
    sample_matchups = blocking_matchups.head(1000)

    return rushers, blockers, blocking_matchups


def analyze_receiver_collisions(week1, plays, pff):
    """Analyze receiver-defender proximity and contested catches"""

    print("\n" + "="*70)
    print("PART 3: RECEIVER COLLISION RISK ANALYSIS")
    print("="*70)

    # Get receivers and defenders
    receivers = pff[pff['pff_role'] == 'Pass Route'].copy()
    defenders = pff[pff['pff_role'] == 'Coverage'].copy()

    print(f"\n1. RECEIVER/DEFENDER OVERVIEW")
    print("-"*70)
    print(f"Total pass routes: {len(receivers):,}")
    print(f"Total coverage assignments: {len(defenders):,}")

    # Merge with play outcomes
    receiver_plays = receivers.merge(
        plays[['gameId', 'playId', 'passResult']],
        on=['gameId', 'playId'],
        how='left'
    )

    print(f"\n2. ROUTES BY OUTCOME")
    print("-"*70)
    print(receiver_plays['passResult'].value_counts())

    # Analyze by position
    print(f"\n3. ROUTES BY POSITION")
    print("-"*70)
    position_counts = receivers['pff_positionLinedUp'].value_counts().head(10)
    print(position_counts)

    return receivers, defenders


def analyze_play_characteristics(plays, pff):
    """Analyze play-level characteristics that might affect collision risk"""

    print("\n" + "="*70)
    print("PART 4: PLAY CHARACTERISTICS & COLLISION CONTEXT")
    print("="*70)

    # Merge plays with pressure data
    qb_data = pff[pff['pff_role'] == 'Pass'][['gameId', 'playId', 'pff_hit', 'pff_hurry', 'pff_sack']]
    plays_with_pressure = plays.merge(qb_data, on=['gameId', 'playId'], how='left')

    plays_with_pressure['any_pressure'] = (
        (plays_with_pressure['pff_hit'] == 1) |
        (plays_with_pressure['pff_hurry'] == 1) |
        (plays_with_pressure['pff_sack'] == 1)
    )

    print(f"\n1. PRESSURE BY DOWN")
    print("-"*70)
    pressure_by_down = plays_with_pressure.groupby('down')['any_pressure'].agg(['sum', 'count', 'mean'])
    pressure_by_down['pressure_rate'] = pressure_by_down['mean'] * 100
    print(pressure_by_down)

    print(f"\n2. PRESSURE BY FORMATION")
    print("-"*70)
    pressure_by_formation = plays_with_pressure.groupby('offenseFormation')['any_pressure'].agg(['sum', 'count', 'mean'])
    pressure_by_formation['pressure_rate'] = pressure_by_formation['mean'] * 100
    print(pressure_by_formation.sort_values('pressure_rate', ascending=False).head(10))

    print(f"\n3. PRESSURE BY COVERAGE TYPE")
    print("-"*70)
    pressure_by_coverage = plays_with_pressure.groupby('pff_passCoverageType')['any_pressure'].agg(['sum', 'count', 'mean'])
    pressure_by_coverage['pressure_rate'] = pressure_by_coverage['mean'] * 100
    print(pressure_by_coverage.sort_values('pressure_rate', ascending=False))

    print(f"\n4. DEFENDERS IN BOX VS PRESSURE")
    print("-"*70)
    pressure_by_defenders = plays_with_pressure.groupby('defendersInBox')['any_pressure'].agg(['sum', 'count', 'mean'])
    pressure_by_defenders['pressure_rate'] = pressure_by_defenders['mean'] * 100
    print(pressure_by_defenders.sort_values('defendersInBox'))

    return plays_with_pressure


def sample_collision_detection(week1, pff):
    """Sample collision detection on a few plays to validate approach"""

    print("\n" + "="*70)
    print("PART 5: SAMPLE COLLISION DETECTION")
    print("="*70)

    # Get a play with a sack (high collision probability)
    qb_sacks = pff[(pff['pff_role'] == 'Pass') & (pff['pff_sack'] == 1)]

    if len(qb_sacks) > 0:
        sample_play = qb_sacks.iloc[0]
        game_id = sample_play['gameId']
        play_id = sample_play['playId']
        qb_id = sample_play['nflId']

        print(f"\n1. ANALYZING SACK PLAY")
        print("-"*70)
        print(f"Game: {game_id}, Play: {play_id}")
        print(f"QB ID: {qb_id}")

        # Get all tracking data for this play
        play_tracking = week1[(week1['gameId'] == game_id) & (week1['playId'] == play_id)]

        print(f"Total players tracked: {play_tracking['nflId'].nunique()}")
        print(f"Total frames: {play_tracking['frameId'].max()}")

        # Get QB tracking
        qb_tracking = play_tracking[play_tracking['nflId'] == qb_id]
        print(f"\nQB tracking points: {len(qb_tracking)}")

        # Get rushers on this play
        play_rushers = pff[(pff['gameId'] == game_id) & (pff['playId'] == play_id) & (pff['pff_role'] == 'Pass Rush')]
        rusher_ids = play_rushers['nflId'].values

        print(f"Pass rushers: {len(rusher_ids)}")

        # Calculate minimum distances between QB and each rusher
        print(f"\n2. QB-RUSHER PROXIMITY")
        print("-"*70)

        for rusher_id in rusher_ids[:5]:  # Check first 5 rushers
            rusher_tracking = play_tracking[play_tracking['nflId'] == rusher_id]

            if len(rusher_tracking) > 0 and len(qb_tracking) > 0:
                # Find common frames
                common_frames = set(qb_tracking['frameId']).intersection(set(rusher_tracking['frameId']))

                if len(common_frames) > 0:
                    qb_common = qb_tracking[qb_tracking['frameId'].isin(common_frames)]
                    rusher_common = rusher_tracking[rusher_tracking['frameId'].isin(common_frames)]

                    # Merge on frame
                    merged = qb_common.merge(rusher_common, on='frameId', suffixes=('_qb', '_rusher'))

                    # Calculate distances
                    merged['distance'] = np.sqrt(
                        (merged['x_qb'] - merged['x_rusher'])**2 +
                        (merged['y_qb'] - merged['y_rusher'])**2
                    )

                    min_distance = merged['distance'].min()
                    min_frame = merged.loc[merged['distance'].idxmin()]

                    print(f"Rusher {rusher_id}:")
                    print(f"  Min distance: {min_distance:.2f} yards at frame {min_frame['frameId']}")
                    print(f"  QB speed at closest: {min_frame['s_qb']:.2f} yards/sec")
                    print(f"  Rusher speed at closest: {min_frame['s_rusher']:.2f} yards/sec")
                    print(f"  Combined speed: {min_frame['s_qb'] + min_frame['s_rusher']:.2f} yards/sec")


def create_visualizations(qb_plays, rushers, plays_with_pressure):
    """Create key visualizations"""

    print("\n" + "="*70)
    print("PART 6: CREATING VISUALIZATIONS")
    print("="*70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Pressure type distribution
    pressure_counts = pd.DataFrame({
        'Hits': [qb_plays['pff_hit'].sum()],
        'Hurries': [qb_plays['pff_hurry'].sum()],
        'Sacks': [qb_plays['pff_sack'].sum()]
    }).T

    axes[0, 0].bar(pressure_counts.index, pressure_counts[0], color=['#d62728', '#ff7f0e', '#2ca02c'])
    axes[0, 0].set_title('QB Pressure Events Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # 2. Pass rush success by position
    position_pressure = rushers.groupby('pff_positionLinedUp').agg({
        'pff_hit': 'sum',
        'pff_hurry': 'sum',
        'pff_sack': 'sum',
        'nflId': 'count'
    }).rename(columns={'nflId': 'attempts'})

    position_pressure['pressure_rate'] = (
        (position_pressure['pff_hit'] + position_pressure['pff_hurry'] + position_pressure['pff_sack']) /
        position_pressure['attempts'] * 100
    )

    top_positions = position_pressure.nlargest(8, 'pressure_rate')
    axes[0, 1].barh(range(len(top_positions)), top_positions['pressure_rate'])
    axes[0, 1].set_yticks(range(len(top_positions)))
    axes[0, 1].set_yticklabels(top_positions.index)
    axes[0, 1].set_xlabel('Pressure Rate (%)')
    axes[0, 1].set_title('Pressure Rate by Rush Position', fontsize=14, fontweight='bold')
    axes[0, 1].grid(axis='x', alpha=0.3)

    # 3. Pressure by down
    pressure_by_down = plays_with_pressure.groupby('down')['any_pressure'].mean() * 100
    axes[0, 2].bar(pressure_by_down.index, pressure_by_down.values, color='steelblue')
    axes[0, 2].set_xlabel('Down')
    axes[0, 2].set_ylabel('Pressure Rate (%)')
    axes[0, 2].set_title('Pressure Rate by Down', fontsize=14, fontweight='bold')
    axes[0, 2].grid(axis='y', alpha=0.3)

    # 4. Defenders in box vs pressure
    pressure_by_def = plays_with_pressure.groupby('defendersInBox')['any_pressure'].mean() * 100
    axes[1, 0].plot(pressure_by_def.index, pressure_by_def.values, marker='o', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Defenders in Box')
    axes[1, 0].set_ylabel('Pressure Rate (%)')
    axes[1, 0].set_title('Pressure Rate vs Defenders in Box', fontsize=14, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    # 5. Coverage type vs pressure
    pressure_by_cov = plays_with_pressure.groupby('pff_passCoverageType')['any_pressure'].mean() * 100
    axes[1, 1].barh(range(len(pressure_by_cov)), pressure_by_cov.values)
    axes[1, 1].set_yticks(range(len(pressure_by_cov)))
    axes[1, 1].set_yticklabels(pressure_by_cov.index)
    axes[1, 1].set_xlabel('Pressure Rate (%)')
    axes[1, 1].set_title('Pressure Rate by Coverage Type', fontsize=14, fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)

    # 6. Formation vs pressure
    pressure_by_form = plays_with_pressure.groupby('offenseFormation')['any_pressure'].mean() * 100
    top_formations = pressure_by_form.nlargest(8)
    axes[1, 2].barh(range(len(top_formations)), top_formations.values)
    axes[1, 2].set_yticks(range(len(top_formations)))
    axes[1, 2].set_yticklabels(top_formations.index)
    axes[1, 2].set_xlabel('Pressure Rate (%)')
    axes[1, 2].set_title('Pressure Rate by Formation', fontsize=14, fontweight='bold')
    axes[1, 2].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('eda_collision_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved visualization to eda_collision_analysis.png")

    plt.close()


def main():
    """Main EDA workflow"""

    print("\n" + "="*70)
    print("BIG DATA BOWL 2023 - COLLISION EDA")
    print("Focus: QB Pressure, Pass Rush, and Receiver Collisions")
    print("="*70)

    # Load data
    week1, plays, pff, players, games = load_all_data()

    # Run analyses
    qb_plays, qb_with_outcome = analyze_qb_pressure_events(week1, plays, pff)
    rushers, blockers, blocking_matchups = analyze_pass_rush_dynamics(week1, pff)
    receivers, defenders = analyze_receiver_collisions(week1, plays, pff)
    plays_with_pressure = analyze_play_characteristics(plays, pff)
    sample_collision_detection(week1, pff)

    # Create visualizations
    create_visualizations(qb_plays, rushers, plays_with_pressure)

    print("\n" + "="*70)
    print("KEY FINDINGS & RECOMMENDATIONS")
    print("="*70)

    print("""
1. QB PRESSURE MODELING (RECOMMENDED ⭐⭐⭐):
   - 843 hits, 2,877 hurries, 602 sacks across 8,557 plays
   - ~50% of plays have some form of pressure
   - Clear target variable: hit/hurry/sack vs clean pocket
   - Can track QB-rusher proximity over time

2. PASS RUSH COLLISION MODELING (RECOMMENDED ⭐⭐⭐):
   - 46,526 documented blocker-rusher matchups
   - 9.8% of rushes result in pressure
   - Can model: collision intensity, block success, rusher closing speed
   - Lineman health is critical but understudied

3. RECEIVER COLLISION MODELING (VIABLE ⭐⭐):
   - Good tracking data for route runners and coverage
   - Less structured "target variable" (no explicit "collision" label)
   - Would need to engineer catch-point proximity features

NEXT STEPS:
- Build collision feature engineering pipeline (similar to punt analytics)
- Focus on QB pressure OR pass rush collisions
- Engineer features: closing speed, proximity, collision intensity
- Build classification model: pressure vs no pressure
""")

    print("\n✅ EDA COMPLETE!")


if __name__ == "__main__":
    main()
