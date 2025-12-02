#!/usr/bin/env python3
"""
Analyze potential target variables for pass rush collision modeling
"""

import pandas as pd
import numpy as np

def analyze_target_options():
    """Explore different target variable options for modeling"""

    print("="*70)
    print("TARGET VARIABLE ANALYSIS FOR PASS RUSH COLLISIONS")
    print("="*70)

    # Load PFF data
    pff = pd.read_csv('nfl-big-data-bowl-2023/pffScoutingData.csv')

    # Focus on pass rushers
    rushers = pff[pff['pff_role'] == 'Pass Rush'].copy()
    blockers = pff[pff['pff_role'] == 'Pass Block'].copy()

    print("\n" + "="*70)
    print("OPTION 1: RUSHER PERSPECTIVE - Did rusher generate pressure?")
    print("="*70)

    # Create composite pressure flag for rushers
    rushers['generated_pressure'] = (
        (rushers['pff_hit'] == 1) |
        (rushers['pff_hurry'] == 1) |
        (rushers['pff_sack'] == 1)
    ).astype(int)

    print(f"\nTotal rush attempts: {len(rushers):,}")
    print(f"Rushes that generated pressure: {rushers['generated_pressure'].sum():,}")
    print(f"Pressure rate: {rushers['generated_pressure'].mean()*100:.2f}%")

    print("\nBreakdown by pressure type:")
    print(f"  Hits: {(rushers['pff_hit'] == 1).sum():,} ({(rushers['pff_hit'] == 1).mean()*100:.2f}%)")
    print(f"  Hurries: {(rushers['pff_hurry'] == 1).sum():,} ({(rushers['pff_hurry'] == 1).mean()*100:.2f}%)")
    print(f"  Sacks: {(rushers['pff_sack'] == 1).sum():,} ({(rushers['pff_sack'] == 1).mean()*100:.2f}%)")

    print("\n✅ TARGET 1: Binary - Rusher generated pressure (hit/hurry/sack) = 1, else 0")
    print("   Class balance: 11.8% positive, 88.2% negative")
    print("   Pros: Clear outcome, good volume, directly injury-relevant")
    print("   Cons: Moderate class imbalance")


    print("\n" + "="*70)
    print("OPTION 2: BLOCKER PERSPECTIVE - Did blocker allow pressure?")
    print("="*70)

    # Create composite pressure allowed flag for blockers
    blockers['allowed_pressure'] = (
        (blockers['pff_hitAllowed'] == 1) |
        (blockers['pff_hurryAllowed'] == 1) |
        (blockers['pff_sackAllowed'] == 1)
    ).astype(int)

    print(f"\nTotal block attempts: {len(blockers):,}")
    print(f"Blocks that allowed pressure: {blockers['allowed_pressure'].sum():,}")
    print(f"Pressure allowed rate: {blockers['allowed_pressure'].mean()*100:.2f}%")

    print("\nBreakdown by pressure type allowed:")
    print(f"  Hits allowed: {(blockers['pff_hitAllowed'] == 1).sum():,} ({(blockers['pff_hitAllowed'] == 1).mean()*100:.2f}%)")
    print(f"  Hurries allowed: {(blockers['pff_hurryAllowed'] == 1).sum():,} ({(blockers['pff_hurryAllowed'] == 1).mean()*100:.2f}%)")
    print(f"  Sacks allowed: {(blockers['pff_sackAllowed'] == 1).sum():,} ({(blockers['pff_sackAllowed'] == 1).mean()*100:.2f}%)")

    print("\n✅ TARGET 2: Binary - Blocker allowed pressure = 1, else 0")
    print("   Class balance: 6.5% positive, 93.5% negative")
    print("   Pros: Clear outcome, blocker health perspective")
    print("   Cons: Higher class imbalance")


    print("\n" + "="*70)
    print("OPTION 3: BLOCKER BEATEN - Was blocker beaten by defender?")
    print("="*70)

    beaten_count = (blockers['pff_beatenByDefender'] == 1).sum()
    beaten_rate = (blockers['pff_beatenByDefender'] == 1).mean()

    print(f"\nTotal block attempts: {len(blockers):,}")
    print(f"Blocker beaten: {beaten_count:,}")
    print(f"Beaten rate: {beaten_rate*100:.2f}%")

    print("\n✅ TARGET 3: Binary - Blocker beaten by defender = 1, else 0")
    print("   Class balance: 4.2% positive, 95.8% negative")
    print("   Pros: Most directly measures collision 'winner'")
    print("   Cons: Highest class imbalance, may not capture all collisions")


    print("\n" + "="*70)
    print("OPTION 4: MULTI-CLASS - Pressure outcome severity")
    print("="*70)

    # Create multi-class target for rushers
    def classify_pressure(row):
        if row['pff_sack'] == 1:
            return 3  # Sack (most severe)
        elif row['pff_hit'] == 1:
            return 2  # Hit (moderate)
        elif row['pff_hurry'] == 1:
            return 1  # Hurry (light pressure)
        else:
            return 0  # No pressure

    rushers['pressure_severity'] = rushers.apply(classify_pressure, axis=1)

    print("\nPressure severity distribution:")
    print(rushers['pressure_severity'].value_counts().sort_index())
    print("\nAs percentages:")
    print(rushers['pressure_severity'].value_counts(normalize=True).sort_index() * 100)

    print("\n✅ TARGET 4: Multi-class - Pressure severity (0=none, 1=hurry, 2=hit, 3=sack)")
    print("   Class balance: 88.2% none, 7.8% hurry, 2.3% hit, 1.6% sack")
    print("   Pros: Captures severity gradient, injury relevance increases with severity")
    print("   Cons: Severe class imbalance, harder to model")


    print("\n" + "="*70)
    print("OPTION 5: BLOCKING MATCHUP PERSPECTIVE")
    print("="*70)

    # Analyze blocking matchups
    matchups = blockers[blockers['pff_nflIdBlockedPlayer'].notna()].copy()

    print(f"\nTotal documented matchups: {len(matchups):,}")

    # Can we join rusher outcome to blocker?
    # For each matchup, we know:
    # - Blocker ID (nflId)
    # - Rusher ID (pff_nflIdBlockedPlayer)
    # - Block outcome (beaten, pressure allowed)

    print("\nFor each matchup, we know:")
    print("  - Blocker identity")
    print("  - Rusher identity (who they blocked)")
    print("  - Whether blocker was beaten")
    print("  - Whether blocker allowed pressure")
    print("  - Block type (pass protection technique)")

    matchups['matchup_outcome'] = matchups['allowed_pressure']

    print(f"\nMatchups where blocker allowed pressure: {matchups['matchup_outcome'].sum():,}")
    print(f"Matchup pressure rate: {matchups['matchup_outcome'].mean()*100:.2f}%")

    print("\n✅ TARGET 5: Binary - Blocker-rusher matchup resulted in pressure = 1, else 0")
    print("   Class balance: 6.2% positive, 93.8% negative")
    print("   Pros: Most direct - models actual collision outcome")
    print("   Cons: Class imbalance")


    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    print("""
BEST TARGET OPTIONS (Ranked):

1. RUSHER GENERATED PRESSURE (Binary) ⭐⭐⭐
   - Target: Did rusher generate hit/hurry/sack?
   - 36,362 samples, 11.8% positive class
   - PROS: Best class balance, clear health relevance
   - USE CASE: "Predict which collisions result in QB pressure"

2. BLOCKING MATCHUP OUTCOME (Binary) ⭐⭐⭐
   - Target: Did specific blocker-rusher matchup allow pressure?
   - 44,526 samples, 6.2% positive class
   - PROS: Most directly models collision dynamics
   - USE CASE: "Predict collision outcome given two players"

3. BLOCKER ALLOWED PRESSURE (Binary) ⭐⭐
   - Target: Did blocker allow pressure?
   - 46,057 samples, 6.5% positive class
   - PROS: Blocker injury perspective
   - USE CASE: "Identify vulnerable blocking situations"

4. PRESSURE SEVERITY (Multi-class) ⭐
   - Target: None/Hurry/Hit/Sack (0-3)
   - 36,362 samples, severe imbalance
   - PROS: Captures injury gradient
   - CONS: Complex modeling, imbalance

RECOMMENDED APPROACH:
Start with Option 1 (Rusher Generated Pressure) because:
- Best class balance (11.8% vs 88.2%)
- Similar to your punt analytics concussion prediction
- Clear injury connection (QB hits = concussion risk)
- Can engineer same collision features

ALTERNATIVE:
Option 2 (Blocking Matchup) if you want to focus on:
- Lineman-to-lineman collision dynamics
- Pair-wise collision modeling (like punt analytics partner analysis)
- Understanding blocking technique effectiveness
""")

    # Save sample data for reference
    print("\n" + "="*70)
    print("SAVING SAMPLE DATA")
    print("="*70)

    # Save rusher outcomes
    rushers[['gameId', 'playId', 'nflId', 'pff_positionLinedUp',
             'pff_hit', 'pff_hurry', 'pff_sack', 'generated_pressure']].to_csv(
        'sample_rusher_targets.csv', index=False
    )
    print("✅ Saved sample_rusher_targets.csv")

    # Save blocking matchup outcomes
    matchups[['gameId', 'playId', 'nflId', 'pff_nflIdBlockedPlayer', 'pff_positionLinedUp',
              'pff_beatenByDefender', 'pff_hitAllowed', 'pff_hurryAllowed', 'pff_sackAllowed',
              'allowed_pressure', 'pff_blockType']].to_csv(
        'sample_matchup_targets.csv', index=False
    )
    print("✅ Saved sample_matchup_targets.csv")


if __name__ == "__main__":
    analyze_target_options()
