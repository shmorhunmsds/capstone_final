#!/usr/bin/env python3
"""
Quick test of collision feature extraction
"""

import sys
sys.path.insert(0, '/home/pshmo/punt_analytics_final/scripts')

from collision_feature_extraction import CollisionFeatureExtractor

def test_extraction():
    """Test basic functionality"""

    print("Testing CollisionFeatureExtractor...")

    # Initialize
    extractor = CollisionFeatureExtractor(collision_threshold='moderate')

    # Load data
    extractor.load_data()

    print("\n✅ Data loaded successfully!")
    print(f"   Motion records: {len(extractor.motion_data):,}")
    print(f"   Video reviews: {len(extractor.video_review):,}")

    # Test injury collision extraction (first 5)
    print("\n🧪 Testing injury collision extraction (first 5 cases)...")

    injury_count = 0
    for idx, injury in extractor.video_review.head(5).iterrows():
        if pd.isna(injury.get('primarypartnergsisid')):
            continue

        season = injury['seasonyear']
        gamekey = injury['gamekey']
        playid = injury['playid']
        injured_player = injury['gsisid']
        partner_player = injury['primarypartnergsisid']

        play_motion = extractor.motion_data[
            (extractor.motion_data['seasonyear'] == season) &
            (extractor.motion_data['gamekey'] == gamekey) &
            (extractor.motion_data['playid'] == playid)
        ]

        injured_motion = play_motion[play_motion['gsisid'] == injured_player]
        partner_motion = play_motion[play_motion['gsisid'] == partner_player]

        if len(injured_motion) >= 3 and len(partner_motion) >= 3:
            features = extractor.calculate_collision_features(injured_motion, partner_motion)
            if features:
                print(f"\n   Case {injury_count + 1}:")
                print(f"      Min distance: {features['min_distance']:.2f} yards")
                print(f"      Max relative speed: {features['max_relative_speed']:.2f} yds/sec")
                print(f"      Collision intensity: {features['collision_intensity']:.2f}")
                print(f"      Collision quality: {features['collision_quality']:.2f}")
                print(f"      Valid collision? {extractor.is_valid_collision(features)}")
                injury_count += 1

    print(f"\n✅ Test complete! Successfully processed {injury_count} injury collisions")

if __name__ == "__main__":
    import pandas as pd
    test_extraction()
