# Pass Rush Collision Dataset
## Big Data Bowl 2023 - Player Health & Safety Analysis

**Generated:** 2025-10-12
**Purpose:** Predict QB pressure events from pass rush collision dynamics

---

## 📊 Dataset Overview

### Files
- **`pass_rush_collision_features_full.csv`** - Complete dataset (all 8 weeks)
- **`week1_features.csv` ... `week8_features.csv`** - Individual week files
- **`dataset_summary.txt`** - Statistical summary
- **`feature_list.txt`** - Complete feature descriptions

### Statistics
- **Total Samples:** 36,362 rush attempts
- **Total Features:** 46
- **Time Period:** NFL 2021 Season, Weeks 1-8
- **Class Balance:** 11.64% pressure, 88.36% no pressure

---

## 🎯 Target Variable

**`generated_pressure`** (Binary)
- **1** = Rusher generated QB pressure (hit, hurry, or sack)
- **0** = Clean rush (no pressure)

### Breakdown
- **Hits:** 823 (2.26%)
- **Hurries:** 2,824 (7.77%)
- **Sacks:** 585 (1.61%)
- **Total Pressure:** 4,232 (11.64%)

---

## 🔧 Feature Categories

### 1. Distance Features (4)
- `min_distance` - Closest approach between rusher and QB
- `avg_distance` - Average distance throughout play
- `distance_at_start` - Initial separation
- `distance_at_end` - Final separation

### 2. Speed Features (6)
- `rusher_max_speed`, `rusher_avg_speed`, `rusher_speed_at_closest`
- `qb_max_speed`, `qb_avg_speed`, `qb_speed_at_closest`

### 3. Acceleration Features (6)
- `rusher_max_accel`, `rusher_avg_accel`, `rusher_accel_at_closest`
- `qb_max_accel`, `qb_avg_accel`, `qb_accel_at_closest`

### 4. Relative Motion (3)
- `combined_speed_at_closest` - Sum of speeds at closest approach
- `max_closing_speed` - Maximum rate of distance decrease
- `avg_closing_speed` - Average closing speed

### 5. Orientation (4)
- `approach_angle` - Angle of approach vector
- `rusher_orientation_at_closest` - Rusher body orientation
- `qb_orientation_at_closest` - QB body orientation
- `rusher_angle_alignment` - Alignment with approach vector

### 6. Temporal (3)
- `time_to_closest_approach` - When closest approach occurred
- `total_frames` - Number of frames tracked
- `play_duration` - Total play length in seconds

### 7. Collision Intensity (3) ⭐ KEY FEATURES
- **`collision_intensity`** - Normalized collision intensity metric
- `collision_intensity_raw` - Raw collision intensity
- `weighted_closing_speed` - Closing speed weighted by proximity

### 8. Play Context (6)
- `down` - Down number (1-4)
- `yardsToGo` - Yards to first down
- `defendersInBox` - Number of defenders in box
- `offenseFormation` - Offensive formation type
- `pff_passCoverageType` - Coverage scheme (Man/Zone)
- `passResult` - Play outcome (C/I/S/R/IN)

### 9. Metadata (10)
- `week`, `gameId`, `playId`
- `rusher_nflId`, `qb_nflId`
- `rusher_position` - Rush position (LEO, DT, etc.)
- `pff_hit`, `pff_hurry`, `pff_sack` - Individual pressure types
- `generated_pressure` - Target variable

---

## 📈 Top 10 Most Predictive Features

| Rank | Feature | Correlation |
|------|---------|-------------|
| 1 | `pff_hurry` | 0.7996 |
| 2 | `weighted_closing_speed` | 0.6252 |
| 3 | **`collision_intensity`** | **0.6192** |
| 4 | `collision_intensity_raw` | 0.6192 |
| 5 | `min_distance` | 0.5033 |
| 6 | `pff_hit` | 0.4193 |
| 7 | `max_closing_speed` | 0.3899 |
| 8 | `combined_speed_at_closest` | 0.3681 |
| 9 | `pff_sack` | 0.3523 |
| 10 | `avg_closing_speed` | 0.3341 |

---

## 🔬 Feature Engineering Methodology

### Collision Feature Calculation
Based on punt analytics collision validation approach:

1. **Extract tracking data** for rusher-QB pairs
2. **Find common frames** where both players tracked
3. **Calculate pairwise distances** at each frame
4. **Identify closest approach** point
5. **Compute collision features** at and around closest approach
6. **Normalize collision intensity** across all plays

### Key Innovation: Collision Intensity
Combines proximity and speed into single risk metric:

```
collision_intensity = (1 / (min_distance + 0.1)) * (combined_speed / max_speed)
```

Higher values indicate more dangerous collision potential.

---

## 🎓 Modeling Applications

### Recommended Models
1. **Logistic Regression** - Baseline, interpretable
2. **Random Forest** - Handle non-linear relationships
3. **XGBoost** - High performance on imbalanced data
4. **SVM** - Effective for high-dimensional features

### Class Imbalance Handling
- **SMOTE** - Synthetic minority oversampling
- **Class weights** - Weight pressure class higher
- **Stratified CV** - Preserve class ratios in folds
- **Precision-Recall focus** - Use PR-AUC instead of ROC-AUC

### Evaluation Metrics
- **Balanced Accuracy** - Equal weight to both classes
- **Precision & Recall** - Trade-off for rare positive class
- **F1 Score** - Harmonic mean of precision/recall
- **PR-AUC** - Area under precision-recall curve

---

## 🔗 Connection to Player Health

This dataset directly connects to player health because:

1. **QB pressure events** → Higher concussion risk
2. **Collision intensity** → Proxy for impact severity
3. **Closing speed** → Related to force of impact
4. **Min distance** → Inverse collision likelihood

**Goal:** Predict which pass rush collisions result in QB pressure, enabling:
- Identification of high-risk rush techniques
- Evaluation of protection scheme effectiveness
- Data-driven coaching interventions
- Rule changes to reduce dangerous plays

---

## 📚 Comparison to Punt Analytics

| Metric | Punt Analytics | Pass Rush |
|--------|----------------|-----------|
| Target | Concussion (1/0) | Pressure (1/0) |
| Positive Class | 9% | 11.64% |
| Key Feature | collision_intensity | collision_intensity |
| Correlation | 0.82 | 0.62 |
| Samples | ~308 collisions | 36,362 rushes |
| Imbalance | 1:10 | 1:8 |

**Advantage:** More balanced class distribution and much larger sample size!

---

## 🚀 Next Steps

1. ✅ **EDA Complete** - Comprehensive exploratory analysis
2. ✅ **Feature Engineering Complete** - 46 collision features
3. ✅ **Dataset Built** - All 8 weeks processed
4. ⏭️ **Model Training** - Build pressure prediction models
5. ⏭️ **Model Evaluation** - Compare performance across algorithms
6. ⏭️ **Feature Importance** - Identify most critical factors
7. ⏭️ **Technical Report** - Document findings for capstone

---

## 💡 Key Insights

1. **Collision intensity is highly predictive** (r=0.62) - Similar to punt analytics success
2. **Weighted closing speed** outperforms raw speed metrics
3. **Minimum distance** more predictive than average distance
4. **Play context** (down, formation) likely adds incremental value
5. **Class balance** is manageable (11.64% vs typical <1% in injury data)

---

## 📝 Citation

Data Source: NFL Big Data Bowl 2023
Kaggle Competition: https://www.kaggle.com/c/nfl-big-data-bowl-2023

Feature Engineering: Based on NFL Punt Analytics collision validation methodology

---

**Ready for modeling!** 🎯
