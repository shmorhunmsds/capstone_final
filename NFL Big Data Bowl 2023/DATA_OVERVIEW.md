# 3. Data Overview

## Purpose

This document provides a comprehensive description of the data used in the NFL Pass Rush Collision Analytics project, including data sources, collection methods, summary statistics, data quality assessment, and ethical considerations.

---

## 3.1 Data Sources and Collection Methods

### Primary Data Source

**NFL Big Data Bowl 2023 (Kaggle Competition)**
- **URL**: https://www.kaggle.com/c/nfl-big-data-bowl-2023
- **Provider**: National Football League (NFL)
- **Focus**: Player health and safety analysis
- **Time Period**: NFL 2021 Season, Weeks 1-8
- **Collection Date**: Downloaded from Kaggle competition dataset (2023)

### Data Collection Methods

#### Player Tracking Data
- **Technology**: Next Gen Stats (NGS) player tracking system
- **Hardware**: Radio-frequency identification (RFID) sensors embedded in player shoulder pads
- **Sampling Rate**: 10 Hz (10 measurements per second)
- **Coordinate System**: Standardized football field coordinates (x, y position in yards)
- **Tracked Metrics**: Position (x, y), speed (s), acceleration (a), distance traveled (dis), orientation (o), direction (dir)
- **Validation**: NFL Quality Control team validates tracking accuracy

#### Professional Football Focus (PFF) Scouting Data
- **Method**: Manual film review by trained analysts
- **Coverage**: All passing plays during weeks 1-8
- **Annotations**: Player roles (Pass Rush, Pass Block, Coverage, etc.), outcomes (hits, hurries, sacks), blocking assignments
- **Quality Assurance**: Multi-analyst review process for consistency

#### Play-Level Metadata
- **Source**: Official NFL game data
- **Collection**: Automated capture from game management systems
- **Content**: Game situation (down, distance, score), formations, coverage schemes, play outcomes

#### Player Information
- **Source**: NFL official roster data
- **Content**: Demographics (height, weight, birthdate), position, college background

---

## 3.2 Data Description

### Dataset Files Overview

| File | Records | Purpose | Size |
|------|---------|---------|------|
| `week1.csv` - `week8.csv` | ~8.5M total | Player tracking data (10Hz sampling) | ~400MB total |
| `plays.csv` | 8,557 | Play-level metadata and context | ~5MB |
| `pffScoutingData.csv` | 188,254 | PFF analyst annotations | ~2MB |
| `players.csv` | 1,679 | Player demographics and positions | ~500KB |
| `games.csv` | 122 | Game-level metadata | ~10KB |

### Engineered Feature Dataset

**Final Dataset**: `pass_rush_collision_features_full.csv`
- **Total Samples**: 36,362 rush attempts
- **Total Features**: 46
- **Data Type**: Structured tabular data (CSV format)
- **Target Variable**: Binary classification (`generated_pressure`)

### Summary Statistics

#### Target Variable Distribution
- **Pressure Events**: 4,232 (11.64%)
- **No Pressure**: 32,130 (88.36%)
- **Class Imbalance Ratio**: 1:7.6 (pressure:no pressure)

#### Pressure Type Breakdown
- **Hits**: 823 (2.26% of all rushes, 19.4% of pressure events)
- **Hurries**: 2,824 (7.77% of all rushes, 66.7% of pressure events)
- **Sacks**: 585 (1.61% of all rushes, 13.8% of pressure events)

#### Temporal Coverage
- **Weeks**: 8 weeks of regular season play
- **Games**: 122 NFL games
- **Teams**: 32 NFL teams
- **Players Tracked**: 1,679 players
- **Unique Rushers**: Multiple defensive positions
- **Unique QBs**: Multiple quarterbacks

#### Data Volume by Week
- Week 1: 1,118,122 tracking records
- Week 2: 1,042,774 tracking records
- Week 3: 1,121,825 tracking records
- Week 4: 1,074,606 tracking records
- Week 5: 1,097,813 tracking records
- Week 6: 973,797 tracking records
- Week 7: 906,292 tracking records
- Week 8: 978,949 tracking records

---

## 3.3 Data Dictionary

### Feature Categories and Descriptions

#### 1. Distance Features (4 features)

| Feature | Type | Description | Unit | Range |
|---------|------|-------------|------|-------|
| `min_distance` | float | Closest approach between rusher and QB | yards | 0.5 - 30 |
| `avg_distance` | float | Average distance throughout play | yards | 2 - 40 |
| `distance_at_start` | float | Initial separation at snap | yards | 5 - 50 |
| `distance_at_end` | float | Final separation at play end | yards | 1 - 50 |

**Relevance**: Proximity metrics are fundamental to collision risk assessment. Minimum distance is the primary indicator of collision likelihood (correlation: 0.50 with pressure).

#### 2. Speed Features (6 features)

| Feature | Type | Description | Unit | Range |
|---------|------|-------------|------|-------|
| `rusher_max_speed` | float | Maximum rusher speed during play | yards/sec | 0 - 12 |
| `rusher_avg_speed` | float | Average rusher speed | yards/sec | 0 - 8 |
| `rusher_speed_at_closest` | float | Rusher speed at closest approach | yards/sec | 0 - 10 |
| `qb_max_speed` | float | Maximum QB speed during play | yards/sec | 0 - 12 |
| `qb_avg_speed` | float | Average QB speed | yards/sec | 0 - 6 |
| `qb_speed_at_closest` | float | QB speed at closest approach | yards/sec | 0 - 8 |

**Relevance**: Speed metrics capture the kinetic energy component of potential collisions. Higher combined speeds indicate greater collision severity.

#### 3. Acceleration Features (6 features)

| Feature | Type | Description | Unit | Range |
|---------|------|-------------|------|-------|
| `rusher_max_accel` | float | Maximum rusher acceleration | yards/sec² | -5 - 8 |
| `rusher_avg_accel` | float | Average rusher acceleration | yards/sec² | -2 - 4 |
| `rusher_accel_at_closest` | float | Rusher acceleration at closest point | yards/sec² | -5 - 6 |
| `qb_max_accel` | float | Maximum QB acceleration | yards/sec² | -5 - 8 |
| `qb_avg_accel` | float | Average QB acceleration | yards/sec² | -2 - 4 |
| `qb_accel_at_closest` | float | QB acceleration at closest point | yards/sec² | -5 - 6 |

**Relevance**: Acceleration indicates changes in velocity and burst movements, important for evasion and pursuit dynamics.

#### 4. Relative Motion Features (3 features)

| Feature | Type | Description | Unit | Range |
|---------|------|-------------|------|-------|
| `combined_speed_at_closest` | float | Sum of rusher and QB speeds at closest point | yards/sec | 0 - 18 |
| `max_closing_speed` | float | Maximum rate of distance decrease | yards/sec | 0 - 10 |
| `avg_closing_speed` | float | Average closing speed when approaching | yards/sec | 0 - 6 |

**Relevance**: Closing dynamics directly measure collision approach velocity, a key component of impact severity (correlation: 0.39-0.63 with pressure).

#### 5. Orientation Features (4 features)

| Feature | Type | Description | Unit | Range |
|---------|------|-------------|------|-------|
| `approach_angle` | float | Angle of approach vector | degrees | -180 - 180 |
| `rusher_orientation_at_closest` | float | Rusher body orientation at closest point | degrees | 0 - 360 |
| `qb_orientation_at_closest` | float | QB body orientation at closest point | degrees | 0 - 360 |
| `rusher_angle_alignment` | float | Alignment between rusher orientation and approach | degrees | 0 - 180 |

**Relevance**: Body orientation affects collision angles and contact surfaces, influencing injury biomechanics.

#### 6. Temporal Features (3 features)

| Feature | Type | Description | Unit | Range |
|---------|------|-------------|------|-------|
| `time_to_closest_approach` | float | Time from snap to closest approach | seconds | 0 - 8 |
| `total_frames` | int | Number of frames tracked for this rush | frames | 3 - 80 |
| `play_duration` | float | Total play length | seconds | 0.3 - 8.0 |

**Relevance**: Temporal dynamics indicate time-to-contact and play development patterns.

#### 7. Collision Intensity Features (3 features) ⭐ KEY FEATURES

| Feature | Type | Description | Unit | Range |
|---------|------|-------------|------|-------|
| `collision_intensity` | float | **Normalized collision intensity metric** | normalized | 0 - 1 |
| `collision_intensity_raw` | float | Raw collision intensity score | raw score | 0 - 50 |
| `weighted_closing_speed` | float | Closing speed weighted by proximity | composite | 0 - 5 |

**Formula**:
```
collision_intensity = (1 / (min_distance + 0.1)) × (combined_speed / max_speed_in_dataset)
```

**Relevance**: The primary predictive feature (correlation: 0.62 with pressure). Combines proximity and speed into a single collision risk metric, adapted from validated punt analytics methodology.

#### 8. Play Context Features (6 features)

| Feature | Type | Description | Values | Purpose |
|---------|------|-------------|--------|---------|
| `down` | int | Down number | 1, 2, 3, 4 | Game situation |
| `yardsToGo` | int | Yards to first down | 1 - 30 | Distance pressure |
| `defendersInBox` | int | Number of defenders in box | 3 - 9 | Rush volume indicator |
| `offenseFormation` | categorical | Offensive formation | SHOTGUN, EMPTY, SINGLEBACK, etc. | Protection scheme |
| `pff_passCoverageType` | categorical | Coverage scheme | Man, Zone, Other | Secondary coverage |
| `passResult` | categorical | Play outcome | C (complete), I (incomplete), S (sack), etc. | Play result |

**Relevance**: Contextual features provide game situation information that influences pressure likelihood and protection schemes.

#### 9. Metadata Features (10 features)

| Feature | Type | Description | Purpose |
|---------|------|-------------|---------|
| `week` | int | Week number (1-8) | Temporal tracking |
| `gameId` | int | Unique game identifier | Join key |
| `playId` | int | Unique play identifier | Join key |
| `rusher_nflId` | int | Rusher player ID | Player identification |
| `qb_nflId` | int | QB player ID | Player identification |
| `rusher_position` | categorical | Rush position (LEO, DT, LILB, etc.) | Position analysis |
| `generated_pressure` | binary | **TARGET VARIABLE** (0/1) | Model target |
| `pff_hit` | binary | QB hit occurred (0/1) | Pressure type |
| `pff_hurry` | binary | QB hurry occurred (0/1) | Pressure type |
| `pff_sack` | binary | Sack occurred (0/1) | Pressure type |

**Target Variable Definition**:
```python
generated_pressure = (pff_hit == 1) OR (pff_hurry == 1) OR (pff_sack == 1)
```

---

## 3.4 Data Quality Assessment

### Missing Values Analysis

#### Raw Data Files
1. **Tracking Data (week1-8.csv)**:
   - **Missing Values**: Minimal (<0.1%)
   - **Pattern**: Occasional missing values in acceleration (`a`) when player is stationary
   - **Handling**: Filled with 0 for stationary periods

2. **Play Metadata (plays.csv)**:
   - **Missing Values**: ~2% in optional fields
   - **Fields Affected**: `penaltyYards`, `foulName1-3`, `foulNFLId1-3`
   - **Impact**: Not used in modeling; no action needed

3. **PFF Scouting Data**:
   - **Missing Values**: ~15% in `pff_nflIdBlockedPlayer` (blocking assignments)
   - **Pattern**: Unassigned blocks or coverage rotations
   - **Handling**: Not used in rush-QB collision features

4. **Player Data**:
   - **Missing Values**: None
   - **Completeness**: 100% for all players tracked

#### Engineered Feature Dataset

**Final Dataset Quality**: `pass_rush_collision_features_full.csv`

| Metric | Value | Assessment |
|--------|-------|------------|
| Total Samples | 36,362 | ✅ Excellent sample size |
| Missing Values | 0 | ✅ Complete dataset |
| Duplicate Rows | 0 | ✅ No duplicates |
| Feature Completeness | 100% | ✅ All features calculated |
| Target Variable Distribution | 11.64% positive | ⚠️ Imbalanced (handled) |

**Exclusion Criteria Applied**:
- Rush attempts with <3 common tracking frames between rusher and QB: ~10% excluded
- Rush attempts without QB tracking data: ~5% excluded
- Invalid tracking coordinates (out of field bounds): <0.1% excluded

### Data Inconsistencies

#### Identified Issues and Resolutions

1. **Coordinate System Inconsistency** (Week 1-2)
   - **Issue**: Play direction not standardized across games
   - **Resolution**: Standardized all plays to "right" direction using `playDirection` field
   - **Impact**: Resolved; no residual issues

2. **Frame Synchronization** (All Weeks)
   - **Issue**: Players not always tracked in same frames
   - **Resolution**: Used frame intersection to find common tracking frames
   - **Impact**: Robust feature calculation; handles missing frames gracefully

3. **Position Label Variations** (PFF Data)
   - **Issue**: Multiple position labels for similar roles (e.g., "LOLB" vs "ROLB")
   - **Resolution**: Maintained original labels; grouped in analysis if needed
   - **Impact**: Minimal; models handle categorical variations

4. **Measurement Units**
   - **Issue**: Mixed documentation of units in original data
   - **Resolution**: Verified all measurements: distance (yards), speed (yards/sec), acceleration (yards/sec²)
   - **Impact**: Resolved; all units standardized

### Data Limitations

#### 1. Sample Selection Bias
- **Limitation**: Only passing plays included; excludes run plays
- **Impact**: Model specific to pass rush scenarios only
- **Mitigation**: Clear scope definition; documented in methodology

#### 2. Temporal Limitation
- **Limitation**: Only 8 weeks of one season (2021)
- **Impact**: May not generalize across full seasons or rule changes
- **Mitigation**: Large sample size (36,362) provides statistical power within this timeframe

#### 3. Class Imbalance
- **Limitation**: 88.36% negative class (no pressure)
- **Impact**: Models may bias toward predicting "no pressure"
- **Mitigation**:
  - Used `balanced_accuracy` as primary metric
  - Applied class weights in models
  - Evaluated with precision-recall curves

#### 4. Tracking Accuracy
- **Limitation**: RFID tracking has ~1 foot margin of error
- **Impact**: Small errors in distance measurements
- **Mitigation**:
  - Error is random, not systematic
  - Smoothed with averaging over multiple frames
  - Impact minimal on aggregate statistics

#### 5. PFF Annotation Subjectivity
- **Limitation**: Human analysts define "hit", "hurry", "sack"
- **Impact**: Some subjectivity in target variable labels
- **Mitigation**:
  - PFF uses standardized definitions
  - Multi-analyst review process
  - Industry-standard labels used across NFL

#### 6. Feature Engineering Assumptions
- **Limitation**: Assumes straight-line distance as collision proxy
- **Impact**: Doesn't account for obstacles (blockers) between rusher and QB
- **Mitigation**:
  - Validated approach against punt analytics
  - Strong feature correlations (r=0.62) validate approach
  - Play context features partially account for blocking

### Data Validation Steps

✅ **Completed Validation Checks**:

1. **Range Validation**: All numeric features within physically plausible ranges
2. **Consistency Checks**: gameId/playId combinations match across files
3. **Distribution Analysis**: No extreme outliers beyond expected NFL performance
4. **Correlation Validation**: Expected relationships confirmed (e.g., min_distance ↔ collision_intensity)
5. **Target Variable Validation**: Pressure rates match PFF season-long statistics (~12%)
6. **Feature Engineering Validation**: Calculated features match manual spot checks
7. **Reproducibility**: Full pipeline rerun produces identical results

---

## 3.5 Ethical Considerations

### Player Privacy and Consent

#### Data Collection
- **Consent**: Players consented to tracking data collection as part of NFL employment agreements
- **Disclosure**: Players informed of NGS system and data usage for health/safety initiatives
- **Ownership**: NFL owns tracking data; players retain personal data rights under CBA

#### Data Sharing
- **Public Release**: NFL released anonymized subset for Big Data Bowl competition
- **Restrictions**: Player names included but limited to public roster information
- **Sensitive Data**: Medical records, personal contact info, contract details NOT included

#### Research Use
- **Purpose**: This project uses publicly released competition data
- **Scope**: Analysis focused on game performance, not personal characteristics
- **Identification**: Player IDs linked only to public roster data (name, position, college)

### Player Health and Safety

#### Injury Risk Assessment
- **Benefit**: Identifying high-risk collision patterns can inform injury prevention
- **Harm Mitigation**: Analysis aims to REDUCE injury risk, not evaluate individual players
- **Application**: Insights for coaching, training, rule changes—not player evaluation

#### Potential Harms
1. **Performance Stigma**: Models should not be used to label players as "injury-prone"
   - **Mitigation**: Focus on collision dynamics, not player characteristics

2. **Employment Discrimination**: Risk scores should not affect roster decisions
   - **Mitigation**: Research-only context; not shared with NFL teams

3. **Misinterpretation**: Pressure prediction ≠ injury prediction
   - **Mitigation**: Clear documentation that models predict pressure events, not injuries

### Fairness and Bias Considerations

#### Position Bias
- **Observation**: Different positions have different pressure rates (8-30%)
- **Consideration**: Models may perform differently across positions
- **Mitigation**: Position included as feature; model evaluation stratified by position

#### Data Representation
- **Sample**: 8 weeks of 2021 season, all 32 teams, diverse player pool
- **Diversity**: NFL player demographics (various colleges, backgrounds, countries)
- **Limitation**: Only 2021 season; may not generalize to other eras

#### Algorithmic Fairness
- **Protected Classes**: Age, race, college background NOT used as features
- **Performance Metrics**: No evidence of biased predictions across player demographics
- **Transparency**: Full methodology and features documented for reproducibility

### Research Ethics

#### Transparency
- **Open Methods**: All code, features, and methods documented in public repository
- **Reproducibility**: Full pipeline reproducible from raw data
- **Limitations**: Known limitations clearly stated in documentation

#### Responsible Communication
- **Accurate Reporting**: Results reported with confidence intervals and limitations
- **Avoid Overgeneralization**: Scope limited to 2021 pass rush scenarios
- **Stakeholder Impact**: Findings shared with focus on health/safety improvements

#### Dual Use Concerns
- **Beneficial Use**: Injury prevention, safer play design, rule improvements
- **Potential Misuse**: Player evaluation, contract negotiations, competitive advantage
- **Mitigation**:
  - Research context emphasized
  - Recommend use for aggregate coaching insights, not individual player assessment
  - Advocate for NFL to control application of collision risk models

### Data Governance

#### Data Security
- **Storage**: Local storage; no cloud uploads of derived datasets
- **Access**: Research use only; no sharing of engineered features beyond academic review
- **Retention**: Data retained only for duration of capstone project

#### Compliance
- **Competition Rules**: Usage compliant with Kaggle competition terms of service
- **Academic Ethics**: Project approved under university research ethics guidelines
- **NFL Policies**: Public data use aligned with NFL Big Data Bowl goals

### Social Impact

#### Positive Impacts
1. **Injury Prevention**: Insights may reduce QB injuries through better protection schemes
2. **Player Safety**: Risk identification can inform equipment and rule improvements
3. **Scientific Knowledge**: Validates collision modeling methodology for other sports

#### Potential Negative Impacts
1. **Risk Compensation**: Teams might take more risks if they believe they can predict outcomes
2. **Performance Pressure**: Players might alter play style based on risk metrics
3. **Legal Liability**: Collision models could be used in injury litigation

#### Recommendations for Responsible Use
1. **Team-Level Analysis Only**: Apply insights to scheme design, not individual player evaluation
2. **Complementary Tool**: Use alongside medical expertise, not as sole decision-maker
3. **Continuous Monitoring**: Update models as game evolves and new data becomes available
4. **Player Education**: Share insights with players to inform their own safety practices
5. **Regulatory Involvement**: Work with NFL Competition Committee for rule adjustments

---

## 3.6 Data Statement Summary

**Dataset Name**: NFL Pass Rush Collision Analytics Dataset
**Version**: 1.0
**Last Updated**: October 12, 2024
**Data Source**: NFL Big Data Bowl 2023 (Kaggle)
**Time Period**: NFL 2021 Season, Weeks 1-8
**Sample Size**: 36,362 rush attempts
**Features**: 46 engineered collision dynamics features
**Target Variable**: Binary (pressure: 11.64%, no pressure: 88.36%)
**Missing Values**: 0% (complete dataset)
**Quality**: High - validated against PFF statistics and NFL performance norms
**Ethical Considerations**: Player-consented public data; research-only use; privacy-preserving
**Limitations**: Single season, pass plays only, class imbalance, tracking measurement error (~1 ft)
**Intended Use**: Academic research, injury prevention insights, player health & safety analysis
**Not Intended For**: Individual player evaluation, contract negotiations, competitive scouting

---

## References

1. NFL Next Gen Stats. (2021). "Player Tracking Methodology." NFL Operations.
2. Professional Football Focus. (2021). "PFF Grading & Data Collection Methodology."
3. NFL Big Data Bowl 2023. Kaggle Competition. https://www.kaggle.com/c/nfl-big-data-bowl-2023
4. NFL Punt Analytics Competition. (2020). Collision detection methodology validation.
5. NFL Players Association. (2020). "Collective Bargaining Agreement: Data Rights and Privacy."

---

**Document Version**: 1.0
**Author**: Patrick Shmorhun
**Project**: Capstone Technical Report - NFL Player Health & Safety Analytics
**Date**: October 13, 2024
