"""
NFL Injury Prediction - Model Comparison

Compares 3 approaches:
1. XGBoost with engineered features (best for small datasets)
2. Simple dense network (simpler deep learning)
3. Cox Proportional Hazards + ML features (survival analysis)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, average_precision_score,
    roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings("ignore")

# Import data loader
from nfl_playing_surface_analytics.scripts.data_loader import load_and_preprocess_data

# Set random seeds
np.random.seed(42)

print("="*70)
print("NFL INJURY PREDICTION - MODEL COMPARISON")
print("="*70)

# Load data
print("\nLoading data...")
InjuryRecord, PlayList, PlayerTrackData, corr_term = load_and_preprocess_data()


def prepare_player_play_dataset(PlayList, PlayerTrackData, injury_ratio=3, max_samples=50000):
    """
    Prepare player-play level dataset with movement feature aggregation.
    """
    print("\n" + "="*70)
    print("PREPARING PLAYER-PLAY DATASET")
    print("="*70)

    # Get injury and non-injury player-plays
    injury_player_plays = PlayerTrackData[PlayerTrackData['DM_M1'] == 1][['PlayKey', 'PlayerKey']].drop_duplicates()
    non_injury_player_plays = PlayerTrackData[PlayerTrackData['DM_M1'] == 0][['PlayKey', 'PlayerKey']].drop_duplicates()

    # Sample non-injury to maintain ratio
    n_injury = len(injury_player_plays)
    n_non_injury = min(n_injury * injury_ratio, max_samples)
    sampled_non_injury = non_injury_player_plays.sample(n=n_non_injury, random_state=42)

    print(f"Injury player-plays: {n_injury}")
    print(f"Non-injury player-plays sampled: {n_non_injury}")
    print(f"Ratio: 1:{injury_ratio}")

    # Combine
    all_player_plays = pd.concat([injury_player_plays, sampled_non_injury])

    # Create identifier for filtering
    PlayerTrackData['player_play_id'] = PlayerTrackData['PlayKey'].astype(str) + '_' + PlayerTrackData['PlayerKey'].astype(str)
    all_player_plays['player_play_id'] = all_player_plays['PlayKey'].astype(str) + '_' + all_player_plays['PlayerKey'].astype(str)

    # Filter tracking data
    selected_data = PlayerTrackData[PlayerTrackData['player_play_id'].isin(all_player_plays['player_play_id'])].copy()

    # Merge with PlayList for additional features
    selected_data = selected_data.merge(
        PlayList[['PlayKey', 'PlayerKey', 'PlayerGamePlay']],
        on=['PlayKey', 'PlayerKey'],
        how='left'
    )

    print(f"Total tracking observations: {len(selected_data)}")

    return selected_data


def engineer_features(selected_data):
    """
    Engineer aggregated features from movement sequences.
    Creates statistical summaries that work well with traditional ML.
    """
    print("\n" + "="*70)
    print("ENGINEERING FEATURES FROM MOVEMENT DATA")
    print("="*70)

    # Aggregate by player-play with statistical features
    features = selected_data.groupby(['PlayKey', 'PlayerKey']).agg({
        # Speed features
        's': ['mean', 'std', 'max', 'min', lambda x: np.percentile(x, 75), lambda x: np.percentile(x, 25)],
        # Acceleration features
        'a': ['mean', 'std', 'max', 'min', lambda x: np.percentile(x, 90)],
        # Velocity components
        'sx': ['mean', 'std', 'max', 'min'],
        'sy': ['mean', 'std', 'max', 'min'],
        # Acceleration components
        'ax': ['mean', 'std', 'max'],
        'ay': ['mean', 'std', 'max'],
        'a_fwd': ['mean', 'max', 'min'],
        'a_sid': ['mean', 'max', 'min', 'std'],
        # Metadata
        'DM_M1': 'first',
        'FieldType': 'first',
        'RosterPosition': 'first',
        'PlayerGamePlay': 'first',
        'time': 'count'  # Number of tracking observations
    })

    # Flatten column names
    features.columns = ['_'.join(col).strip() for col in features.columns.values]
    features = features.reset_index()

    # Rename time_count to seq_length
    features = features.rename(columns={'time_count': 'seq_length'})

    print(f"Engineered features shape: {features.shape}")
    print(f"Number of features: {features.shape[1] - 2}")  # Minus PlayKey, PlayerKey

    # Calculate additional derived features
    features['speed_range'] = features['s_max'] - features['s_min']
    features['accel_range'] = features['a_max'] - features['a_min']
    features['speed_cv'] = features['s_std'] / (features['s_mean'] + 1e-7)  # Coefficient of variation
    features['lateral_dominance'] = np.abs(features['a_sid_mean']) / (features['a_fwd_mean'] + 1e-7)

    # Encode categorical variables
    features['FieldSynthetic'] = (features['FieldType_first'] == 'Synthetic').astype(int)

    # Encode position
    top_positions = ['Wide Receiver', 'Cornerback', 'Linebacker', 'Safety',
                     'Offensive Lineman', 'Defensive Lineman', 'Running Back']
    features['RosterPosition_clean'] = features['RosterPosition_first'].apply(
        lambda x: x if x in top_positions else 'Other'
    )
    position_encoder = LabelEncoder()
    features['Position_encoded'] = position_encoder.fit_transform(features['RosterPosition_clean'])

    print(f"Final features shape: {features.shape}")

    return features, position_encoder


# ============================================================================
# OPTION 1: XGBoost with Engineered Features
# ============================================================================

def train_xgboost_model(features, n_folds=5):
    """
    Train XGBoost classifier on engineered features.
    Best approach for small datasets with tabular data.
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("\nXGBoost not installed. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'xgboost'])
        import xgboost as xgb

    print("\n" + "="*70)
    print("OPTION 1: XGBoost with Engineered Features")
    print("="*70)

    # Select feature columns (exclude metadata)
    exclude_cols = ['PlayKey', 'PlayerKey', 'DM_M1_first', 'FieldType_first',
                    'RosterPosition_first', 'RosterPosition_clean']
    feature_cols = [col for col in features.columns if col not in exclude_cols]

    X = features[feature_cols].values
    y = features['DM_M1_first'].values
    player_keys = features['PlayerKey'].values

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution: Injury={y.sum()}, No Injury={(1-y).sum()}")

    # Handle any NaN/Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Cross-validation
    kf = GroupKFold(n_splits=n_folds)
    oof_predictions = np.zeros(len(y))
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y, groups=player_keys)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'='*70}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"Train: {len(y_train)} samples (Injury: {y_train.sum()})")
        print(f"Val: {len(y_val)} samples (Injury: {y_val.sum()})")

        # Calculate scale_pos_weight for imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42 + fold,
            eval_metric='aucpr',
            early_stopping_rounds=20
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Predictions
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs > 0.5).astype(int)
        oof_predictions[val_idx] = val_probs

        # Metrics
        acc = accuracy_score(y_val, val_preds)
        auc = roc_auc_score(y_val, val_probs)
        ap = average_precision_score(y_val, val_probs)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, val_preds, average='binary', zero_division=0
        )

        print(f"\nFold {fold + 1} Results:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  ROC-AUC: {auc:.4f}")
        print(f"  PR-AUC: {ap:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")

        fold_results.append({
            'fold': fold + 1,
            'accuracy': acc,
            'auc': auc,
            'pr_auc': ap,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

        # Feature importance (only for first fold)
        if fold == 0:
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\nTop 10 Most Important Features:")
            print(feature_importance.head(10).to_string(index=False))

    # Overall results
    cv_results = pd.DataFrame(fold_results)

    print(f"\n{'='*70}")
    print("XGBoost Overall Results")
    print(f"{'='*70}")
    print(f"\nCross-Validation Metrics:")
    for metric in ['accuracy', 'auc', 'pr_auc', 'precision', 'recall', 'f1']:
        mean = cv_results[metric].mean()
        std = cv_results[metric].std()
        print(f"  {metric.upper():12s}: {mean:.4f} ± {std:.4f}")

    return {
        'model_name': 'XGBoost',
        'oof_predictions': oof_predictions,
        'cv_results': cv_results,
        'y_true': y
    }


# ============================================================================
# OPTION 3: Simple Dense Network
# ============================================================================

def train_simple_dense_model(selected_data, n_folds=5):
    """
    Train simple dense network on flattened movement sequences.
    Simpler than LSTM but still leverages deep learning.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    print("\n" + "="*70)
    print("OPTION 3: Simple Dense Network on Flattened Sequences")
    print("="*70)

    # Prepare sequences (simpler than before)
    print("\nPreparing sequences...")

    # Aggregate by player-play
    player_play_agg = selected_data.groupby(['PlayKey', 'PlayerKey']).agg({
        's': lambda x: list(x)[:30],  # Take only first 30 timesteps
        'a': lambda x: list(x)[:30],
        'sx': lambda x: list(x)[:30],
        'sy': lambda x: list(x)[:30],
        'DM_M1': 'first',
        'FieldType': 'first',
        'RosterPosition': 'first',
        'PlayerGamePlay': 'first',
    }).reset_index()

    # Pad sequences to fixed length
    pad_len = 30

    def pad_seq(seq):
        if len(seq) < pad_len:
            return seq + [0.0] * (pad_len - len(seq))
        return seq[:pad_len]

    s_padded = np.array([pad_seq(x) for x in player_play_agg['s'].values])
    a_padded = np.array([pad_seq(x) for x in player_play_agg['a'].values])
    sx_padded = np.array([pad_seq(x) for x in player_play_agg['sx'].values])
    sy_padded = np.array([pad_seq(x) for x in player_play_agg['sy'].values])

    # Flatten all features into single vector
    X_flattened = np.column_stack([
        s_padded, a_padded, sx_padded, sy_padded
    ])

    # Add static features
    field_type = (player_play_agg['FieldType'] == 'Synthetic').astype(float).values.reshape(-1, 1)

    # Encode position
    top_positions = ['Wide Receiver', 'Cornerback', 'Linebacker', 'Safety']
    position = player_play_agg['RosterPosition'].apply(
        lambda x: top_positions.index(x) if x in top_positions else len(top_positions)
    ).values.reshape(-1, 1)

    exposure = (player_play_agg['PlayerGamePlay'].values / 100.0).reshape(-1, 1)

    X = np.column_stack([X_flattened, field_type, position, exposure])
    y = player_play_agg['DM_M1'].values
    player_keys = player_play_agg['PlayerKey'].values

    print(f"Input shape: {X.shape}")
    print(f"Features: {pad_len * 4} movement + 3 static = {X.shape[1]} total")

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Cross-validation
    kf = GroupKFold(n_splits=n_folds)
    oof_predictions = np.zeros(len(y))
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y, groups=player_keys)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'='*70}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"Train: {len(y_train)} (Injury: {y_train.sum()})")
        print(f"Val: {len(y_val)} (Injury: {y_val.sum()})")

        # Build simple model
        model = keras.Sequential([
            layers.Input(shape=(X.shape[1],)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])

        # Class weights
        class_weight = {0: 1.0, 1: (y_train == 0).sum() / (y_train == 1).sum()}

        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss='binary_crossentropy',
            metrics=[keras.metrics.AUC(name='auc'), keras.metrics.AUC(curve='PR', name='pr_auc')]
        )

        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            class_weight=class_weight,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_pr_auc', patience=10, mode='max', restore_best_weights=True)
            ]
        )

        # Predictions
        val_probs = model.predict(X_val, verbose=0).flatten()
        val_preds = (val_probs > 0.5).astype(int)
        oof_predictions[val_idx] = val_probs

        # Metrics
        acc = accuracy_score(y_val, val_preds)
        auc = roc_auc_score(y_val, val_probs)
        ap = average_precision_score(y_val, val_probs)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, val_preds, average='binary', zero_division=0
        )

        print(f"\nFold {fold + 1} Results:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  ROC-AUC: {auc:.4f}")
        print(f"  PR-AUC: {ap:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")

        fold_results.append({
            'fold': fold + 1,
            'accuracy': acc,
            'auc': auc,
            'pr_auc': ap,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    cv_results = pd.DataFrame(fold_results)

    print(f"\n{'='*70}")
    print("Simple Dense Network Overall Results")
    print(f"{'='*70}")
    print(f"\nCross-Validation Metrics:")
    for metric in ['accuracy', 'auc', 'pr_auc', 'precision', 'recall', 'f1']:
        mean = cv_results[metric].mean()
        std = cv_results[metric].std()
        print(f"  {metric.upper():12s}: {mean:.4f} ± {std:.4f}")

    return {
        'model_name': 'Simple Dense Network',
        'oof_predictions': oof_predictions,
        'cv_results': cv_results,
        'y_true': y
    }


# ============================================================================
# OPTION 4: Cox PH + ML Features Documentation
# ============================================================================

def document_cox_ml_approach():
    """
    Document how to use Cox Proportional Hazards with Machine Learning.
    This is a hybrid approach combining survival analysis with feature learning.
    """
    print("\n" + "="*70)
    print("OPTION 4: Cox Proportional Hazards + Machine Learning")
    print("="*70)

    explanation = """

Cox Proportional Hazards (Cox PH) is a SURVIVAL ANALYSIS model, not traditional ML.
However, you CAN combine Cox PH with ML in several ways:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
APPROACH 1: Cox PH with ML-Engineered Features
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use ML (Random Forest, autoencoders) to create features, then feed to Cox PH:

    from lifelines import CoxPHFitter
    from sklearn.ensemble import RandomForestRegressor

    # Step 1: Use RF to create learned features
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_movement, y_dummy)  # Unsupervised or weakly supervised

    learned_features = rf.apply(X_movement)  # Leaf indices as features

    # Step 2: Combine with static features
    cox_features = pd.concat([
        pd.DataFrame(learned_features),
        df[['FieldSynthetic', 'PlayerGamePlay']]
    ], axis=1)

    # Step 3: Fit Cox PH
    cph = CoxPHFitter()
    cph.fit(cox_features, duration_col='PlayerGamePlay', event_col='DM_M1')

    # Advantages:
    - Properly handles censoring (players who don't get injured)
    - Naturally models time-to-event
    - Interpretable hazard ratios

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
APPROACH 2: DeepSurv (Deep Learning + Cox PH)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use neural networks to learn features, optimize Cox partial likelihood:

    from pycox.models import CoxPH
    import torchtuples as tt

    # Neural network for feature learning
    net = tt.practical.MLPVanilla(
        in_features=X.shape[1],
        num_nodes=[128, 64],
        out_features=1,  # Outputs log-hazard
        batch_norm=True,
        dropout=0.3
    )

    # DeepSurv model (Cox PH loss with neural net)
    model = CoxPH(net, tt.optim.Adam)

    # Train with Cox partial likelihood loss
    model.fit(X_train, (durations_train, events_train),
              epochs=100, batch_size=64)

    # Predict hazards
    hazards = model.predict_surv_df(X_test)

    # Advantages:
    - Learns complex non-linear relationships
    - Still outputs interpretable survival curves
    - Handles censoring properly

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
APPROACH 3: Two-Stage Model (Classification → Survival)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stage 1: ML classifier predicts injury risk
Stage 2: Cox PH refines timing

    # Stage 1: XGBoost predicts "will player get injured in next N plays?"
    xgb_model.fit(X_train, y_train)
    injury_risk_score = xgb_model.predict_proba(X)[:, 1]

    # Stage 2: Cox PH models WHEN injury occurs given risk score
    cox_data = df[['PlayerGamePlay', 'DM_M1', 'FieldSynthetic']].copy()
    cox_data['ML_RiskScore'] = injury_risk_score

    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col='PlayerGamePlay', event_col='DM_M1')

    # Advantages:
    - Combines strengths of both approaches
    - ML captures complex patterns
    - Cox PH handles time-to-event properly

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY COX PH FOR THIS PROBLEM?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. HANDLES CENSORING:
   - Most players DON'T get injured (censored observations)
   - Standard ML treats "no injury" = "will never get injured" (wrong!)
   - Cox PH treats it as "hasn't gotten injured YET" (correct!)

2. TIME-TO-EVENT MODELING:
   - Natural fit: "How many plays until injury?"
   - Accounts for cumulative exposure (PlayerGamePlay)
   - Can incorporate time-varying covariates

3. INTERPRETABLE HAZARD RATIOS:
   - "Synthetic turf increases injury hazard by 1.66x"
   - Directly answers research question
   - Publishable, clinically meaningful

4. BETTER FOR RARE EVENTS:
   - Designed for low event rates (104 injuries)
   - More statistical power than binary classification
   - Proper uncertainty quantification

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDED HYBRID APPROACH FOR YOUR DATA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    1. Engineer movement features (max_speed, accel_bursts, cutting_events)
    2. Use XGBoost to select most important features
    3. Feed selected features to Cox PH for final model
    4. Compare hazards: Synthetic vs. Natural grass

This gives you:
✓ ML's pattern recognition power
✓ Cox PH's proper survival modeling
✓ Interpretable hazard ratios for publication
✓ Handles censoring and time-to-event correctly

Would you like me to implement this hybrid approach?
"""

    print(explanation)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Prepare data
    selected_data = prepare_player_play_dataset(PlayList, PlayerTrackData, injury_ratio=3)

    # Engineer features for XGBoost
    features, position_encoder = engineer_features(selected_data)

    # Run all models
    results = {}

    # Option 1: XGBoost
    results['xgboost'] = train_xgboost_model(features, n_folds=5)

    # Option 3: Simple Dense Network
    results['dense'] = train_simple_dense_model(selected_data, n_folds=5)

    # Option 4: Document Cox PH approach
    document_cox_ml_approach()

    # Compare results
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)

    comparison = pd.DataFrame({
        'Model': [results['xgboost']['model_name'], results['dense']['model_name']],
        'PR-AUC': [
            results['xgboost']['cv_results']['pr_auc'].mean(),
            results['dense']['cv_results']['pr_auc'].mean()
        ],
        'ROC-AUC': [
            results['xgboost']['cv_results']['auc'].mean(),
            results['dense']['cv_results']['auc'].mean()
        ],
        'F1': [
            results['xgboost']['cv_results']['f1'].mean(),
            results['dense']['cv_results']['f1'].mean()
        ],
        'Precision': [
            results['xgboost']['cv_results']['precision'].mean(),
            results['dense']['cv_results']['precision'].mean()
        ],
        'Recall': [
            results['xgboost']['cv_results']['recall'].mean(),
            results['dense']['cv_results']['recall'].mean()
        ]
    })

    print("\n" + comparison.to_string(index=False))

    best_model = comparison.loc[comparison['PR-AUC'].idxmax(), 'Model']
    print(f"\n🏆 Best Model: {best_model} (Highest PR-AUC)")

    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    print("""
    1. XGBoost likely performs best for this small dataset
    2. Consider Cox PH + ML hybrid for publication-quality analysis
    3. Deep learning needs 10-100x more data to be effective
    4. Feature engineering is more important than model complexity here
    """)


if __name__ == "__main__":
    main()
