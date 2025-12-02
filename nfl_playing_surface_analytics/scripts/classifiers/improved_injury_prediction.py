"""
Improved NFL Injury Prediction Model

Key improvements over original approach:
1. Player-play level aggregation (not play-level)
2. Field type and position as features
3. Reduced padding (60 timesteps instead of 200)
4. Focal loss for class imbalance
5. Proper masking of padded sequences
6. Better evaluation metrics (PR-AUC, F1, confusion matrix)
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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, average_precision_score
)
import warnings
warnings.filterwarnings("ignore")

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU'))} GPU(s)")

# Import data loader
from nfl_playing_surface_analytics.scripts.data_loader import load_and_preprocess_data

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)


def prepare_injury_dataset(PlayList, PlayerTrackData,
                          injury_ratio=3, max_samples=50000, pad_length=60):
    """
    Prepare injury prediction dataset with player-play level aggregation.

    Parameters:
    -----------
    PlayList : pd.DataFrame
        Play-level data with injury labels
    PlayerTrackData : pd.DataFrame
        Player tracking data with movement features
    injury_ratio : int
        Ratio of non-injury to injury samples (e.g., 3 = 1:3 balance)
    max_samples : int
        Maximum number of non-injury samples to include
    pad_length : int
        Sequence padding length (timesteps)

    Returns:
    --------
    dict with X_sequence, X_static, y, player_keys, play_keys
    """

    print("="*70)
    print("PREPARING IMPROVED DATASET (PLAYER-PLAY LEVEL)")
    print("="*70)

    # Get unique player-play combinations with injuries
    injury_player_plays = PlayerTrackData[PlayerTrackData['DM_M1'] == 1][['PlayKey', 'PlayerKey']].drop_duplicates()
    print(f"\nInjury player-plays: {len(injury_player_plays)}")

    # Get non-injury player-play combinations
    non_injury_player_plays = PlayerTrackData[PlayerTrackData['DM_M1'] == 0][['PlayKey', 'PlayerKey']].drop_duplicates()

    # Sample non-injury to maintain desired ratio
    n_injury = len(injury_player_plays)
    n_non_injury = min(n_injury * injury_ratio, max_samples)

    sampled_non_injury = non_injury_player_plays.sample(n=n_non_injury, random_state=42)
    print(f"Non-injury player-plays sampled: {len(sampled_non_injury)}")
    print(f"Injury ratio: 1:{injury_ratio} (injury:non-injury)")

    # Combine injury and non-injury samples
    all_player_plays = pd.concat([injury_player_plays, sampled_non_injury])

    # Create unique identifier for filtering
    PlayerTrackData['player_play_id'] = PlayerTrackData['PlayKey'].astype(str) + '_' + PlayerTrackData['PlayerKey'].astype(str)
    all_player_plays['player_play_id'] = all_player_plays['PlayKey'].astype(str) + '_' + all_player_plays['PlayerKey'].astype(str)

    # Filter tracking data to selected player-plays
    selected_data = PlayerTrackData[PlayerTrackData['player_play_id'].isin(all_player_plays['player_play_id'])].copy()

    print(f"\nTotal player-play samples: {len(all_player_plays)}")
    print(f"Total tracking observations: {len(selected_data)}")

    # Merge with PlayList to get PlayerGamePlay (RosterPosition already in PlayerTrackData)
    print("\nMerging with PlayList for PlayerGamePlay...")
    selected_data = selected_data.merge(
        PlayList[['PlayKey', 'PlayerKey', 'PlayerGamePlay']],
        on=['PlayKey', 'PlayerKey'],
        how='left'
    )

    # Aggregate by player-play (NOT just play!)
    print("\nAggregating movement features by PLAYER-PLAY...")
    player_play_agg = selected_data.groupby(['PlayKey', 'PlayerKey']).agg({
        's': lambda x: list(x),          # Speed sequence for THIS player
        'a': lambda x: list(x),          # Acceleration sequence
        'sx': lambda x: list(x),         # X velocity
        'sy': lambda x: list(x),         # Y velocity
        'ax': lambda x: list(x),         # X acceleration
        'ay': lambda x: list(x),         # Y acceleration
        'DM_M1': 'first',                # Injury label for THIS player
        'FieldType': 'first',            # Field type
        'RosterPosition': 'first',       # Player position
        'PlayerGamePlay': 'first',       # Cumulative plays (exposure)
    }).reset_index()

    print(f"Aggregated dataset shape: {player_play_agg.shape}")
    print(f"Injury rate: {player_play_agg['DM_M1'].mean():.4f}")

    # Pad sequences
    print(f"\nPadding sequences to {pad_length} timesteps...")

    def pad_sequence(seq, pad_len=pad_length, pad_value=-1.0):
        """Pad or truncate sequence to fixed length."""
        seq = seq[:pad_len]  # Truncate if too long
        if len(seq) < pad_len:
            seq = seq + [pad_value] * (pad_len - len(seq))
        return np.array(seq, dtype=np.float32)

    # Apply padding to all movement features
    s_padded = np.array([pad_sequence(x) for x in player_play_agg['s'].values])
    a_padded = np.array([pad_sequence(x) for x in player_play_agg['a'].values])
    sx_padded = np.array([pad_sequence(x) for x in player_play_agg['sx'].values])
    sy_padded = np.array([pad_sequence(x) for x in player_play_agg['sy'].values])
    ax_padded = np.array([pad_sequence(x) for x in player_play_agg['ax'].values])
    ay_padded = np.array([pad_sequence(x) for x in player_play_agg['ay'].values])

    # Stack into sequence array (samples, timesteps, features)
    X_sequence = np.stack([s_padded, a_padded, sx_padded, sy_padded, ax_padded, ay_padded], axis=2)
    print(f"Sequence shape: {X_sequence.shape}")

    # Create static features (field type, position, exposure)
    print("\nCreating static features...")

    # Encode field type (0=Natural, 1=Synthetic)
    field_type_encoded = (player_play_agg['FieldType'] == 'Synthetic').astype(int).values

    # Encode position (one-hot for top positions, "Other" for rest)
    top_positions = ['Wide Receiver', 'Cornerback', 'Linebacker', 'Safety',
                     'Offensive Lineman', 'Defensive Lineman', 'Running Back']

    position_encoder = LabelEncoder()
    position_labels = player_play_agg['RosterPosition'].apply(
        lambda x: x if x in top_positions else 'Other'
    )
    position_encoded = position_encoder.fit_transform(position_labels)

    # Normalize exposure (cumulative plays)
    player_game_play_norm = player_play_agg['PlayerGamePlay'].values / 100.0  # Scale to ~0-1

    # Combine static features
    X_static = np.column_stack([
        field_type_encoded,
        position_encoded,
        player_game_play_norm
    ]).astype(np.float32)

    print(f"Static features shape: {X_static.shape}")
    print(f"Static features: [field_type, position, cumulative_plays]")

    # Create target (binary classification)
    y = player_play_agg['DM_M1'].values.astype(int)

    # Store metadata for GroupKFold
    player_keys = player_play_agg['PlayerKey'].values
    play_keys = player_play_agg['PlayKey'].values

    # Summary statistics
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"Total samples: {len(y)}")
    print(f"Injury samples: {y.sum()} ({y.mean()*100:.2f}%)")
    print(f"Non-injury samples: {(1-y).sum()} ({(1-y.mean())*100:.2f}%)")
    print(f"Unique players: {len(np.unique(player_keys))}")
    print(f"Unique plays: {len(np.unique(play_keys))}")
    print(f"\nSequence shape: {X_sequence.shape}")
    print(f"Static shape: {X_static.shape}")
    print(f"Average sequence length (non-padded): {(s_padded != -1).sum(axis=1).mean():.1f} timesteps")
    print("="*70)

    return {
        'X_sequence': X_sequence,
        'X_static': X_static,
        'y': y,
        'player_keys': player_keys,
        'play_keys': play_keys,
        'position_encoder': position_encoder,
        'field_type_names': ['Natural', 'Synthetic']
    }


def build_hybrid_lstm_model(seq_length, n_seq_features, n_static_features,
                            hidden_size=128, dropout=0.4):
    """
    Build hybrid LSTM model with sequence and static inputs.

    Architecture:
    - Bidirectional LSTM with attention for movement sequences
    - Dense layers for static features (field type, position, exposure)
    - Combined classifier
    """

    # Sequence input (movement trajectories)
    sequence_input = Input(shape=(seq_length, n_seq_features), name='sequence')

    # Masking layer to ignore padding (-1 values)
    masked = layers.Masking(mask_value=-1.0)(sequence_input)

    # Bidirectional LSTM layers
    lstm1 = layers.Bidirectional(
        layers.LSTM(hidden_size, return_sequences=True, dropout=dropout, recurrent_dropout=0.2)
    )(masked)

    lstm2 = layers.Bidirectional(
        layers.LSTM(hidden_size // 2, return_sequences=True, dropout=dropout, recurrent_dropout=0.2)
    )(lstm1)

    # Multi-head attention to focus on critical moments
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=hidden_size // 2)(lstm2, lstm2)
    attention = layers.GlobalAveragePooling1D()(attention)

    # Static features input (field type, position, exposure)
    static_input = Input(shape=(n_static_features,), name='static')
    static_dense = layers.Dense(32, activation='relu')(static_input)
    static_dense = layers.Dropout(dropout * 0.5)(static_dense)

    # Combine sequence and static features
    combined = layers.Concatenate()([attention, static_dense])

    # Classification head
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout * 0.75)(x)

    # Output layer (sigmoid for binary classification)
    output = layers.Dense(1, activation='sigmoid', name='injury')(x)

    # Build model
    model = Model(inputs=[sequence_input, static_input], outputs=output)

    return model


def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Focuses training on hard examples and down-weights easy negatives.
    """
    def loss(y_true, y_pred):
        # Cast y_true to float32 to match y_pred
        y_true = tf.cast(y_true, tf.float32)

        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Calculate focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy

        return tf.reduce_mean(loss)

    return loss


def train_with_cross_validation(X_sequence, X_static, y, player_keys,
                                n_folds=5, epochs=100, batch_size=64):
    """
    Train model with GroupKFold cross-validation.
    """

    print("\n" + "="*70)
    print("TRAINING WITH GROUP K-FOLD CROSS-VALIDATION")
    print("="*70)

    kf = GroupKFold(n_splits=n_folds)

    fold_results = []
    oof_predictions = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_sequence, y, groups=player_keys)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'='*70}")

        # Split data
        X_seq_train, X_seq_val = X_sequence[train_idx], X_sequence[val_idx]
        X_stat_train, X_stat_val = X_static[train_idx], X_static[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"Train samples: {len(y_train)} (Injury: {y_train.sum()}, {y_train.mean()*100:.2f}%)")
        print(f"Val samples: {len(y_val)} (Injury: {y_val.sum()}, {y_val.mean()*100:.2f}%)")

        # Build model
        model = build_hybrid_lstm_model(
            seq_length=X_sequence.shape[1],
            n_seq_features=X_sequence.shape[2],
            n_static_features=X_static.shape[1]  # X_static is 2D: (samples, features)
        )

        # Calculate class weights
        n_injury = y_train.sum()
        n_non_injury = len(y_train) - n_injury
        class_weight = {0: 1.0, 1: n_non_injury / n_injury}

        print(f"Class weights: {{0: 1.0, 1: {class_weight[1]:.2f}}}")

        # Compile with focal loss
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=focal_loss(gamma=2.0, alpha=0.25),
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )

        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_auc',
            patience=15,
            restore_best_weights=True,
            mode='max',
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            mode='max',
            verbose=1
        )

        # Train
        history = model.fit(
            [X_seq_train, X_stat_train],
            y_train,
            validation_data=([X_seq_val, X_stat_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        # Predict on validation fold
        val_preds = model.predict([X_seq_val, X_stat_val], verbose=0).flatten()
        oof_predictions[val_idx] = val_preds

        # Calculate metrics
        val_preds_binary = (val_preds > 0.5).astype(int)

        acc = accuracy_score(y_val, val_preds_binary)
        auc = roc_auc_score(y_val, val_preds)
        ap = average_precision_score(y_val, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, val_preds_binary, average='binary', zero_division=0
        )

        cm = confusion_matrix(y_val, val_preds_binary)

        print(f"\n{'='*70}")
        print(f"FOLD {fold + 1} RESULTS")
        print(f"{'='*70}")
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC: {auc:.4f}")
        print(f"PR-AUC: {ap:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

        fold_results.append({
            'fold': fold + 1,
            'accuracy': acc,
            'auc': auc,
            'pr_auc': ap,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        })

        # Save model
        model.save(f'injury_model_fold{fold+1}.keras')
        print(f"\nModel saved to: injury_model_fold{fold+1}.keras")

    return fold_results, oof_predictions


def evaluate_overall_performance(y_true, oof_predictions, fold_results):
    """
    Evaluate overall out-of-fold performance.
    """

    print("\n" + "="*70)
    print("OVERALL OUT-OF-FOLD PERFORMANCE")
    print("="*70)

    # Calculate overall metrics
    oof_binary = (oof_predictions > 0.5).astype(int)

    overall_acc = accuracy_score(y_true, oof_binary)
    overall_auc = roc_auc_score(y_true, oof_predictions)
    overall_ap = average_precision_score(y_true, oof_predictions)
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        y_true, oof_binary, average='binary', zero_division=0
    )

    overall_cm = confusion_matrix(y_true, oof_binary)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {overall_acc:.4f}")
    print(f"  ROC-AUC: {overall_auc:.4f}")
    print(f"  PR-AUC (Average Precision): {overall_ap:.4f}")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall: {overall_recall:.4f}")
    print(f"  F1-Score: {overall_f1:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              No Injury  Injury")
    print(f"Actual No Inj    {overall_cm[0,0]:5d}    {overall_cm[0,1]:5d}")
    print(f"       Injury    {overall_cm[1,0]:5d}    {overall_cm[1,1]:5d}")

    # Cross-validation statistics
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION STATISTICS")
    print(f"{'='*70}")

    cv_metrics = pd.DataFrame(fold_results)
    print(f"\n{cv_metrics[['fold', 'accuracy', 'auc', 'pr_auc', 'f1']].to_string(index=False)}")

    print(f"\nMean ± Std:")
    for metric in ['accuracy', 'auc', 'pr_auc', 'precision', 'recall', 'f1']:
        mean = cv_metrics[metric].mean()
        std = cv_metrics[metric].std()
        print(f"  {metric.capitalize():12s}: {mean:.4f} ± {std:.4f}")

    # Classification report
    print(f"\n{'='*70}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*70}")
    print(classification_report(y_true, oof_binary, target_names=['No Injury', 'Injury']))

    return {
        'overall_accuracy': overall_acc,
        'overall_auc': overall_auc,
        'overall_pr_auc': overall_ap,
        'overall_f1': overall_f1,
        'confusion_matrix': overall_cm,
        'cv_results': cv_metrics
    }


def main():
    """
    Main training pipeline.
    """

    print("\n" + "="*70)
    print("IMPROVED NFL INJURY PREDICTION MODEL")
    print("="*70)

    # Load data
    print("\nLoading data...")
    InjuryRecord, PlayList, PlayerTrackData, corr_term = load_and_preprocess_data()

    # Prepare dataset
    dataset = prepare_injury_dataset(
        PlayList,
        PlayerTrackData,
        injury_ratio=3,      # 1:3 injury:non-injury ratio
        max_samples=50000,   # Max non-injury samples
        pad_length=60        # Reduced from 200
    )

    # Train with cross-validation
    fold_results, oof_predictions = train_with_cross_validation(
        X_sequence=dataset['X_sequence'],
        X_static=dataset['X_static'],
        y=dataset['y'],
        player_keys=dataset['player_keys'],
        n_folds=5,
        epochs=100,
        batch_size=64
    )

    # Evaluate overall performance
    results = evaluate_overall_performance(
        y_true=dataset['y'],
        oof_predictions=oof_predictions,
        fold_results=fold_results
    )

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nBest Overall PR-AUC: {results['overall_pr_auc']:.4f}")
    print(f"Best Overall F1: {results['overall_f1']:.4f}")
    print("\nModels saved: injury_model_fold1.keras through injury_model_fold5.keras")


if __name__ == "__main__":
    main()
