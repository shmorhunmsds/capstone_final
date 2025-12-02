"""
NFL Playing Surface Analytics - Data Loading and Preprocessing
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import gc
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

DTYPE = np.float32
FREQ = 3

def load_and_preprocess_data(data_dir='nfl-playing-surface-analytics'):
    """
    Load and preprocess NFL playing surface analytics data.

    Parameters:
    -----------
    data_dir : str
        Path to the directory containing the data files

    Returns:
    --------
    tuple : (InjuryRecord, PlayList, PlayerTrackData, corr_term)
    """
    print("Loading data...")

    # Load InjuryRecord
    InjuryRecord = pd.read_csv(f'{data_dir}/InjuryRecord.csv')

    # Load PlayList
    PlayList = pd.read_csv(f'{data_dir}/PlayList.csv')
    last_plays = PlayList.groupby('GameID')['PlayKey'].last()
    InjuryRecord.loc[InjuryRecord['PlayKey'].isnull(), 'PlayKey'] = \
        InjuryRecord['GameID'].map(last_plays)[InjuryRecord['PlayKey'].isnull()].values
    InjuryRecord = InjuryRecord.groupby('PlayKey').first().reset_index()

    PlayList = PlayList.merge(InjuryRecord[['PlayKey', 'DM_M1']],
                              left_on='PlayKey', right_on='PlayKey', how='left')
    PlayList['DM_M1'] = PlayList['DM_M1'].fillna(0)

    print(f"PlayList shape: {PlayList.shape}")
    print(f"InjuryRecord shape: {InjuryRecord.shape}")

    # Load PlayerTrackData
    print("Loading PlayerTrackData...")
    PlayerTrackData = pd.read_csv(f'{data_dir}/PlayerTrackData.csv',
                                  usecols=["PlayKey", "time", "x", "y", "dir"],
                                  dtype={"time": DTYPE, "x": DTYPE, "y": DTYPE, "dir": DTYPE})
    PlayerTrackData['time'] = (PlayerTrackData['time'] * 10).astype(np.int16)
    PlayerTrackData = PlayerTrackData[PlayerTrackData['time'] % FREQ == 0]

    gc.collect()

    # Merge with PlayList to get RosterPosition
    PlayerTrackData = PlayerTrackData.merge(PlayList[['PlayKey', 'RosterPosition']],
                                            left_on='PlayKey', right_on='PlayKey', how='left')

    # Calculate velocities
    PlayerTrackData['sx'] = PlayerTrackData['x'].diff().astype(DTYPE) * 10 / FREQ
    PlayerTrackData['sy'] = PlayerTrackData['y'].diff().astype(DTYPE) * 10 / FREQ
    PlayerTrackData['s'] = np.sqrt(PlayerTrackData['sx']**2 + PlayerTrackData['sy']**2).astype(DTYPE)

    # Calculate accelerations
    PlayerTrackData['ax'] = PlayerTrackData['sx'].diff().astype(DTYPE)
    PlayerTrackData['ay'] = PlayerTrackData['sy'].diff().astype(DTYPE)
    PlayerTrackData['a'] = np.sqrt(PlayerTrackData['ax']**2 + PlayerTrackData['ay']**2).astype(DTYPE)

    # Previous velocity for angle calculations
    PlayerTrackData['sx_prev'] = PlayerTrackData['sx'].shift()
    PlayerTrackData['sy_prev'] = PlayerTrackData['sy'].shift()
    PlayerTrackData['s_prev'] = PlayerTrackData['s'].shift()

    PlayerTrackData = PlayerTrackData[PlayerTrackData['time'] > FREQ].drop(['dir'], axis=1)

    # Merge with PlayList and InjuryRecord
    PlayerTrackData = PlayerTrackData.merge(PlayList[['PlayKey', 'PlayerKey', 'FieldType']],
                                            left_on='PlayKey', right_on='PlayKey')
    PlayerTrackData = PlayerTrackData.merge(InjuryRecord[['PlayKey', 'DM_M1']],
                                            left_on='PlayKey', right_on='PlayKey', how='left')
    PlayerTrackData['DM_M1'] = PlayerTrackData['DM_M1'].fillna(0)

    gc.collect()

    # Calculate forward and sideways acceleration
    PlayerTrackData['cos_th'] = ((PlayerTrackData['ax']*PlayerTrackData['sx_prev'] +
                                  PlayerTrackData['ay']*PlayerTrackData['sy_prev']) /
                                 PlayerTrackData['a'] / PlayerTrackData['s_prev']).fillna(1)
    PlayerTrackData['a_fwd'] = PlayerTrackData['a'] * PlayerTrackData['cos_th']
    PlayerTrackData['a_sid'] = np.sqrt(PlayerTrackData['a']**2 - PlayerTrackData['a_fwd']**2)
    PlayerTrackData['a_sid'] = PlayerTrackData['a_sid'].fillna(0) * \
        np.sign(PlayerTrackData['ax']*PlayerTrackData['sy_prev'] -
                PlayerTrackData['ay']*PlayerTrackData['sx_prev'])

    print(f"PlayerTrackData shape: {PlayerTrackData.shape}")

    # Calculate correction term
    corr_term = (32 * 53 - 250) / 250
    print(f"Correction factor: {corr_term}")

    return InjuryRecord, PlayList, PlayerTrackData, corr_term


def csq_test(df, grp=None, target='DM_M1', c_term=1):
    """
    Chi-square test for comparing injury rates between surface types.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    grp : str, optional
        Column name to group by (e.g., 'GameID')
    target : str
        Target variable column name
    c_term : float
        Correction term

    Returns:
    --------
    tuple : (p-value, total count, total injuries)
    """
    df0 = df[df[target] >= df.groupby('GameID')[target].cumsum()]

    if grp is not None:
        df0 = df0.groupby(grp)[['FieldType', target]].max().reset_index()

    cond = df0['FieldType'] == "Synthetic"

    e1 = df0[cond][target].sum()
    n1 = cond.sum() * c_term
    e2 = df0[~cond][target].sum()
    n2 = (~cond).sum() * c_term
    p_hat = (e1 + e2) / (n1 + n2)
    z = (e1/n1 - e2/n2) / np.sqrt(p_hat * (1-p_hat) * (1/n1 + 1/n2))

    return 1-norm.cdf(z), int(n1+n2), int(e1+e2)


if __name__ == "__main__":
    # Load data
    InjuryRecord, PlayList, PlayerTrackData, corr_term = load_and_preprocess_data()

    # Run statistical tests
    print("\n" + "="*70)
    print("STATISTICAL TESTS")
    print("="*70)

    res = csq_test(PlayList, c_term=1)
    print(f"p-value by play: {res[0]:<6.4f} ({res[2]} injuries from {res[1]} plays)")

    res = csq_test(PlayList, grp='GameID', c_term=1)
    print(f"p-value by game: {res[0]:<6.4f} ({res[2]} injuries from {res[1]} games)")

    res = csq_test(PlayList, c_term=corr_term)
    print(f"Bias corrected p-value by play: {res[0]:<6.4f} ({res[2]} injuries from {res[1]} plays)")

    res = csq_test(PlayList, grp='GameID', c_term=corr_term)
    print(f"Bias corrected p-value by game: {res[0]:<6.4f} ({res[2]} injuries from {res[1]} games)")
    print("="*70)
