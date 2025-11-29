# preprocess.py
import pandas as pd

def create_target(df: pd.DataFrame, src_col: str = "booking_status") -> pd.DataFrame:
    df = df.copy()
    df['is_canceled'] = df[src_col].apply(
        lambda x: 1 if str(x).strip().lower() == "canceled" else 0
    )
    return df

def drop_ids(df: pd.DataFrame, id_cols=None) -> pd.DataFrame:
    if id_cols is None:
        id_cols = ['Booking_ID']
    cols_to_drop = [c for c in id_cols if c in df.columns]
    # Always remove booking_status to avoid accidental leakage
    if 'booking_status' in df.columns:
        cols_to_drop.append('booking_status')
    return df.drop(columns=cols_to_drop, errors='ignore')
