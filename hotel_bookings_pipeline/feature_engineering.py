# feature_engineering.py
import pandas as pd

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # total stay nights
    df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']

    # total guests
    df['total_guests'] = df['no_of_adults'] + df['no_of_children']
    df['total_guests'] = df['total_guests'].replace(0, 1)

    # lead_time category
    # bins: <=30 short, 31-180 medium, >180 long
    bins = [-1, 30, 180, df['lead_time'].max() + 1]
    labels = ['short', 'medium', 'long']
    df['lead_time_cat'] = pd.cut(df['lead_time'], bins=bins, labels=labels)

    # avg price per person
    df['avg_price_per_person'] = df['avg_price_per_room'] / df['total_guests']

    # weekend booking flag
    df['weekend_booking'] = (df['no_of_weekend_nights'] > 0).astype(int)

    return df
