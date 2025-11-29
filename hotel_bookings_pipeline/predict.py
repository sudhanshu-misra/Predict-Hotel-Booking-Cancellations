# predict.py

import pandas as pd
import joblib

from hotel_bookings_pipeline.preprocess import create_target, drop_ids
from hotel_bookings_pipeline.feature_engineering import add_basic_features

def load_model(model_path: str):
    """Load a trained joblib model."""
    return joblib.load(model_path)

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_ids(df)             
    df = add_basic_features(df)    
    df = df.dropna().reset_index(drop=True)
    return df

def predict_from_csv(model_path: str, csv_path: str) -> pd.DataFrame:
    """Run predictions on a new CSV file."""
    model = load_model(model_path)
    new_df = pd.read_csv(csv_path)

    processed_df = prepare_data(new_df)

    # predict
    preds = model.predict(processed_df)
    probs = model.predict_proba(processed_df)[:, 1]

    out_df = processed_df.copy()
    out_df["prediction"] = preds
    out_df["probability"] = probs

    return out_df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="predictions.csv")

    args = parser.parse_args()

    result = predict_from_csv(args.model_path, args.csv_path)
    result.to_csv(args.output_path, index=False)
    print(f"Predictions saved to {args.output_path}")
