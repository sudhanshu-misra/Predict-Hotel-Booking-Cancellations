# main.py
import argparse, joblib
from hotel_bookings_pipeline.data_loader import load_csv
from hotel_bookings_pipeline.preprocess import create_target, drop_ids
from hotel_bookings_pipeline.feature_engineering import add_basic_features
from hotel_bookings_pipeline.train import build_preprocessor, build_rf_pipeline, tune_random_forest
from hotel_bookings_pipeline.evaluate import print_metrics
from hotel_bookings_pipeline.utils import get_logger
from sklearn.model_selection import train_test_split

logger = get_logger('main')

def run(data_path, model_out="final_random_forest_model.joblib", tune=False):
    logger.info("Loading data")
    df = load_csv(data_path)

    logger.info("Creating target")
    df = create_target(df)

    logger.info("Dropping ids (and booking_status)")
    df = drop_ids(df)

    logger.info("Feature engineering")
    df = add_basic_features(df)

    # Drop NA rows (safety)
    df = df.dropna().reset_index(drop=True)

    # Train/Val/Test split
    X = df.drop(columns=['is_canceled'], errors='ignore')
    y = df['is_canceled']
    logger.info("Splitting data (70/15/15)")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    logger.info("Building preprocessor")
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)
    logger.info(f"Numeric cols: {numeric_cols}")
    logger.info(f"Categorical cols: {categorical_cols}")

    if tune:
        logger.info("Starting Random Forest hyperparameter tuning...")
        rnd = tune_random_forest(
            preprocessor, X_train, y_train,
            n_iter=10,  # Adjust as needed
            cv=3,
            random_state=42,
            n_jobs=-1,
            scoring='f1'
        )
        logger.info(f"Best params: {rnd.best_params_}")
        model = rnd.best_estimator_

    else:
        logger.info("Training baseline Random Forest")
        model = build_rf_pipeline(preprocessor)
        model.fit(X_train, y_train)

    logger.info("Evaluating on test set")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    print_metrics(y_test, y_pred, y_proba)

    logger.info("Saving model")
    joblib.dump(model, model_out)
    logger.info(f"Saved model to {model_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to Hotel Reservations CSV")
    parser.add_argument("--model_out", type=str, default="final_random_forest_model.joblib",
                        help="Output model path")
    parser.add_argument("--tune", action="store_true",
                        help="Enable Random Forest hyperparameter tuning")
    args = parser.parse_args()

    run(args.data_path, args.model_out, tune=args.tune)
