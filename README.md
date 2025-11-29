# Hotel Booking Cancellations Prediction Pipeline

A fully modular, production-ready machine learning pipeline for predicting hotel booking cancellations.  
The project includes data processing, feature engineering, model training, hyperparameter tuning, evaluation, prediction interface, and CI/CD automation using GitHub Actions.


## Key Features

- **Modular ML Pipeline** (data loading → preprocessing → feature engineering → training → evaluation → prediction)
- **Clean, reusable code structure**
- **Feature engineering for real booking behavior patterns**
- **Random Forest with hyperparameter tuning**
- **Reproducible training using CLI (`python -m` style)**
- **Model saving (joblib) and reusable predict module**
- **CI/CD with GitHub Actions**  
- **Complete reproducibility & testability**


## Project Structure

```
Predict-Hotel-Booking-Cancellations/
├── hotel_bookings_pipeline/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   ├── main.py
│   └── predict.py
├── screenshots/
│   ├── baseline_run_in_powershell.png
│   ├── final_build.png
│   ├── first_few_columns_of_preds.csv_.png
│   ├── predict_run.png
│   ├── tuned_run_in_powershell (1).png
│   ├── tuned_run_in_powershell (2).png
│   └── tuned_run_in_powershell (3).png
├── Hotel Reservations.csv
├── requirements.txt
└── README.md
```

## Dataset Summary
- Total rows: 36,275
- No missing values after preprocessing
- Features include guest counts, meal plan, room type, lead time, special requests, pricing, and booking status
- booking_status transformed into binary target is_canceled

## Feature Engineering

The pipeline automatically adds:
- total_nights
- total_guests
- avg_price_per_person
- lead_time_cat (short, medium, long)
- weekend_booking flag

These significantly improve predictive performance.

## How to Run

**Baseline training:**
python -m hotel_bookings_pipeline.main --data_path "Hotel Reservations.csv" --model_out baseline_rf.joblib

**Hyperparameter tuning:**
python -m hotel_bookings_pipeline.main --data_path "Hotel Reservations.csv" --model_out tuned_rf.joblib --tune

**Run predictions:**
python -m hotel_bookings_pipeline.predict --model_path tuned_rf.joblib --csv_path "Hotel Reservations.csv" --output_path preds.csv

## Screenshots
**Baseline Run**
(screenshots/baseline_run_in_powershell.png)

**Tuning Run**
(screenshots/tuned_run_in_powershell (1).png)
(screenshots/tuned_run_in_powershell (2).png)
(screenshots/tuned_run_in_powershell (3).png)

**Predictions Output**
(screenshots/first_few_columns_of_preds.csv_.png)
(screenshots/predict_run.png)

**GitHub Actions Successful Build**
(screenshots/final_build.png)

## Model Performance (Tuned RF – Test Set)

- Accuracy: **0.8969**
- Precision: **0.8646**
- Recall: **0.8127**
- F1 Score: **0.8378**
- ROC-AUC: **0.9552**

Best model: Tuned Random Forest

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-Learn
- Joblib
- Seaborn / Matplotlib
- PyTest
- GitHub Actions

## Installation

pip install -r requirements.txt

## Running Tests

pytest -q

## Contact

Sudhanshu Misra |
GitHub: https://github.com/sudhanshu-misra
