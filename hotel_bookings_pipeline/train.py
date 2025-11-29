# train.py
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# for tuning
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'is_canceled']
    categorical_cols = X.select_dtypes(include=['object','category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop')

    return preprocessor, numeric_cols, categorical_cols

def build_rf_pipeline(preprocessor):
    """Random Forest pipeline"""
    return Pipeline([('pre', preprocessor),
                     ('clf', RandomForestClassifier(n_estimators=300,
                                                    class_weight='balanced',
                                                    random_state=42,
                                                    n_jobs=-1))])

def build_logreg_pipeline(preprocessor):
    """Logistic Regression pipeline"""
    return Pipeline([('pre', preprocessor),
                     ('clf', LogisticRegression(max_iter=1000,
                                                class_weight='balanced',
                                                solver='saga',
                                                random_state=42))])

def build_gb_pipeline(preprocessor):
    """Gradient Boosting pipeline"""
    return Pipeline([('pre', preprocessor),
                     ('clf', GradientBoostingClassifier(random_state=42))])

# --------------------------
# Hyperparameter tuning helper
# --------------------------
def tune_random_forest(preprocessor, X_train, y_train, n_iter=20, cv=3, random_state=42, n_jobs=-1, scoring='f1'):
    rf_pipe = Pipeline([('pre', preprocessor),
                        ('clf', RandomForestClassifier(class_weight='balanced', random_state=random_state))])

    param_dist = {
        'clf__n_estimators': [200, 300, 500, 700],
        'clf__max_depth': [None, 10, 20, 30],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__max_features': ['sqrt', 'log2', None]
    }

    rnd = RandomizedSearchCV(
        estimator=rf_pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        verbose=2,
        random_state=random_state,
        n_jobs=n_jobs,
        error_score='raise' 
    )

    rnd.fit(X_train, y_train)
    return rnd
