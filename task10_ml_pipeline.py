"""
Task 10 – End-to-End ML Pipeline
==================================
A production-ready sklearn Pipeline covering:
  1. Data cleaning
  2. Feature engineering
  3. Model training
  4. Model saving (.pkl via joblib)
  5. Prediction script

Usage:
    python task10_ml_pipeline.py --train          # train & save model
    python task10_ml_pipeline.py --predict input.csv  # load & predict
"""

import pandas as pd
import numpy as np
import joblib
import os
import argparse
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

MODEL_PATH = "ml_pipeline_model.pkl"
FEATURE_COLS = ['age', 'income', 'loan_amount', 'credit_score',
                'employment_type', 'loan_purpose']
TARGET_COL = 'default'


# ─── Data generation ──────────────────────────────────────────────────────────

def generate_data(n: int = 500, with_nulls: bool = True) -> pd.DataFrame:
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(20, 65, n).astype(float),
        'income': np.random.normal(55000, 20000, n).clip(15000),
        'loan_amount': np.random.uniform(1000, 50000, n).round(0),
        'credit_score': np.random.randint(300, 850, n).astype(float),
        'employment_type': np.random.choice(['Salaried', 'Self-Employed', 'Unemployed'], n),
        'loan_purpose': np.random.choice(['Education', 'Home', 'Car', 'Personal'], n),
    })
    default_prob = (
        0.1 + 0.001*(65 - df['age']) + 0.000005*(50000 - df['income'])
        + 0.000002*df['loan_amount'] + 0.0003*(700 - df['credit_score'])
        + np.where(df['employment_type'] == 'Unemployed', 0.25, 0)
        + np.random.normal(0, 0.05, n)
    ).clip(0, 1)
    df[TARGET_COL] = (np.random.rand(n) < default_prob).astype(int)
    if with_nulls:
        for col in ['income', 'credit_score']:
            df.loc[np.random.choice(n, 20, replace=False), col] = np.nan
    return df


# ─── Pipeline construction ────────────────────────────────────────────────────

def build_pipeline(numeric_features: list, categorical_features: list) -> Pipeline:
    """
    Build a sklearn ColumnTransformer + estimator Pipeline.
    Handles: imputation → encoding → scaling → model
    """
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42
        )),
    ])

    return model


def train(data: pd.DataFrame, output_path: str = MODEL_PATH) -> None:
    """Train pipeline on data and save model."""
    X = data[FEATURE_COLS]
    y = data[TARGET_COL]

    numeric_features = ['age', 'income', 'loan_amount', 'credit_score']
    categorical_features = ['employment_type', 'loan_purpose']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"[INFO] Training on {len(X_train)} rows...")

    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')

    print(f"\n[METRICS]")
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))
    print(f"  Test F1      : {f1:.4f}")
    print(f"  CV F1 (mean) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save model
    joblib.dump(pipeline, output_path)
    print(f"\n[SUCCESS] Model saved to '{output_path}'")


def predict(input_path: str, model_path: str = MODEL_PATH) -> pd.DataFrame:
    """Load saved model and make predictions on new data."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: '{model_path}'. Run --train first.")

    pipeline = joblib.load(model_path)
    print(f"[INFO] Loaded model from '{model_path}'")

    df = pd.read_csv(input_path)
    X = df[FEATURE_COLS]
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)[:, 1]

    df['predicted_default'] = predictions
    df['default_probability'] = probabilities.round(4)

    output_path = input_path.replace('.csv', '_predictions.csv')
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Predictions saved to '{output_path}'")
    print(f"  Predicted defaults: {predictions.sum()} / {len(predictions)}")
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="End-to-End ML Pipeline")
    parser.add_argument('--train', action='store_true', help='Train and save model')
    parser.add_argument('--predict', type=str, help='CSV file to run predictions on')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    args = parser.parse_args()

    if args.train or (not args.train and not args.predict):
        print("[INFO] Generating training data...")
        df = generate_data(n=600)
        train(df, output_path=args.model_path)

    if args.predict:
        predict(args.predict, model_path=args.model_path)


if __name__ == "__main__":
    main()
