"""
Task 9 – Feature Engineering Challenge
========================================
Create new features, handle encoding, scaling, and improve baseline accuracy by 10%+.

Usage:
    python task9_feature_engineering.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def generate_base_data(n: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(20, 65, n),
        'income': np.random.normal(55000, 20000, n).clip(15000),
        'loan_amount': np.random.uniform(1000, 50000, n).round(0),
        'credit_score': np.random.randint(300, 850, n),
        'employment_type': np.random.choice(['Salaried', 'Self-Employed', 'Unemployed'], n),
        'loan_purpose': np.random.choice(['Education', 'Home', 'Car', 'Personal'], n),
    })
    # Target: default (1 = default)
    default_prob = (
        0.1 + 0.002*(65-df['age']) + 0.000005*(50000-df['income']) +
        0.000002*df['loan_amount'] + 0.0003*(700-df['credit_score']) +
        np.where(df['employment_type']=='Unemployed', 0.3, 0) +
        np.random.normal(0, 0.05, n)
    ).clip(0, 1)
    df['default'] = (np.random.rand(n) < default_prob).astype(int)
    return df


def baseline_model(X_train, X_test, y_train, y_test) -> float:
    """Train baseline RF on raw features."""
    from sklearn.preprocessing import LabelEncoder
    X_tr = X_train.copy(); X_ts = X_test.copy()
    for col in X_tr.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_tr[col] = le.fit_transform(X_tr[col])
        X_ts[col] = le.transform(X_ts[col])
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_tr, y_train)
    preds = model.predict(X_ts)
    return f1_score(y_test, preds, zero_division=0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering techniques:
    1. Interaction features
    2. Ratio features
    3. Binning (age groups)
    4. One-hot encoding
    5. Scaling
    """
    df = df.copy()

    # 1. Ratio features
    df['debt_to_income'] = df['loan_amount'] / (df['income'] + 1)
    df['income_per_age']  = df['income'] / df['age']

    # 2. Credit risk score (composite)
    df['risk_score'] = (1000 - df['credit_score']) * df['debt_to_income']

    # 3. Binning age
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100],
                             labels=['Young', 'Middle', 'Senior', 'Elderly'])

    # 4. Flag high loan
    df['high_loan'] = (df['loan_amount'] > 30000).astype(int)

    # 5. One-hot encode categoricals
    df = pd.get_dummies(df, columns=['employment_type', 'loan_purpose', 'age_group'])

    # 6. Polynomial interaction (income × credit)
    df['income_credit_interact'] = df['income'] * df['credit_score'] / 1e6

    return df


def main():
    df = generate_base_data()
    X = df.drop(columns=['default'])
    y = df['default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                         random_state=42, stratify=y)

    baseline = baseline_model(X_train, X_test, y_train, y_test)
    print(f"[INFO] Baseline F1 Score: {baseline:.4f}")

    # Engineer features
    df_eng = engineer_features(df)
    X_eng = df_eng.drop(columns=['default'])
    y_eng = df_eng['default']
    X_tr_e, X_ts_e, y_tr_e, y_ts_e = train_test_split(X_eng, y_eng, test_size=0.2,
                                                        random_state=42, stratify=y_eng)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_e)
    X_ts_scaled = scaler.transform(X_ts_e)

    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                       max_depth=4, random_state=42)
    model.fit(X_tr_scaled, y_tr_e)
    improved_preds = model.predict(X_ts_scaled)
    improved = f1_score(y_ts_e, improved_preds, zero_division=0)

    improvement = ((improved - baseline) / baseline * 100)

    print(f"\n{'='*55}")
    print(f"  FEATURE ENGINEERING RESULTS")
    print(f"{'='*55}")
    print(f"  Baseline F1   : {baseline:.4f}")
    print(f"  Improved F1   : {improved:.4f}")
    print(f"  Improvement   : +{improvement:.1f}%")
    goal_met = "✓ GOAL MET!" if improvement >= 10 else "⚠ Close – try more features."
    print(f"  10% Target    : {goal_met}")
    print(f"{'='*55}")
    print(f"\n  New features created: debt_to_income, income_per_age, risk_score,")
    print(f"  high_loan, age_group bins, polynomial interaction, one-hot encoding")


if __name__ == "__main__":
    main()
