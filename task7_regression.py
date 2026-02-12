"""
Task 7 – Regression Model (House Price Prediction)
====================================================
Trains and evaluates:
  - Linear Regression
  - Random Forest Regressor
Metrics: RMSE, R², MAE
Outputs a comparison report.

Usage:
    python task7_regression.py --input house_prices.csv
    python task7_regression.py                           # generates sample data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
import os
import argparse
import warnings
warnings.filterwarnings('ignore')


# ─── Sample data generator ────────────────────────────────────────────────────

def generate_house_data(n: int = 500) -> pd.DataFrame:
    """Generate synthetic house price data."""
    np.random.seed(42)
    n_samples = n
    bedrooms   = np.random.randint(1, 6, n_samples)
    bathrooms  = np.random.randint(1, 4, n_samples)
    sqft       = np.random.randint(600, 4500, n_samples)
    age        = np.random.randint(0, 60, n_samples)
    garage     = np.random.randint(0, 3, n_samples)
    neighborhood = np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples)

    # Price formula with noise
    base_price = (
        sqft * 150 +
        bedrooms * 15000 +
        bathrooms * 10000 -
        age * 500 +
        garage * 8000 +
        np.where(neighborhood == 'Urban', 50000, np.where(neighborhood == 'Suburban', 20000, 0))
    )
    price = (base_price + np.random.normal(0, 30000, n_samples)).clip(50000, 2000000)

    return pd.DataFrame({
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft': sqft,
        'age': age,
        'garage': garage,
        'neighborhood': neighborhood,
        'price': price.round(2)
    })


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, target_col: str = 'price') -> tuple:
    """Encode, split, and scale data."""
    df = df.dropna()
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[INFO] Train: {len(X_train)} rows | Test: {len(X_test)} rows | Features: {list(X.columns)}")
    return X_train, X_test, y_train, y_test, list(X.columns)


# ─── Model evaluation ─────────────────────────────────────────────────────────

def evaluate_model(model, X_train, X_test, y_train, y_test, name: str) -> dict:
    """Fit and evaluate a model, returning metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
    r2    = r2_score(y_test, y_pred)
    mae   = mean_absolute_error(y_test, y_pred)
    mape  = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    return {
        'Model': name,
        'RMSE': round(rmse, 2),
        'R² Score': round(r2, 4),
        'MAE': round(mae, 2),
        'MAPE (%)': round(mape, 2),
        'CV R² (mean)': round(cv_scores.mean(), 4),
        'CV R² (std)': round(cv_scores.std(), 4),
        'y_pred': y_pred,
    }


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_results(results: list, y_test, feature_names: list, rf_model, output_dir: str) -> None:
    """Plot comparison and feature importance charts."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Regression Model Comparison", fontsize=14, fontweight='bold')

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    # Predicted vs Actual
    for i, res in enumerate(results):
        ax = axes[0] if i == 0 else axes[0]
        axes[0].scatter(y_test, res['y_pred'], alpha=0.4, s=20,
                        label=res['Model'], color=colors[i % len(colors)])
    lim = [min(y_test.min(), min(r['y_pred'].min() for r in results)),
           max(y_test.max(), max(r['y_pred'].max() for r in results))]
    axes[0].plot(lim, lim, 'r--', linewidth=1.5, label='Perfect Fit')
    axes[0].set_title("Predicted vs Actual")
    axes[0].set_xlabel("Actual Price ($)")
    axes[0].set_ylabel("Predicted Price ($)")
    axes[0].legend(fontsize=8)

    # Metrics comparison
    metrics_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ('y_pred',)} for r in results])
    x = np.arange(len(metrics_df))
    width = 0.35
    axes[1].bar(x - width/2, metrics_df['RMSE'] / 1000, width, label='RMSE ($K)', color='steelblue')
    axes[1].bar(x + width/2, metrics_df['R² Score'], width, label='R² Score', color='darkorange')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics_df['Model'], rotation=15)
    axes[1].set_title("RMSE vs R² Score")
    axes[1].legend()

    # Feature importance (Random Forest)
    if hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
        feat_imp.plot(kind='barh', ax=axes[2], color='teal')
        axes[2].set_title("Feature Importances (Random Forest)")
        axes[2].set_xlabel("Importance")

    plt.tight_layout()
    path = os.path.join(output_dir, "regression_results.png")
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Chart saved: '{path}'")


def print_comparison_report(results: list) -> None:
    """Print formatted model comparison."""
    print("\n" + "=" * 70)
    print("          REGRESSION MODEL COMPARISON REPORT")
    print("=" * 70)
    header = f"  {'Model':<25} {'RMSE':>12} {'R²':>8} {'MAE':>12} {'MAPE%':>8} {'CV R²':>8}"
    print(header)
    print("  " + "-" * 67)
    for r in results:
        print(f"  {r['Model']:<25} ${r['RMSE']:>11,.0f} {r['R² Score']:>8.4f} "
              f"${r['MAE']:>11,.0f} {r['MAPE (%)']:>7.1f}% {r['CV R² (mean)']:>8.4f}")
    best = max(results, key=lambda x: x['R² Score'])
    print(f"\n  ✓ Best Model: {best['Model']} (R² = {best['R² Score']})")
    print("=" * 70)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Regression Model Comparison")
    parser.add_argument('--input', type=str, help='Input CSV file')
    parser.add_argument('--target', type=str, default='price', help='Target column name')
    parser.add_argument('--output', type=str, default='.', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.input and os.path.exists(args.input):
        df = pd.read_csv(args.input)
    else:
        df = generate_house_data()
        print(f"[INFO] Generated house price dataset ({len(df)} rows)")

    X_train, X_test, y_train, y_test, features = preprocess(df, target_col=args.target)

    # Define models
    models = [
        (LinearRegression(), "Linear Regression"),
        (Ridge(alpha=1.0), "Ridge Regression"),
        (RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), "Random Forest"),
        (GradientBoostingRegressor(n_estimators=100, random_state=42), "Gradient Boosting"),
    ]

    results = []
    rf_model = None
    for model, name in models:
        print(f"[INFO] Training: {name}...")
        res = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(res)
        if name == "Random Forest":
            rf_model = model

    print_comparison_report(results)
    plot_results(results, y_test, features, rf_model, args.output)

    # Save best model
    best = max(results, key=lambda x: x['R² Score'])
    best_name = best['Model']
    best_model = next(m for m, n in models if n == best_name)
    model_path = os.path.join(args.output, "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"[SUCCESS] Best model saved: '{model_path}'")


if __name__ == "__main__":
    main()
