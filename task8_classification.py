"""
Task 8 – Classification Model (Customer Churn / Spam Detection)
=================================================================
Trains and evaluates:
  - Logistic Regression
  - Decision Tree
  - Random Forest
Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix

Usage:
    python task8_classification.py --input churn_data.csv
    python task8_classification.py                       # uses generated sample data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_curve, roc_auc_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
import os
import argparse
import warnings
warnings.filterwarnings('ignore')


# ─── Sample data generator ────────────────────────────────────────────────────

def generate_churn_data(n: int = 600) -> pd.DataFrame:
    """Generate synthetic customer churn dataset."""
    np.random.seed(42)
    tenure         = np.random.randint(1, 72, n)
    monthly_charges = np.random.uniform(20, 120, n).round(2)
    total_charges  = (tenure * monthly_charges + np.random.normal(0, 50, n)).clip(0).round(2)
    support_calls  = np.random.randint(0, 8, n)
    contract_type  = np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n)
    internet_service = np.random.choice(['DSL', 'Fiber', 'No'], n)

    # Churn probability based on features
    churn_prob = (
        0.1 +
        0.003 * (120 - tenure) +
        0.005 * (monthly_charges - 50) +
        0.04  * support_calls +
        np.where(contract_type == 'Month-to-Month', 0.2, 0) +
        np.where(internet_service == 'Fiber', 0.1, 0) +
        np.random.normal(0, 0.05, n)
    ).clip(0, 1)

    churn = (np.random.rand(n) < churn_prob).astype(int)

    return pd.DataFrame({
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'support_calls': support_calls,
        'contract_type': contract_type,
        'internet_service': internet_service,
        'churn': churn
    })


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, target_col: str = 'churn') -> tuple:
    """Encode categoricals and split data."""
    df = df.dropna()
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)} | "
          f"Churn rate: {y.mean():.1%}")
    return X_train, X_test, y_train, y_test, list(X.columns)


# ─── Model evaluation ─────────────────────────────────────────────────────────

def evaluate_model(model, X_train, X_test, y_train, y_test, name: str) -> dict:
    """Fit and evaluate a classifier."""
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    acc       = accuracy_score(y_test, y_pred)
    prec      = precision_score(y_test, y_pred, zero_division=0)
    rec       = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc       = roc_auc_score(y_test, y_proba) if y_proba is not None else 0
    cm        = confusion_matrix(y_test, y_pred)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv  = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1')

    return {
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1 Score': round(f1, 4),
        'AUC-ROC': round(auc, 4),
        'CV F1 (mean)': round(cv.mean(), 4),
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'model_obj': model,
    }


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_all(results: list, y_test, feature_names: list, output_dir: str) -> None:
    """Generate classification plots."""
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Classification Model Comparison", fontsize=14, fontweight='bold')

    # Confusion matrices
    n_models = len(results)
    for i, res in enumerate(results):
        ax = fig.add_subplot(2, n_models, i + 1)
        sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'], ax=ax)
        ax.set_title(res['Model'])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    # ROC Curves
    ax_roc = fig.add_subplot(2, 2, 3)
    colors = ['blue', 'green', 'red', 'purple']
    for i, res in enumerate(results):
        if res['y_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
            ax_roc.plot(fpr, tpr, color=colors[i], linewidth=2,
                        label=f"{res['Model']} (AUC={res['AUC-ROC']:.3f})")
    ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax_roc.set_title("ROC Curves")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(fontsize=9)

    # Metrics bar chart
    ax_bar = fig.add_subplot(2, 2, 4)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    x = np.arange(len(metrics))
    width = 0.8 / n_models
    for i, res in enumerate(results):
        vals = [res[m] for m in metrics]
        ax_bar.bar(x + i * width, vals, width, label=res['Model'], color=colors[i], alpha=0.85)
    ax_bar.set_xticks(x + width * (n_models - 1) / 2)
    ax_bar.set_xticklabels(metrics)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.set_title("Metrics Comparison")
    ax_bar.legend(fontsize=9)
    ax_bar.set_ylabel("Score")

    plt.tight_layout()
    path = os.path.join(output_dir, "classification_results.png")
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Classification chart saved: '{path}'")


def print_report(results: list) -> None:
    """Print model metrics summary."""
    print("\n" + "=" * 75)
    print("           CLASSIFICATION MODEL COMPARISON REPORT")
    print("=" * 75)
    header = (f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} "
              f"{'Recall':>8} {'F1':>8} {'AUC':>8} {'CV F1':>8}")
    print(header)
    print("  " + "-" * 72)
    for r in results:
        print(f"  {r['Model']:<22} {r['Accuracy']:>9.4f} {r['Precision']:>10.4f} "
              f"{r['Recall']:>8.4f} {r['F1 Score']:>8.4f} {r['AUC-ROC']:>8.4f} "
              f"{r['CV F1 (mean)']:>8.4f}")
    best = max(results, key=lambda x: x['F1 Score'])
    print(f"\n  ✓ Best Model: {best['Model']} (F1 = {best['F1 Score']})")
    print("=" * 75)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Classification Model")
    parser.add_argument('--input', type=str, help='Input CSV file path')
    parser.add_argument('--target', type=str, default='churn', help='Target column')
    parser.add_argument('--output', type=str, default='.', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.input and os.path.exists(args.input):
        df = pd.read_csv(args.input)
    else:
        df = generate_churn_data()
        print(f"[INFO] Generated customer churn dataset ({len(df)} rows)")

    X_train, X_test, y_train, y_test, features = preprocess(df, target_col=args.target)

    models = [
        (Pipeline([('scaler', StandardScaler()),
                   ('clf', LogisticRegression(max_iter=500, random_state=42))]),
         "Logistic Regression"),
        (DecisionTreeClassifier(max_depth=5, random_state=42), "Decision Tree"),
        (RandomForestClassifier(n_estimators=100, max_depth=6,
                                random_state=42, n_jobs=-1), "Random Forest"),
    ]

    results = []
    for model, name in models:
        print(f"[INFO] Training: {name}...")
        res = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(res)

    print_report(results)
    plot_all(results, y_test, features, args.output)

    # Save best model
    best = max(results, key=lambda x: x['F1 Score'])
    joblib.dump(best['model_obj'], os.path.join(args.output, "best_classifier.pkl"))
    print(f"[SUCCESS] Best classifier saved: {best['Model']}")


if __name__ == "__main__":
    main()
