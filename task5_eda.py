"""
Task 5 – Exploratory Data Analysis (EDA)
==========================================
Performs comprehensive EDA on any CSV dataset including:
  - Summary statistics
  - Missing values report
  - Correlation analysis
  - Outlier detection
  - Visual insights

Usage:
    python task5_eda.py --input dataset.csv --output eda_report/
    python task5_eda.py                      # uses generated sample data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import argparse
import warnings
warnings.filterwarnings('ignore')


# ─── Sample data ──────────────────────────────────────────────────────────────

def generate_sample_dataset(n: int = 300) -> pd.DataFrame:
    """Generate a synthetic dataset for EDA demonstration."""
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(22, 65, n),
        'salary': np.random.normal(60000, 15000, n).round(2),
        'experience_years': np.random.randint(0, 30, n),
        'department': np.random.choice(['Engineering', 'Sales', 'HR', 'Marketing', 'Finance'], n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'performance_score': np.random.randint(1, 6, n),
        'hours_per_week': np.random.normal(40, 5, n).round(1),
        'attrition': np.random.choice([0, 1], n, p=[0.8, 0.2]),
    })
    # Inject some nulls
    df.loc[np.random.choice(n, 15, replace=False), 'salary'] = np.nan
    df.loc[np.random.choice(n, 10, replace=False), 'hours_per_week'] = np.nan
    # Inject outliers
    df.loc[np.random.choice(n, 5, replace=False), 'salary'] = 250000
    return df


# ─── EDA Functions ────────────────────────────────────────────────────────────

def summary_statistics(df: pd.DataFrame) -> dict:
    """Compute descriptive statistics."""
    numeric_df = df.select_dtypes(include=[np.number])
    stats = numeric_df.describe().round(2)
    skew = numeric_df.skew().round(3)
    kurt = numeric_df.kurtosis().round(3)
    return {'describe': stats, 'skewness': skew, 'kurtosis': kurt}


def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a missing value summary."""
    total = df.isnull().sum()
    pct = (df.isnull().mean() * 100).round(2)
    report = pd.DataFrame({'Missing Count': total, 'Missing %': pct})
    report = report[report['Missing Count'] > 0].sort_values('Missing %', ascending=False)
    return report


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr().round(3)


def detect_outliers(df: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers using IQR method.
    Returns a summary of outlier counts per numeric column.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    results = []
    for col in numeric_df.columns:
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        n_outliers = ((numeric_df[col] < lower) | (numeric_df[col] > upper)).sum()
        results.append({
            'Column': col, 'Q1': round(Q1, 2), 'Q3': round(Q3, 2),
            'IQR': round(IQR, 2), 'Lower Bound': round(lower, 2),
            'Upper Bound': round(upper, 2), 'Outlier Count': n_outliers,
            'Outlier %': round(n_outliers / len(df) * 100, 2)
        })
    return pd.DataFrame(results).sort_values('Outlier Count', ascending=False)


# ─── Visualizations ───────────────────────────────────────────────────────────

def plot_eda_visuals(df: pd.DataFrame, output_dir: str = ".") -> None:
    """Generate a comprehensive EDA visual report."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    sns.set_theme(style="whitegrid", palette="muted")

    # ── Figure 1: Distributions ────────────────────────────────────────
    n = len(numeric_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    fig.suptitle("Numeric Feature Distributions", fontsize=15, fontweight='bold')
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color='steelblue')
        axes[i].set_title(col)
        axes[i].set_xlabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_distributions.png"), dpi=130, bbox_inches='tight')
    plt.close()

    # ── Figure 2: Correlation Heatmap ─────────────────────────────────
    corr = correlation_analysis(df)
    if len(corr) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                    center=0, square=True, linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eda_correlation.png"), dpi=130, bbox_inches='tight')
        plt.close()

    # ── Figure 3: Box plots (outliers) ────────────────────────────────
    if numeric_cols:
        fig, axes = plt.subplots(1, len(numeric_cols[:5]), figsize=(4 * min(5, len(numeric_cols)), 5))
        if len(numeric_cols) == 1:
            axes = [axes]
        axes = np.array(axes).flatten()
        for i, col in enumerate(numeric_cols[:5]):
            sns.boxplot(y=df[col].dropna(), ax=axes[i], color='lightcoral')
            axes[i].set_title(col)
        plt.suptitle("Outlier Detection (Box Plots)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eda_outliers.png"), dpi=130, bbox_inches='tight')
        plt.close()

    # ── Figure 4: Categorical bar charts ─────────────────────────────
    if cat_cols:
        fig, axes = plt.subplots(1, min(3, len(cat_cols)), figsize=(5 * min(3, len(cat_cols)), 5))
        if len(cat_cols) == 1:
            axes = [axes]
        axes = np.array(axes).flatten()
        for i, col in enumerate(cat_cols[:3]):
            vc = df[col].value_counts()
            sns.barplot(x=vc.index, y=vc.values, ax=axes[i], palette='Set2')
            axes[i].set_title(f"{col} Distribution")
            axes[i].tick_params(axis='x', rotation=30)
        plt.suptitle("Categorical Feature Distributions", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eda_categorical.png"), dpi=130, bbox_inches='tight')
        plt.close()

    print(f"[SUCCESS] EDA charts saved to '{output_dir}'")


def print_eda_report(df: pd.DataFrame) -> None:
    """Print EDA findings to console."""
    print("\n" + "=" * 60)
    print("          EXPLORATORY DATA ANALYSIS REPORT")
    print("=" * 60)
    print(f"\n  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")

    stats = summary_statistics(df)
    print("\n── Summary Statistics ──────────────────────────────────")
    print(stats['describe'].to_string())

    mv = missing_values_report(df)
    print("\n── Missing Values ──────────────────────────────────────")
    if mv.empty:
        print("  ✓ No missing values found.")
    else:
        print(mv.to_string())

    outliers = detect_outliers(df)
    print("\n── Outlier Detection (IQR Method) ──────────────────────")
    print(outliers[['Column', 'Lower Bound', 'Upper Bound', 'Outlier Count', 'Outlier %']].to_string(index=False))

    print("\n── Skewness ────────────────────────────────────────────")
    for col, val in stats['skewness'].items():
        flag = "⚠ Highly skewed" if abs(val) > 1 else ""
        print(f"  {col:<25} {val:>8.3f}  {flag}")

    print("\n[SUCCESS] EDA complete.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EDA Script")
    parser.add_argument('--input', type=str, help='Input CSV file path')
    parser.add_argument('--output', type=str, default='eda_output', help='Output directory')
    args = parser.parse_args()

    if args.input and os.path.exists(args.input):
        df = pd.read_csv(args.input)
        print(f"[INFO] Loaded: '{args.input}'")
    else:
        df = generate_sample_dataset()
        print("[INFO] Using generated sample dataset.")

    os.makedirs(args.output, exist_ok=True)
    print_eda_report(df)
    plot_eda_visuals(df, output_dir=args.output)


if __name__ == "__main__":
    main()
