"""
Task 2 – Employee Salary Analyzer
===================================
Reads employee data, computes salary statistics, and prints a formatted report.

Usage:
    python task2_salary_analyzer.py --input employees.csv
    python task2_salary_analyzer.py  # uses built-in sample data
"""

import pandas as pd
import numpy as np
import argparse
import os


# ─── Sample data generator (used if no CSV is provided) ───────────────────────
def get_sample_data() -> pd.DataFrame:
    np.random.seed(42)
    departments = ['Engineering', 'Marketing', 'HR', 'Sales', 'Finance']
    names = [
        'Alice', 'Bob', 'Carol', 'David', 'Eve',
        'Frank', 'Grace', 'Hank', 'Ivy', 'Jack',
        'Karen', 'Leo', 'Mia', 'Nate', 'Olivia'
    ]
    data = {
        'employee_id': range(1001, 1016),
        'name': names,
        'department': np.random.choice(departments, size=15),
        'salary': np.random.randint(40000, 150000, size=15),
        'years_experience': np.random.randint(1, 20, size=15),
    }
    return pd.DataFrame(data)


# ─── Analysis functions ────────────────────────────────────────────────────────

def calculate_overall_stats(df: pd.DataFrame) -> dict:
    """Compute company-wide salary statistics."""
    return {
        'total_employees': len(df),
        'total_salary_expense': df['salary'].sum(),
        'average_salary': df['salary'].mean(),
        'median_salary': df['salary'].median(),
        'min_salary': df['salary'].min(),
        'max_salary': df['salary'].max(),
        'std_deviation': df['salary'].std(),
    }


def department_salary_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-department salary breakdown."""
    dept_stats = df.groupby('department')['salary'].agg(
        Employee_Count='count',
        Average_Salary='mean',
        Median_Salary='median',
        Min_Salary='min',
        Max_Salary='max',
        Total_Salary='sum'
    ).reset_index()
    dept_stats.columns = ['Department', 'Employees', 'Avg Salary', 'Median', 'Min', 'Max', 'Total']
    dept_stats = dept_stats.sort_values('Avg Salary', ascending=False)
    return dept_stats


def get_top_earners(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Return top N highest paid employees."""
    cols = ['name', 'department', 'salary', 'years_experience'] if 'years_experience' in df.columns else ['name', 'department', 'salary']
    return df.nlargest(n, 'salary')[cols].reset_index(drop=True)


def get_lowest_earners(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Return bottom N lowest paid employees."""
    cols = ['name', 'department', 'salary', 'years_experience'] if 'years_experience' in df.columns else ['name', 'department', 'salary']
    return df.nsmallest(n, 'salary')[cols].reset_index(drop=True)


def salary_bracket_distribution(df: pd.DataFrame) -> pd.Series:
    """Bin employees into salary brackets."""
    bins = [0, 50000, 80000, 100000, 130000, float('inf')]
    labels = ['< $50K', '$50K–$80K', '$80K–$100K', '$100K–$130K', '> $130K']
    df['bracket'] = pd.cut(df['salary'], bins=bins, labels=labels)
    return df['bracket'].value_counts().sort_index()


# ─── Report printing ──────────────────────────────────────────────────────────

def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_overall_stats(stats: dict) -> None:
    print_header("OVERALL SALARY STATISTICS")
    print(f"  {'Total Employees':<30} {stats['total_employees']}")
    print(f"  {'Total Salary Expense':<30} ${stats['total_salary_expense']:>12,.2f}")
    print(f"  {'Average Salary':<30} ${stats['average_salary']:>12,.2f}")
    print(f"  {'Median Salary':<30} ${stats['median_salary']:>12,.2f}")
    print(f"  {'Highest Salary':<30} ${stats['max_salary']:>12,.2f}")
    print(f"  {'Lowest Salary':<30} ${stats['min_salary']:>12,.2f}")
    print(f"  {'Std Deviation':<30} ${stats['std_deviation']:>12,.2f}")


def print_department_stats(dept_df: pd.DataFrame) -> None:
    print_header("DEPARTMENT-WISE SALARY BREAKDOWN")
    header = f"  {'Department':<15} {'Emp':>4}  {'Avg Salary':>12}  {'Min':>10}  {'Max':>10}"
    print(header)
    print("  " + "-" * 58)
    for _, row in dept_df.iterrows():
        print(f"  {row['Department']:<15} {int(row['Employees']):>4}  ${row['Avg Salary']:>11,.0f}  ${row['Min']:>9,.0f}  ${row['Max']:>9,.0f}")


def print_top_earners(top_df: pd.DataFrame, label: str) -> None:
    print_header(label)
    for i, row in top_df.iterrows():
        exp_str = f"  ({int(row['years_experience'])} yrs exp)" if 'years_experience' in top_df.columns else ""
        print(f"  {i+1}. {row['name']:<15} | {row['department']:<15} | ${row['salary']:>10,.0f}{exp_str}")


def print_salary_distribution(dist: pd.Series) -> None:
    print_header("SALARY BRACKET DISTRIBUTION")
    for bracket, count in dist.items():
        bar = "█" * count
        print(f"  {bracket:<15}  {bar} ({count})")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Employee Salary Analyzer")
    parser.add_argument('--input', type=str, help='Path to employee CSV file')
    args = parser.parse_args()

    if args.input and os.path.exists(args.input):
        df = pd.read_csv(args.input)
        print(f"[INFO] Loaded data from '{args.input}'")
    else:
        df = get_sample_data()
        print("[INFO] Using built-in sample data (15 employees)")

    # Validate required columns
    required = ['name', 'department', 'salary']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Run analyses
    stats = calculate_overall_stats(df)
    dept_stats = department_salary_analysis(df)
    top = get_top_earners(df, n=3)
    bottom = get_lowest_earners(df, n=3)
    distribution = salary_bracket_distribution(df)

    # Print formatted report
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║       EMPLOYEE SALARY ANALYSIS REPORT" + " " * 20 + "║")
    print("╚" + "═" * 58 + "╝")

    print_overall_stats(stats)
    print_department_stats(dept_stats)
    print_top_earners(top, "TOP 3 HIGHEST PAID EMPLOYEES")
    print_top_earners(bottom, "BOTTOM 3 LOWEST PAID EMPLOYEES")
    print_salary_distribution(distribution)

    print("\n[SUCCESS] Report generated successfully.")


if __name__ == "__main__":
    main()
