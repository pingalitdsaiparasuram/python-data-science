"""
Task 1 – Data Cleaning Script
==============================
Cleans a CSV file by removing nulls, fixing wrong values, and standardizing dates.

Usage:
    python task1_data_cleaning.py --input raw_data.csv --output cleaned_data.csv
"""

import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV file into a DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded {len(df)} rows, {len(df.columns)} columns from '{filepath}'")
    return df


def remove_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with critical null values.
    For non-critical columns, fill nulls with appropriate defaults.
    """
    before = len(df)

    # Drop rows where all values are null
    df.dropna(how='all', inplace=True)

    # For numeric columns → fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"[INFO] Filled nulls in '{col}' with median: {median_val:.2f}")

    # For string/object columns → fill with 'Unknown'
    str_cols = df.select_dtypes(include=['object']).columns
    for col in str_cols:
        df[col].fillna('Unknown', inplace=True)

    after = len(df)
    print(f"[INFO] Removed {before - after} all-null rows. {after} rows remain.")
    return df


def replace_wrong_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and replace obviously wrong/invalid values.
    - Negative ages → set to NaN then fill with median
    - Negative salaries → set to NaN then fill with median
    - Remove duplicate rows
    - Strip leading/trailing whitespace from strings
    """
    # Strip whitespace from all string columns
    str_cols = df.select_dtypes(include=['object']).columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Fix negative numeric values for specific columns
    for col in df.select_dtypes(include=[np.number]).columns:
        if col.lower() in ['age', 'salary', 'price', 'amount', 'quantity']:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                df.loc[df[col] < 0, col] = np.nan
                df[col].fillna(df[col].median(), inplace=True)
                print(f"[INFO] Replaced {neg_count} negative values in '{col}'")

    # Remove duplicate rows
    dup_count = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f"[INFO] Removed {dup_count} duplicate rows.")

    return df


def standardize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect columns that look like dates and standardize them to YYYY-MM-DD format.
    """
    date_formats = [
        "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y",
        "%Y/%m/%d", "%d %b %Y", "%B %d, %Y", "%d.%m.%Y"
    ]

    for col in df.select_dtypes(include=['object']).columns:
        # Heuristic: column name or sample values look like dates
        if any(keyword in col.lower() for keyword in ['date', 'dob', 'created', 'updated', 'time']):
            converted = False
            for fmt in date_formats:
                try:
                    df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
                    converted = True
                    break
                except Exception:
                    continue
            if not converted:
                # Try pandas auto-detect
                try:
                    df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                except Exception:
                    pass
            # Convert to standard string format
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime('%Y-%m-%d')
                print(f"[INFO] Standardized date column: '{col}' → YYYY-MM-DD")

    return df


def save_data(df: pd.DataFrame, filepath: str) -> None:
    """Save cleaned DataFrame to CSV."""
    df.to_csv(filepath, index=False)
    print(f"[SUCCESS] Cleaned data saved to '{filepath}' ({len(df)} rows)")


def generate_report(original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> None:
    """Print a cleaning summary report."""
    print("\n" + "=" * 50)
    print("         DATA CLEANING REPORT")
    print("=" * 50)
    print(f"Original rows   : {len(original_df)}")
    print(f"Cleaned rows    : {len(cleaned_df)}")
    print(f"Rows removed    : {len(original_df) - len(cleaned_df)}")
    print(f"Columns         : {list(cleaned_df.columns)}")
    print(f"Null values left: {cleaned_df.isnull().sum().sum()}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Data Cleaning Script")
    parser.add_argument('--input', type=str, default='raw_data.csv', help='Input CSV file path')
    parser.add_argument('--output', type=str, default='cleaned_data.csv', help='Output CSV file path')
    args = parser.parse_args()

    # Load
    df_original = load_data(args.input)
    df = df_original.copy()

    # Clean steps
    df = remove_nulls(df)
    df = replace_wrong_values(df)
    df = standardize_dates(df)

    # Save
    save_data(df, args.output)
    generate_report(df_original, df)


if __name__ == "__main__":
    main()
