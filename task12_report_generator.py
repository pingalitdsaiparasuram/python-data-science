"""
Task 12 – Automated Report Generator
======================================
Reads data, computes summary stats, and exports a PDF/Excel report automatically.

Usage:
    python task12_report_generator.py --input data.csv --format excel
    python task12_report_generator.py --input data.csv --format pdf
    python task12_report_generator.py                   # generates sample report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import os
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def generate_sample_data(n: int = 200) -> pd.DataFrame:
    np.random.seed(1)
    return pd.DataFrame({
        'Month': np.tile(['Jan','Feb','Mar','Apr','May','Jun',
                          'Jul','Aug','Sep','Oct','Nov','Dec'], n // 12 + 1)[:n],
        'Region': np.random.choice(['North','South','East','West'], n),
        'Sales': np.random.randint(5000, 50000, n),
        'Expenses': np.random.randint(2000, 30000, n),
        'Units': np.random.randint(10, 200, n),
    })


def compute_summary(df: pd.DataFrame) -> dict:
    """Compute key business metrics."""
    numeric = df.select_dtypes(include=[np.number])
    month_sales = df.groupby('Month')['Sales'].sum() if 'Month' in df.columns else None
    region_sales = df.groupby('Region')['Sales'].sum() if 'Region' in df.columns else None
    df['Profit'] = df['Sales'] - df['Expenses']
    return {
        'total_sales': df['Sales'].sum(),
        'total_expenses': df['Expenses'].sum() if 'Expenses' in df.columns else 0,
        'total_profit': df['Profit'].sum() if 'Profit' in df.columns else 0,
        'avg_sales': df['Sales'].mean(),
        'describe': numeric.describe(),
        'month_sales': month_sales,
        'region_sales': region_sales,
    }


def export_excel_report(df: pd.DataFrame, summary: dict, output_path: str) -> None:
    """Generate multi-sheet Excel report."""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Raw data
        df.to_excel(writer, sheet_name='Raw Data', index=False)

        # Summary stats
        summary['describe'].to_excel(writer, sheet_name='Summary Stats')

        # Monthly sales
        if summary['month_sales'] is not None:
            summary['month_sales'].reset_index().to_excel(
                writer, sheet_name='Monthly Sales', index=False)

        # Region performance
        if summary['region_sales'] is not None:
            summary['region_sales'].reset_index().to_excel(
                writer, sheet_name='Region Performance', index=False)

        # KPI Overview
        kpi = pd.DataFrame([{
            'Metric': 'Total Sales', 'Value': f"${summary['total_sales']:,.2f}"},
            {'Metric': 'Total Expenses', 'Value': f"${summary['total_expenses']:,.2f}"},
            {'Metric': 'Total Profit', 'Value': f"${summary['total_profit']:,.2f}"},
            {'Metric': 'Avg Sales', 'Value': f"${summary['avg_sales']:,.2f}"},
            {'Metric': 'Report Date', 'Value': datetime.now().strftime('%Y-%m-%d')},
        ])
        kpi.to_excel(writer, sheet_name='KPI Overview', index=False)

    print(f"[SUCCESS] Excel report saved: '{output_path}'")


def export_pdf_report(df: pd.DataFrame, summary: dict, output_path: str) -> None:
    """Generate a multi-page PDF report with charts."""
    with pdf_backend.PdfPages(output_path) as pdf:
        # Page 1: Title + KPI
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.axis('off')
        title = "Automated Data Report"
        subtitle = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        kpis = [
            f"Total Sales:    ${summary['total_sales']:>12,.0f}",
            f"Total Expenses: ${summary['total_expenses']:>12,.0f}",
            f"Net Profit:     ${summary['total_profit']:>12,.0f}",
            f"Avg Sale:       ${summary['avg_sales']:>12,.0f}",
            f"Records:        {len(df):>13,}",
        ]
        ax.text(0.5, 0.9, title, ha='center', fontsize=22, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.82, subtitle, ha='center', fontsize=10, color='gray', transform=ax.transAxes)
        for i, kpi in enumerate(kpis):
            ax.text(0.25, 0.65 - i * 0.1, kpi, fontsize=13,
                    fontfamily='monospace', transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 2: Monthly sales chart
        if summary['month_sales'] is not None:
            fig, ax = plt.subplots(figsize=(10, 5))
            summary['month_sales'].plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title("Monthly Sales", fontweight='bold')
            ax.set_ylabel("Sales ($)")
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        # Page 3: Region performance
        if summary['region_sales'] is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            summary['region_sales'].plot(kind='bar', ax=axes[0], color='coral')
            axes[0].set_title("Region Sales", fontweight='bold')
            axes[0].tick_params(axis='x', rotation=30)
            summary['region_sales'].plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
            axes[1].set_title("Region Share")
            axes[1].set_ylabel("")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f"[SUCCESS] PDF report saved: '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Automated Report Generator")
    parser.add_argument('--input', type=str, help='Input CSV file')
    parser.add_argument('--format', type=str, default='excel',
                        choices=['excel', 'pdf', 'both'], help='Output format')
    parser.add_argument('--output', type=str, default='.', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    df = pd.read_csv(args.input) if args.input and os.path.exists(args.input) else generate_sample_data()
    if 'Profit' not in df.columns and 'Expenses' in df.columns:
        df['Profit'] = df['Sales'] - df['Expenses']

    print(f"[INFO] Data loaded: {df.shape[0]} rows × {df.shape[1]} cols")
    summary = compute_summary(df)

    if args.format in ('excel', 'both'):
        export_excel_report(df, summary, os.path.join(args.output, "report.xlsx"))
    if args.format in ('pdf', 'both'):
        export_pdf_report(df, summary, os.path.join(args.output, "report.pdf"))


if __name__ == "__main__":
    main()
