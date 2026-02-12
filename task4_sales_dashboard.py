"""
Task 4 – Sales Data Dashboard
================================
Analyzes sales data and generates charts for:
  - Monthly revenue trend
  - Top 5 products
  - Region-wise performance

Usage:
    python task4_sales_dashboard.py --input sales.csv
    python task4_sales_dashboard.py          # uses built-in sample data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import argparse
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ─── Sample data generator ────────────────────────────────────────────────────

def generate_sample_sales_data(n_rows: int = 500) -> pd.DataFrame:
    """Generate realistic synthetic sales data."""
    np.random.seed(42)
    products = ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Monitor',
                'Keyboard', 'Mouse', 'Webcam', 'Speaker', 'Charger']
    regions = ['North', 'South', 'East', 'West', 'Central']

    # Simulate dates over 12 months
    start = datetime(2023, 1, 1)
    dates = [start + timedelta(days=np.random.randint(0, 365)) for _ in range(n_rows)]

    # Product weights (some sell more)
    product_weights = [0.2, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.03, 0.02]
    chosen_products = np.random.choice(products, size=n_rows, p=product_weights)

    # Price by product
    price_map = {
        'Laptop': 1200, 'Phone': 800, 'Tablet': 500, 'Headphones': 150,
        'Monitor': 400, 'Keyboard': 80, 'Mouse': 40, 'Webcam': 90,
        'Speaker': 120, 'Charger': 25
    }
    quantities = np.random.randint(1, 6, size=n_rows)
    prices = np.array([price_map[p] * (1 + np.random.uniform(-0.1, 0.1)) for p in chosen_products])
    revenues = prices * quantities

    df = pd.DataFrame({
        'date': dates,
        'product': chosen_products,
        'region': np.random.choice(regions, size=n_rows),
        'quantity': quantities,
        'unit_price': prices.round(2),
        'revenue': revenues.round(2),
    })
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


# ─── Analysis functions ────────────────────────────────────────────────────────

def monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue by month."""
    df = df.copy()
    df['month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('month')['revenue'].sum().reset_index()
    monthly['month_str'] = monthly['month'].astype(str)
    return monthly


def top_products(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Get top N products by total revenue."""
    return df.groupby('product')['revenue'].sum().nlargest(n).reset_index()


def region_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Region-wise revenue and quantity breakdown."""
    region = df.groupby('region').agg(
        Total_Revenue=('revenue', 'sum'),
        Orders=('revenue', 'count'),
        Avg_Order_Value=('revenue', 'mean')
    ).reset_index()
    region.columns = ['Region', 'Total Revenue', 'Orders', 'Avg Order Value']
    return region.sort_values('Total Revenue', ascending=False)


# ─── Visualization ────────────────────────────────────────────────────────────

def create_dashboard(df: pd.DataFrame, output_dir: str = ".") -> None:
    """Create a 2×2 dashboard of charts."""
    sns.set_theme(style="whitegrid", palette="muted")
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Sales Data Dashboard", fontsize=20, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Chart 1: Monthly Revenue Trend ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])  # Full-width top row
    monthly = monthly_revenue(df)
    ax1.plot(monthly['month_str'], monthly['revenue'] / 1000,
             marker='o', linewidth=2.5, color='#2196F3', markersize=6)
    ax1.fill_between(monthly['month_str'], monthly['revenue'] / 1000, alpha=0.15, color='#2196F3')
    ax1.set_title("Monthly Revenue Trend", fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Revenue ($K)")
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels
    for _, row in monthly.iterrows():
        ax1.annotate(f"${row['revenue']/1000:.1f}K",
                     xy=(row['month_str'], row['revenue']/1000),
                     xytext=(0, 8), textcoords='offset points',
                     ha='center', fontsize=7.5, color='#1565C0')

    # ── Chart 2: Top 5 Products by Revenue ─────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    top = top_products(df, n=5)
    colors = sns.color_palette("Blues_d", n_colors=5)
    bars = ax2.barh(top['product'], top['revenue'] / 1000, color=colors[::-1])
    ax2.set_title("Top 5 Products by Revenue", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Revenue ($K)")
    ax2.invert_yaxis()
    for bar, val in zip(bars, top['revenue']):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f"${val/1000:.1f}K", va='center', fontsize=8.5)

    # ── Chart 3: Region-wise Performance ───────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    region = region_performance(df)
    wedge_colors = sns.color_palette("Set2", n_colors=len(region))
    wedges, texts, autotexts = ax3.pie(
        region['Total Revenue'],
        labels=region['Region'],
        autopct='%1.1f%%',
        startangle=90,
        colors=wedge_colors,
        pctdistance=0.8
    )
    for text in autotexts:
        text.set_fontsize(9)
    ax3.set_title("Region-wise Revenue Distribution", fontsize=13, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    chart_path = os.path.join(output_dir, "sales_dashboard.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Dashboard saved to '{chart_path}'")


def save_summary_report(df: pd.DataFrame, output_dir: str = ".") -> None:
    """Save summary stats to CSV."""
    summary = pd.DataFrame({
        'Metric': ['Total Revenue', 'Total Orders', 'Avg Order Value', 'Date Range', 'Products', 'Regions'],
        'Value': [
            f"${df['revenue'].sum():,.2f}",
            str(len(df)),
            f"${df['revenue'].mean():,.2f}",
            f"{df['date'].min().date()} to {df['date'].max().date()}",
            str(df['product'].nunique()),
            str(df['region'].nunique()),
        ]
    })
    summary.to_csv(os.path.join(output_dir, "sales_summary.csv"), index=False)

    top_products(df).to_csv(os.path.join(output_dir, "top_products.csv"), index=False)
    region_performance(df).to_csv(os.path.join(output_dir, "region_performance.csv"), index=False)
    print("[SUCCESS] Summary CSVs saved.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sales Data Dashboard")
    parser.add_argument('--input', type=str, help='Path to sales CSV file')
    parser.add_argument('--output', type=str, default='.', help='Output directory for charts')
    args = parser.parse_args()

    if args.input and os.path.exists(args.input):
        df = pd.read_csv(args.input, parse_dates=['date'])
        print(f"[INFO] Loaded data from '{args.input}'")
    else:
        df = generate_sample_sales_data()
        print(f"[INFO] Generated sample data: {len(df)} rows")

    os.makedirs(args.output, exist_ok=True)

    print("\n[INFO] Running analysis...")
    print(f"  Total Revenue : ${df['revenue'].sum():,.2f}")
    print(f"  Total Orders  : {len(df)}")
    print(f"  Date Range    : {df['date'].min().date()} → {df['date'].max().date()}")

    create_dashboard(df, output_dir=args.output)
    save_summary_report(df, output_dir=args.output)
    print("\n[INFO] All outputs generated in:", args.output)


if __name__ == "__main__":
    main()
