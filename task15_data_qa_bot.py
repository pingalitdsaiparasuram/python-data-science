"""
Task 15 – Data Q&A Bot
========================
Upload a CSV and ask natural language questions about the data.
Returns answers using pandas query parsing.

Examples:
  "Show top 3 months by revenue"
  "What is the average salary?"
  "How many rows are there?"
  "Which department has the highest total salary?"
  "Show rows where salary > 100000"

Usage:
    python task15_data_qa_bot.py --csv sales.csv
    python task15_data_qa_bot.py             # uses sample data, runs demo
"""

import pandas as pd
import numpy as np
import re
import argparse
import os
import warnings
warnings.filterwarnings('ignore')


# ─── Sample data ──────────────────────────────────────────────────────────────

def generate_sample() -> pd.DataFrame:
    np.random.seed(42)
    n = 100
    months = ['January','February','March','April','May','June',
              'July','August','September','October','November','December']
    return pd.DataFrame({
        'month':      np.random.choice(months, n),
        'department': np.random.choice(['Engineering','Sales','HR','Marketing'], n),
        'employee':   [f"Employee_{i}" for i in range(1, n + 1)],
        'salary':     np.random.randint(40000, 160000, n),
        'revenue':    np.random.randint(10000, 500000, n),
        'units_sold': np.random.randint(5, 300, n),
    })


# ─── Query parser ─────────────────────────────────────────────────────────────

class DataQABot:
    """
    Natural language interface to a pandas DataFrame.
    Parses common query patterns into pandas operations.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.columns = list(df.columns)
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    def _detect_column(self, text: str) -> str | None:
        """Find which column is referenced in the query."""
        text_lower = text.lower()
        for col in self.columns:
            if col.lower() in text_lower or col.lower().replace('_', ' ') in text_lower:
                return col
        return None

    def _detect_agg(self, text: str) -> str:
        """Detect aggregation keyword."""
        text = text.lower()
        if any(w in text for w in ['average', 'avg', 'mean']):
            return 'mean'
        if any(w in text for w in ['total', 'sum']):
            return 'sum'
        if any(w in text for w in ['max', 'highest', 'largest', 'biggest', 'most']):
            return 'max'
        if any(w in text for w in ['min', 'lowest', 'smallest', 'least']):
            return 'min'
        if any(w in text for w in ['count', 'how many', 'number of']):
            return 'count'
        return 'mean'

    def _detect_n(self, text: str, default: int = 5) -> int:
        """Extract numeric N from 'top N' or 'bottom N'."""
        match = re.search(r'\b(\d+)\b', text)
        return int(match.group(1)) if match else default

    def answer(self, question: str) -> str:
        """Parse the natural language question and return an answer."""
        q = question.strip().lower()
        df = self.df

        try:
            # ── Row count ─────────────────────────────────────────
            if re.search(r'how many rows|row count|total rows|shape|size', q):
                return f"The dataset has {len(df)} rows and {len(df.columns)} columns.\nColumns: {self.columns}"

            # ── Column list ───────────────────────────────────────
            if re.search(r'column|field|variable', q) and 'list' in q or 'what are' in q:
                return f"Columns ({len(self.columns)}): {self.columns}"

            # ── Top N ─────────────────────────────────────────────
            if re.search(r'\btop\b', q):
                n = self._detect_n(q)
                col = self._detect_column(q) or (self.numeric_cols[0] if self.numeric_cols else None)
                group_col = None
                for cat in self.cat_cols:
                    if cat.lower() in q:
                        group_col = cat
                        break
                if group_col and col:
                    result = df.groupby(group_col)[col].sum().nlargest(n)
                    return f"Top {n} {group_col} by total {col}:\n{result.to_string()}"
                elif col:
                    result = df.nlargest(n, col)[self.columns[:5]]
                    return f"Top {n} rows by {col}:\n{result.to_string(index=False)}"

            # ── Bottom N ──────────────────────────────────────────
            if re.search(r'\bbottom\b|\blowest\b|\bworst\b', q):
                n = self._detect_n(q)
                col = self._detect_column(q) or (self.numeric_cols[0] if self.numeric_cols else None)
                if col:
                    result = df.nsmallest(n, col)[self.columns[:5]]
                    return f"Bottom {n} rows by {col}:\n{result.to_string(index=False)}"

            # ── Group by + aggregation ────────────────────────────
            if re.search(r'\bby\b|\bper\b|\beach\b|\bgroup\b', q):
                agg = self._detect_agg(q)
                # Find group column
                group_col = None
                for cat in self.cat_cols:
                    if cat.lower() in q or cat.lower().replace('_', ' ') in q:
                        group_col = cat
                        break
                # Find measure column
                measure_col = None
                for col in self.numeric_cols:
                    if col.lower() in q or col.lower().replace('_', ' ') in q:
                        measure_col = col
                        break
                measure_col = measure_col or (self.numeric_cols[0] if self.numeric_cols else None)
                group_col = group_col or (self.cat_cols[0] if self.cat_cols else None)

                if group_col and measure_col:
                    result = getattr(df.groupby(group_col)[measure_col], agg)()
                    result = result.sort_values(ascending=False)
                    return f"{agg.capitalize()} of {measure_col} by {group_col}:\n{result.to_string()}"

            # ── Filter rows (e.g., "show rows where salary > 100000") ──
            filter_match = re.search(
                r'where\s+(\w+)\s*(>|<|>=|<=|==|=|!=)\s*([\d.,]+)', q)
            if filter_match:
                col_name = filter_match.group(1)
                op = filter_match.group(2).replace('=', '==')
                if op == '===':
                    op = '=='
                val = float(filter_match.group(3).replace(',', ''))
                # Find actual column name (case-insensitive)
                matched_col = next((c for c in self.columns if c.lower() == col_name.lower()), None)
                if matched_col:
                    filtered = df.query(f"`{matched_col}` {op} {val}")
                    return (f"Rows where {matched_col} {op} {val}: {len(filtered)} found\n"
                            f"{filtered.head(10).to_string(index=False)}")

            # ── Simple aggregation ────────────────────────────────
            agg = self._detect_agg(q)
            col = self._detect_column(q) or (self.numeric_cols[0] if self.numeric_cols else None)
            if col and col in self.numeric_cols:
                value = getattr(df[col], agg)()
                return f"The {agg} of {col} is: {value:,.2f}"

            # ── Describe / summary ────────────────────────────────
            if re.search(r'describe|summary|statistics|stat', q):
                return df.describe().round(2).to_string()

            # ── Missing values ────────────────────────────────────
            if re.search(r'missing|null|nan|empty', q):
                mv = df.isnull().sum()
                mv = mv[mv > 0]
                if mv.empty:
                    return "No missing values found in the dataset."
                return f"Missing values per column:\n{mv.to_string()}"

            return (f"I couldn't understand that question. Try:\n"
                    f"  - 'Show top 5 rows by revenue'\n"
                    f"  - 'What is the average salary?'\n"
                    f"  - 'Total revenue by department'\n"
                    f"  - 'Show rows where salary > 80000'\n"
                    f"  - 'How many rows are there?'")

        except Exception as e:
            return f"Error processing query: {e}"

    def run_interactive(self):
        print("\n" + "=" * 60)
        print(f"  DATA Q&A BOT  |  {len(self.df)} rows × {len(self.columns)} columns")
        print(f"  Columns: {self.columns}")
        print("=" * 60)
        while True:
            try:
                q = input("\n  Ask: ").strip()
                if q.lower() in ('quit', 'exit', 'q'):
                    print("  Bye!")
                    break
                print(f"\n  Answer:\n  {self.answer(q)}")
            except (KeyboardInterrupt, EOFError):
                break

    def run_demo(self):
        print("\n" + "=" * 60)
        print("  DATA Q&A BOT - DEMO")
        print("=" * 60)
        questions = [
            "How many rows are there?",
            "What is the average salary?",
            "Show top 3 months by revenue",
            "Total revenue by department",
            "Show rows where salary > 100000",
            "Which department has the highest total salary?",
        ]
        for q in questions:
            print(f"\n  Q: {q}")
            print(f"  A: {self.answer(q)}")


def main():
    parser = argparse.ArgumentParser(description="Data Q&A Bot")
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    parser.add_argument('--demo', action='store_true', help='Run demo with preset questions')
    args = parser.parse_args()

    if args.csv and os.path.exists(args.csv):
        df = pd.read_csv(args.csv)
        print(f"[INFO] Loaded: '{args.csv}' ({len(df)} rows)")
    else:
        df = generate_sample()
        print("[INFO] Using generated sample dataset.")

    bot = DataQABot(df)

    if args.demo or not args.csv:
        bot.run_demo()
    else:
        bot.run_interactive()


if __name__ == "__main__":
    main()
