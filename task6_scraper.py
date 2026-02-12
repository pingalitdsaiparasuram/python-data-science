"""
Task 6 – Web Scraper
======================
Scrapes job listings from a public jobs API / HTML source,
stores results in CSV, and analyzes trends.

Usage:
    python task6_scraper.py --source jobs --output scraped_jobs.csv
    python task6_scraper.py --source mock         # Uses mock data (no internet needed)

Skills: requests, BeautifulSoup, pandas
"""

import requests
import csv
import os
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# ─── Mock Data (fallback when network is unavailable) ─────────────────────────

MOCK_JOBS = [
    {"title": "Data Scientist", "company": "Google", "location": "San Francisco, CA",
     "salary": "$120K–$160K", "tags": "Python,ML,TensorFlow", "date_posted": "2024-01-15"},
    {"title": "ML Engineer", "company": "Meta", "location": "New York, NY",
     "salary": "$140K–$180K", "tags": "Python,PyTorch,AWS", "date_posted": "2024-01-20"},
    {"title": "Data Analyst", "company": "Amazon", "location": "Seattle, WA",
     "salary": "$90K–$120K", "tags": "SQL,Python,Tableau", "date_posted": "2024-01-18"},
    {"title": "Data Scientist", "company": "Netflix", "location": "Los Angeles, CA",
     "salary": "$130K–$170K", "tags": "Python,R,Statistics", "date_posted": "2024-01-22"},
    {"title": "AI Researcher", "company": "OpenAI", "location": "San Francisco, CA",
     "salary": "$200K+", "tags": "Python,Research,ML", "date_posted": "2024-01-25"},
    {"title": "Data Engineer", "company": "Uber", "location": "San Francisco, CA",
     "salary": "$130K–$160K", "tags": "Python,Spark,Airflow", "date_posted": "2024-01-10"},
    {"title": "Business Analyst", "company": "Microsoft", "location": "Redmond, WA",
     "salary": "$100K–$130K", "tags": "Excel,SQL,PowerBI", "date_posted": "2024-01-12"},
    {"title": "ML Engineer", "company": "Apple", "location": "Cupertino, CA",
     "salary": "$150K–$200K", "tags": "Python,CoreML,Swift", "date_posted": "2024-01-08"},
    {"title": "Data Scientist", "company": "Airbnb", "location": "San Francisco, CA",
     "salary": "$125K–$155K", "tags": "Python,A/BTesting,SQL", "date_posted": "2024-01-14"},
    {"title": "Data Analyst", "company": "Twitter", "location": "New York, NY",
     "salary": "$95K–$115K", "tags": "SQL,Python,Analytics", "date_posted": "2024-01-16"},
    {"title": "MLOps Engineer", "company": "Stripe", "location": "New York, NY",
     "salary": "$140K–$170K", "tags": "Python,Docker,Kubernetes", "date_posted": "2024-01-11"},
    {"title": "Data Scientist", "company": "LinkedIn", "location": "Sunnyvale, CA",
     "salary": "$120K–$150K", "tags": "Python,ML,NLP", "date_posted": "2024-01-19"},
]


# ─── Scraping functions ────────────────────────────────────────────────────────

def scrape_remotive_jobs(keyword: str = "data science", limit: int = 20) -> list:
    """
    Scrape remote tech jobs from the Remotive public API.
    Returns a list of job dicts.
    """
    print(f"[INFO] Fetching jobs for keyword: '{keyword}'...")
    url = f"https://remotive.com/api/remote-jobs?category=data&limit={limit}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; DataScienceBot/1.0)"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        jobs = data.get('jobs', [])

        parsed = []
        for job in jobs[:limit]:
            parsed.append({
                'title': job.get('title', 'N/A'),
                'company': job.get('company_name', 'N/A'),
                'location': job.get('candidate_required_location', 'Remote'),
                'salary': job.get('salary', 'Not Disclosed'),
                'tags': ','.join(job.get('tags', [])),
                'date_posted': job.get('publication_date', 'N/A')[:10]
                               if job.get('publication_date') else 'N/A',
            })

        print(f"[INFO] Successfully scraped {len(parsed)} jobs.")
        return parsed

    except requests.exceptions.RequestException as e:
        print(f"[WARNING] Network request failed: {e}")
        print("[INFO] Falling back to mock data...")
        return MOCK_JOBS


def scrape_quotes_demo() -> list:
    """
    Scrape quotes from quotes.toscrape.com as a HTML scraping demo.
    Returns list of quote dicts.
    """
    print("[INFO] Scraping quotes.toscrape.com (HTML scraping demo)...")
    quotes = []
    try:
        for page in range(1, 4):
            url = f"http://quotes.toscrape.com/page/{page}/"
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            for q in soup.find_all('div', class_='quote'):
                text = q.find('span', class_='text').get_text(strip=True)
                author = q.find('small', class_='author').get_text(strip=True)
                tags = [t.get_text(strip=True) for t in q.find_all('a', class_='tag')]
                quotes.append({'text': text[:100], 'author': author, 'tags': ','.join(tags)})
            time.sleep(0.5)
        print(f"[INFO] Scraped {len(quotes)} quotes.")
    except Exception as e:
        print(f"[WARNING] Could not scrape quotes: {e}")
    return quotes


# ─── Analysis ─────────────────────────────────────────────────────────────────

def analyze_jobs(df: pd.DataFrame, output_dir: str = ".") -> None:
    """Analyze job trends and save charts."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Job Market Analysis", fontsize=15, fontweight='bold')

    # 1. Top job titles
    title_counts = df['title'].value_counts().head(8)
    sns.barplot(x=title_counts.values, y=title_counts.index, ax=axes[0], palette='Blues_d')
    axes[0].set_title("Top Job Titles")
    axes[0].set_xlabel("Count")

    # 2. Top locations
    loc_counts = df['location'].value_counts().head(6)
    sns.barplot(x=loc_counts.values, y=loc_counts.index, ax=axes[1], palette='Greens_d')
    axes[1].set_title("Top Locations")
    axes[1].set_xlabel("Count")

    # 3. Top skills from tags
    all_tags = []
    for tags in df['tags'].dropna():
        all_tags.extend([t.strip() for t in str(tags).split(',') if t.strip()])
    skill_counts = pd.Series(Counter(all_tags)).nlargest(10)
    sns.barplot(x=skill_counts.values, y=skill_counts.index, ax=axes[2], palette='Oranges_d')
    axes[2].set_title("Top Skills in Demand")
    axes[2].set_xlabel("Mentions")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "job_trends.png")
    plt.savefig(chart_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Job trends chart saved: '{chart_path}'")


# ─── Save to CSV ──────────────────────────────────────────────────────────────

def save_to_csv(data: list, filepath: str) -> None:
    """Save list of dicts to CSV."""
    if not data:
        print("[WARNING] No data to save.")
        return
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"[SUCCESS] {len(df)} records saved to '{filepath}'")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Web Scraper for Job Listings")
    parser.add_argument('--source', type=str, default='jobs',
                        choices=['jobs', 'quotes', 'mock'],
                        help="Data source to scrape")
    parser.add_argument('--output', type=str, default='scraped_data.csv', help='Output CSV path')
    parser.add_argument('--chart_dir', type=str, default='.', help='Directory to save charts')
    args = parser.parse_args()

    os.makedirs(args.chart_dir, exist_ok=True)

    if args.source == 'mock':
        data = MOCK_JOBS
        print(f"[INFO] Using {len(data)} mock job records.")
    elif args.source == 'quotes':
        data = scrape_quotes_demo()
    else:
        data = scrape_remotive_jobs()

    save_to_csv(data, args.output)

    if args.source in ('jobs', 'mock'):
        df = pd.DataFrame(data)
        analyze_jobs(df, output_dir=args.chart_dir)

        # Print summary
        print("\n── Job Listings Summary ───────────────────────────")
        print(f"  Total jobs scraped : {len(df)}")
        print(f"  Unique companies   : {df['company'].nunique()}")
        print(f"  Unique titles      : {df['title'].nunique()}")
        print(f"  Top title          : {df['title'].mode()[0]}")


if __name__ == "__main__":
    main()
