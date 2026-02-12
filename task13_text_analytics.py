"""
Task 13 – Text Analytics (Sentiment Analysis + Word Cloud)
===========================================================
Performs sentiment analysis on product reviews,
generates a word cloud, and extracts top keywords.

Usage:
    python task13_text_analytics.py
"""

import pandas as pd
import numpy as np
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')


# ─── Sample reviews ───────────────────────────────────────────────────────────

SAMPLE_REVIEWS = [
    "This product is absolutely amazing! Best purchase I've ever made.",
    "Terrible quality. Broke after just 2 days of use. Very disappointed.",
    "Good value for the price. Shipping was fast. Satisfied overall.",
    "Not bad, but not great either. Average product for the price.",
    "Excellent build quality! The design is modern and sleek. Highly recommend.",
    "Waste of money. Customer service was useless. Never buying again.",
    "Works perfectly as described. Very happy with this purchase.",
    "The product is okay. Nothing special but gets the job done.",
    "Outstanding performance! Exceeded all my expectations.",
    "Poor packaging, item arrived damaged. Terrible experience.",
    "Amazing product! Love it. Will buy more from this brand.",
    "Mediocre quality. Expected better for this price point.",
    "Great product. Fast delivery. Excellent customer support.",
    "Defective unit received. Had to return immediately.",
    "Fantastic! Very impressed with the quality and durability.",
]


def rule_based_sentiment(text: str) -> dict:
    """
    Simple rule-based sentiment analysis using word lexicons.
    Returns: {'label': 'positive/negative/neutral', 'score': float, 'confidence': float}
    """
    positive_words = {
        'amazing', 'excellent', 'fantastic', 'great', 'good', 'love', 'best',
        'outstanding', 'impressed', 'happy', 'satisfied', 'perfect', 'wonderful',
        'awesome', 'superb', 'recommend', 'fast', 'quality', 'pleased', 'brilliant'
    }
    negative_words = {
        'terrible', 'horrible', 'awful', 'bad', 'poor', 'disappointed', 'waste',
        'broken', 'defective', 'useless', 'damaged', 'mediocre', 'worst', 'terrible',
        'never', 'refund', 'return', 'avoid', 'disappointing', 'overpriced'
    }
    # Negation flips sentiment
    negation = {'not', 'never', 'no', "n't"}

    words = re.findall(r'\b\w+\b', text.lower())

    pos_count = neg_count = 0
    i = 0
    while i < len(words):
        word = words[i]
        negate = i > 0 and words[i - 1] in negation
        if word in positive_words:
            neg_count += 1 if negate else 0
            pos_count += 0 if negate else 1
        elif word in negative_words:
            pos_count += 1 if negate else 0
            neg_count += 0 if negate else 1
        i += 1

    total = pos_count + neg_count
    if total == 0:
        return {'label': 'neutral', 'score': 0.0, 'confidence': 0.5}

    score = (pos_count - neg_count) / total
    if score > 0.1:
        label = 'positive'
    elif score < -0.1:
        label = 'negative'
    else:
        label = 'neutral'

    confidence = abs(score) * 0.8 + 0.2
    return {'label': label, 'score': round(score, 3), 'confidence': round(confidence, 3)}


def extract_keywords(texts: list, top_n: int = 15) -> list:
    """Extract top N keywords by frequency (removing stopwords)."""
    stopwords = {
        'the', 'a', 'an', 'is', 'it', 'i', 'my', 'this', 'was', 'and', 'or', 'but',
        'for', 'with', 'to', 'of', 'in', 'on', 'at', 'by', 'as', 'are', 'be', 'been',
        'have', 'has', 'had', 'do', 'did', 'will', 'would', 'could', 'should', 'not',
        'from', 'that', 'very', 'just', 'all', 'got', 'get', 'after', 'days', 'use',
        'its', 'ever', 'also', 'me', 'so'
    }
    words = []
    for text in texts:
        tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
        words.extend([w for w in tokens if w not in stopwords])

    return Counter(words).most_common(top_n)


def generate_wordcloud_text(keywords: list) -> str:
    """Generate ASCII-style word cloud from keyword frequencies."""
    if not keywords:
        return ""
    max_freq = keywords[0][1]
    sizes = [3, 2, 1]
    cloud = "\n  ASCII Word Cloud (size ∝ frequency):\n"
    cloud += "  " + "─" * 50 + "\n  "
    for word, freq in keywords:
        size_idx = 0 if freq >= max_freq * 0.7 else (1 if freq >= max_freq * 0.4 else 2)
        formatted = word.upper() if size_idx == 0 else (word.title() if size_idx == 1 else word)
        cloud += f"{formatted}  "
    return cloud


def run_sentiment_analysis():
    reviews = SAMPLE_REVIEWS
    results = []
    for review in reviews:
        sentiment = rule_based_sentiment(review)
        results.append({
            'review': review[:70] + ('...' if len(review) > 70 else ''),
            'sentiment': sentiment['label'],
            'score': sentiment['score'],
            'confidence': sentiment['confidence']
        })

    df = pd.DataFrame(results)
    keywords = extract_keywords(reviews)

    print("\n" + "=" * 65)
    print("         TEXT ANALYTICS REPORT")
    print("=" * 65)

    print("\n── Sentiment Distribution ───────────────────────────────────")
    dist = df['sentiment'].value_counts()
    for label, count in dist.items():
        bar = "█" * count
        print(f"  {label:<10} {bar} ({count}/{len(df)})")

    print("\n── Sample Predictions ───────────────────────────────────────")
    for _, row in df.head(6).iterrows():
        icon = "😊" if row['sentiment'] == 'positive' else ("😞" if row['sentiment'] == 'negative' else "😐")
        print(f"  {icon} [{row['sentiment'].upper():<8} | {row['score']:+.2f}] {row['review']}")

    print("\n── Top Keywords ─────────────────────────────────────────────")
    for word, freq in keywords[:10]:
        print(f"  {word:<20} {'▓' * freq} ({freq})")

    print(generate_wordcloud_text(keywords))
    df.to_csv("sentiment_results.csv", index=False)
    print("\n[SUCCESS] Results saved to 'sentiment_results.csv'")


if __name__ == "__main__":
    run_sentiment_analysis()
