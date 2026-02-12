"""
Task 3 – Build Your Own Utility Library
=========================================
A reusable Python utility module with:
  - remove_duplicates()
  - normalize_text()
  - calculate_zscore()
  - date_formatter()

Usage (import in other scripts):
    from task3_utils import remove_duplicates, normalize_text, calculate_zscore, date_formatter

Usage (run standalone demo):
    python task3_utils.py
"""

import re
import unicodedata
from datetime import datetime
from typing import List, Union, Optional
import statistics


# ──────────────────────────────────────────────────────────────────────────────
# 1. remove_duplicates
# ──────────────────────────────────────────────────────────────────────────────

def remove_duplicates(data: list, key=None, case_sensitive: bool = True) -> list:
    """
    Remove duplicates from a list while preserving original order.

    Args:
        data          : Input list (strings, dicts, or any hashable items)
        key           : Optional function to extract comparison key (for dicts/objects)
        case_sensitive: If False, treats strings as equal regardless of case

    Returns:
        List with duplicates removed (order preserved)

    Examples:
        >>> remove_duplicates([1, 2, 2, 3, 1])
        [1, 2, 3]
        >>> remove_duplicates(['Apple', 'apple', 'APPLE'], case_sensitive=False)
        ['Apple']
        >>> remove_duplicates([{'id':1,'name':'A'},{'id':1,'name':'B'}], key=lambda x: x['id'])
        [{'id': 1, 'name': 'A'}]
    """
    seen = set()
    result = []
    for item in data:
        if key:
            compare_val = key(item)
        else:
            compare_val = item

        if isinstance(compare_val, str) and not case_sensitive:
            compare_val = compare_val.lower()

        if compare_val not in seen:
            seen.add(compare_val)
            result.append(item)

    removed = len(data) - len(result)
    if removed:
        print(f"[remove_duplicates] Removed {removed} duplicate(s). {len(result)} item(s) remain.")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 2. normalize_text
# ──────────────────────────────────────────────────────────────────────────────

def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_extra_spaces: bool = True,
    remove_accents: bool = True,
    remove_numbers: bool = False,
    remove_stopwords: bool = False,
    custom_stopwords: Optional[List[str]] = None
) -> str:
    """
    Normalize a text string with configurable options.

    Args:
        text               : Input string
        lowercase          : Convert to lowercase
        remove_punctuation : Strip all punctuation
        remove_extra_spaces: Collapse multiple spaces to one
        remove_accents     : Strip diacritics (é→e, ñ→n, etc.)
        remove_numbers     : Strip numeric characters
        remove_stopwords   : Remove common English stopwords
        custom_stopwords   : Optional extra words to remove

    Returns:
        Cleaned/normalized string

    Examples:
        >>> normalize_text("  Hello,  Wörld!  ")
        'hello world'
        >>> normalize_text("The cat sat on the mat.", remove_stopwords=True)
        'cat sat mat'
    """
    if not isinstance(text, str):
        text = str(text)

    # Remove accents
    if remove_accents:
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')

    # Lowercase
    if lowercase:
        text = text.lower()

    # Remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)

    # Remove punctuation (keep spaces and letters)
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'_', ' ', text)

    # Remove stopwords
    default_stopwords = {
        'a', 'an', 'the', 'is', 'it', 'in', 'on', 'at', 'to', 'of',
        'and', 'or', 'but', 'for', 'with', 'by', 'from', 'was', 'are',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'
    }
    if remove_stopwords:
        stop = default_stopwords
        if custom_stopwords:
            stop = stop.union(set(w.lower() for w in custom_stopwords))
        text = ' '.join(w for w in text.split() if w not in stop)

    # Collapse whitespace
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text).strip()

    return text


# ──────────────────────────────────────────────────────────────────────────────
# 3. calculate_zscore
# ──────────────────────────────────────────────────────────────────────────────

def calculate_zscore(
    data: List[Union[int, float]],
    return_outliers: bool = False,
    threshold: float = 2.0
) -> Union[List[float], dict]:
    """
    Compute the Z-score for each value in a numeric list.

    Z-score = (x - mean) / std_deviation

    Args:
        data           : List of numeric values
        return_outliers: If True, also return values beyond ±threshold
        threshold      : Z-score threshold for outlier detection (default ±2.0)

    Returns:
        List of Z-scores, OR a dict with 'zscores' and 'outliers' if return_outliers=True

    Examples:
        >>> calculate_zscore([10, 20, 30, 40, 50])
        [-1.41, -0.71, 0.0, 0.71, 1.41]
        >>> calculate_zscore([10, 20, 100], return_outliers=True, threshold=1.5)
        {'zscores': [...], 'outliers': [{'index': 2, 'value': 100, 'zscore': 1.56}]}
    """
    if len(data) < 2:
        raise ValueError("Data must contain at least 2 values.")

    if not all(isinstance(x, (int, float)) for x in data):
        raise TypeError("All values must be numeric (int or float).")

    mean = statistics.mean(data)
    std = statistics.stdev(data)  # sample std

    if std == 0:
        zscores = [0.0] * len(data)
    else:
        zscores = [round((x - mean) / std, 4) for x in data]

    if return_outliers:
        outliers = [
            {'index': i, 'value': data[i], 'zscore': z}
            for i, z in enumerate(zscores)
            if abs(z) > threshold
        ]
        return {'zscores': zscores, 'mean': round(mean, 4), 'std': round(std, 4), 'outliers': outliers}

    return zscores


# ──────────────────────────────────────────────────────────────────────────────
# 4. date_formatter
# ──────────────────────────────────────────────────────────────────────────────

def date_formatter(
    date_input: Union[str, datetime],
    output_format: str = "%Y-%m-%d",
    input_formats: Optional[List[str]] = None
) -> str:
    """
    Parse a date string (in various formats) and return it in a standardized format.

    Args:
        date_input    : Date string or datetime object
        output_format : Desired output format (default: YYYY-MM-DD)
        input_formats : List of possible input formats to try

    Returns:
        Formatted date string

    Raises:
        ValueError: If the date cannot be parsed

    Examples:
        >>> date_formatter("25/12/2023")
        '2023-12-25'
        >>> date_formatter("December 25, 2023", output_format="%d %b %Y")
        '25 Dec 2023'
        >>> date_formatter("2023-12-25", output_format="%B %d, %Y")
        'December 25, 2023'
    """
    if isinstance(date_input, datetime):
        return date_input.strftime(output_format)

    if not isinstance(date_input, str):
        raise TypeError(f"Expected str or datetime, got {type(date_input)}")

    date_input = date_input.strip()

    # Default formats to try
    default_formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y",
        "%Y/%m/%d", "%d %b %Y", "%d %B %Y", "%B %d, %Y", "%b %d, %Y",
        "%d.%m.%Y", "%Y%m%d", "%b %d %Y", "%d/%m/%y", "%m/%d/%y",
    ]

    formats_to_try = (input_formats or []) + default_formats

    for fmt in formats_to_try:
        try:
            parsed = datetime.strptime(date_input, fmt)
            return parsed.strftime(output_format)
        except ValueError:
            continue

    raise ValueError(
        f"Could not parse date: '{date_input}'. "
        f"Tried formats: {formats_to_try[:5]} ... Pass `input_formats` with the correct format."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Demo / Self-test
# ──────────────────────────────────────────────────────────────────────────────

def run_demo():
    SEP = "─" * 55

    print("\n" + "=" * 55)
    print("         UTILITY LIBRARY DEMO")
    print("=" * 55)

    # 1. remove_duplicates
    print(f"\n{SEP}")
    print("1. remove_duplicates()")
    print(SEP)
    nums = [1, 2, 2, 3, 3, 3, 4]
    print(f"  Input : {nums}")
    print(f"  Output: {remove_duplicates(nums)}")

    words = ['Hello', 'world', 'HELLO', 'World']
    print(f"\n  Input (case-insensitive): {words}")
    print(f"  Output: {remove_duplicates(words, case_sensitive=False)}")

    records = [{'id': 1, 'val': 'a'}, {'id': 2, 'val': 'b'}, {'id': 1, 'val': 'c'}]
    print(f"\n  Input (dicts, key=id): {records}")
    print(f"  Output: {remove_duplicates(records, key=lambda x: x['id'])}")

    # 2. normalize_text
    print(f"\n{SEP}")
    print("2. normalize_text()")
    print(SEP)
    raw = "  The Café Déjà Vu has 3 great deals!  "
    print(f"  Input : '{raw}'")
    print(f"  Basic : '{normalize_text(raw)}'")
    print(f"  No num: '{normalize_text(raw, remove_numbers=True)}'")
    print(f"  No stop: '{normalize_text(raw, remove_stopwords=True)}'")

    # 3. calculate_zscore
    print(f"\n{SEP}")
    print("3. calculate_zscore()")
    print(SEP)
    scores = [50, 55, 60, 62, 65, 200]
    print(f"  Input: {scores}")
    result = calculate_zscore(scores, return_outliers=True, threshold=1.5)
    print(f"  Z-scores: {result['zscores']}")
    print(f"  Mean: {result['mean']}, Std: {result['std']}")
    print(f"  Outliers: {result['outliers']}")

    # 4. date_formatter
    print(f"\n{SEP}")
    print("4. date_formatter()")
    print(SEP)
    dates = ["25/12/2023", "December 25, 2023", "2023-12-25", "25.12.2023"]
    for d in dates:
        print(f"  '{d}' → '{date_formatter(d)}'")
    print(f"\n  Custom output: '{date_formatter('2023-12-25', output_format='%B %d, %Y')}'")

    print(f"\n{SEP}")
    print("  All utility functions working correctly!")
    print(SEP)


if __name__ == "__main__":
    run_demo()
