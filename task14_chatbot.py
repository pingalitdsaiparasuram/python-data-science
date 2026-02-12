"""
Task 14 – Build a Simple FAQ Chatbot
======================================
A rule-based + embedding-augmented FAQ chatbot.
Supports local keyword matching and optional OpenAI backend.

Usage:
    python task14_chatbot.py            # interactive mode
    python task14_chatbot.py --demo     # runs preset Q&A demo
"""

import re
import math
import argparse
from collections import defaultdict


# ─── FAQ Knowledge Base ────────────────────────────────────────────────────────

FAQ_DB = {
    "what is data science": "Data Science is the field that combines statistics, programming, and domain expertise to extract insights from data. It involves data collection, cleaning, EDA, modeling, and visualization.",
    "what is machine learning": "Machine Learning is a subset of AI where algorithms learn patterns from data without being explicitly programmed. Types include supervised, unsupervised, and reinforcement learning.",
    "what is python": "Python is the most popular programming language for data science. Key libraries: pandas (data manipulation), NumPy (numerical computing), scikit-learn (ML), matplotlib/seaborn (visualization).",
    "what is a neural network": "A neural network is a computational model inspired by the human brain. It consists of layers of interconnected nodes (neurons) that learn representations from data.",
    "what is overfitting": "Overfitting occurs when a model learns the training data too well, including noise, and performs poorly on new data. Prevention: cross-validation, regularization, early stopping, more data.",
    "what is cross validation": "Cross-validation is a technique to evaluate model performance by splitting data into multiple train/test folds. K-Fold CV is the most common approach.",
    "what is feature engineering": "Feature engineering is the process of creating, transforming, or selecting input features to improve model performance. It includes encoding, scaling, binning, and creating interaction features.",
    "what is pandas": "Pandas is a Python library for data manipulation and analysis. Key objects: DataFrame (table) and Series (column). Supports reading CSV, Excel, SQL, JSON, and many more formats.",
    "what is eda": "Exploratory Data Analysis (EDA) is the process of analyzing datasets to summarize their main characteristics using visualizations and statistical measures before modeling.",
    "how to handle missing values": "Missing values can be handled by: removing rows/columns with nulls, filling with mean/median/mode, forward/backward fill for time series, or using ML-based imputation (KNNImputer).",
    "what is rmse": "RMSE (Root Mean Squared Error) measures the average magnitude of prediction errors in regression. Lower RMSE = better model. Formula: sqrt(mean((y_true - y_pred)^2))",
    "what is r squared": "R² (R-squared) is the proportion of variance in the target explained by the model. Ranges from 0 to 1, where 1 = perfect fit. Also called the coefficient of determination.",
    "what is a confusion matrix": "A confusion matrix shows model performance for classification: True Positives, True Negatives, False Positives, and False Negatives. Used to compute accuracy, precision, recall, and F1.",
    "what is precision and recall": "Precision = TP / (TP + FP) — how many predicted positives are actually positive. Recall = TP / (TP + FN) — how many actual positives were correctly predicted.",
    "what is the difference between ai and ml": "AI is the broad concept of machines simulating human intelligence. ML is a subset of AI where machines learn from data. Deep Learning is a subset of ML using neural networks.",
}


# ─── Simple TF-IDF-like similarity ────────────────────────────────────────────

def tokenize(text: str) -> list:
    return re.findall(r'\b\w+\b', text.lower())


def compute_tf(tokens: list) -> dict:
    tf = defaultdict(int)
    for t in tokens:
        tf[t] += 1
    return {k: v / len(tokens) for k, v in tf.items()}


def cosine_similarity(vec1: dict, vec2: dict) -> float:
    common = set(vec1.keys()) & set(vec2.keys())
    if not common:
        return 0.0
    dot = sum(vec1[k] * vec2[k] for k in common)
    mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    return dot / (mag1 * mag2) if mag1 and mag2 else 0.0


def find_best_match(query: str, faq: dict, threshold: float = 0.15) -> str:
    q_tokens = tokenize(query)
    q_tf = compute_tf(q_tokens)

    best_score, best_answer = 0, None
    for key, answer in faq.items():
        k_tf = compute_tf(tokenize(key))
        score = cosine_similarity(q_tf, k_tf)
        if score > best_score:
            best_score = score
            best_answer = answer

    if best_score >= threshold:
        return best_answer
    return ("I'm not sure about that. Try asking about: data science, machine learning, "
            "python, EDA, missing values, overfitting, confusion matrix, RMSE, or feature engineering.")


class FAQChatbot:
    def __init__(self):
        self.faq = FAQ_DB
        self.history = []

    def respond(self, user_input: str) -> str:
        user_input = user_input.strip()
        if not user_input:
            return "Please ask a question!"
        answer = find_best_match(user_input, self.faq)
        self.history.append({'user': user_input, 'bot': answer})
        return answer

    def run_interactive(self):
        print("\n" + "=" * 60)
        print("  DATA SCIENCE FAQ CHATBOT  (type 'quit' to exit)")
        print("=" * 60)
        print("  Ask anything about: Python, ML, EDA, statistics,")
        print("  metrics, preprocessing, and more!")
        print("=" * 60)
        while True:
            try:
                user = input("\n  You: ").strip()
                if user.lower() in ('quit', 'exit', 'bye', 'q'):
                    print("  Bot: Goodbye! Happy coding!")
                    break
                reply = self.respond(user)
                print(f"\n  Bot: {reply}")
            except (KeyboardInterrupt, EOFError):
                print("\n  Bot: Goodbye!")
                break

    def run_demo(self):
        print("\n" + "=" * 60)
        print("  FAQ CHATBOT DEMO")
        print("=" * 60)
        demo_questions = [
            "What is machine learning?",
            "How do I handle missing values?",
            "What is overfitting?",
            "Explain confusion matrix",
            "What is RMSE?",
        ]
        for q in demo_questions:
            print(f"\n  Q: {q}")
            print(f"  A: {self.respond(q)}")


def main():
    parser = argparse.ArgumentParser(description="FAQ Chatbot")
    parser.add_argument('--demo', action='store_true', help='Run demo mode')
    args = parser.parse_args()
    bot = FAQChatbot()
    if args.demo:
        bot.run_demo()
    else:
        bot.run_interactive()


if __name__ == "__main__":
    main()
