# Python Data Science
# 🐍 Python for Data Science – All 15 Tasks

> **Author:** [Pingali TirumalaDatta Sai Parasuram]  
> **Submission:** Python for Data Science – Level 1 through Advanced/GenAI  
> **Tools:** Python 3.10+, pandas, scikit-learn, matplotlib, seaborn, FastAPI

---

## 📁 Project Structure

```
python-data-science/
│
├── tasks/
│   ├── task1_data_cleaning.py          # CSV cleaning: nulls, wrong values, dates
│   ├── task2_salary_analyzer.py        # Employee salary statistics & report
│   ├── task3_utils.py                  # Reusable utility library
│   ├── task4_sales_dashboard.py        # Sales analysis + charts
│   ├── task5_eda.py                    # Exploratory Data Analysis
│   ├── task6_scraper.py                # Web scraper (jobs/products)
│   ├── task7_regression.py             # Regression: Linear vs Random Forest
│   ├── task8_classification.py         # Classification: Logistic/DTree/RF
│   ├── task9_feature_engineering.py    # Feature engineering challenge
│   ├── task10_ml_pipeline.py           # End-to-end sklearn Pipeline + joblib
│   ├── task11_api_deployment.py        # FastAPI model serving
│   ├── task12_report_generator.py      # Automated Excel/PDF reports
│   ├── task13_text_analytics.py        # Sentiment analysis + keywords
│   ├── task14_chatbot.py               # FAQ chatbot (NLP)
│   └── task15_data_qa_bot.py           # Natural language CSV Q&A bot
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/python-data-science.git
cd python-data-science
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running Each Task

All tasks include built-in sample data, so they run out of the box without any external CSV file.

### Level 1 – Python Foundations

| Task | Command |
|------|---------|
| Task 1 – Data Cleaning | `python tasks/task1_data_cleaning.py --input raw.csv --output cleaned.csv` |
| Task 2 – Salary Analyzer | `python tasks/task2_salary_analyzer.py` |
| Task 3 – Utility Library | `python tasks/task3_utils.py` |

### Data Analysis

| Task | Command |
|------|---------|
| Task 4 – Sales Dashboard | `python tasks/task4_sales_dashboard.py` |
| Task 5 – EDA | `python tasks/task5_eda.py --output eda_output/` |
| Task 6 – Web Scraper | `python tasks/task6_scraper.py --source mock` |

### Machine Learning

| Task | Command |
|------|---------|
| Task 7 – Regression | `python tasks/task7_regression.py` |
| Task 8 – Classification | `python tasks/task8_classification.py` |
| Task 9 – Feature Engineering | `python tasks/task9_feature_engineering.py` |

### Real Industry Projects

| Task | Command |
|------|---------|
| Task 10 – ML Pipeline | `python tasks/task10_ml_pipeline.py --train` |
| Task 11 – API Deployment | `python tasks/task11_api_deployment.py` |
| Task 12 – Report Generator | `python tasks/task12_report_generator.py --format both` |

### Advanced / GenAI

| Task | Command |
|------|---------|
| Task 13 – Text Analytics | `python tasks/task13_text_analytics.py` |
| Task 14 – Chatbot | `python tasks/task14_chatbot.py --demo` |
| Task 15 – Data Q&A Bot | `python tasks/task15_data_qa_bot.py --demo` |

---

## 📊 Task Descriptions

### Task 1 – Data Cleaning Script
- Removes null rows, fills numeric nulls with median, fills string nulls with "Unknown"
- Detects and fixes negative values in columns like `age`, `salary`, `price`
- Standardizes all date columns to `YYYY-MM-DD` format
- Removes duplicate rows
- Prints a cleaning summary report

### Task 2 – Employee Salary Analyzer
- Computes overall salary stats: average, median, min, max, std deviation
- Groups by department to show department-wise breakdown
- Identifies top 3 highest and lowest paid employees
- Shows salary bracket distribution with ASCII bar chart

### Task 3 – Utility Library
- `remove_duplicates()`: Preserves order, supports `key=` and `case_sensitive=` args
- `normalize_text()`: Removes accents, punctuation, stopwords; case normalizes
- `calculate_zscore()`: Returns Z-scores + optional outlier detection
- `date_formatter()`: Parses 15+ date formats → outputs any target format

### Task 4 – Sales Data Dashboard
- Monthly revenue trend line chart
- Top 5 products by revenue (horizontal bar chart)
- Region-wise revenue distribution (pie chart)
- Exports PNG dashboard + summary CSVs

### Task 5 – EDA
- Summary statistics (mean, std, quartiles, skewness, kurtosis)
- Missing values report with percentages
- Correlation heatmap
- Outlier detection using IQR method
- Saves: distribution plots, correlation heatmap, boxplots, categorical charts

### Task 6 – Web Scraper
- Scrapes job listings from Remotive API (falls back to mock data)
- HTML scraping demo using BeautifulSoup
- Stores results in CSV
- Analyzes: top job titles, locations, in-demand skills

### Task 7 – Regression
- Trains 4 models: Linear Regression, Ridge, Random Forest, Gradient Boosting
- Metrics: RMSE, R², MAE, MAPE, 5-fold CV
- Feature importance chart
- Saves best model as `.pkl` via joblib

### Task 8 – Classification
- Trains: Logistic Regression, Decision Tree, Random Forest
- Metrics: Accuracy, Precision, Recall, F1, AUC-ROC
- Confusion matrices + ROC curves + metric comparison bar chart

### Task 9 – Feature Engineering
- Creates: debt-to-income ratio, income per age, risk score composite
- Bins age groups, flags high loan amounts
- One-hot encodes categoricals, polynomial interaction features
- Demonstrates 10%+ F1 improvement over baseline

### Task 10 – End-to-End ML Pipeline
- sklearn ColumnTransformer for numeric + categorical features
- Handles imputation, scaling, encoding in a single Pipeline
- Saves trained pipeline to `.pkl` with joblib
- `--predict` flag for loading saved model and making predictions

### Task 11 – API Deployment (FastAPI)
- `GET /health` → API health check
- `POST /predict` → single prediction with probability + risk level
- `POST /predict/batch` → batch predictions
- Interactive Swagger docs at `http://localhost:8000/docs`
- Install: `pip install fastapi uvicorn`

### Task 12 – Automated Report Generator
- Reads any sales CSV and computes KPIs automatically
- Exports multi-sheet Excel report (`openpyxl`)
- Exports multi-page PDF with charts (`matplotlib`)
- Outputs: KPI overview, monthly sales, region performance

### Task 13 – Text Analytics
- Rule-based + lexicon sentiment analysis (positive/negative/neutral)
- Handles negation ("not great" → negative)
- Extracts top keywords using frequency analysis
- ASCII word cloud visualization

### Task 14 – FAQ Chatbot
- Custom TF-IDF-like similarity for FAQ matching
- 15+ data science Q&A pairs in knowledge base
- Interactive and demo modes
- Easily extensible knowledge base

### Task 15 – Data Q&A Bot
- Natural language interface to any CSV/DataFrame
- Handles: aggregations, groupby, filtering, top-N, describe
- Example: "Show top 3 months by revenue" → groupby + nlargest
- Interactive and demo modes

---

## 🔧 Requirements

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
joblib>=1.3
requests>=2.31
beautifulsoup4>=4.12
openpyxl>=3.1
fastapi>=0.104          # Task 11 only
uvicorn>=0.24           # Task 11 only
```

---

## 📸 Screenshots

> Screenshots of outputs are located in the `/screenshots` folder of this repository.

---

## 📝 Submission Notes

- All 15 tasks are fully functional and include built-in sample data
- Each task can run standalone with `python tasks/taskN_*.py`
- Clean, PEP8-compliant code with docstrings and type hints
- Tasks 7–11 save models and outputs to disk
- See the video walkthrough for a live demonstration of all tasks

---

## 📄 License

MIT License — free to use and modify.
