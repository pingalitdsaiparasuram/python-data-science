# Python Data Science
Task 1 – Data Cleaning Script
Objective
The goal is to build a robust, reusable script that ingests a raw CSV file and produces a clean, analysis-ready version. Real-world datasets are almost always messy — they contain null values, incorrect entries, inconsistent date formats, and duplicate rows. This script systematically addresses all of these issues.

What the Script Does
Loads any CSV file using pandas
Removes all-null rows and fills numeric nulls with column medians
Fills missing string values with 'Unknown'
Detects and corrects negative values in columns like age, salary, or price
Removes duplicate rows automatically
Detects date columns by name heuristics and standardizes them to YYYY-MM-DD
Prints a final cleaning report showing before/after row counts

Key Functions
remove_nulls(df)
This function handles missing data in two ways. First, it drops any row where every single column is null (completely empty rows). Then for numeric columns, it fills remaining nulls with the median of that column — which is robust to outliers. For string or object columns, nulls are replaced with the placeholder 'Unknown'.
replace_wrong_values(df)
Handles data integrity issues: strips leading/trailing whitespace from all string fields, detects negative values in logically non-negative columns (age, salary, price, amount, quantity), replaces them with the column median, and then drops all exact duplicate rows.
standardize_dates(df)
Automatically detects date columns using keyword matching in the column name (e.g., 'date', 'dob', 'created'). It then tries 10+ common date formats (DD/MM/YYYY, MM-DD-YYYY, etc.) and converts all dates to the universal ISO standard YYYY-MM-DD.

How to Run
python task1_data_cleaning.py --input raw_data.csv --output cleaned_data.csv

Output
A cleaned CSV file at the specified output path, plus a console report showing the original row count, cleaned row count, rows removed, and final null value count.

Task 2 – Employee Salary Analyzer
Objective
Build a program that reads employee data and produces a rich, formatted salary analysis report. This simulates real HR analytics work where you need to quickly understand pay structures, identify disparities, and summarize workforce compensation.

Analysis Performed
Overall Company Stats	Department-wise Stats
Average salary	Department avg/median/min/max
Median salary	Department headcount
Min and Max salary	Top 3 highest earners
Std deviation	Bottom 3 earners
Total salary expense	Salary bracket distribution

Key Functions
calculate_overall_stats(df)
Computes total employees, total salary expense, average, median, min, max, and standard deviation of salaries across the entire company.
department_salary_analysis(df)
Uses pandas groupby to compute per-department salary statistics. Results are sorted by average salary in descending order for easy reading.
salary_bracket_distribution(df)
Bins employees into salary ranges (< $50K, $50K–$80K, etc.) using pd.cut and displays the distribution as an ASCII bar chart.

How to Run
python task2_salary_analyzer.py --input employees.csv
python task2_salary_analyzer.py          # runs with built-in sample data

Sample Output
The report is formatted in a professional console layout with aligned columns, dollar signs, separator lines, and an ASCII bar chart showing salary distribution.

Task 3 – Utility Library
Objective
Create a standalone Python module of reusable, well-documented utility functions that can be imported into any other project. This promotes the Don't Repeat Yourself (DRY) principle in data science workflows.

Functions Implemented
remove_duplicates(data, key=None, case_sensitive=True)
Removes duplicate items from a list while strictly preserving the original order. Supports case-insensitive deduplication for strings, and a key= function for deduplicating complex objects (like dicts) based on a specific field.
remove_duplicates([1, 2, 2, 3]) → [1, 2, 3]
remove_duplicates(["Apple","apple"], case_sensitive=False) → ["Apple"]

normalize_text(text, ...)
Cleans and normalizes any text string. Options include: converting to lowercase, removing punctuation, collapsing whitespace, stripping Unicode accents (é→e, ñ→n), removing numbers, and filtering stopwords. All options are configurable with boolean flags.
normalize_text("  The Café DÉJÀ VU!  ") → "cafe deja vu"

calculate_zscore(data, return_outliers=False, threshold=2.0)
Computes the Z-score for every value in a numeric list using sample standard deviation. Z-score = (x - mean) / std. Optionally detects outliers beyond ±threshold and returns them with their index, value, and Z-score.
calculate_zscore([10, 20, 100], return_outliers=True)
→ {zscores: [...], outliers: [{index: 2, value: 100, zscore: 1.56}]}

date_formatter(date_input, output_format='%Y-%m-%d')
Intelligently parses a date string in any of 15+ common formats and converts it to a target output format. Raises a helpful ValueError if the date cannot be parsed.
date_formatter("25/12/2023") → "2023-12-25"
date_formatter("Dec 25 2023", "%B %d, %Y") → "December 25, 2023"

How to Run
python task3_utils.py                   # runs full demo
from task3_utils import normalize_text  # import in other scripts

Task 4 – Sales Data Dashboard
Objective
Analyze a sales dataset and produce a multi-panel visual dashboard using matplotlib and seaborn. This mimics the kind of business intelligence work done at companies like Amazon, Flipkart, or any e-commerce platform.

Charts Produced
Dashboard Panels
• Monthly Revenue Trend – Line chart showing revenue trajectory over 12 months with data labels
• Top 5 Products by Revenue – Horizontal bar chart sorted by revenue descending
• Region-wise Revenue Distribution – Pie chart showing each region's share of total revenue

Data Used
The script generates 500 rows of synthetic sales data with columns: date, product, region, quantity, unit_price, and revenue. It includes realistic product price distributions and seasonal variation. You can also pass your own CSV with --input.

Key Analysis Functions
monthly_revenue(df) — Groups by calendar month, sums revenue, returns a sorted monthly series
top_products(df, n=5) — Groups by product name, returns top N by total revenue
region_performance(df) — Groups by region, computes total revenue, order count, and average order value

How to Run
python task4_sales_dashboard.py --input sales.csv --output charts/
python task4_sales_dashboard.py          # uses generated sample data

Outputs
sales_dashboard.png — A 2×2 grid PNG dashboard. sales_summary.csv, top_products.csv, region_performance.csv — Summary CSVs.

Task 5 – Exploratory Data Analysis (EDA)
Objective
Perform a comprehensive Exploratory Data Analysis on any dataset. EDA is the critical first step in every data science project — it reveals data quality issues, feature distributions, correlations, and anomalies that guide all subsequent modeling decisions.

EDA Components
Statistical Analysis	Visualizations
Summary statistics (count, mean, std, quartiles)	Correlation heatmap (lower triangle)
Missing values count and percentage per column	Distribution histograms with KDE curves
Skewness and kurtosis per numeric column	Boxplots for outlier visualization
Data types and shape	Categorical bar charts

Outlier Detection Method
Uses the IQR (Interquartile Range) method: outliers are values below Q1 – 1.5×IQR or above Q3 + 1.5×IQR. This is more robust than Z-score for skewed distributions. The report shows outlier count and percentage per column.

Skewness Interpretation
Skewness between -0.5 and 0.5: approximately symmetric — safe for most models
Skewness between 0.5 and 1 (or -1 to -0.5): moderately skewed — consider log transform
Skewness above 1 or below -1: highly skewed — log/sqrt transform strongly recommended

How to Run
python task5_eda.py --input my_dataset.csv --output eda_output/
python task5_eda.py          # uses generated HR dataset

Output Files
eda_distributions.png, eda_correlation.png, eda_outliers.png, eda_categorical.png — plus a detailed printed console report.

Task 6 – Web Scraper
Objective
Build a web scraper that collects job listings or product prices from the internet, stores the results in CSV format, and analyzes emerging trends. Web scraping is a core data collection skill used by data scientists when public APIs are not available.

Libraries Used
Library	Purpose
requests	HTTP requests to web pages and APIs
BeautifulSoup (bs4)	HTML parsing and data extraction
pandas	Data storage and analysis
matplotlib/seaborn	Trend visualization

Data Sources
Source 1: Remotive API (Remote Jobs)
Fetches remote tech job listings from the Remotive public API. Extracts: job title, company name, candidate location, salary, required skills/tags, and posting date. If the network is unavailable, the script automatically falls back to 20 built-in mock job records so you can always run it offline.
Source 2: BeautifulSoup HTML Demo
Demonstrates traditional HTML scraping on quotes.toscrape.com. Uses BeautifulSoup's CSS selector pattern to extract quote text, author name, and tags — showing how to navigate DOM structure.

Trend Analysis
Top Job Titles — Which roles appear most frequently (Data Scientist, ML Engineer, Data Analyst, etc.)
Top Locations — Cities and countries with the most remote-friendly jobs
Most In-Demand Skills — Technologies appearing most in job tags (Python, SQL, TensorFlow, etc.)

How to Run
python task6_scraper.py --source jobs     # live scrape
python task6_scraper.py --source mock     # offline demo mode

Task 7 – Regression Model
Objective
Build and compare multiple regression models to predict house prices. Regression is the most fundamental supervised learning task, where the goal is to predict a continuous numeric value. This task demonstrates the full modeling lifecycle.

Dataset Features
Synthetic house price data with features: bedrooms, bathrooms, square footage, age, garage spaces, and neighborhood type (Urban/Suburban/Rural). The target variable is price.

Models Compared
Regression Models
• Linear Regression — baseline model; assumes linear relationship between features and target
• Ridge Regression — adds L2 regularization to prevent overfitting; best when features are correlated
• Random Forest Regressor — ensemble of 100 decision trees; handles non-linear patterns well
• Gradient Boosting — sequential tree boosting; typically the highest accuracy on tabular data

Evaluation Metrics
Metric	Meaning
RMSE (Root Mean Squared Error)	Average prediction error in dollars — lower is better
R² Score	% of variance explained — higher is better (max 1.0)
MAE (Mean Absolute Error)	Average absolute error — more interpretable than RMSE
MAPE (Mean Absolute % Error)	Error as a percentage of actual value
5-Fold Cross-Validation R²	Generalization estimate across 5 train/test splits

Outputs
Console table comparing all 4 models across all metrics
regression_results.png — predicted vs actual scatter, RMSE/R² bar chart, feature importance
best_model.pkl — saved best-performing model for deployment

How to Run
python task7_regression.py --input house_prices.csv
python task7_regression.py          # uses generated house data

Task 8 – Classification Model
Objective
Train and evaluate multiple classification models on a customer churn dataset. Classification predicts a discrete category (churn/no-churn). This task covers the full workflow: data preparation, model training, metric evaluation, and visual comparison.

Dataset
Synthetic customer churn data with features: tenure (months as customer), monthly charges, total charges, support call count, contract type (Month-to-Month, One Year, Two Year), internet service type, and churn label (0 or 1).

Models
Model	Characteristics
Logistic Regression (with StandardScaler)	Fast, interpretable, good baseline for binary classification
Decision Tree (max_depth=5)	Highly interpretable; shows decision rules; can overfit
Random Forest (100 trees)	Most robust; reduces variance via ensemble; best F1

Metrics Explained
Accuracy
The percentage of all predictions that were correct. Misleading when classes are imbalanced (e.g., 80% non-churners always predicted would give 80% accuracy).
Precision
Of all customers predicted to churn, what fraction actually churned? High precision = few false alarms. Important when follow-up actions are costly.
Recall
Of all customers who actually churned, what fraction did we catch? High recall = few missed churners. Important when missing churn is costly.
F1 Score
The harmonic mean of precision and recall. The best single metric for imbalanced classification. Ranges 0–1, higher is better.
AUC-ROC
Area Under the ROC Curve. Measures model discrimination ability across all threshold settings. AUC = 1.0 is perfect; AUC = 0.5 is no better than random.

Confusion Matrix
A 2×2 grid showing: True Positives (correctly predicted churn), True Negatives (correctly predicted no-churn), False Positives (predicted churn but didn't), False Negatives (missed churners). Visual heatmaps are generated for each model.

How to Run
python task8_classification.py --input churn_data.csv
python task8_classification.py          # uses generated churn data

Task 9 – Feature Engineering Challenge
Objective
Demonstrate how thoughtful feature engineering can significantly boost model performance. The challenge: improve baseline F1 score by at least 10% through creating new features, handling categoricals, and scaling.

Feature Engineering Techniques Applied
1. Ratio Features
debt_to_income = loan_amount / income — captures the borrower's debt burden relative to earnings, which is one of the strongest default predictors. income_per_age = income / age — normalizes income by life stage.
2. Composite Risk Score
risk_score = (1000 - credit_score) × debt_to_income — combines two strong predictors into a single engineered feature, giving the model a direct risk signal.
3. Binning
age groups into Young (< 30), Middle (30–45), Senior (45–60), Elderly (60+) using pd.cut. Binning can help non-linear models by reducing noise in age.
4. Boolean Flag
high_loan = 1 if loan_amount > $30,000. Binary flags help tree-based models create clean splits on important thresholds.
5. One-Hot Encoding
pd.get_dummies converts categorical columns like employment_type and loan_purpose into binary indicator columns, making them usable by linear models.
6. Interaction Feature
income_credit_interact = income × credit_score / 1e6 — captures the combined effect of two features that individually might not tell the full story.

How to Run
python task9_feature_engineering.py

Expected Output
Prints baseline F1, improved F1, and the percentage improvement. The goal of 10% improvement is typically achieved by the combination of Gradient Boosting + the engineered features above.

Task 10 – End-to-End ML Pipeline
Objective
Build a production-grade machine learning pipeline using scikit-learn's Pipeline and ColumnTransformer APIs. This ensures reproducibility, prevents data leakage, and makes deployment clean. Save the fitted pipeline as a .pkl file using joblib.

Pipeline Architecture
Pipeline Steps (in order)
• Step 1 — Numeric preprocessing: SimpleImputer (median) → StandardScaler
• Step 2 — Categorical preprocessing: SimpleImputer (most frequent) → OneHotEncoder
• Step 3 — ColumnTransformer combines numeric and categorical pipelines in parallel
• Step 4 — GradientBoostingClassifier receives transformed feature matrix

Why Use sklearn Pipelines?
Prevents data leakage: the scaler is only fitted on training data, never on test data
Single fit() call handles preprocessing + training in one step
Single predict() call handles all transformations + inference
The entire pipeline (including scaler and encoder) is saved in one .pkl file
Easy to swap the estimator: just change the last step

Saving and Loading with joblib
import joblib
joblib.dump(pipeline, 'ml_pipeline_model.pkl')   # save
pipeline = joblib.load('ml_pipeline_model.pkl')   # load
predictions = pipeline.predict(new_data)           # predict

How to Run
python task10_ml_pipeline.py --train
python task10_ml_pipeline.py --predict new_data.csv

Task 11 – API Deployment (FastAPI)
Objective
Deploy the trained ML model as a REST API so that any application (web app, mobile app, dashboard) can request predictions over HTTP. FastAPI is the modern industry standard for Python ML APIs — it is faster than Flask and auto-generates Swagger documentation.

API Endpoints
Endpoint	Description
GET /	API welcome message and links
GET /health	Health check: model loaded status, version
POST /predict	Single customer prediction with probability + risk level
POST /predict/batch	Batch prediction for multiple customers at once

Request/Response Example
POST /predict — Request Body
{
  "age": 35,
  "income": 60000,
  "loan_amount": 15000,
  "credit_score": 720,
  "employment_type": "Salaried",
  "loan_purpose": "Home"
}

Response
{
  "prediction": 0,
  "probability": 0.1834,
  "risk_level": "Low",
  "message": "Customer has low default risk (unlikely to default)."
}

Installation and Running
pip install fastapi uvicorn
python task11_api_deployment.py
# OR
uvicorn task11_api_deployment:app --reload --port 8000

Swagger UI
Visit http://localhost:8000/docs for interactive API documentation. You can test all endpoints directly from the browser without writing any code.
Input Validation
All input fields use Pydantic models with type hints and validators. For example, age must be between 18 and 100, credit_score between 300 and 850. Invalid input returns a 422 Unprocessable Entity error with a clear message.

Task 12 – Automated Report Generator
Objective
Build a script that automatically reads any sales CSV, computes business KPIs, and exports a professional report — either as a multi-sheet Excel workbook or a multi-page PDF with charts. This simulates automation tasks common in BI and data engineering roles.

Excel Report (openpyxl)
The Excel output contains 5 sheets: Raw Data (complete dataset), Summary Stats (describe() output), Monthly Sales (aggregated monthly revenue), Region Performance (regional breakdown), and KPI Overview (key metrics + report date). Uses openpyxl for direct .xlsx generation.

PDF Report (matplotlib PdfPages)
The PDF output uses matplotlib's PdfPages backend to create a multi-page report. Page 1 shows the title and KPI table. Page 2 shows monthly sales bar chart. Page 3 shows regional performance with both a bar chart and pie chart side-by-side.

How to Run
python task12_report_generator.py --input data.csv --format excel
python task12_report_generator.py --input data.csv --format pdf
python task12_report_generator.py --input data.csv --format both

Automation Use Case
This script can be scheduled with cron (Linux/Mac) or Task Scheduler (Windows) to run automatically every day/week, pulling fresh data and emailing the report to stakeholders. Combined with smtplib, this becomes a fully automated reporting system.

Task 13 – Text Analytics
Objective
Perform sentiment analysis on product reviews, extract top keywords, and generate a word cloud. Text analytics is used heavily in e-commerce (Amazon, Flipkart), social media monitoring, and customer feedback analysis.

Sentiment Analysis Approach
This implementation uses a rule-based lexicon approach with negation handling. A positive word lexicon (amazing, excellent, great, etc.) and negative word lexicon (terrible, broken, waste, etc.) are defined. Each review is tokenized and scored based on word matches. Negation words (not, never, no) flip the polarity of the following word — so 'not great' correctly registers as negative.

Scoring Formula
score = (positive_count - negative_count) / total_sentiment_words
label = 'positive' if score > 0.1
label = 'negative' if score < -0.1
label = 'neutral'  otherwise

Keyword Extraction
Tokenizes all reviews into words, removes a comprehensive stopword list, and counts frequency using Python's Counter. Returns the top 15 most significant words. These keywords reveal what customers talk about most (quality, delivery, price, etc.).

Word Cloud
An ASCII word cloud is generated in the console, where more frequent words appear in ALL CAPS (high frequency), Title Case (medium), or lowercase (low). For a graphical word cloud, install the wordcloud library and add: WordCloud().generate(text).to_image().show().

How to Run
python task13_text_analytics.py
Results are also saved to sentiment_results.csv with columns: review, sentiment label, score, and confidence.

Task 14 – Build a Simple Chatbot
Objective
Build an FAQ chatbot that answers data science questions using natural language matching. This demonstrates the fundamentals of NLP-based information retrieval without requiring an external API or GPU.

How It Works
The chatbot uses a custom TF-IDF-like similarity algorithm. When a user asks a question, the input is tokenized into words. Each FAQ question key is also tokenized. The chatbot computes cosine similarity between the input's term frequency vector and each FAQ question's term frequency vector. The highest-scoring FAQ answer is returned.

Algorithm Details
Tokenization
Input is lowercased and split into word tokens using regex (r'\b\w+\b'). This handles punctuation and mixed case naturally.
Term Frequency (TF)
TF(word) = count of word in text / total words in text. This normalizes for text length so longer questions don't have an unfair advantage.
Cosine Similarity
Similarity = dot_product(query_tf, faq_tf) / (|query_tf| × |faq_tf|). Returns a value between 0 (no overlap) and 1 (identical). A threshold of 0.15 is used — below this, the chatbot says it doesn't know.

Knowledge Base
Contains 15 Q&A pairs covering: data science, machine learning, Python, neural networks, overfitting, cross-validation, feature engineering, pandas, EDA, missing values, RMSE, R², confusion matrix, precision/recall, and AI vs ML.

How to Run
python task14_chatbot.py          # interactive mode
python task14_chatbot.py --demo   # demo with 5 preset questions

Task 15 – Data Q&A Bot
Objective
Build a natural language interface to any CSV dataset. Users can upload a CSV and ask questions in plain English — the bot parses the intent and translates it into pandas operations, returning the answer.

Supported Query Types
Query Examples
• Row counts: 'How many rows are there?' → df.shape
• Aggregations: 'What is the average salary?' → df['salary'].mean()
• Top N: 'Show top 3 months by revenue' → groupby + nlargest
• Groupby: 'Total revenue by department' → df.groupby('department')['revenue'].sum()
• Filters: 'Show rows where salary > 100000' → df.query('salary > 100000')
• Summary: 'Describe the data' → df.describe()
• Missing: 'How many nulls?' → df.isnull().sum()

NLP Parsing Strategy
The parser uses regular expression pattern matching on the lowercased query. It detects aggregation keywords (average/total/max/min/count), groupby indicators (by/per/each/group), top-N patterns (top N, bottom N), comparison filters (where column > value), and column names by scanning for known column name matches in the query text.

Extending the Bot
To support new query patterns, add a new regex check in the answer() method. For production use, replace the regex parser with a fine-tuned LLM (e.g., GPT-3.5 or a local LLaMA model) with a system prompt like: 'You are a pandas assistant. Convert the user question into a Python pandas one-liner.'

How to Run
python task15_data_qa_bot.py --csv sales.csv    # your data
python task15_data_qa_bot.py --demo            # auto demo

