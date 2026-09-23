# 📡 Telco Customer Churn Analysis & Retention Dashboard

A single-file Streamlit application that helps a telecom company understand **why customers leave**, **which customers are at risk**, and **what actions to take** — following the flow: *Facts → Insights → Risks & Opportunities → Recommended Actions*.

---

## 🧩 Business Problem

Customer churn is one of the costliest problems in the telecom industry. Acquiring a new customer is widely cited as costing several times more than retaining an existing one. This dashboard provides:

- Real-time KPIs on churn health
- Data-driven analysis of churn drivers across contract types, payment methods, internet service, tenure, and demographics
- A trained ML model (Logistic Regression + Random Forest) to identify at-risk customers before they leave
- Actionable retention strategies with quantified revenue impact

---

## 📂 Dataset

| Property | Value |
|----------|-------|
| **Name** | WA_Fn-UseC_-Telco-Customer-Churn.csv |
| **Source** | [Kaggle – IBM Sample Data](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Rows** | 7,043 |
| **Columns** | 21 |
| **Target** | `Churn` (Yes / No → 1 / 0) |

> **Note on TotalCharges:** 11 rows have a blank `TotalCharges` value — all correspond to new customers with `tenure = 0`. These are filled with `0.0` (their total spend is genuinely zero). No rows are dropped.

---

## 🛠️ Tech Stack

| Layer | Library |
|-------|---------|
| Dashboard / UI | [Streamlit](https://streamlit.io/) |
| Data Manipulation | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| Visualisations | [Plotly](https://plotly.com/python/) |
| Machine Learning | [scikit-learn](https://scikit-learn.org/) |

---

## 🤖 Models Used

| Model | Notes |
|-------|-------|
| **Logistic Regression** | Baseline — `class_weight="balanced"`, `max_iter=1000`, features scaled with `StandardScaler` |
| **Random Forest** | 200 trees, `max_depth=10`, `class_weight="balanced"` |

The model with the higher ROC-AUC on the held-out test set (20% split, stratified) is automatically selected as the **best model** for the risk table and single-customer predictions.

---

## 📊 Dashboard Pages

### 1. Executive Overview
- 6 KPI cards: total customers, churn rate %, avg monthly charges, avg tenure, monthly revenue at risk, model ROC-AUC
- Churn vs Retained donut chart
- Churn rate by contract type bar chart
- 4 key insight cards with computed numbers

### 2. Churn Drivers
- Churn rate by: Internet Service, Contract Type, Payment Method, Tenure Group, Tech Support, Senior Citizen
- Every chart accompanied by a data-driven explanation

### 3. Risk & Actions
- Side-by-side LR vs RF metrics table
- Confusion matrix + Feature Importance chart for best model
- Top-20 highest-risk **active** customers table (customers who have not yet churned)
- Single-customer churn probability form
- Risks / Opportunities / Recommended Actions tabs (all numbers computed live)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- The dataset CSV (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) in the **same folder** as the script

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the dashboard
```bash
streamlit run churn_retention_app.py
```

Open your browser at `http://localhost:8501`

---

## 🗂️ Project Structure

```
telco-customer-churn-retention-dashboard/
├── churn_retention_app.py               # Single combined Python file (data + models + dashboard)
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
├── Telco_Churn_Project_Report.docx      # Full project report with findings and charts
└── WA_Fn-UseC_-Telco-Customer-Churn.csv # Dataset (also downloadable from the Kaggle link above)
```

---

## 📋 Key Findings (from the data)

- **26.5% overall churn rate** across 7,043 customers (1,869 churned)
- Month-to-month contract customers churn at **42.7%** vs 11.3% on one-year and 2.8% on two-year contracts
- Fiber optic internet customers churn at **41.9%**, more than double the DSL rate (19.0%)
- Electronic check payers churn at **45.3%** vs ~15–17% for automatic payment methods
- Customers in their first 12 months churn at **47.4%**
- Customers without tech support churn at **41.6%** vs **15.2%** with support (about 2.7x)
- Senior citizens churn at **41.7%** vs 23.6% for non-seniors
- Month-to-month + fiber optic + first-year customers form the riskiest segment: **916 customers, 70.2% churn**
- Model quality (held-out 20% test set): Logistic Regression and Random Forest both reach a ROC-AUC of about **0.84**; the dashboard picks whichever scores higher

> Exact model metrics can vary slightly with library versions because the two models are nearly tied. All descriptive percentages above are computed directly from the dataset and do not change.

---

## 📄 Report

See `Telco_Churn_Project_Report.docx` for the full project report including methodology, model evaluation metrics, embedded charts, and strategic recommendations.
