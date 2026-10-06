# Data-analytics-with-ai
# Supermarket Sales Analysis

A complete, reproducible data analytics project on 500 supermarket sales
transactions across 4 branches, 8 product categories, and 20 products. The
project cleans the raw data, computes KPIs and group-wise summaries,
generates 10 charts, and writes every result out to structured files that a
report or dashboard can be built from.

## Problem Statement

Analyze the supermarket sales data to find useful information about
products, branches, categories, customers, payments, and ratings, and turn
it into actionable business insights — which branches and categories drive
revenue, how customer segments and payment methods differ, whether higher
prices or larger baskets relate to satisfaction, and how sales move across
months and weekdays.

## Dataset

**File:** `data/SUPER_MARKET_DATA_-_supermarket_sales_500_rows.csv`
**Rows:** 500 transactions, one row per invoice, no missing values,
no duplicate `Invoice ID`s, and `Sales` matches `Quantity * Unit Price`
exactly on every row.

| Column | Description |
|---|---|
| `Invoice ID` | Unique identifier for the transaction |
| `Date` | Date of the transaction (2026-01-01 to 2026-07-01) |
| `Branch` | Branch code (A, B, C, D) |
| `City` | City the branch operates in (Jaipur, Mumbai, Delhi, Bengaluru) |
| `Customer Type` | Member or Normal (walk-in) customer |
| `Gender` | Customer gender (Male, Female) |
| `Product` | Product purchased (20 distinct products) |
| `Category` | Product category (8 distinct categories) |
| `Quantity` | Units purchased |
| `Unit Price` | Price per unit |
| `Payment` | Payment method (UPI, Card, Cash, Net Banking) |
| `Rating` | Customer satisfaction rating (1–5) |
| `Sales` | Total transaction value (`Quantity * Unit Price`) |

## Repository Structure

```
supermarket-sales-analysis/
├── data/
│   └── SUPER_MARKET_DATA_-_supermarket_sales_500_rows.csv
├── src/
│   └── analyze.py              # single runnable analysis script
├── outputs/
│   ├── figures/                # 10 generated PNG charts
│   ├── tables/                 # generated CSV summary tables
│   └── summary.json            # machine-readable KPI summary
├── README.md
├── REPORT.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Setup & Run

```bash
# 1. Clone the repository and move into it
git clone <this-repo-url>
cd supermarket-sales-analysis

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the analysis
python src/analyze.py
```

Running the script regenerates every file under `outputs/` from the raw CSV
in `data/`, so the pipeline is fully reproducible on any machine. It uses
only `pandas` and `matplotlib` (headless `Agg` backend — no display needed)
and relative paths, so it works from a fresh clone with no edits.

## Headline Results

*(from an actual run of `src/analyze.py` against the dataset — see
`outputs/summary.json` for the full machine-readable output)*

- **500 transactions**, **₹244,411.08** total revenue, spanning
  **2026-01-01 to 2026-07-01**
- **Average order value:** ₹488.82 · **Average rating:** 3.99 / 5 ·
  **Total units sold:** 2,768
- **Top branch by revenue:** Branch C, Mumbai — ₹72,469.45 (143 orders)
- **Top category by revenue:** Beverages — ₹56,108.24
- **Top product by revenue:** Cheese — ₹27,906.30
- **Members outspend Normal customers in total** (₹143,009.30 vs.
  ₹101,401.78), driven by more orders (296 vs. 204) rather than a much
  higher average order value
- **Unit Price correlates with Sales far more than Quantity does**
  (r = 0.722 vs. r = 0.573), and **Rating has essentially no linear
  relationship with Sales, Quantity, or Unit Price** (all |r| < 0.09)

See `REPORT.md` for the full write-up with charts embedded and
section-by-section discussion of every finding.
