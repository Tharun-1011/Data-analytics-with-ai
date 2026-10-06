"""
analyze.py
==========

Supermarket Sales Analysis
---------------------------
Loads, cleans, and analyzes the supermarket sales dataset, then produces:
  - Cleaned summary tables (outputs/tables/*.csv)
  - Charts (outputs/figures/*.png)
  - A machine-readable KPI summary (outputs/summary.json)
  - A human-readable console report

Run from the repository root with:
    python src/analyze.py

Requires only pandas and matplotlib (see requirements.txt).
"""

import os
import json
import warnings

import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed

import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths (all relative to this script's location, so it works from any machine)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = SCRIPT_DIR
DATA_PATH = os.path.join(
    REPO_ROOT, "data", "SUPER_MARKET_DATA_-_supermarket_sales_500_rows.csv"
)
FIGURES_DIR = os.path.join(REPO_ROOT, "outputs", "figures")
TABLES_DIR = os.path.join(REPO_ROOT, "outputs", "tables")
SUMMARY_JSON_PATH = os.path.join(REPO_ROOT, "outputs", "summary.json")

CHART_DPI = 150
plt.rcParams["figure.autolayout"] = True


# ---------------------------------------------------------------------------
# 1. Load and clean
# ---------------------------------------------------------------------------
def load_and_clean(path: str) -> pd.DataFrame:
    """Load the raw CSV, validate it, coerce types, and engineer date features.

    Steps performed:
      - Parse Date as datetime
      - Coerce Quantity, Unit Price, Rating, Sales to numeric
      - Drop exact duplicate rows and duplicate Invoice IDs (keep first)
      - Drop rows with nulls in any required column
      - Recompute Quantity * Unit Price and flag/report any mismatch
        against the stored Sales column (tolerance of 0.01 for rounding)
      - Engineer Month, Weekday, and IsWeekend from Date
    """
    df = pd.read_csv(path)

    # --- type coercion -----------------------------------------------------
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    numeric_cols = ["Quantity", "Unit Price", "Rating", "Sales"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- duplicates ----------------------------------------------------
    n_before = len(df)
    df = df.drop_duplicates()
    n_after_full_dupe_drop = len(df)

    if "Invoice ID" in df.columns:
        n_before_invoice_dupe_drop = len(df)
        df = df.drop_duplicates(subset=["Invoice ID"], keep="first")
        n_after_invoice_dupe_drop = len(df)
    else:
        n_before_invoice_dupe_drop = n_after_invoice_dupe_drop = n_after_full_dupe_drop

    # --- nulls -----------------------------------------------------------
    n_before_null_drop = len(df)
    required_cols = [
        "Invoice ID", "Date", "Branch", "City", "Customer Type", "Gender",
        "Product", "Category", "Quantity", "Unit Price", "Payment",
        "Rating", "Sales",
    ]
    required_cols = [c for c in required_cols if c in df.columns]
    df = df.dropna(subset=required_cols)
    n_after_null_drop = len(df)

    # --- Sales integrity check --------------------------------------------
    computed_sales = (df["Quantity"] * df["Unit Price"]).round(2)
    mismatch_mask = (computed_sales - df["Sales"]).abs() > 0.01
    n_mismatches = int(mismatch_mask.sum())
    if n_mismatches > 0:
        print(f"WARNING: {n_mismatches} rows where Sales != Quantity * Unit Price "
              f"(tolerance 0.01). These rows are kept but flagged.")
    df["Sales_Mismatch_Flag"] = mismatch_mask

    # --- feature engineering ------------------------------------------------
    df["Month"] = df["Date"].dt.strftime("%Y-%m")
    df["Weekday"] = df["Date"].dt.day_name()
    df["IsWeekend"] = df["Date"].dt.weekday >= 5

    df = df.reset_index(drop=True)

    # Stash cleaning stats on the DataFrame's attrs for later reporting
    df.attrs["cleaning_stats"] = {
        "rows_loaded": n_before,
        "rows_after_full_duplicate_drop": n_after_full_dupe_drop,
        "rows_before_invoice_id_dedupe": n_before_invoice_dupe_drop,
        "rows_after_invoice_id_dedupe": n_after_invoice_dupe_drop,
        "rows_before_null_drop": n_before_null_drop,
        "rows_after_null_drop": n_after_null_drop,
        "rows_final": len(df),
        "sales_formula_mismatches": n_mismatches,
    }

    return df


# ---------------------------------------------------------------------------
# 2. KPIs and group-wise summaries
# ---------------------------------------------------------------------------
def compute_summary(df: pd.DataFrame) -> dict:
    """Compute top-line KPIs and every group-wise summary table.

    Returns a dict with:
      - "kpis": dict of scalar KPI values
      - "tables": dict of {name: DataFrame} group-wise summaries
      - "correlation": DataFrame correlation matrix
    """
    kpis = {
        "total_transactions": int(len(df)),
        "total_revenue": round(float(df["Sales"].sum()), 2),
        "average_order_value": round(float(df["Sales"].mean()), 2),
        "average_rating": round(float(df["Rating"].mean()), 2),
        "total_units_sold": int(df["Quantity"].sum()),
        "date_range_start": df["Date"].min().strftime("%Y-%m-%d"),
        "date_range_end": df["Date"].max().strftime("%Y-%m-%d"),
        "num_branches": int(df["Branch"].nunique()),
        "num_cities": int(df["City"].nunique()),
        "num_categories": int(df["Category"].nunique()),
        "num_products": int(df["Product"].nunique()),
    }

    def agg_table(group_col, extra_agg=None):
        aggs = {
            "Sales": ["sum", "mean"],
            "Invoice ID": "count",
            "Quantity": "sum",
            "Rating": "mean",
        }
        g = df.groupby(group_col).agg(aggs)
        g.columns = ["Total_Revenue", "Avg_Order_Value", "Num_Orders",
                     "Total_Units", "Avg_Rating"]
        g["Total_Revenue"] = g["Total_Revenue"].round(2)
        g["Avg_Order_Value"] = g["Avg_Order_Value"].round(2)
        g["Avg_Rating"] = g["Avg_Rating"].round(2)
        g = g.sort_values("Total_Revenue", ascending=False).reset_index()
        return g

    tables = {}
    tables["by_branch_city"] = (
        df.groupby(["Branch", "City"])
        .agg(
            Total_Revenue=("Sales", "sum"),
            Avg_Order_Value=("Sales", "mean"),
            Num_Orders=("Invoice ID", "count"),
            Total_Units=("Quantity", "sum"),
            Avg_Rating=("Rating", "mean"),
        )
        .round(2)
        .sort_values("Total_Revenue", ascending=False)
        .reset_index()
    )
    tables["by_category"] = agg_table("Category")
    tables["by_product"] = agg_table("Product")
    tables["by_customer_type"] = agg_table("Customer Type")
    tables["by_gender"] = agg_table("Gender")
    tables["by_payment"] = agg_table("Payment")
    tables["by_month"] = agg_table("Month").sort_values("Month").reset_index(drop=True)
    # order weekdays Mon-Sun rather than alphabetically / by revenue
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                      "Friday", "Saturday", "Sunday"]
    by_weekday = agg_table("Weekday")
    by_weekday["Weekday"] = pd.Categorical(
        by_weekday["Weekday"], categories=weekday_order, ordered=True
    )
    tables["by_weekday"] = by_weekday.sort_values("Weekday").reset_index(drop=True)

    correlation = df[["Quantity", "Unit Price", "Rating", "Sales"]].corr().round(3)

    return {"kpis": kpis, "tables": tables, "correlation": correlation}


# ---------------------------------------------------------------------------
# 3. Charts
# ---------------------------------------------------------------------------
def make_charts(df: pd.DataFrame, summary: dict, out_dir: str) -> None:
    """Generate and save all required PNG charts to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    tables = summary["tables"]

    # 1. Revenue by branch
    fig, ax = plt.subplots(figsize=(7, 5))
    branch_rev = df.groupby("Branch")["Sales"].sum().sort_values(ascending=False)
    ax.bar(branch_rev.index.astype(str), branch_rev.values, color="#4C72B0")
    ax.set_title("Revenue by Branch")
    ax.set_xlabel("Branch")
    ax.set_ylabel("Total Revenue")
    fig.savefig(os.path.join(out_dir, "revenue_by_branch.png"), dpi=CHART_DPI)
    plt.close(fig)

    # 2. Revenue by category
    fig, ax = plt.subplots(figsize=(8, 5))
    cat_rev = tables["by_category"].set_index("Category")["Total_Revenue"]
    ax.barh(cat_rev.index[::-1], cat_rev.values[::-1], color="#55A868")
    ax.set_title("Revenue by Category")
    ax.set_xlabel("Total Revenue")
    fig.savefig(os.path.join(out_dir, "revenue_by_category.png"), dpi=CHART_DPI)
    plt.close(fig)

    # 3. Top 10 products by revenue
    fig, ax = plt.subplots(figsize=(8, 6))
    top10 = tables["by_product"].head(10).set_index("Product")["Total_Revenue"]
    ax.barh(top10.index[::-1], top10.values[::-1], color="#C44E52")
    ax.set_title("Top 10 Products by Revenue")
    ax.set_xlabel("Total Revenue")
    fig.savefig(os.path.join(out_dir, "top10_products_by_revenue.png"), dpi=CHART_DPI)
    plt.close(fig)

    # 4. Average spend by customer type
    fig, ax = plt.subplots(figsize=(6, 5))
    ct = tables["by_customer_type"].set_index("Customer Type")["Avg_Order_Value"]
    ax.bar(ct.index.astype(str), ct.values, color="#8172B2")
    ax.set_title("Average Order Value by Customer Type")
    ax.set_ylabel("Average Order Value")
    fig.savefig(os.path.join(out_dir, "avg_spend_by_customer_type.png"), dpi=CHART_DPI)
    plt.close(fig)

    # 5. Revenue share by gender (pie)
    fig, ax = plt.subplots(figsize=(6, 6))
    gender_rev = df.groupby("Gender")["Sales"].sum()
    ax.pie(gender_rev.values, labels=gender_rev.index.astype(str), autopct="%1.1f%%",
           colors=["#4C72B0", "#DD8452"])
    ax.set_title("Revenue Share by Gender")
    fig.savefig(os.path.join(out_dir, "revenue_share_by_gender.png"), dpi=CHART_DPI)
    plt.close(fig)

    # 6. Payment method share (pie)
    fig, ax = plt.subplots(figsize=(6, 6))
    pay_counts = df["Payment"].value_counts()
    ax.pie(pay_counts.values, labels=pay_counts.index.astype(str), autopct="%1.1f%%",
           colors=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set_title("Payment Method Share (by Order Count)")
    fig.savefig(os.path.join(out_dir, "payment_method_share.png"), dpi=CHART_DPI)
    plt.close(fig)

    # 7. Rating distribution (histogram)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["Rating"], bins=12, color="#55A868", edgecolor="white")
    ax.set_title("Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Orders")
    fig.savefig(os.path.join(out_dir, "rating_distribution.png"), dpi=CHART_DPI)
    plt.close(fig)

    # 8. Monthly sales trend (line)
    fig, ax = plt.subplots(figsize=(8, 5))
    monthly = tables["by_month"].set_index("Month")["Total_Revenue"]
    ax.plot(monthly.index.astype(str), monthly.values, marker="o", color="#4C72B0")
    ax.set_title("Monthly Sales Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Revenue")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.savefig(os.path.join(out_dir, "monthly_sales_trend.png"), dpi=CHART_DPI)
    plt.close(fig)

    # 9. Weekday sales (bar)
    fig, ax = plt.subplots(figsize=(8, 5))
    wk = tables["by_weekday"].set_index("Weekday")["Total_Revenue"]
    ax.bar(wk.index.astype(str), wk.values, color="#C44E52")
    ax.set_title("Revenue by Weekday")
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Total Revenue")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.savefig(os.path.join(out_dir, "weekday_sales.png"), dpi=CHART_DPI)
    plt.close(fig)

    # 10. Correlation heatmap (matplotlib only, no seaborn)
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = summary["correlation"]
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                     color="black", fontsize=9)
    ax.set_title("Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(os.path.join(out_dir, "correlation_heatmap.png"), dpi=CHART_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Save outputs
# ---------------------------------------------------------------------------
def save_tables(summary: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, table in summary["tables"].items():
        table.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    summary["correlation"].to_csv(os.path.join(out_dir, "correlation_matrix.csv"))


def save_summary_json(df: pd.DataFrame, summary: dict, path: str) -> None:
    payload = {
        "kpis": summary["kpis"],
        "cleaning_stats": df.attrs.get("cleaning_stats", {}),
        "tables": {
            name: table.to_dict(orient="records")
            for name, table in summary["tables"].items()
        },
        "correlation_matrix": summary["correlation"].to_dict(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def print_console_summary(df: pd.DataFrame, summary: dict) -> None:
    kpis = summary["kpis"]
    stats = df.attrs.get("cleaning_stats", {})
    print("=" * 60)
    print("SUPERMARKET SALES ANALYSIS - SUMMARY")
    print("=" * 60)
    print(f"Rows loaded:              {stats.get('rows_loaded')}")
    print(f"Rows after cleaning:      {stats.get('rows_final')}")
    print(f"Sales formula mismatches: {stats.get('sales_formula_mismatches')}")
    print("-" * 60)
    print(f"Date range:          {kpis['date_range_start']} to {kpis['date_range_end']}")
    print(f"Total transactions:  {kpis['total_transactions']}")
    print(f"Total revenue:       {kpis['total_revenue']:,.2f}")
    print(f"Average order value: {kpis['average_order_value']:,.2f}")
    print(f"Average rating:      {kpis['average_rating']} / 5")
    print(f"Total units sold:    {kpis['total_units_sold']}")
    print(f"Branches / Cities:   {kpis['num_branches']} / {kpis['num_cities']}")
    print(f"Categories / Products: {kpis['num_categories']} / {kpis['num_products']}")
    print("-" * 60)
    print("Top branch by revenue:")
    print(summary["tables"]["by_branch_city"].head(1).to_string(index=False))
    print("Top category by revenue:")
    print(summary["tables"]["by_category"].head(1).to_string(index=False))
    print("Top product by revenue:")
    print(summary["tables"]["by_product"].head(1).to_string(index=False))
    print("=" * 60)
    print(f"Tables saved to:  outputs/tables/")
    print(f"Charts saved to:  outputs/figures/")
    print(f"KPI JSON saved:   outputs/summary.json")
    print("=" * 60)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    df = load_and_clean(DATA_PATH)
    summary = compute_summary(df)
    save_tables(summary, TABLES_DIR)
    make_charts(df, summary, FIGURES_DIR)
    save_summary_json(df, summary, SUMMARY_JSON_PATH)
    print_console_summary(df, summary)


if __name__ == "__main__":
    main()
