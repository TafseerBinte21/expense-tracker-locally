"""
========================================================================
Personal Expense Tracker — Complete Pipeline
========================================================================

Reads a messy bank-statement CSV, cleans it, uses a local AI model
(Ollama) to categorize each transaction, and prints a monthly report.

SETUP (one time):
    1. pip install pandas ollama
    2. Install Ollama from https://ollama.com/download
    3. ollama pull llama3.2:3b

RUN:
    python main.py
========================================================================
"""

import pandas as pd
import sqlite3
import ollama
import webbrowser
from pathlib import Path


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent
RAW_CSV     = PROJECT_DIR / "data" / "bank_statement.csv"
DB_PATH     = PROJECT_DIR / "expenses.db"
MODEL       = "llama3.2:3b"

CATEGORIES = [
    "Food & Dining", "Groceries", "Transport", "Bills & Utilities",
    "Entertainment", "Shopping", "Healthcare", "Personal Care",
    "Income", "Transfer", "Other",
]


# ==================================================================
# STEP 1 — EXTRACT
# ==================================================================
def extract():
    """Read the raw CSV."""
    print("\n[1/5] EXTRACT — reading raw CSV...")
    df = pd.read_csv(RAW_CSV)
    print(f"  Loaded {len(df)} rows.")
    return df


# ==================================================================
# STEP 2 — TRANSFORM (clean the data)
# ==================================================================
def clean_dates(df):
    """Handle multiple date formats by trying each until one works."""
    formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d",
        "%d-%m-%Y", "%d-%b-%Y", "%d-%b-%y",
    ]
    def try_parse(value):
        for fmt in formats:
            try:
                return pd.to_datetime(value, format=fmt)
            except (ValueError, TypeError):
                continue
        return pd.NaT
    df["Date"] = df["Date"].apply(try_parse)
    return df


def clean_amounts(df):
    """Turn strings like '৳420' and '-2,340.50' into real numbers."""
    df["Amount"] = (
        df["Amount"].astype(str)
        .str.replace(r"[^\d.\-]", "", regex=True)
        .replace("", None)
        .astype(float)
    )
    return df


def clean_descriptions(df):
    """Standardize casing and whitespace so duplicates become findable."""
    df["Description"] = (
        df["Description"].astype(str)
        .str.strip().str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )
    df.loc[df["Description"].isin(["", "nan"]), "Description"] = "unknown"
    return df


def fix_types(df):
    """Fill missing Type and reconcile sign vs type."""
    df["Type"] = df["Type"].fillna("").str.strip()
    df.loc[df["Type"] == "", "Type"] = df["Amount"].apply(
        lambda x: "Credit" if x > 0 else "Debit"
    )
    df.loc[(df["Type"] == "Debit")  & (df["Amount"] > 0), "Amount"] *= -1
    df.loc[(df["Type"] == "Credit") & (df["Amount"] < 0), "Amount"] *= -1
    return df


def transform(df):
    """Run all cleaning steps."""
    print("\n[2/5] TRANSFORM — cleaning the data...")
    print(f"  Before: {len(df)} rows, {df.duplicated().sum()} duplicates.")
    df = clean_dates(df)
    df = clean_amounts(df)
    df = clean_descriptions(df)
    df = fix_types(df)
    df = df.drop_duplicates(subset=["Date", "Description", "Amount"])
    df = df.dropna(subset=["Date", "Amount"])
    print(f"  After:  {len(df)} rows.")
    return df


# ==================================================================
# STEP 3 — LOAD (save to SQLite)
# ==================================================================
def load_to_db(df):
    """Save the cleaned dataframe to a SQLite database."""
    print("\n[3/5] LOAD — saving to SQLite...")
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.close()
    print(f"  Saved to {DB_PATH.name}")


# ==================================================================
# STEP 4 — CATEGORIZE (using local AI via Ollama)
# ==================================================================
def categorize_one(description: str) -> str:
    """Ask the AI to pick one category for a single transaction."""
    prompt = f"""You are a personal finance categorizer.

Categorize this bank transaction into EXACTLY ONE of these categories:
{', '.join(CATEGORIES)}

Transaction: "{description}"

Rules:
- Respond with ONLY the category name, exactly as spelled above.
- No explanation, no punctuation, no extra words.
"""
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    result = response["message"]["content"].strip()
    if result in CATEGORIES:
        return result
    for cat in CATEGORIES:
        if cat.lower() in result.lower():
            return cat
    return "Other"


def add_categories(df):
    """Categorize every unique description, then map to all rows."""
    print(f"\n[4/5] CATEGORIZE — using local AI ({MODEL})...")
    uniques = df["Description"].unique()
    print(f"  Categorizing {len(uniques)} unique descriptions...\n")

    mapping = {}
    for i, desc in enumerate(uniques, start=1):
        cat = categorize_one(desc)
        mapping[desc] = cat
        print(f"    [{i:>2}/{len(uniques)}]  {desc:<35} → {cat}")

    df["Category"] = df["Description"].map(mapping)
    return df


# ==================================================================
# STEP 5 — REPORT (print a monthly summary)
# ==================================================================
def print_report(df):
    """Print a human-readable monthly summary."""
    print("\n[5/5] REPORT")
    print("=" * 60)
    print(f"  Monthly Report — {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"  {len(df)} transactions across {df['Category'].nunique()} categories")
    print("=" * 60)

    # Income vs spending
    income   = df.loc[df["Amount"] > 0, "Amount"].sum()
    spending = -df.loc[df["Amount"] < 0, "Amount"].sum()
    net      = income - spending
    print("\n  MONEY IN vs OUT")
    print(f"    Income:    {income:>12,.2f}")
    print(f"    Spending:  {spending:>12,.2f}")
    print(f"    Net:       {net:>12,.2f}   {'📈' if net > 0 else '📉'}")

    # Spending by category
    debits = df[df["Amount"] < 0].copy()
    debits["Amount"] = debits["Amount"].abs()
    by_cat = (
        debits.groupby("Category")["Amount"]
        .agg(["sum", "count"])
        .sort_values("sum", ascending=False)
    )
    total = by_cat["sum"].sum()
    print("\n  SPENDING BY CATEGORY")
    print(f"    {'Category':<20} {'Amount':>10} {'Txns':>6} {'%':>7}")
    print(f"    {'-'*20} {'-'*10} {'-'*6} {'-'*7}")
    for cat, row in by_cat.iterrows():
        pct = row["sum"] / total * 100
        print(f"    {cat:<20} {row['sum']:>10,.2f} {int(row['count']):>6} {pct:>6.1f}%")

    # Top merchants
    top = (
        debits.groupby("Description")["Amount"].sum()
        .sort_values(ascending=False).head(5)
    )
    print("\n  TOP 5 MERCHANTS")
    for merchant, amount in top.items():
        print(f"    {merchant:<35} {amount:>10,.2f}")

    print("\n" + "=" * 60)


# ==================================================================
# STEP 6 — HTML REPORT (opens in your browser)
# ==================================================================
def generate_html_report(df):
    """Generate a styled HTML dashboard and open it in the browser."""
    print("\n[6/6] HTML REPORT — generating dashboard...")

    # ---- calculate all the numbers ----
    income   = df.loc[df["Amount"] > 0, "Amount"].sum()
    spending = -df.loc[df["Amount"] < 0, "Amount"].sum()
    net      = income - spending

    debits = df[df["Amount"] < 0].copy()
    debits["Amount"] = debits["Amount"].abs()

    by_cat = (
        debits.groupby("Category")["Amount"]
        .agg(["sum", "count"])
        .sort_values("sum", ascending=False)
    )
    max_cat_amount = by_cat["sum"].max()
    total_spend    = by_cat["sum"].sum()

    top_merchants = (
        debits.groupby("Description")["Amount"].sum()
        .sort_values(ascending=False).head(5)
    )

    # ---- build the HTML fragments ----
    period = f"{df['Date'].min().strftime('%b %d')} – {df['Date'].max().strftime('%b %d, %Y')}"
    net_class = "positive" if net > 0 else "negative"
    net_sign  = "+" if net > 0 else ""

    bars_html = ""
    for cat, row in by_cat.iterrows():
        width = row["sum"] / max_cat_amount * 100
        pct   = row["sum"] / total_spend * 100
        bars_html += f"""
        <div class="bar-row">
          <div class="cat-label">{cat}</div>
          <div class="bar-container"><div class="bar" style="width:{width:.1f}%"></div></div>
          <div class="amount">৳{row["sum"]:,.0f} <span class="pct">({pct:.0f}%)</span></div>
        </div>"""

    merchants_html = ""
    for merchant, amount in top_merchants.items():
        merchants_html += f"""
        <tr><td>{merchant.title()}</td><td>৳{amount:,.2f}</td></tr>"""

    # ---- assemble the full HTML page ----
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Monthly Expense Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f7fa; color: #1a1a1a; padding: 2rem;
    max-width: 1000px; margin: 0 auto; line-height: 1.5;
  }}
  header {{
    text-align: center; margin-bottom: 1.5rem; padding: 2rem;
    background: white; border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  }}
  header h1 {{ font-size: 2rem; color: #16a34a; margin-bottom: 0.5rem; }}
  header .period {{ color: #666; font-size: 1rem; }}
  .cards {{
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 1rem; margin-bottom: 1.5rem;
  }}
  .card {{
    background: white; padding: 1.5rem; border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05); text-align: center;
  }}
  .card .label {{
    font-size: 0.8rem; color: #888; text-transform: uppercase;
    letter-spacing: 0.05em; margin-bottom: 0.5rem;
  }}
  .card .value {{ font-size: 1.75rem; font-weight: 700; }}
  .card.income .value {{ color: #16a34a; }}
  .card.spending .value {{ color: #dc2626; }}
  .card.net.positive .value {{ color: #16a34a; }}
  .card.net.negative .value {{ color: #dc2626; }}
  section {{
    background: white; padding: 1.5rem; border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 1.5rem;
  }}
  section h2 {{
    font-size: 1.1rem; color: #1a1a1a; margin-bottom: 1rem;
    padding-bottom: 0.5rem; border-bottom: 2px solid #f0f0f0;
  }}
  .bar-row {{
    display: grid; grid-template-columns: 160px 1fr 140px;
    align-items: center; gap: 1rem; padding: 0.5rem 0;
  }}
  .cat-label {{ font-weight: 500; font-size: 0.9rem; }}
  .bar-container {{
    background: #f0f0f0; border-radius: 4px; height: 24px; overflow: hidden;
  }}
  .bar {{
    background: linear-gradient(90deg, #16a34a, #22c55e);
    height: 100%; border-radius: 4px;
    animation: grow 0.6s ease-out;
  }}
  @keyframes grow {{ from {{ width: 0; }} }}
  .amount {{ text-align: right; font-family: monospace; font-size: 0.9rem; color: #333; }}
  .pct {{ color: #999; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #f0f0f0; }}
  th {{ font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }}
  td:last-child, th:last-child {{ text-align: right; font-family: monospace; }}
  footer {{ text-align: center; color: #999; font-size: 0.85rem; margin-top: 2rem; }}
</style>
</head>
<body>
  <header>
    <h1>🟢 Monthly Expense Report</h1>
    <p class="period">{period} · {len(df)} transactions · {df['Category'].nunique()} categories</p>
  </header>

  <div class="cards">
    <div class="card income">
      <div class="label">Income</div>
      <div class="value">৳{income:,.0f}</div>
    </div>
    <div class="card spending">
      <div class="label">Spending</div>
      <div class="value">৳{spending:,.0f}</div>
    </div>
    <div class="card net {net_class}">
      <div class="label">Net Change</div>
      <div class="value">{net_sign}৳{net:,.0f}</div>
    </div>
  </div>

  <section>
    <h2>Spending by Category</h2>
    {bars_html}
  </section>

  <section>
    <h2>Top 5 Merchants</h2>
    <table>
      <thead><tr><th>Merchant</th><th>Amount</th></tr></thead>
      <tbody>{merchants_html}
      </tbody>
    </table>
  </section>

  <footer>Built with Python · Local AI via Ollama · Data Operations Analyst series</footer>
</body>
</html>
"""

    # ---- save and open ----
    output_path = PROJECT_DIR / "report.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"  Saved to {output_path.name}")
    print(f"  Opening in your browser...")
    webbrowser.open(output_path.absolute().as_uri())


# ==================================================================
# RUN EVERYTHING
# ==================================================================
if __name__ == "__main__":
    raw          = extract()
    clean        = transform(raw)
    load_to_db(clean)
    categorized  = add_categories(clean)
    print_report(categorized)
    generate_html_report(categorized)
    print("\n✅ Done!\n")