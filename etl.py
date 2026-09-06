"""
Chapter 4 — Personal Expense Tracker: ETL Pipeline
====================================================

ETL stands for Extract, Transform, Load — the three stages every data pipeline
goes through:

    1. EXTRACT  → read raw data from somewhere (a CSV, an API, a database)
    2. TRANSFORM → clean it, reshape it, fix quality issues
    3. LOAD      → save the clean version somewhere useful (a database)

We're going to take a messy bank statement (bank_statement.csv), clean it up,
and load it into a SQLite database (expenses.db). SQLite is a tiny database
that lives in a single file — no server to install, perfect for local work.
"""

import pandas as pd
import sqlite3
from pathlib import Path

# ------------------------------------------------------------------
# Paths — where our files live
# ------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent          # folder this script sits in
RAW_CSV     = PROJECT_DIR / "data" / "bank_statement.csv"
DB_PATH     = PROJECT_DIR / "expenses.db"


# ==================================================================
# STEP 1 — EXTRACT
# ==================================================================
def extract():
    """Read the raw CSV into a pandas DataFrame."""
    print("\n[1/3] EXTRACT — reading raw CSV...")
    df = pd.read_csv(RAW_CSV)
    print(f"  Loaded {len(df)} rows and {len(df.columns)} columns.")
    print(f"  Columns: {list(df.columns)}")
    return df


# ==================================================================
# STEP 2 — TRANSFORM (this is where the real DataOps work happens)
# ==================================================================
def profile(df):
    """Quickly look at what's wrong with the data BEFORE cleaning.
    In real life, this step is called 'data profiling'."""
    print("\n  Data quality snapshot (before cleaning):")
    print(f"    - Total rows:        {len(df)}")
    print(f"    - Duplicate rows:    {df.duplicated().sum()}")
    print(f"    - Missing values per column:")
    for col in df.columns:
        # Count blanks and NaN as missing
        missing = df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
        print(f"        {col:<15} → {missing}")


def clean_dates(df):
    """Dates come in many formats — 2026-01-02, 02/01/2026, 03-Jan-2026,
    13-Jan-26, etc. pandas' default parser only likes ISO format, so we
    try several formats in order until one works for each value.

    IMPORTANT DataOps lesson: real-world data is messy. A one-liner like
    pd.to_datetime(col) usually looks like it works — but silently turns
    half your rows into missing values. Always check!"""

    # Formats we'll try, in order of preference
    formats = [
        "%Y-%m-%d",   # 2026-01-02
        "%d/%m/%Y",   # 02/01/2026  (day/month/year — common outside the US)
        "%Y/%m/%d",   # 2026/01/07
        "%d-%m-%Y",   # 06-01-2026
        "%d-%b-%Y",   # 03-Jan-2026
        "%d-%b-%y",   # 13-Jan-26   (two-digit year)
    ]

    def try_parse(value):
        for fmt in formats:
            try:
                return pd.to_datetime(value, format=fmt)
            except (ValueError, TypeError):
                continue
        return pd.NaT  # nothing worked — mark as missing

    df["Date"] = df["Date"].apply(try_parse)
    return df


def clean_amounts(df):
    """Amounts are strings like '৳420', '-$8.50', '-2,340.50'.
    We need real numbers. Strategy: strip out anything that's not a digit,
    a dot, or a minus sign, then convert to float."""
    df["Amount"] = (
        df["Amount"]
        .astype(str)                         # make sure it's a string first
        .str.replace(r"[^\d.\-]", "", regex=True)  # keep only 0-9, ., -
        .replace("", None)                   # blank strings → missing
        .astype(float)                       # finally, convert to number
    )
    return df


def clean_descriptions(df):
    """Descriptions have extra spaces, weird casing, and near-duplicates like
    'STAR KABAB DHANMONDI' vs 'STAR kabab dhanmondi'. We standardize to lower
    case and trim whitespace so duplicates become detectable."""
    df["Description"] = (
        df["Description"]
        .astype(str)
        .str.strip()          # remove leading/trailing spaces
        .str.lower()          # everything lowercase
        .str.replace(r"\s+", " ", regex=True)  # collapse multiple spaces to one
    )
    # Empty descriptions become "unknown" so we don't lose the row
    df.loc[df["Description"].isin(["", "nan"]), "Description"] = "unknown"
    return df


def fill_missing_type(df):
    """The 'Type' column is sometimes blank. We can infer it: negative amount
    → Debit (money going out), positive amount → Credit (money coming in).

    We also fix sign mismatches: if a row says 'Debit' but the amount is
    positive (or vice versa), we make them agree. This is a classic
    reconciliation task in real ETL work."""
    df["Type"] = df["Type"].fillna("").str.strip()

    # Fill blank types based on amount sign
    df.loc[df["Type"] == "", "Type"] = df["Amount"].apply(
        lambda x: "Credit" if x > 0 else "Debit"
    )

    # Reconcile sign with type: debits should be negative, credits positive
    df.loc[(df["Type"] == "Debit")  & (df["Amount"] > 0), "Amount"] *= -1
    df.loc[(df["Type"] == "Credit") & (df["Amount"] < 0), "Amount"] *= -1
    return df


def drop_duplicates_and_bad_rows(df):
    """Now that everything is standardized, real duplicates are easy to spot."""
    before = len(df)
    df = df.drop_duplicates(subset=["Date", "Description", "Amount"])
    df = df.dropna(subset=["Date", "Amount"])   # drop rows with no date or amount
    print(f"    Dropped {before - len(df)} duplicate or invalid rows.")
    return df


def transform(df):
    """Run all the cleaning steps in order."""
    print("\n[2/3] TRANSFORM — cleaning the data...")
    profile(df)                     # show the mess
    df = clean_dates(df)
    df = clean_amounts(df)
    df = clean_descriptions(df)
    df = fill_missing_type(df)
    df = drop_duplicates_and_bad_rows(df)
    print(f"  After cleaning: {len(df)} rows remain.")
    return df


# ==================================================================
# STEP 3 — LOAD
# ==================================================================
def load(df):
    """Save the clean DataFrame into a SQLite database."""
    print("\n[3/3] LOAD — writing to SQLite database...")
    conn = sqlite3.connect(DB_PATH)
    # 'replace' means: if the table already exists, wipe it and start fresh.
    # Good for a tutorial; in real DataOps you'd usually 'append' new rows.
    df.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.close()
    print(f"  Saved to {DB_PATH.name} → table 'transactions'.")


# ==================================================================
# Run the whole pipeline
# ==================================================================
if __name__ == "__main__":
    raw = extract()
    clean = transform(raw)
    load(clean)

    print("\n✅ Pipeline finished. Preview of clean data:")
    print(clean.head(10).to_string(index=False))
