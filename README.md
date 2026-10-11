# Personal Expense Tracker — Chapters 4 & 5

A small local project built for the *Data Operations Analyst* series on Medium.

- **Chapter 4** → Build an ETL pipeline that cleans a messy bank statement CSV and loads it into a SQLite database.
- **Chapter 5** → Use Ollama (a local AI model) to auto-categorize each transaction and generate a monthly report.

Everything runs on your laptop. No cloud, no API key.

---

## Folder layout

```
expense-tracker/
├── data/
│   └── bank_statement.csv     # the messy raw input
├── etl.py                     # Chapter 4 script
├── expenses.db                # created after you run etl.py
├── requirements.txt
└── README.md
```

---

## One-time setup

**1. Install Python 3.10+**
Check with `python --version`. If missing, download from [python.org](https://www.python.org/downloads/).

**2. Create a virtual environment** (keeps this project's packages separate from the rest of your system)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install the required packages**

```bash
pip install -r requirements.txt
```

---

## Chapter 4 — run the pipeline

```bash
python etl.py
```

You should see three stages print out (Extract → Transform → Load), followed by a preview of the cleaned data. A file called `expenses.db` will appear in the folder — that's your SQLite database.

To poke at the database, install [DB Browser for SQLite](https://sqlitebrowser.org/) (free) or query it directly:

```python
import sqlite3, pandas as pd
conn = sqlite3.connect("expenses.db")
print(pd.read_sql("SELECT * FROM transactions LIMIT 5", conn))
```

---

## Chapter 5 — coming next

We'll install Ollama, pull a small model, and use it to categorize each transaction. 

pip install pandas ollama
Go to ollama.com/download and install it for your OS.

After it's installed, open a terminal and run:

ollama pull llama3.2:3b

Then final run : python main.py