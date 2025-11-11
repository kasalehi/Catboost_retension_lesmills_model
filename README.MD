# Churn Snapshots & Final Comparison — README

A lightweight pipeline to:
1. **Extract & label** raw data by month-number churn status → `datasource/*.csv`
2. **Build snapshots** from project logic → `snapshots/*.csv`
3. **Compare & compile** final results → `final/*.csv`

> The model currently runs **manually**. Next phase: deploy to cloud and orchestrate with **Airflow (Composer)**.

---

## Prerequisites

- **Python 3.13.1+**
- Optional: **Git** for versioning

> If you previously saw “python 13.0.1”, use **Python 3.13.1+** (major version 3).

---

## Quick Start

### 1) Create & activate a virtual environment

**Windows (PowerShell)**
```powershell
py -3.13 -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Repository Layout (expected)

```
.
├─ sql.py            # Pulls data from SQL; labels churn by month number; writes datasource/*.csv
├─ app.py            # Core model logic
├─ src.py            # Snapshot + compare utilities / CLI wrappers
├─ requirements.txt
├─ datasource/       # Output of sql.py (raw labeled CSVs)
├─ snapshots/        # Snapshot outputs built from app/src
└─ final/            # Final compared/merged CSVs
```

---

## How to Run (Manual)

### A) Extract & label data
Generates labeled CSVs in `datasource/`.
```bash
python sql.py --out ./datasource
```
> Add flags for date ranges or DB connection if your script supports them (e.g., `--start 2024-01-01 --end 2025-03-25`).

### B) Create snapshots
Builds snapshot CSVs into `snapshots/` using project logic.
```bash
python app.py --in ./datasource --out ./snapshots
# or, if snapshots are driven from src.py:
python src.py snapshots --in ./datasource --out ./snapshots
```

### C) Compare & generate finals
Combines snapshots and produces `final/*.csv`.
```bash
python src.py compare --in ./snapshots --out ./final
```

---

## Configuration

If scripts read environment variables, create a `.env` file in the project root:

```
# .env (example)
SQL_SERVER=your_server
SQL_DATABASE=your_db
SQL_USER=your_user
SQL_PASSWORD=your_password
OUTPUT_ROOT=./
```

Ensure your code loads this (e.g., via `python-dotenv`) or pass equivalent CLI flags.

---

## Commands Summary

```bash
# 1) Extract
python sql.py --out ./datasource

# 2) Snapshots
python app.py --in ./datasource --out ./snapshots
# or
python src.py snapshots --in ./datasource --out ./snapshots

# 3) Final compare
python src.py compare --in ./snapshots --out ./final
```

---

## SharePoint Project Folder

Full docs & code:
https://lmnz.sharepoint.com/sites/POWERBIBROS/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FPOWERBIBROS%2FShared%20Documents%2FGeneral%2FBI%20Reports%20Backup%2Fretension%5Fmodel%5Fdocs%2FretensionCode&viewid=4ad21b17%2Da44b%2D4218%2D955a%2D412602ccd6e8&e=5%3Ae97b1aa859a04a5abc4a87b5935e696c&sharingv2=true&fromShare=true&at=9&CID=95d5fcea%2D40cc%2D4bf0%2Da565%2D16ee2c771697&FolderCTID=0x0120007A86EF74E5504B48960064A5290CA8D1

---

## Next Phase (Cloud & Orchestration)

- Package the same three steps into a DAG and run on **Google Cloud Composer (Airflow)**.
- Keep CLI flags identical so local and cloud runs behave the same.

---

## Tips

- Clear `datasource/`, `snapshots/`, and `final/` between runs to avoid mixing outputs.
- Pin versions in `requirements.txt` for reproducibility.
- Expose `--date` / `--test-date` flags in scripts for consistent backtesting.
