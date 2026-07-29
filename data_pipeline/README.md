# Q1 Data Pipeline

This module scrapes product-style catalogue data from [Books to Scrape](https://books.toscrape.com/), cleans it, converts prices from GBP to INR, stores the data in a normalized SQLite database, and runs SQL plus pandas checks.

The required fixed conversion rate is:

```text
1 GBP = 105.50 INR
```

This is a project-defined constant, not a live or historical exchange rate.

## Files

- `scrape-clean-store-data.py`: complete pipeline script.
- `requirements.txt`: Python packages needed for this module.
- `cleaned_books.csv`: generated cleaned dataset.
- `books_catalog.db`: generated SQLite database.
- `query_outputs.md`: generated SQL outputs and pandas comparison.

## Install

From the repository root:

```bash
cd data_pipeline
python -m pip install -r requirements.txt
```

If your system has multiple Python versions, use the Python command that works on your machine, for example `py`, `python3`, or a virtual environment Python.

## Run

From the repository root:

```bash
python data_pipeline/scrape-clean-store-data.py
```

Or from inside this folder:

```bash
python scrape-clean-store-data.py
```

The script runs end to end without manual copy-pasting. It scrapes categories from the website sidebar until it has at least 3 categories and at least 60 books.

## Cleaning Decisions

The raw scraped fields are cleaned as follows:

- `price`: the `£` symbol is removed and the value is converted to `price_gbp` as a float.
- `star_rating`: words like `One`, `Two`, and `Three` are converted to integer `rating` values from 1 to 5.
- `availability`: text containing `In stock` becomes `in_stock = True`; text containing `Out of stock` becomes `False`.
- `price_inr`: calculated as `price_gbp * 105.50`.

If numeric fields fail to parse, the script uses median imputation. This keeps one messy numeric row from crashing the whole pipeline. Rows with invalid title, category, or stock status are dropped because these fields identify the product and category and should not be guessed.

## SQLite Schema

The database is normalized into two related tables:

```sql
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE books (
    book_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    in_stock INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
```

`books.category_id` references `categories.category_id`, so category names are stored once and reused through the foreign key.

## SQL Query Coverage

The generated `query_outputs.md` includes five query result sections:

- Highest priced in-stock books: covers `SELECT`, `WHERE`, `ORDER BY`, `LIMIT`, and `JOIN`.
- Distinct categories: covers `DISTINCT`.
- Highly rated books: covers `IN`.
- Mid-price books: covers `BETWEEN`.
- Category summary: covers aggregation and another `JOIN`.

At least two outputs are loaded through `pd.read_sql(...)` because all five query outputs are read into pandas DataFrames using `pd.read_sql(...)`.

## pandas Merge Check

The script recreates the highest priced in-stock books join using `pd.merge(...)` on the in-memory `books_df` and `categories_df`.

It then compares the pandas result with the SQL join result. The generated output file ends with:

```text
MATCH: True
```

when both approaches produce the same table.
