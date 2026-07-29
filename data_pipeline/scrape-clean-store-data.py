"""
Q1 Data Pipeline: scrape, clean, convert, store, and query book catalogue data.

This data pipeline includes:
1. Scrape book data from books.toscrape.com.
2. Clean raw text fields into useful Python/pandas types.
3. Convert GBP prices to INR with the fixed project rate.
4. Store the result in a normalized SQLite database.
5. Run SQL queries and compare one SQL join with a pandas merge.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


# The given project's fixed baseline rate.
GBP_TO_INR_RATE = 105.50

# To keep all generated files inside the data_pipeline module folder.
MODULE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = MODULE_DIR / "books_catalog.db"
CLEANED_CSV_PATH = MODULE_DIR / "cleaned_books.csv"
QUERY_OUTPUT_PATH = MODULE_DIR / "query_outputs.md"

# URL to scrape books from a public scraping-practice site, the minimum criteria for scraping given in the project.
BASE_URL = "https://books.toscrape.com/"
MIN_CATEGORIES = 3
MIN_BOOKS = 60


def fetch_soup(url: str) -> BeautifulSoup:
    """Download one page and return a BeautifulSoup parser object."""
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return BeautifulSoup(response.content, "html.parser", from_encoding="utf-8")


def clean_text(value: str) -> str:
    """Collapse repeated spaces/newlines so scraped text is easier to work with."""
    return " ".join(value.split())


def get_category_links() -> list[dict[str, str]]:
    """Read the home page sidebar and return all category names with their URLs."""
    soup = fetch_soup(BASE_URL)
    category_links: list[dict[str, str]] = []

    for link in soup.select(".side_categories ul li ul li a"):
        category_links.append(
            {
                "category": clean_text(link.get_text()),
                "url": urljoin(BASE_URL, link.get("href", "")),
            }
        )

    if not category_links:
        raise RuntimeError("No category links were found on the Books to Scrape home page.")

    return category_links


def scrape_category(category_name: str, category_url: str) -> list[dict[str, str]]:
    """Scrape every paginated listing page for one category."""
    books: list[dict[str, str]] = []
    page_url: str | None = category_url

    while page_url:
        soup = fetch_soup(page_url)

        # Each book appears inside an article.product_pod card.
        for article in soup.select("article.product_pod"):
            title_link = article.select_one("h3 a")
            price_tag = article.select_one(".price_color")
            rating_tag = article.select_one(".star-rating")
            availability_tag = article.select_one(".availability")

            rating_text = ""
            if rating_tag:
                rating_classes = rating_tag.get("class", [])
                rating_text = next((item for item in rating_classes if item != "star-rating"), "")

            books.append(
                {
                    "title": title_link.get("title", "").strip() if title_link else "",
                    "price": clean_text(price_tag.get_text()) if price_tag else "",
                    "star_rating": rating_text,
                    "availability": clean_text(availability_tag.get_text()) if availability_tag else "",
                    "category": category_name,
                }
            )

        # Follow the next-page link until the category has no more pages.
        next_link = soup.select_one("li.next a")
        page_url = urljoin(page_url, next_link["href"]) if next_link else None

    return books


def scrape_books() -> pd.DataFrame:
    """Scrape categories until the dataset satisfies the required minimums."""
    all_books: list[dict[str, str]] = []
    categories_used = 0

    for category_info in get_category_links():
        category_books = scrape_category(category_info["category"], category_info["url"])
        all_books.extend(category_books)
        categories_used += 1

        if categories_used >= MIN_CATEGORIES and len(all_books) >= MIN_BOOKS:
            break

    raw_df = pd.DataFrame(all_books)

    if raw_df.empty:
        raise RuntimeError("Scraping completed, but no book rows were collected.")

    if raw_df["category"].nunique() < MIN_CATEGORIES or len(raw_df) < MIN_BOOKS:
        raise RuntimeError(
            f"Dataset is too small: {len(raw_df)} rows across "
            f"{raw_df['category'].nunique()} categories."
        )

    return raw_df


def parse_price(price_text: Any) -> float | None:
    """Convert text such as '£51.77' into a float value such as 51.77."""
    match = re.search(r"\d+(?:\.\d+)?", str(price_text))
    return float(match.group()) if match else None


def parse_rating(rating_text: Any) -> int | None:
    """Convert star rating words into integer values from 1 to 5."""
    rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
    return rating_map.get(str(rating_text).strip())


def parse_stock_status(availability_text: Any) -> bool | None:
    """Convert availability text into True for in stock and False for out of stock."""
    text = str(availability_text).strip().lower()

    if "in stock" in text:
        return True
    if "out of stock" in text:
        return False
    return None


def clean_books(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean scraped strings and return a DataFrame ready for database loading."""
    df = raw_df.copy()
    notes: dict[str, int] = {}

    # Parse raw text columns into clean typed columns.
    df["price_gbp"] = df["price"].apply(parse_price)
    df["rating"] = df["star_rating"].apply(parse_rating)
    df["in_stock"] = df["availability"].apply(parse_stock_status)

    # Text fields are identifiers/descriptions, so invalid rows are dropped instead of guessed.
    required_text_mask = (
        df["title"].astype(str).str.strip().ne("")
        & df["category"].astype(str).str.strip().ne("")
        & df["in_stock"].notna()
    )
    notes["dropped_invalid_text_or_stock_rows"] = int((~required_text_mask).sum())
    df = df.loc[required_text_mask].copy()

    # Numeric fields use median imputation so one messy row does not crash the pipeline.
    for column in ["price_gbp", "rating"]:
        missing_count = int(df[column].isna().sum())
        notes[f"{column}_median_imputed_rows"] = missing_count

        if missing_count:
            median_value = df[column].median()
            if pd.isna(median_value):
                raise RuntimeError(f"Cannot impute {column}; every value is missing.")
            df[column] = df[column].fillna(median_value)

    # Convert columns to the final required types.
    df["price_gbp"] = df["price_gbp"].astype(float).round(2)
    df["rating"] = df["rating"].round().clip(1, 5).astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR_RATE).round(2)

    cleaned_df = df[
        ["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]
    ].reset_index(drop=True)

    if len(cleaned_df) < MIN_BOOKS or cleaned_df["category"].nunique() < MIN_CATEGORIES:
        raise RuntimeError(
            f"Cleaned dataset is too small: {len(cleaned_df)} rows across "
            f"{cleaned_df['category'].nunique()} categories."
        )

    return cleaned_df, notes


def create_database(cleaned_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the normalized SQLite database and insert cleaned book rows."""
    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()

    # Build a category lookup table first so books can reference category_id.
    categories_df = (
        cleaned_df[["category"]]
        .drop_duplicates()
        .rename(columns={"category": "category_name"})
        .reset_index(drop=True)
    )
    categories_df.insert(0, "category_id", range(1, len(categories_df) + 1))

    books_df = cleaned_df.merge(
        categories_df,
        left_on="category",
        right_on="category_name",
        how="left",
    )
    books_df = books_df[
        ["title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"]
    ].copy()
    books_df.insert(0, "book_id", range(1, len(books_df) + 1))
    books_df["in_stock"] = books_df["in_stock"].astype(int)

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON;")
        connection.execute(
            """
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY,
                category_name TEXT UNIQUE NOT NULL
            );
            """
        )
        connection.execute(
            """
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
            """
        )

        categories_df.to_sql("categories", connection, if_exists="append", index=False)
        books_df.to_sql("books", connection, if_exists="append", index=False)

    return books_df, categories_df


def run_sql_queries() -> dict[str, pd.DataFrame]:
    """Run the required SQL examples and return each result as a DataFrame."""
    queries = {
        "highest_priced_in_stock_books": """
            SELECT
                b.title,
                c.category_name,
                b.price_gbp,
                b.price_inr,
                b.rating
            FROM books AS b
            JOIN categories AS c ON b.category_id = c.category_id
            WHERE b.in_stock = 1
            ORDER BY b.price_gbp DESC
            LIMIT 10;
        """,
        "distinct_categories": """
            SELECT DISTINCT
                c.category_name
            FROM categories AS c
            ORDER BY c.category_name;
        """,
        "highly_rated_books_using_in": """
            SELECT
                b.title,
                c.category_name,
                b.rating,
                b.price_gbp
            FROM books AS b
            JOIN categories AS c ON b.category_id = c.category_id
            WHERE b.rating IN (4, 5)
            ORDER BY b.rating DESC, b.price_gbp DESC
            LIMIT 15;
        """,
        "mid_price_books_using_between": """
            SELECT
                b.title,
                c.category_name,
                b.price_gbp,
                b.price_inr
            FROM books AS b
            JOIN categories AS c ON b.category_id = c.category_id
            WHERE b.price_gbp BETWEEN 20 AND 40
            ORDER BY b.price_gbp ASC
            LIMIT 15;
        """,
        "category_summary": """
            SELECT
                c.category_name,
                COUNT(*) AS book_count,
                ROUND(AVG(b.price_gbp), 2) AS avg_price_gbp,
                ROUND(AVG(b.price_inr), 2) AS avg_price_inr,
                ROUND(AVG(b.rating), 2) AS avg_rating
            FROM categories AS c
            JOIN books AS b ON c.category_id = b.category_id
            GROUP BY c.category_id, c.category_name
            ORDER BY book_count DESC, avg_rating DESC;
        """,
    }

    results: dict[str, pd.DataFrame] = {}
    with sqlite3.connect(DATABASE_PATH) as connection:
        for name, query in queries.items():
            results[name] = pd.read_sql(query, connection)

    return results


def compare_sql_with_pandas_merge(
    books_df: pd.DataFrame, categories_df: pd.DataFrame, sql_join_df: pd.DataFrame
) -> tuple[pd.DataFrame, bool]:
    """Recreate the join query with pandas.merge and compare it with SQL output."""
    merged_df = books_df.merge(categories_df, on="category_id", how="inner")
    pandas_join_df = (
        merged_df.loc[merged_df["in_stock"] == 1]
        .sort_values("price_gbp", ascending=False)
        .head(10)[["title", "category_name", "price_gbp", "price_inr", "rating"]]
        .reset_index(drop=True)
    )

    sql_compare_df = sql_join_df.reset_index(drop=True)
    matches = pandas_join_df.equals(sql_compare_df)
    return pandas_join_df, matches


def write_query_outputs(
    results: dict[str, pd.DataFrame],
    pandas_join_df: pd.DataFrame,
    join_outputs_match: bool,
    cleaning_notes: dict[str, int],
    cleaned_df: pd.DataFrame,
) -> None:
    """Save SQL query strings/results and pandas comparison output to Markdown."""
    lines: list[str] = [
        "# Q1 Data Pipeline Query Outputs",
        "",
        "This file is generated by `scrape-clean-store-data.py`.",
        "",
        "## Dataset Summary",
        "",
        f"- Cleaned rows: {len(cleaned_df)}",
        f"- Categories: {cleaned_df['category'].nunique()}",
        f"- Fixed conversion rate: 1 GBP = {GBP_TO_INR_RATE:.2f} INR",
        "",
        "## Cleaning Notes",
        "",
    ]

    for key, value in cleaning_notes.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## SQL Query Results", ""])

    for name, dataframe in results.items():
        lines.extend(
            [
                f"### {name}",
                "",
                dataframe.to_markdown(index=False),
                "",
            ]
        )

    lines.extend(
        [
            "## pandas.read_sql vs pandas.merge Join Check",
            "",
            "### SQL join output",
            "",
            results["highest_priced_in_stock_books"].to_markdown(index=False),
            "",
            "### pandas.merge output",
            "",
            pandas_join_df.to_markdown(index=False),
            "",
            f"**MATCH: {join_outputs_match}**",
            "",
        ]
    )

    QUERY_OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the full data pipeline from scraping to query output generation."""
    print("Starting Q1 data pipeline...")

    print("1. Scraping book category pages...")
    raw_df = scrape_books()
    print(f"   Scraped {len(raw_df)} raw rows from {raw_df['category'].nunique()} categories.")

    print("2. Cleaning fields and converting GBP prices to INR...")
    cleaned_df, cleaning_notes = clean_books(raw_df)
    cleaned_df.to_csv(CLEANED_CSV_PATH, index=False)
    print(f"   Saved cleaned CSV to {CLEANED_CSV_PATH}.")

    print("3. Creating normalized SQLite database...")
    books_df, categories_df = create_database(cleaned_df)
    print(f"   Saved SQLite database to {DATABASE_PATH}.")

    print("4. Running SQL queries with pandas.read_sql...")
    sql_results = run_sql_queries()

    print("5. Recreating the join result with pandas.merge...")
    pandas_join_df, join_outputs_match = compare_sql_with_pandas_merge(
        books_df,
        categories_df,
        sql_results["highest_priced_in_stock_books"],
    )

    print("6. Writing query outputs and comparison notes...")
    write_query_outputs(
        sql_results,
        pandas_join_df,
        join_outputs_match,
        cleaning_notes,
        cleaned_df,
    )

    print(f"   Saved query outputs to {QUERY_OUTPUT_PATH}.")
    print(f"Pipeline complete. SQL and pandas join outputs match: {join_outputs_match}")


if __name__ == "__main__":
    main()
