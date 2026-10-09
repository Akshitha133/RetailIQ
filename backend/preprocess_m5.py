import pandas as pd
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "M5"

CALENDAR_FILE = DATA_DIR / "calendar.csv"
SALES_FILE = DATA_DIR / "sales_train_validation.csv"
PRICES_FILE = DATA_DIR / "sell_prices.csv"

OUTPUT_FILE = DATA_DIR / "retailiq_sales.csv"


print("Loading M5 datasets...")

calendar = pd.read_csv(CALENDAR_FILE)
sales = pd.read_csv(SALES_FILE)
prices = pd.read_csv(PRICES_FILE)

print("Calendar:", calendar.shape)
print("Sales:", sales.shape)
print("Prices:", prices.shape)


# Convert sales from wide format to long format
print("\nConverting sales data...")

id_columns = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id"
]

sales_long = sales.melt(
    id_vars=id_columns,
    var_name="d",
    value_name="quantity"
)


# Merge calendar
print("Merging calendar...")

sales_long = sales_long.merge(
    calendar[
        [
            "d",
            "date",
            "wm_yr_wk",
            "weekday",
            "wday",
            "month",
            "year"
        ]
    ],
    on="d",
    how="left"
)


# Merge prices
print("Merging prices...")

sales_long = sales_long.merge(
    prices[
        [
            "store_id",
            "item_id",
            "wm_yr_wk",
            "sell_price"
        ]
    ],
    on=[
        "store_id",
        "item_id",
        "wm_yr_wk"
    ],
    how="left"
)


# Rename columns
sales_long = sales_long.rename(
    columns={
        "date": "sale_date",
        "item_id": "product",
        "sell_price": "price"
    }
)


# Calculate sales amount
sales_long["sales_amount"] = (
    sales_long["quantity"] *
    sales_long["price"]
)


# Keep useful columns
retailiq_sales = sales_long[
    [
        "sale_date",
        "product",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "quantity",
        "price",
        "sales_amount"
    ]
].copy()


# Clean data
retailiq_sales["sale_date"] = pd.to_datetime(
    retailiq_sales["sale_date"]
)

retailiq_sales["quantity"] = pd.to_numeric(
    retailiq_sales["quantity"],
    errors="coerce"
)

retailiq_sales["price"] = pd.to_numeric(
    retailiq_sales["price"],
    errors="coerce"
)

retailiq_sales["sales_amount"] = pd.to_numeric(
    retailiq_sales["sales_amount"],
    errors="coerce"
)


retailiq_sales = retailiq_sales.dropna(
    subset=["sale_date", "product", "quantity"]
)


# Save
print("\nSaving processed dataset...")

retailiq_sales.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n===================================")
print("M5 preprocessing completed!")
print("===================================")

print("Processed shape:", retailiq_sales.shape)
print("Output:", OUTPUT_FILE)

print("\nFirst 5 rows:")
print(retailiq_sales.head())