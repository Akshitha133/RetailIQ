import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "retail_store_inventory.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "retail_store_inventory_forecasting_features.csv"
)


# ============================================================
# LOAD REAL RETAIL DATASET
# ============================================================

print("\n==============================================")
print("REAL RETAIL DEMAND FEATURE ENGINEERING")
print("==============================================")

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Date",
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Demand Forecast",
    "Price",
    "Discount",
    "Weather Condition",
    "Holiday/Promotion",
    "Competitor Pricing",
    "Seasonality"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nAll required columns found.")


# ============================================================
# DATE CONVERSION
# ============================================================

print("\nConverting dates...")

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

if df["Date"].isna().any():
    print(
        "Warning:",
        df["Date"].isna().sum(),
        "invalid dates found."
    )

df = df.dropna(
    subset=["Date"]
).copy()


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Demand Forecast",
    "Price",
    "Discount",
    "Holiday/Promotion",
    "Competitor Pricing"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# REMOVE INVALID RECORDS
# ============================================================

df = df.dropna(
    subset=[
        "Units Sold",
        "Inventory Level",
        "Price",
        "Discount",
        "Competitor Pricing"
    ]
).copy()


# ============================================================
# SORT DATA
# ============================================================

print("\nSorting data...")

df = df.sort_values(
    [
        "Store ID",
        "Product ID",
        "Date"
    ]
).reset_index(drop=True)


# ============================================================
# TIME FEATURES
# ============================================================

print("Creating time features...")

df["day_of_week"] = (
    df["Date"].dt.dayofweek
)

df["day_of_month"] = (
    df["Date"].dt.day
)

df["week_of_year"] = (
    df["Date"]
    .dt.isocalendar()
    .week
    .astype(int)
)

df["month"] = (
    df["Date"].dt.month
)

df["quarter"] = (
    df["Date"].dt.quarter
)

df["year"] = (
    df["Date"].dt.year
)

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)


# ============================================================
# HISTORICAL DEMAND FEATURES
# ============================================================

print("Creating demand lag features...")

group_columns = [
    "Store ID",
    "Product ID"
]

grouped_sales = (
    df.groupby(group_columns)["Units Sold"]
)

# Previous observation
df["lag_1"] = (
    grouped_sales.shift(1)
)

# Previous 7 observations
df["lag_7"] = (
    grouped_sales.shift(7)
)

# Previous 28 observations
df["lag_28"] = (
    grouped_sales.shift(28)
)


# ============================================================
# ROLLING DEMAND FEATURES
# ============================================================

print("Creating rolling demand features...")

df["rolling_mean_7"] = (
    df.groupby(group_columns)["Units Sold"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=7,
            min_periods=7
        )
        .mean()
    )
)

df["rolling_mean_28"] = (
    df.groupby(group_columns)["Units Sold"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=28,
            min_periods=28
        )
        .mean()
    )
)

df["rolling_std_7"] = (
    df.groupby(group_columns)["Units Sold"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=7,
            min_periods=7
        )
        .std()
    )
)


# ============================================================
# DEMAND TREND
# ============================================================

print("Creating demand trend features...")

df["demand_change_7"] = (
    df["rolling_mean_7"]
    -
    df["rolling_mean_28"]
)

df["demand_growth_pct"] = (
    (
        df["rolling_mean_7"]
        -
        df["rolling_mean_28"]
    )
    /
    df["rolling_mean_28"].replace(
        0,
        np.nan
    )
) * 100


# ============================================================
# PRICE RELATIONSHIP
# ============================================================

print("Creating pricing features...")

df["price_difference"] = (
    df["Price"]
    -
    df["Competitor Pricing"]
)

df["price_difference_pct"] = (
    (
        df["Price"]
        -
        df["Competitor Pricing"]
    )
    /
    df["Competitor Pricing"].replace(
        0,
        np.nan
    )
) * 100


# ============================================================
# INVENTORY PRESSURE INDICATOR
# ============================================================
# This is an explanatory feature.
# It is NOT the final inventory-risk model.

df["inventory_to_recent_demand"] = (
    df["Inventory Level"]
    /
    df["rolling_mean_7"].replace(
        0,
        np.nan
    )
)


# ============================================================
# CLEAN INFINITE VALUES
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# REMOVE ROWS WITHOUT SUFFICIENT HISTORY
# ============================================================

print(
    "\nRemoving rows without sufficient historical demand..."
)

history_columns = [
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28"
]

before = len(df)

df = df.dropna(
    subset=history_columns
).copy()

after = len(df)

print(
    "Rows removed:",
    before - after
)

print(
    "Rows remaining:",
    after
)


# ============================================================
# SAVE FEATURE DATASET
# ============================================================

print("\nSaving forecasting feature dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n==============================================")
print("FORECASTING FEATURE DATASET CREATED")
print("==============================================")

print(
    "Final shape:",
    df.shape
)

print("\nOutput file:")

print(
    OUTPUT_FILE
)

print("\nColumns:")

print(
    df.columns.tolist()
)

print("\nDate range:")

print(
    df["Date"].min(),
    "to",
    df["Date"].max()
)

print(
    "\nStores:",
    df["Store ID"].nunique()
)

print(
    "Products:",
    df["Product ID"].nunique()
)

print("\nSample:")

print(
    df.head()
)

print(
    "\nFeature engineering completed successfully."
)