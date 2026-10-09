import os
import joblib
import numpy as np
import pandas as pd

print("=" * 50)
print("REAL RETAIL FUTURE INVENTORY RISK FEATURE ENGINEERING")
print("=" * 50)


# =========================================================
# PATHS
# =========================================================

INPUT_FILE = "../data/retail_store_inventory_forecasting_features.csv"
OUTPUT_FILE = "../data/retail_inventory_features.csv"

XGB_MODEL_FILE = "models/xgboost_sales_model.pkl"
XGB_FEATURES_FILE = "models/forecasting_feature_names.pkl"
XGB_MAPPINGS_FILE = "models/forecast_category_mappings.pkl"


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading real retail dataset...")

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

print("Dataset shape:", df.shape)


# =========================================================
# LOAD XGBOOST MODEL
# =========================================================

print("\nLoading XGBoost demand forecasting model...")

xgb_model = joblib.load(XGB_MODEL_FILE)
forecast_features = joblib.load(XGB_FEATURES_FILE)

print("XGBoost model loaded successfully.")
print("Forecast features:", len(forecast_features))


# =========================================================
# APPLY SAME CATEGORY MAPPINGS
# =========================================================

print("\nPreparing categorical features...")

if os.path.exists(XGB_MAPPINGS_FILE):

    category_mappings = joblib.load(
        XGB_MAPPINGS_FILE
    )

    for column, mapping in category_mappings.items():

        if column in df.columns:

            df[column] = (
                df[column]
                .map(mapping)
                .fillna(-1)
                .astype(int)
            )

    print("Saved category mappings applied.")

else:

    print(
        "Warning: category mappings file not found."
    )

    categorical_columns = [
        "Category",
        "Region",
        "Weather Condition",
        "Seasonality"
    ]

    for column in categorical_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .astype("category")
                .cat.codes
            )


# =========================================================
# XGBOOST PREDICTED DEMAND
# =========================================================

print("\nGenerating predicted demand...")

X_forecast = df[
    forecast_features
]

df["Predicted Demand"] = (
    xgb_model.predict(X_forecast)
)

df["Predicted Demand"] = (
    df["Predicted Demand"]
    .clip(lower=0)
)


# =========================================================
# SORT DATA
# =========================================================

df = df.sort_values(
    [
        "Store ID",
        "Product ID",
        "Date"
    ]
).reset_index(drop=True)


# =========================================================
# DEMAND GROWTH
# =========================================================

print("\nCalculating demand growth...")

df["Demand Growth %"] = (
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

df["Demand Growth %"] = (
    df["Demand Growth %"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)


# =========================================================
# FUTURE 7-DAY DEMAND
# =========================================================

print("\nCalculating future 7-day demand...")

df["Future_7_Day_Demand"] = (
    df.groupby(
        [
            "Store ID",
            "Product ID"
        ]
    )["Units Sold"]
    .transform(
        lambda x:
        x.shift(-1)
        .rolling(
            window=7,
            min_periods=7
        )
        .sum()
    )
)


# =========================================================
# FUTURE INVENTORY BALANCE
# =========================================================

print("\nCalculating future inventory balance...")

df["Future_Inventory_Balance"] = (
    df["Inventory Level"]
    -
    df["Future_7_Day_Demand"]
)


# =========================================================
# FUTURE STOCK RATIO
# =========================================================

print("\nCalculating future stock ratio...")

df["Future_Stock_Ratio"] = (
    df["Inventory Level"]
    /
    df["Future_7_Day_Demand"].replace(
        0,
        np.nan
    )
)


# =========================================================
# REMOVE ROWS WITHOUT FUTURE INFORMATION
# =========================================================

before_rows = len(df)

df = df.dropna(
    subset=[
        "Future_7_Day_Demand",
        "Future_Stock_Ratio"
    ]
).copy()

after_rows = len(df)

print(
    "\nRows removed because future demand "
    "was unavailable:",
    before_rows - after_rows
)

print(
    "Rows remaining:",
    after_rows
)


# =========================================================
# CREATE BALANCED FUTURE-RISK TARGET
# =========================================================

print("\nCreating future inventory-risk target...")

# Rank first so qcut works even when many values are repeated.
ranked_ratio = (
    df["Future_Stock_Ratio"]
    .rank(
        method="first"
    )
)

df["inventory_risk"] = pd.qcut(
    ranked_ratio,
    q=[
        0.00,
        0.25,
        0.75,
        1.00
    ],
    labels=[
        "High",
        "Medium",
        "Low"
    ]
)


# =========================================================
# CURRENT INVENTORY FEATURES
# =========================================================

print("\nCreating inventory features...")

df["Demand Trend"] = (
    df["rolling_mean_7"]
    -
    df["rolling_mean_28"]
)


df["Demand Volatility"] = (
    df[
        [
            "lag_1",
            "lag_7",
            "lag_28"
        ]
    ]
    .std(axis=1)
)


df["Stock Pressure"] = (
    df["Predicted Demand"]
    /
    (
        df["Inventory Level"]
        + 1
    )
)


# =========================================================
# CURRENT INVENTORY COVERAGE
# =========================================================

df["Current Inventory Coverage Days"] = (
    df["Inventory Level"]
    /
    df["Predicted Demand"].replace(
        0,
        np.nan
    )
)


# =========================================================
# FINAL DATASET
# =========================================================

inventory_columns = [

    # Identification
    "Date",
    "Store ID",
    "Product ID",
    "Category",
    "Region",

    # Current inventory
    "Inventory Level",

    # Current demand
    "Units Sold",

    # Historical demand
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",

    # Demand behaviour
    "Demand Trend",
    "Demand Growth %",
    "Demand Volatility",

    # XGBoost prediction
    "Predicted Demand",

    # Current inventory indicators
    "Stock Pressure",
    "Current Inventory Coverage Days",

    # Future information
    # These are retained for evaluation/analysis,
    # but MUST NOT be used as Random Forest features.
    "Future_7_Day_Demand",
    "Future_Inventory_Balance",
    "Future_Stock_Ratio",

    # Target
    "inventory_risk"
]


inventory_df = df[
    inventory_columns
].copy()


# =========================================================
# SAVE
# =========================================================

inventory_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 50)
print("FUTURE INVENTORY RISK DATASET CREATED")
print("=" * 50)

print(
    "\nFinal shape:",
    inventory_df.shape
)


print("\nRisk distribution:")

print(
    inventory_df[
        "inventory_risk"
    ].value_counts()
)


print("\nRisk distribution (%):")

print(
    (
        inventory_df[
            "inventory_risk"
        ]
        .value_counts(
            normalize=True
        )
        * 100
    ).round(2)
)


print("\nFuture stock ratio statistics:")

print(
    inventory_df[
        "Future_Stock_Ratio"
    ].describe()
)


print("\nSample records:")

print(
    inventory_df
    .head(10)
    .to_string(index=False)
)


# =========================================================
# SAVE LOCATION
# =========================================================

print("\nSaved to:")

print(
    os.path.abspath(
        OUTPUT_FILE
    )
)


print("\n" + "=" * 50)
print("INVENTORY FEATURE ENGINEERING COMPLETE")
print("=" * 50)