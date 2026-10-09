import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "retail_store_inventory_forecasting_features.csv"
)

MODEL_DIR = BASE_DIR / "backend" / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_FILE = (
    MODEL_DIR
    / "xgboost_sales_model.pkl"
)

ENCODER_FILE = (
    MODEL_DIR
    / "forecast_category_mappings.pkl"
)

FEATURE_FILE = (
    MODEL_DIR
    / "forecasting_feature_names.pkl"
)


# ============================================================
# HEADER
# ============================================================

print("\n==============================================")
print("XGBOOST RETAIL DEMAND FORECASTING")
print("==============================================")


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading forecasting dataset...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["Date"]
)

print(
    "Dataset shape:",
    df.shape
)


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

print("\nSorting data chronologically...")

df = df.sort_values(
    "Date"
).reset_index(
    drop=True
)


# ============================================================
# CATEGORICAL ENCODING
# ============================================================

print("\nEncoding categorical features...")

categorical_columns = [
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality"
]

category_mappings = {}

for column in categorical_columns:

    df[column] = (
        df[column]
        .astype("category")
    )

    category_mappings[column] = {
        category: code
        for code, category
        in enumerate(
            df[column].cat.categories
        )
    }

    df[column] = (
        df[column]
        .cat.codes
    )


# ============================================================
# FEATURE LIST
# ============================================================

features = [

    # --------------------------------------------------------
    # Business identity
    # --------------------------------------------------------
    "Store ID",
    "Product ID",
    "Category",
    "Region",

    # --------------------------------------------------------
    # Pricing
    # --------------------------------------------------------
    "Price",
    "Discount",
    "Competitor Pricing",
    "price_difference",
    "price_difference_pct",

    # --------------------------------------------------------
    # External conditions
    # --------------------------------------------------------
    "Weather Condition",
    "Holiday/Promotion",
    "Seasonality",

    # --------------------------------------------------------
    # Calendar
    # --------------------------------------------------------
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_weekend",

    # --------------------------------------------------------
    # Historical demand
    # --------------------------------------------------------
    "lag_1",
    "lag_7",
    "lag_28",

    # --------------------------------------------------------
    # Rolling demand
    # --------------------------------------------------------
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",

    # --------------------------------------------------------
    # Demand trend
    # --------------------------------------------------------
    "demand_change_7",
    "demand_growth_pct"
]


# ============================================================
# TARGET
# ============================================================

target = "Units Sold"

print("\nTarget variable:")
print(target)

print("\nNumber of features:")
print(len(features))


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    column
    for column in features
    if column not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing model features: "
        f"{missing_features}"
    )


# ============================================================
# REMOVE INVALID VALUES
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=features + [target]
).copy()

print(
    "\nRows available for training:",
    len(df)
)


# ============================================================
# TIME-BASED TRAIN / TEST SPLIT
# ============================================================

print("\nCreating chronological train/test split...")

unique_dates = (
    df["Date"]
    .sort_values()
    .unique()
)

split_position = int(
    len(unique_dates) * 0.80
)

split_date = unique_dates[
    split_position
]

train_df = df[
    df["Date"] < split_date
].copy()

test_df = df[
    df["Date"] >= split_date
].copy()


print(
    "Training period:",
    train_df["Date"].min(),
    "to",
    train_df["Date"].max()
)

print(
    "Testing period:",
    test_df["Date"].min(),
    "to",
    test_df["Date"].max()
)

print(
    "Training rows:",
    len(train_df)
)

print(
    "Testing rows:",
    len(test_df)
)


# ============================================================
# CREATE X AND Y
# ============================================================

X_train = train_df[
    features
]

y_train = train_df[
    target
]

X_test = test_df[
    features
]

y_test = test_df[
    target
]


# ============================================================
# TRAIN XGBOOST
# ============================================================

print("\n==============================================")
print("Training XGBoost...")
print("==============================================")

model = XGBRegressor(

    n_estimators=500,

    max_depth=7,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    eval_metric="mae",

    random_state=42,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train,

    eval_set=[
        (
            X_test,
            y_test
        )
    ],

    verbose=False
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    X_test
)

# Demand cannot be negative
predictions = np.maximum(
    predictions,
    0
)


# ============================================================
# MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


# ============================================================
# MAPE
# ============================================================

actual = y_test.to_numpy()

valid_mape = actual != 0

if valid_mape.any():

    mape = np.mean(
        np.abs(
            (
                actual[valid_mape]
                -
                predictions[valid_mape]
            )
            /
            actual[valid_mape]
        )
    ) * 100

else:

    mape = np.nan


# ============================================================
# RESULTS
# ============================================================

print("\n==============================================")
print("XGBOOST MODEL RESULTS")
print("==============================================")

print(
    f"MAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"R²   : {r2:.4f}"
)

if not np.isnan(mape):

    print(
        f"MAPE : {mape:.2f}%"
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n==============================================")
print("TOP FEATURE IMPORTANCE")
print("==============================================")

importance_df = pd.DataFrame({

    "feature": features,

    "importance": model.feature_importances_

})

importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
)

print(
    importance_df
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving XGBoost model...")

joblib.dump(
    model,
    MODEL_FILE
)

print(
    "Model saved:"
)

print(
    MODEL_FILE
)


# ============================================================
# SAVE CATEGORY MAPPINGS
# ============================================================

print("\nSaving categorical mappings...")

joblib.dump(
    category_mappings,
    ENCODER_FILE
)

print(
    "Mappings saved:"
)

print(
    ENCODER_FILE
)


# ============================================================
# SAVE MODEL FEATURE LIST
# ============================================================

joblib.dump(
    features,
    FEATURE_FILE
)

print(
    "\nFeature list saved:"
)

print(
    FEATURE_FILE
)


# ============================================================
# FINAL
# ============================================================

print("\n==============================================")
print("XGBOOST TRAINING COMPLETED SUCCESSFULLY")
print("==============================================")