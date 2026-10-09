import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


print("=" * 50)
print("RANDOM FOREST FUTURE INVENTORY RISK MODEL")
print("=" * 50)


# =========================================================
# PATHS
# =========================================================

INPUT_FILE = "../data/retail_inventory_features.csv"

MODEL_FILE = "models/random_forest_inventory_model.pkl"
FEATURE_FILE = "models/inventory_feature_names.pkl"
IMPORTANCE_FILE = "models/inventory_feature_importance.csv"


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading inventory dataset...")

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

print("Dataset shape:", df.shape)


# =========================================================
# FEATURES
# =========================================================
#
# IMPORTANT:
#
# We intentionally DO NOT use:
#
# Future_7_Day_Demand
# Future_Inventory_Balance
# Future_Stock_Ratio
#
# These values belong to the future and were only used
# to construct the target.
#
# Giving them to the model would cause data leakage.
# =========================================================

INVENTORY_FEATURES = [

    "Inventory Level",

    "Units Sold",

    "lag_1",

    "lag_7",

    "lag_28",

    "rolling_mean_7",

    "rolling_mean_28",

    "rolling_std_7",

    "Demand Trend",

    "Demand Growth %",

    "Demand Volatility",

    "Predicted Demand",

    "Stock Pressure",

    "Current Inventory Coverage Days"
]


TARGET = "inventory_risk"


# =========================================================
# CHECK FEATURES
# =========================================================

print("\nChecking required features...")

missing_features = [
    feature
    for feature in INVENTORY_FEATURES
    if feature not in df.columns
]

if missing_features:

    print(
        "Missing features:",
        missing_features
    )

    raise ValueError(
        "Required inventory features are missing."
    )


if TARGET not in df.columns:

    raise ValueError(
        "Target column 'inventory_risk' not found."
    )


print(
    "All required features found."
)


# =========================================================
# REMOVE INVALID VALUES
# =========================================================

model_df = df[
    INVENTORY_FEATURES + [TARGET, "Date"]
].copy()

model_df = model_df.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

model_df = model_df.dropna()

print(
    "\nRows available after cleaning:",
    len(model_df)
)


# =========================================================
# SORT CHRONOLOGICALLY
# =========================================================

model_df = model_df.sort_values(
    "Date"
).reset_index(drop=True)


# =========================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# =========================================================

print(
    "\nCreating chronological train/test split..."
)

unique_dates = sorted(
    model_df["Date"].unique()
)

split_index = int(
    len(unique_dates) * 0.80
)

split_date = unique_dates[
    split_index
]


train_df = model_df[
    model_df["Date"] < split_date
].copy()

test_df = model_df[
    model_df["Date"] >= split_date
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


# =========================================================
# X / Y
# =========================================================

X_train = train_df[
    INVENTORY_FEATURES
]

y_train = train_df[
    TARGET
]

X_test = test_df[
    INVENTORY_FEATURES
]

y_test = test_df[
    TARGET
]


# =========================================================
# DISPLAY CLASS DISTRIBUTION
# =========================================================

print(
    "\nTraining risk distribution:"
)

print(
    y_train.value_counts()
)

print(
    "\nTesting risk distribution:"
)

print(
    y_test.value_counts()
)


# =========================================================
# TRAIN RANDOM FOREST
# =========================================================

print(
    "\n" + "=" * 50
)

print(
    "Training Random Forest..."
)

print(
    "=" * 50
)


model = RandomForestClassifier(

    n_estimators=400,

    max_depth=12,

    min_samples_split=10,

    min_samples_leaf=4,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print(
    "Model training complete!"
)


# =========================================================
# PREDICTIONS
# =========================================================

print(
    "\nGenerating predictions..."
)

y_pred = model.predict(
    X_test
)


# =========================================================
# EVALUATION
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    "\n" + "=" * 50
)

print(
    "INVENTORY RISK MODEL RESULTS"
)

print(
    "=" * 50
)

print(
    f"Accuracy: {accuracy:.4f}"
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_pred
    )
)


print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

print(
    "\n" + "=" * 50
)

print(
    "FEATURE IMPORTANCE"
)

print(
    "=" * 50
)


importance_df = pd.DataFrame({

    "feature":
        INVENTORY_FEATURES,

    "importance":
        model.feature_importances_

})


importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    importance_df.to_string(
        index=False
    )
)


# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(
    model,
    MODEL_FILE
)


# =========================================================
# SAVE FEATURE LIST
# =========================================================

joblib.dump(
    INVENTORY_FEATURES,
    FEATURE_FILE
)


# =========================================================
# SAVE FEATURE IMPORTANCE
# =========================================================

importance_df.to_csv(
    IMPORTANCE_FILE,
    index=False
)


# =========================================================
# FINAL OUTPUT
# =========================================================

print(
    "\nModel saved successfully:"
)

print(
    os.path.abspath(
        MODEL_FILE
    )
)


print(
    "\nFeature importance saved to:"
)

print(
    os.path.abspath(
        IMPORTANCE_FILE
    )
)


print(
    "\nFeature list saved to:"
)

print(
    os.path.abspath(
        FEATURE_FILE
    )
)


print(
    "\n" + "=" * 50
)

print(
    "RANDOM FOREST INVENTORY TRAINING COMPLETE"
)

print(
    "=" * 50
)