import pandas as pd
import numpy as np
from pathlib import Path

import joblib
import shap
import matplotlib.pyplot as plt


print("=" * 60)
print("XGBOOST DEMAND FORECASTING - SHAP EXPLAINABILITY")
print("=" * 60)


# =========================================================
# 1. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "retail_store_inventory_forecasting_features.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "backend"
    / "models"
    / "xgboost_sales_model.pkl"
)

FEATURE_FILE = (
    BASE_DIR
    / "backend"
    / "models"
    / "forecasting_feature_names.pkl"
)

OUTPUT_DIR = (
    BASE_DIR
    / "backend"
    / "models"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 2. LOAD DATA
# =========================================================

print("\nLoading forecasting dataset...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["Date"]
)

print("Dataset shape:", df.shape)


# =========================================================
# 3. LOAD MODEL
# =========================================================

print("\nLoading trained XGBoost model...")

model = joblib.load(MODEL_FILE)

print("XGBoost model loaded successfully.")


# =========================================================
# 4. LOAD EXACT TRAINING FEATURES
# =========================================================

print("\nLoading exact XGBoost feature list...")

features = joblib.load(FEATURE_FILE)

print("Number of features:", len(features))

print("\nFeatures used by model:")

for i, feature in enumerate(features, start=1):
    print(f"{i}. {feature}")


# =========================================================
# 5. CHECK FEATURES
# =========================================================

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]

if missing_features:

    print("\nMissing features:")
    print(missing_features)

    raise ValueError(
        "Forecasting dataset does not contain all model features."
    )

print("\nAll model features found.")


# =========================================================
# 6. PREPARE MODEL DATA
# =========================================================

X = df[features].copy()


# ---------------------------------------------------------
# Use the SAME category mappings used during training
# ---------------------------------------------------------

MAPPING_FILE = (
    BASE_DIR
    / "backend"
    / "models"
    / "forecast_category_mappings.pkl"
)

if MAPPING_FILE.exists():

    print(
        "\nLoading saved category mappings..."
    )

    category_mappings = joblib.load(
        MAPPING_FILE
    )

    for column, mapping in category_mappings.items():

        if column in X.columns:

            X[column] = (
                X[column]
                .map(mapping)
                .fillna(-1)
            )

    print(
        "Saved category mappings applied."
    )

else:

    print(
        "\nWARNING: category mapping file not found."
    )

    # Fallback only if mappings are unavailable
    for column in X.columns:

        if X[column].dtype == "object":

            X[column] = (
                X[column]
                .astype("category")
                .cat.codes
            )


# =========================================================
# 7. CLEAN NUMERIC VALUES
# =========================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)


# =========================================================
# 8. SAMPLE DATA
# =========================================================

sample_size = min(
    200,
    len(X)
)

X_sample = X.iloc[:sample_size].copy()

print(
    f"\nCalculating SHAP values for "
    f"{sample_size} rows..."
)


# =========================================================
# 9. CREATE SHAP EXPLAINER
# =========================================================

print(
    "\nCreating SHAP TreeExplainer..."
)

explainer = shap.TreeExplainer(
    model
)

print(
    "SHAP explainer created successfully."
)


# =========================================================
# 10. CALCULATE SHAP VALUES
# =========================================================

shap_values = explainer.shap_values(
    X_sample
)

print(
    "SHAP values calculated successfully."
)


# =========================================================
# 11. GLOBAL SHAP IMPORTANCE
# =========================================================

shap_array = np.asarray(
    shap_values
)

if shap_array.ndim == 3:

    mean_abs_shap = np.abs(
        shap_array
    ).mean(axis=(0, 2))

else:

    mean_abs_shap = np.abs(
        shap_array
    ).mean(axis=0)


feature_importance = pd.DataFrame({

    "feature": features,

    "mean_abs_shap": mean_abs_shap

})


feature_importance = (
    feature_importance
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    "\n" + "=" * 50
)

print(
    "SHAP FEATURE IMPORTANCE"
)

print(
    "=" * 50
)

print(
    feature_importance.to_string(
        index=False
    )
)


# =========================================================
# 12. SAVE SHAP IMPORTANCE
# =========================================================

importance_file = (
    OUTPUT_DIR
    / "forecast_shap_feature_importance.csv"
)

feature_importance.to_csv(
    importance_file,
    index=False
)

print(
    "\nSHAP importance saved to:"
)

print(
    importance_file
)


# =========================================================
# 13. CREATE SHAP BAR PLOT
# =========================================================

print(
    "\nCreating SHAP importance plot..."
)

plt.figure(
    figsize=(10, 7)
)

shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=features,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plot_file = (
    OUTPUT_DIR
    / "forecast_shap_feature_importance.png"
)

plt.savefig(
    plot_file,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    "SHAP plot saved to:"
)

print(
    plot_file
)


# =========================================================
# 14. INDIVIDUAL PREDICTION
# =========================================================

print(
    "\n" + "=" * 50
)

print(
    "INDIVIDUAL DEMAND PREDICTION EXPLANATION"
)

print(
    "=" * 50
)


example = X.iloc[[0]]

prediction = model.predict(
    example
)[0]


example_shap = explainer.shap_values(
    example
)

example_shap = np.asarray(
    example_shap
)

if example_shap.ndim == 3:

    example_shap = (
        example_shap[0]
        .mean(axis=1)
    )

else:

    example_shap = (
        example_shap[0]
    )


base_value = explainer.expected_value

if isinstance(
    base_value,
    np.ndarray
):

    base_value = float(
        np.asarray(
            base_value
        ).reshape(-1)[0]
    )

else:

    base_value = float(
        base_value
    )


# =========================================================
# 15. CREATE EXPLANATION TABLE
# =========================================================

explanation = pd.DataFrame({

    "feature":
        features,

    "feature_value":
        example.iloc[0].values,

    "shap_value":
        example_shap

})


explanation["impact"] = np.where(

    explanation["shap_value"] > 0,

    "Increases predicted demand",

    "Decreases predicted demand"

)


explanation = (
    explanation
    .sort_values(
        "shap_value",
        key=lambda x: np.abs(x),
        ascending=False
    )
    .reset_index(drop=True)
)


# =========================================================
# 16. DISPLAY RESULT
# =========================================================

print(
    "\nStore ID:",
    df.iloc[0]["Store ID"]
)

print(
    "Product ID:",
    df.iloc[0]["Product ID"]
)

print(
    "Date:",
    df.iloc[0]["Date"]
)

print(
    "\nPredicted demand:",
    round(
        float(prediction),
        2
    )
)

print(
    "\nBase value:",
    round(
        base_value,
        2
    )
)

print(
    "\nTop feature contributions:"
)

print(
    explanation[
        [
            "feature",
            "feature_value",
            "shap_value",
            "impact"
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# =========================================================
# 17. SAVE INDIVIDUAL EXPLANATION
# =========================================================

explanation_file = (
    OUTPUT_DIR
    / "forecast_shap_prediction_explanation.csv"
)

explanation.to_csv(
    explanation_file,
    index=False
)

print(
    "\nIndividual explanation saved to:"
)

print(
    explanation_file
)


# =========================================================
# COMPLETE
# =========================================================

print(
    "\n" + "=" * 60
)

print(
    "FORECAST SHAP EXPLAINABILITY COMPLETE!"
)

print(
    "=" * 60
)