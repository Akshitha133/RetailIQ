import pandas as pd
import numpy as np
from pathlib import Path

import joblib
import shap
import matplotlib.pyplot as plt


print("=" * 60)
print("RANDOM FOREST INVENTORY RISK - SHAP EXPLAINABILITY")
print("=" * 60)


# =========================================================
# 1. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "retail_inventory_features.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "backend"
    / "models"
    / "random_forest_inventory_model.pkl"
)

FEATURE_FILE = (
    BASE_DIR
    / "backend"
    / "models"
    / "inventory_feature_names.pkl"
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

print("\nLoading inventory dataset...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["Date"]
)

print("Dataset shape:", df.shape)


# =========================================================
# 3. LOAD TRAINED RANDOM FOREST
# =========================================================

print("\nLoading trained Random Forest model...")

model = joblib.load(MODEL_FILE)

print(
    "Random Forest model loaded successfully."
)


# =========================================================
# 4. LOAD EXACT MODEL FEATURES
# =========================================================

print(
    "\nLoading exact Random Forest feature list..."
)

features = joblib.load(
    FEATURE_FILE
)

print(
    "Number of features:",
    len(features)
)

print(
    "\nFeatures used by model:"
)

for i, feature in enumerate(
    features,
    start=1
):

    print(
        f"{i}. {feature}"
    )


# =========================================================
# 5. CHECK FEATURES
# =========================================================

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]

if missing_features:

    print(
        "\nMissing features:"
    )

    print(
        missing_features
    )

    raise ValueError(
        "Inventory dataset does not contain all "
        "Random Forest model features."
    )

print(
    "\nAll model features found."
)


# =========================================================
# 6. PREPARE DATA
# =========================================================

X = df[
    features
].copy()


X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)


# =========================================================
# 7. SELECT SHAP SAMPLE
# =========================================================

sample_size = min(
    200,
    len(X)
)

X_sample = X.iloc[
    :sample_size
].copy()

print(
    f"\nCalculating SHAP values for "
    f"{sample_size} rows..."
)


# =========================================================
# 8. CREATE SHAP EXPLAINER
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
# 9. CALCULATE SHAP VALUES
# =========================================================

shap_result = explainer(
    X_sample
)

print(
    "SHAP values calculated successfully."
)


# =========================================================
# 10. GET SHAP ARRAY
# =========================================================

shap_array = np.asarray(
    shap_result.values
)

print(
    "\nSHAP array shape:",
    shap_array.shape
)


# =========================================================
# 11. GLOBAL SHAP IMPORTANCE
# =========================================================

# Random Forest is multiclass:
#
# samples × features × classes
#
# We calculate average absolute SHAP
# across samples and classes.

if shap_array.ndim == 3:

    mean_abs_shap = np.abs(
        shap_array
    ).mean(
        axis=(0, 2)
    )

else:

    mean_abs_shap = np.abs(
        shap_array
    ).mean(
        axis=0
    )


feature_importance = pd.DataFrame({

    "feature":
        features,

    "mean_abs_shap":
        mean_abs_shap

})


feature_importance = (
    feature_importance
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# =========================================================
# 12. DISPLAY GLOBAL IMPORTANCE
# =========================================================

print(
    "\n" + "=" * 50
)

print(
    "INVENTORY SHAP FEATURE IMPORTANCE"
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
# 13. SAVE GLOBAL IMPORTANCE
# =========================================================

importance_file = (
    OUTPUT_DIR
    / "inventory_shap_feature_importance.csv"
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
# 14. CREATE SHAP BAR PLOT
# =========================================================

print(
    "\nCreating SHAP importance plot..."
)

plot_values = (
    np.abs(shap_array)
    .mean(axis=2)
)

mean_plot_values = (
    plot_values.mean(axis=0)
)

plot_df = pd.DataFrame({

    "feature":
        features,

    "importance":
        mean_plot_values

})

plot_df = (
    plot_df
    .sort_values(
        "importance",
        ascending=True
    )
)


plt.figure(
    figsize=(10, 7)
)

plt.barh(
    plot_df["feature"],
    plot_df["importance"]
)

plt.xlabel(
    "Mean Absolute SHAP Value"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Inventory Risk - SHAP Feature Importance"
)

plt.tight_layout()


plot_file = (
    OUTPUT_DIR
    / "inventory_shap_feature_importance.png"
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
# 15. INDIVIDUAL PREDICTION EXPLANATION
# =========================================================

print(
    "\n" + "=" * 50
)

print(
    "INDIVIDUAL INVENTORY RISK EXPLANATION"
)

print(
    "=" * 50
)


example = X.iloc[
    [0]
]


actual_risk = df.iloc[
    0
]["inventory_risk"]


predicted_risk = model.predict(
    example
)[0]


probabilities = model.predict_proba(
    example
)[0]


# =========================================================
# 16. INDIVIDUAL SHAP VALUES
# =========================================================

example_result = explainer(
    example
)

example_values = np.asarray(
    example_result.values[0]
)


# For multiclass:
#
# features × classes

if example_values.ndim == 2:

    predicted_class_index = list(
        model.classes_
    ).index(
        predicted_risk
    )

    predicted_class_shap = (
        example_values[
            :,
            predicted_class_index
        ]
    )

else:

    predicted_class_shap = (
        example_values
    )


# =========================================================
# 17. CREATE EXPLANATION TABLE
# =========================================================

explanation = pd.DataFrame({

    "feature":
        features,

    "feature_value":
        example.iloc[0].values,

    "shap_value":
        predicted_class_shap

})


explanation["impact"] = np.where(

    explanation["shap_value"] > 0,

    "Supports predicted risk",

    "Reduces predicted risk"

)


explanation = (
    explanation
    .sort_values(
        "shap_value",
        key=lambda x: np.abs(x),
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# =========================================================
# 18. DISPLAY PREDICTION
# =========================================================

print(
    "\nDate:"
)

print(
    df.iloc[0]["Date"]
)

print(
    "\nStore ID:"
)

print(
    df.iloc[0]["Store ID"]
)

print(
    "\nProduct ID:"
)

print(
    df.iloc[0]["Product ID"]
)

print(
    "\nActual inventory risk:"
)

print(
    actual_risk
)

print(
    "\nPredicted inventory risk:"
)

print(
    predicted_risk
)


print(
    "\nRisk probabilities:"
)

for class_name, probability in zip(
    model.classes_,
    probabilities
):

    print(
        f"{class_name}: "
        f"{probability:.2%}"
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
# 19. SAVE INDIVIDUAL EXPLANATION
# =========================================================

explanation_file = (
    OUTPUT_DIR
    / "inventory_shap_prediction_explanation.csv"
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
    "INVENTORY SHAP EXPLAINABILITY COMPLETE!"
)

print(
    "=" * 60
)