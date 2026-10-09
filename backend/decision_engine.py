# ============================================================
# RETAILIQ - BUSINESS DECISION ENGINE
# Explainable AI-Driven Business Decision Intelligence Platform
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "backend" / "models"


# ------------------------------------------------------------
# Real retail dataset
# ------------------------------------------------------------

RETAIL_DATA = (
    DATA_DIR
    / "retail_store_inventory.csv"
)


# ------------------------------------------------------------
# Forecast feature dataset
# ------------------------------------------------------------

FORECAST_FEATURE_DATA = (
    DATA_DIR
    / "retail_store_inventory_forecasting_features.csv"
)


# ------------------------------------------------------------
# Inventory feature dataset
# ------------------------------------------------------------

INVENTORY_FEATURE_DATA = (
    DATA_DIR
    / "retail_inventory_features.csv"
)


# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

FORECAST_MODEL_PATH = (
    MODEL_DIR
    / "xgboost_sales_model.pkl"
)

FORECAST_MAPPING_PATH = (
    MODEL_DIR
    / "forecast_category_mappings.pkl"
)

FORECAST_FEATURE_NAMES_PATH = (
    MODEL_DIR
    / "forecasting_feature_names.pkl"
)

INVENTORY_MODEL_PATH = (
    MODEL_DIR
    / "random_forest_inventory_model.pkl"
)

INVENTORY_FEATURE_NAMES_PATH = (
    MODEL_DIR
    / "inventory_feature_names.pkl"
)


# ------------------------------------------------------------
# Optional sentiment models
# ------------------------------------------------------------

REVIEW_MODEL_PATH = (
    MODEL_DIR
    / "review_sentiment_model.pkl"
)

TFIDF_MODEL_PATH = (
    MODEL_DIR
    / "review_tfidf_vectorizer.pkl"
)


# ============================================================
# 2. LOAD MODELS
# ============================================================

print("\n==============================================")
print("Loading RetailIQ Decision Engine...")
print("==============================================")

forecast_model = joblib.load(
    FORECAST_MODEL_PATH
)

forecast_mappings = joblib.load(
    FORECAST_MAPPING_PATH
)

forecast_feature_names = joblib.load(
    FORECAST_FEATURE_NAMES_PATH
)

inventory_model = joblib.load(
    INVENTORY_MODEL_PATH
)

inventory_feature_names = joblib.load(
    INVENTORY_FEATURE_NAMES_PATH
)


# ------------------------------------------------------------
# Optional sentiment model
# ------------------------------------------------------------

review_model = None
tfidf_vectorizer = None

if (
    REVIEW_MODEL_PATH.exists()
    and TFIDF_MODEL_PATH.exists()
):

    try:

        review_model = joblib.load(
            REVIEW_MODEL_PATH
        )

        tfidf_vectorizer = joblib.load(
            TFIDF_MODEL_PATH
        )

        print(
            "Review sentiment model loaded."
        )

    except Exception as e:

        print(
            "Warning: Review model could not be loaded:",
            e
        )


print("Forecast model loaded.")
print("Inventory model loaded.")

print(
    "Forecast feature count:",
    len(forecast_feature_names)
)

print(
    "Inventory feature count:",
    len(inventory_feature_names)
)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\nLoading real retail datasets...")

retail_df = pd.read_csv(
    RETAIL_DATA,
    parse_dates=["Date"]
)

forecast_df = pd.read_csv(
    FORECAST_FEATURE_DATA,
    parse_dates=["Date"]
)

inventory_df = pd.read_csv(
    INVENTORY_FEATURE_DATA,
    parse_dates=["Date"]
)


print(
    "Retail dataset shape:",
    retail_df.shape
)

print(
    "Forecast feature dataset shape:",
    forecast_df.shape
)

print(
    "Inventory feature dataset shape:",
    inventory_df.shape
)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def convert_product_to_inventory_code(product):
    """
    Convert real product IDs such as:

        P0001 -> 0
        P0002 -> 1
        P0020 -> 19

    The Random Forest inventory dataset stores
    Product ID using these numeric category codes.
    """

    product = str(product).strip().upper()

    if product.startswith("P"):

        number = int(
            product.replace("P", "")
        )

        return number - 1

    return int(product)


def convert_store_to_inventory_code(store_id):
    """
    Convert real store IDs such as:

        S001 -> 0
        S002 -> 1
        S005 -> 4

    The inventory feature dataset stores
    Store ID using these numeric category codes.
    """

    store_id = str(store_id).strip().upper()

    if store_id.startswith("S"):

        number = int(
            store_id.replace("S", "")
        )

        return number - 1

    return int(store_id)


def inventory_product_label(code):
    """
    Convert numeric inventory Product ID
    back to the real business ID.
    """

    try:

        return f"P{int(code) + 1:04d}"

    except Exception:

        return str(code)


def inventory_store_label(code):
    """
    Convert numeric inventory Store ID
    back to the real business ID.
    """

    try:

        return f"S{int(code) + 1:03d}"

    except Exception:

        return str(code)


# ============================================================
# 5. FORECAST ENCODING
# ============================================================

def encode_forecast_row(row):
    """
    Encode categorical values exactly according
    to the mappings used during XGBoost training.
    """

    row = row.copy()

    categorical_columns = [
        "Store ID",
        "Product ID",
        "Category",
        "Region",
        "Weather Condition",
        "Seasonality"
    ]

    for column in categorical_columns:

        if column not in row.index:
            continue

        value = row[column]

        mapping = forecast_mappings.get(
            column,
            {}
        )

        if value in mapping:

            row[column] = mapping[value]

        else:

            # Try string representation
            value_string = str(value)

            if value_string in mapping:

                row[column] = mapping[
                    value_string
                ]

            else:

                # Unknown category fallback
                row[column] = -1

    return row


# ============================================================
# 6. SALES / DEMAND FORECAST
# ============================================================

def forecast_product(
    product,
    store_id=None
):
    """
    Forecast demand for a real product.

    Example:

        forecast_product("P0001")

    or:

        forecast_product(
            "P0001",
            "S001"
        )
    """

    product = str(product).strip().upper()

    # --------------------------------------------------------
    # Filter real product
    # --------------------------------------------------------

    df = forecast_df[
        forecast_df["Product ID"].astype(str).str.upper()
        == product
    ].copy()

    if store_id is not None:

        store_id = (
            str(store_id)
            .strip()
            .upper()
        )

        df = df[
            df["Store ID"].astype(str).str.upper()
            == store_id
        ].copy()

    if df.empty:

        return None

    # --------------------------------------------------------
    # Latest real observation
    # --------------------------------------------------------

    df = df.sort_values(
        "Date"
    )

    row = df.iloc[-1].copy()

    # --------------------------------------------------------
    # Encode categories
    # --------------------------------------------------------

    encoded_row = encode_forecast_row(
        row
    )

    # --------------------------------------------------------
    # Build model input using EXACT
    # feature list from training
    # --------------------------------------------------------

    values = []

    for feature in forecast_feature_names:

        if feature not in encoded_row.index:

            raise ValueError(
                f"Forecast feature '{feature}' "
                "is missing from the dataset."
            )

        values.append(
            encoded_row[feature]
        )

    X = pd.DataFrame(
        [values],
        columns=forecast_feature_names
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = forecast_model.predict(
        X
    )[0]

    prediction = max(
        float(prediction),
        0.0
    )

    # --------------------------------------------------------
    # Recent demand information
    # --------------------------------------------------------

    recent_demand = None

    if "rolling_mean_7" in row.index:

        try:

            recent_demand = float(
                row["rolling_mean_7"]
            )

        except Exception:

            recent_demand = None

    demand_change = None

    if (
        "demand_change_7" in row.index
    ):

        try:

            demand_change = float(
                row["demand_change_7"]
            )

        except Exception:

            demand_change = None

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {

        "product": product,

        "store": (
            store_id
            if store_id is not None
            else str(row["Store ID"])
        ),

        "forecast_demand": round(
            prediction,
            2
        ),

        "date": str(
            row["Date"].date()
        )
    }

    if recent_demand is not None:

        result[
            "recent_7_day_average"
        ] = round(
            recent_demand,
            2
        )

    if demand_change is not None:

        result[
            "demand_change_7"
        ] = round(
            demand_change,
            2
        )

    return result


# ============================================================
# 7. INVENTORY RISK
# ============================================================

def inventory_risk(
    product,
    store_id=None
):
    """
    Predict inventory risk for a real product.

    Important:
    The inventory feature dataset stores Product ID
    and Store ID as numeric category codes.

    Therefore:

        P0001 -> 0
        S001  -> 0
    """

    product = str(product).strip().upper()

    # --------------------------------------------------------
    # Convert real Product ID to numeric code
    # --------------------------------------------------------

    try:

        product_code = (
            convert_product_to_inventory_code(
                product
            )
        )

    except Exception:

        return None

    # --------------------------------------------------------
    # Filter product
    # --------------------------------------------------------

    df = inventory_df[
        inventory_df["Product ID"]
        == product_code
    ].copy()

    if df.empty:

        return None

    # --------------------------------------------------------
    # Convert Store ID if supplied
    # --------------------------------------------------------

    if store_id is not None:

        try:

            store_code = (
                convert_store_to_inventory_code(
                    store_id
                )
            )

        except Exception:

            return None

        df = df[
            df["Store ID"]
            == store_code
        ].copy()

    if df.empty:

        return None

    # --------------------------------------------------------
    # Latest observation
    # --------------------------------------------------------

    df = df.sort_values(
        "Date"
    )

    row = df.iloc[-1].copy()

    # --------------------------------------------------------
    # Build exact RF feature vector
    # --------------------------------------------------------

    values = []

    for feature in inventory_feature_names:

        if feature not in row.index:

            raise ValueError(
                f"Inventory feature '{feature}' "
                "is missing from the dataset."
            )

        values.append(
            row[feature]
        )

    X = pd.DataFrame(
        [values],
        columns=inventory_feature_names
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = inventory_model.predict(
        X
    )[0]

    probabilities = (
        inventory_model.predict_proba(
            X
        )[0]
    )

    probability_map = {

        str(class_name):
            round(
                float(probability),
                4
            )

        for class_name, probability
        in zip(
            inventory_model.classes_,
            probabilities
        )
    }

    # --------------------------------------------------------
    # Real business IDs
    # --------------------------------------------------------

    actual_product = inventory_product_label(
        row["Product ID"]
    )

    actual_store = inventory_store_label(
        row["Store ID"]
    )

    if store_id is not None:

        actual_store = str(
            store_id
        ).upper()

    # --------------------------------------------------------
    # Additional inventory information
    # --------------------------------------------------------

    result = {

        "product": actual_product,

        "store": actual_store,

        "risk": str(
            prediction
        ),

        "probabilities": probability_map,

        "date": str(
            row["Date"].date()
        )
    }

    # Add useful operational values
    # if available.

    if "Inventory Level" in row.index:

        result[
            "inventory_level"
        ] = float(
            row["Inventory Level"]
        )

    if "Predicted Demand" in row.index:

        result[
            "predicted_demand"
        ] = float(
            row["Predicted Demand"]
        )

    if (
        "Current Inventory Coverage Days"
        in row.index
    ):

        result[
            "inventory_coverage_days"
        ] = round(
            float(
                row[
                    "Current Inventory Coverage Days"
                ]
            ),
            2
        )

    if "Stock Pressure" in row.index:

        result[
            "stock_pressure"
        ] = round(
            float(
                row["Stock Pressure"]
            ),
            4
        )

    return result


# ============================================================
# 8. SHAP - DEMAND FORECAST EXPLANATION
# ============================================================

def explain_forecast(
    product,
    store_id=None
):
    """
    Generate a local SHAP explanation for
    the XGBoost demand forecast.
    """

    product = str(product).strip().upper()

    df = forecast_df[
        forecast_df["Product ID"].astype(str).str.upper()
        == product
    ].copy()

    if store_id is not None:

        store_id = (
            str(store_id)
            .strip()
            .upper()
        )

        df = df[
            df["Store ID"].astype(str).str.upper()
            == store_id
        ].copy()

    if df.empty:

        return None

    df = df.sort_values(
        "Date"
    )

    row = df.iloc[-1].copy()

    encoded_row = encode_forecast_row(
        row
    )

    values = []

    for feature in forecast_feature_names:

        values.append(
            encoded_row[feature]
        )

    X = pd.DataFrame(
        [values],
        columns=forecast_feature_names
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        forecast_model
    )

    shap_values = explainer.shap_values(
        X
    )

    shap_array = np.asarray(
        shap_values
    )

    # XGBoost regression normally gives:
    # (samples, features)

    if shap_array.ndim == 1:

        contributions = shap_array

    else:

        contributions = shap_array[0]

    # --------------------------------------------------------
    # Base value
    # --------------------------------------------------------

    expected_value = explainer.expected_value

    if np.isscalar(
        expected_value
    ):

        base_value = float(
            expected_value
        )

    else:

        base_value = float(
            np.asarray(
                expected_value
            ).reshape(-1)[0]
        )

    prediction = float(
        forecast_model.predict(X)[0]
    )

    # --------------------------------------------------------
    # Feature contributions
    # --------------------------------------------------------

    explanations = []

    for feature, value, shap_value in zip(
        forecast_feature_names,
        X.iloc[0].values,
        contributions
    ):

        explanations.append({

            "feature": feature,

            "value": (
                float(value)
                if isinstance(
                    value,
                    (np.integer, np.floating)
                )
                else str(value)
            ),

            "shap_value": round(
                float(shap_value),
                6
            ),

            "effect": (
                "increases forecast"
                if shap_value > 0
                else "decreases forecast"
            )
        })

    # Strongest contributors first

    explanations.sort(
        key=lambda item:
            abs(item["shap_value"]),
        reverse=True
    )

    return {

        "product": product,

        "store": (
            store_id
            if store_id is not None
            else str(row["Store ID"])
        ),

        "date": str(
            row["Date"].date()
        ),

        "predicted_demand": round(
            prediction,
            2
        ),

        "base_value": round(
            base_value,
            2
        ),

        "top_features": explanations[:10],

        "all_features": explanations
    }


# ============================================================
# 9. SHAP - INVENTORY RISK EXPLANATION
# ============================================================

def explain_inventory(
    product,
    store_id=None
):
    """
    Generate a local SHAP explanation for
    the Random Forest inventory-risk prediction.
    """

    product = str(product).strip().upper()

    # --------------------------------------------------------
    # Convert Product ID
    # --------------------------------------------------------

    try:

        product_code = (
            convert_product_to_inventory_code(
                product
            )
        )

    except Exception:

        return None

    # --------------------------------------------------------
    # Filter Product
    # --------------------------------------------------------

    df = inventory_df[
        inventory_df["Product ID"]
        == product_code
    ].copy()

    if df.empty:

        return None

    # --------------------------------------------------------
    # Filter Store
    # --------------------------------------------------------

    if store_id is not None:

        try:

            store_code = (
                convert_store_to_inventory_code(
                    store_id
                )
            )

        except Exception:

            return None

        df = df[
            df["Store ID"]
            == store_code
        ].copy()

    if df.empty:

        return None

    # --------------------------------------------------------
    # Latest row
    # --------------------------------------------------------

    df = df.sort_values(
        "Date"
    )

    row = df.iloc[-1].copy()

    # --------------------------------------------------------
    # Exact RF features
    # --------------------------------------------------------

    values = []

    for feature in inventory_feature_names:

        if feature not in row.index:

            raise ValueError(
                f"Inventory feature '{feature}' "
                "is missing from the dataset."
            )

        values.append(
            row[feature]
        )

    X = pd.DataFrame(
        [values],
        columns=inventory_feature_names
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = inventory_model.predict(
        X
    )[0]

    probabilities = (
        inventory_model.predict_proba(
            X
        )[0]
    )

    probability_map = {

        str(class_name):
            round(
                float(probability),
                4
            )

        for class_name, probability
        in zip(
            inventory_model.classes_,
            probabilities
        )
    }

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        inventory_model
    )

    shap_values = explainer.shap_values(
        X
    )

    # --------------------------------------------------------
    # Handle Random Forest multiclass SHAP
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        class_index = list(
            inventory_model.classes_
        ).index(
            prediction
        )

        contributions = np.asarray(
            shap_values[class_index]
        )[0]

    else:

        shap_array = np.asarray(
            shap_values
        )

        if shap_array.ndim == 3:

            # Shape:
            # samples, features, classes

            class_index = list(
                inventory_model.classes_
            ).index(
                prediction
            )

            contributions = (
                shap_array[
                    0,
                    :,
                    class_index
                ]
            )

        elif shap_array.ndim == 2:

            contributions = (
                shap_array[0]
            )

        else:

            contributions = (
                shap_array.reshape(-1)
            )

    # --------------------------------------------------------
    # Feature explanations
    # --------------------------------------------------------

    explanations = []

    for feature, value, shap_value in zip(
        inventory_feature_names,
        X.iloc[0].values,
        contributions
    ):

        explanations.append({

            "feature": feature,

            "value": (
                float(value)
                if isinstance(
                    value,
                    (np.integer, np.floating)
                )
                else str(value)
            ),

            "shap_value": round(
                float(shap_value),
                6
            ),

            "effect": (
                "supports predicted risk"
                if shap_value > 0
                else "reduces predicted risk"
            )
        })

    explanations.sort(
        key=lambda item:
            abs(item["shap_value"]),
        reverse=True
    )

    return {

        "product": inventory_product_label(
            row["Product ID"]
        ),

        "store": (
            str(store_id).upper()
            if store_id is not None
            else inventory_store_label(
                row["Store ID"]
            )
        ),

        "date": str(
            row["Date"].date()
        ),

        "predicted_risk": str(
            prediction
        ),

        "probabilities": probability_map,

        "top_features": explanations[:10],

        "all_features": explanations
    }


# ============================================================
# 10. OPTIONAL CUSTOMER REVIEW SENTIMENT
# ============================================================

def analyze_review(review_text):
    """
    Analyze a customer review using the trained
    TF-IDF + Logistic Regression model.

    This is optional.

    The retailer does NOT need to provide a review
    for the main business decision pipeline.
    """

    if review_text is None:

        return None

    if (
        review_model is None
        or tfidf_vectorizer is None
    ):

        return {

            "sentiment": "Unavailable",

            "confidence": 0.0,

            "message":
                "Review sentiment model is not loaded."
        }

    text = str(
        review_text
    ).strip()

    if not text:

        return None

    features = (
        tfidf_vectorizer.transform(
            [text]
        )
    )

    prediction = review_model.predict(
        features
    )[0]

    probabilities = (
        review_model.predict_proba(
            features
        )[0]
    )

    classes = list(
        review_model.classes_
    )

    predicted_index = classes.index(
        prediction
    )

    confidence = float(
        probabilities[
            predicted_index
        ]
    )

    # --------------------------------------------------------
    # Handle common 0/1 sentiment labels
    # --------------------------------------------------------

    if str(prediction) == "1":

        sentiment = "Positive"

    elif str(prediction) == "0":

        sentiment = "Negative"

    else:

        sentiment = str(
            prediction
        )

    return {

        "sentiment": sentiment,

        "confidence": round(
            confidence,
            4
        )
    }


# ============================================================
# 11. BUSINESS HEALTH SCORE
# ============================================================

def calculate_health_score(
    forecast,
    inventory,
    sentiment=None
):
    """
    Decision-support score combining:

    - Demand outlook
    - Inventory risk
    - Optional customer sentiment

    This is NOT an ML target.
    It is a business decision-support score.
    """

    score = 70

    # --------------------------------------------------------
    # Demand
    # --------------------------------------------------------

    demand = forecast[
        "forecast_demand"
    ]

    if demand >= 20:

        score += 10

    elif demand < 5:

        score -= 10

    # --------------------------------------------------------
    # Inventory
    # --------------------------------------------------------

    risk = inventory[
        "risk"
    ]

    if risk == "High":

        score -= 15

    elif risk == "Medium":

        score -= 5

    elif risk == "Low":

        score += 5

    # --------------------------------------------------------
    # Sentiment if available
    # --------------------------------------------------------

    if sentiment is not None:

        if (
            sentiment.get("sentiment")
            == "Positive"
        ):

            score += 10

        elif (
            sentiment.get("sentiment")
            == "Negative"
        ):

            score -= 10

    # --------------------------------------------------------
    # Keep between 0 and 100
    # --------------------------------------------------------

    score = max(
        0,
        min(
            score,
            100
        )
    )

    return score


# ============================================================
# 12. RECOMMENDATION ENGINE
# ============================================================

def generate_recommendations(
    forecast,
    inventory,
    sentiment=None
):
    """
    Convert ML outputs into business-oriented
    recommendations.
    """

    recommendations = []

    # ========================================================
    # INVENTORY
    # ========================================================

    if inventory["risk"] == "High":

        recommendations.append({

            "priority": "High",

            "type": "Inventory",

            "message": (
                "High inventory risk detected "
                "from the current stock and demand "
                "patterns. Review stock levels and "
                "replenishment decisions."
            )
        })

    elif inventory["risk"] == "Medium":

        recommendations.append({

            "priority": "Medium",

            "type": "Inventory",

            "message": (
                "Moderate inventory risk detected. "
                "Monitor stock levels and recent "
                "demand before the next replenishment."
            )
        })

    else:

        recommendations.append({

            "priority": "Low",

            "type": "Inventory",

            "message": (
                "Inventory risk is currently low. "
                "Continue monitoring demand and stock "
                "coverage."
            )
        })

    # ========================================================
    # DEMAND
    # ========================================================

    forecast_demand = (
        forecast[
            "forecast_demand"
        ]
    )

    if forecast_demand >= 20:

        recommendations.append({

            "priority": "High",

            "type": "Demand",

            "message": (
                "Expected demand is relatively high. "
                "Review product availability and "
                "replenishment plans before the next "
                "selling period."
            )
        })

    elif forecast_demand < 5:

        recommendations.append({

            "priority": "Low",

            "type": "Demand",

            "message": (
                "Expected demand is relatively low. "
                "Avoid unnecessary replenishment and "
                "review whether current stock levels "
                "are appropriate."
            )
        })

    else:

        recommendations.append({

            "priority": "Medium",

            "type": "Demand",

            "message": (
                "Expected demand is within a moderate "
                "range. Continue monitoring recent "
                "demand trends."
            )
        })

    # ========================================================
    # CUSTOMER SENTIMENT
    # ========================================================

    if sentiment is not None:

        if (
            sentiment.get("sentiment")
            == "Negative"
        ):

            recommendations.append({

                "priority": "Medium",

                "type": "Customer Feedback",

                "message": (
                    "Recent customer feedback is "
                    "negative. Review customer complaints "
                    "and identify possible product or "
                    "service issues."
                )
            })

        elif (
            sentiment.get("sentiment")
            == "Positive"
        ):

            recommendations.append({

                "priority": "Low",

                "type": "Customer Feedback",

                "message": (
                    "Customer sentiment is positive. "
                    "Continue monitoring feedback for "
                    "changes in customer experience."
                )
            })

    return recommendations


# ============================================================
# 13. COMPLETE BUSINESS DECISION
# ============================================================

def generate_business_decision(
    product,
    review_text=None,
    store_id=None
):
    """
    Main RetailIQ decision pipeline.

    Inputs:
        product
        optional store_id
        optional review_text

    Main pipeline:

        Real Business Data
                ↓
        XGBoost Demand Forecast
                ↓
        Random Forest Inventory Risk
                ↓
        SHAP Explanations
                ↓
        Business Health
                ↓
        Recommendations

    Customer review analysis is optional.
    """

    # --------------------------------------------------------
    # Forecast
    # --------------------------------------------------------

    forecast = forecast_product(
        product,
        store_id
    )

    if forecast is None:

        return {

            "error": (
                f"Product '{product}' "
                "was not found in the "
                "forecasting dataset."
            )
        }

    # --------------------------------------------------------
    # Inventory
    # --------------------------------------------------------

    inventory = inventory_risk(
        product,
        store_id
    )

    if inventory is None:

        return {

            "error": (
                f"Inventory data for "
                f"'{product}'"
                + (
                    f" at store '{store_id}'"
                    if store_id
                    else ""
                )
                + " was not found."
            )
        }

    # --------------------------------------------------------
    # Optional review sentiment
    # --------------------------------------------------------

    sentiment = None

    if review_text is not None:

        sentiment = analyze_review(
            review_text
        )

    # --------------------------------------------------------
    # Health score
    # --------------------------------------------------------

    health_score = (
        calculate_health_score(
            forecast,
            inventory,
            sentiment
        )
    )

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    recommendations = (
        generate_recommendations(
            forecast,
            inventory,
            sentiment
        )
    )

    # --------------------------------------------------------
    # SHAP explanations
    # --------------------------------------------------------

    forecast_explanation = (
        explain_forecast(
            product,
            store_id
        )
    )

    inventory_explanation = (
        explain_inventory(
            product,
            store_id
        )
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    result = {

        "product": str(
            product
        ).upper(),

        "store": (
            str(store_id).upper()
            if store_id is not None
            else forecast.get("store")
        ),

        "sales_forecast": forecast,

        "inventory_risk": inventory,

        "customer_sentiment": sentiment,

        "business_health_score":
            health_score,

        "recommendations":
            recommendations,

        "explainability": {

            "forecast": (
                forecast_explanation
            ),

            "inventory": (
                inventory_explanation
            )
        }
    }

    return result


# ============================================================
# 14. DEMO / TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("====================================================")
    print("       RETAILIQ BUSINESS DECISION ENGINE")
    print("====================================================")

    # --------------------------------------------------------
    # Real product and store from your dataset
    # --------------------------------------------------------

    demo_product = "P0001"

    demo_store = "S001"

    print(
        "\nTesting real product:",
        demo_product
    )

    print(
        "Testing real store:",
        demo_store
    )

    # --------------------------------------------------------
    # Run decision engine
    # --------------------------------------------------------

    result = generate_business_decision(
        product=demo_product,
        store_id=demo_store
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print("\n")
    print("====================================================")
    print("RESULT")
    print("====================================================")

    if "error" in result:

        print(
            "\nERROR:",
            result["error"]
        )

    else:

        print("\nPRODUCT:")

        print(
            result["product"]
        )

        print("\nSTORE:")

        print(
            result["store"]
        )

        print("\nSALES FORECAST:")

        print(
            result["sales_forecast"]
        )

        print("\nINVENTORY RISK:")

        print(
            result["inventory_risk"]
        )

        print("\nBUSINESS HEALTH SCORE:")

        print(
            result[
                "business_health_score"
            ]
        )

        print("\nRECOMMENDATIONS:")

        for recommendation in result[
            "recommendations"
        ]:

            print(
                f"- "
                f"[{recommendation['priority']}] "
                f"{recommendation['type']}: "
                f"{recommendation['message']}"
            )

        # ----------------------------------------------------
        # SHAP forecast
        # ----------------------------------------------------

        print(
            "\nFORECAST EXPLANATION:"
        )

        forecast_exp = (
            result[
                "explainability"
            ][
                "forecast"
            ]
        )

        if forecast_exp:

            print(
                "Predicted demand:",
                forecast_exp[
                    "predicted_demand"
                ]
            )

            print(
                "Base value:",
                forecast_exp[
                    "base_value"
                ]
            )

            print(
                "\nTop forecast factors:"
            )

            for item in forecast_exp[
                "top_features"
            ]:

                print(
                    f"  "
                    f"{item['feature']}: "
                    f"{item['shap_value']} "
                    f"({item['effect']})"
                )

        # ----------------------------------------------------
        # SHAP inventory
        # ----------------------------------------------------

        print(
            "\nINVENTORY EXPLANATION:"
        )

        inventory_exp = (
            result[
                "explainability"
            ][
                "inventory"
            ]
        )

        if inventory_exp:

            print(
                "Predicted risk:",
                inventory_exp[
                    "predicted_risk"
                ]
            )

            print(
                "\nTop inventory factors:"
            )

            for item in inventory_exp[
                "top_features"
            ]:

                print(
                    f"  "
                    f"{item['feature']}: "
                    f"{item['shap_value']} "
                    f"({item['effect']})"
                )

    print("\n")
    print("====================================================")
    print("       DECISION ENGINE TEST COMPLETE")
    print("====================================================")