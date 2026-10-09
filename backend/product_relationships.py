from pathlib import Path
import pandas as pd


# ============================================================
# RetailIQ - Product Demand Relationship Engine
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "retail_store_inventory.csv"
)


# ============================================================
# LOAD REAL RETAIL DATA
# ============================================================

print("\n==============================================")
print("RetailIQ Product Relationship Engine")
print("==============================================")

retail_df = pd.read_csv(
    DATA_FILE,
    parse_dates=["Date"]
)

print(
    f"Retail dataset shape: {retail_df.shape}"
)


# ============================================================
# VALIDATE DATA
# ============================================================

REQUIRED_COLUMNS = [
    "Date",
    "Product ID",
    "Units Sold"
]

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in retail_df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# ============================================================
# CREATE DAILY PRODUCT DEMAND MATRIX
# ============================================================

daily_product_sales = (
    retail_df
    .groupby(
        ["Date", "Product ID"],
        as_index=False
    )["Units Sold"]
    .sum()
)

DEMAND_MATRIX = (
    daily_product_sales
    .pivot(
        index="Date",
        columns="Product ID",
        values="Units Sold"
    )
    .fillna(0)
)


# ============================================================
# RELATIONSHIP STRENGTH
# ============================================================

def relationship_strength(correlation):

    absolute_value = abs(correlation)

    if absolute_value >= 0.70:
        return "Strong"

    if absolute_value >= 0.40:
        return "Moderate"

    if absolute_value >= 0.20:
        return "Weak"

    return "Very Weak"


# ============================================================
# FIND RELATED PRODUCTS
# ============================================================

def get_product_relationships(
    product_id,
    limit=5
):

    product_id = str(product_id).upper()

    if product_id not in DEMAND_MATRIX.columns:

        return {
            "error": (
                f"Product '{product_id}' "
                "was not found in the retail dataset."
            )
        }

    correlations = (
        DEMAND_MATRIX
        .corr()[product_id]
        .drop(labels=[product_id])
        .dropna()
    )

    correlations = correlations.reindex(
        correlations
        .abs()
        .sort_values(
            ascending=False
        )
        .index
    )

    correlations = correlations.head(
        int(limit)
    )

    relationships = []

    for related_product, correlation in correlations.items():

        correlation = float(correlation)

        relationships.append(
            {
                "product": str(
                    related_product
                ),

                "correlation": round(
                    correlation,
                    4
                ),

                "relationship_strength":
                    relationship_strength(
                        correlation
                    ),

                "relationship_type":
                    (
                        "Positive"
                        if correlation >= 0
                        else "Negative"
                    )
            }
        )

    return {
        "product": product_id,

        "method":
            "Pearson correlation of historical daily Units Sold",

        "data_source":
            "Real Retail Inventory Dataset",

        "relationships":
            relationships
    }


# ============================================================
# PRODUCT DEMAND PROFILE
# ============================================================

def get_product_demand_profile(
    product_id
):

    product_id = str(product_id).upper()

    if product_id not in DEMAND_MATRIX.columns:

        return {
            "error":
                f"Product '{product_id}' was not found."
        }

    series = DEMAND_MATRIX[
        product_id
    ]

    return {

        "product":
            product_id,

        "average_daily_demand":
            round(
                float(series.mean()),
                2
            ),

        "maximum_daily_demand":
            round(
                float(series.max()),
                2
            ),

        "minimum_daily_demand":
            round(
                float(series.min()),
                2
            ),

        "demand_volatility":
            round(
                float(series.std()),
                2
            ),

        "number_of_days":
            int(series.shape[0])
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_product = "P0001"

    result = get_product_relationships(
        test_product,
        limit=5
    )

    print("\nSelected Product:")
    print(
        result.get(
            "product",
            test_product
        )
    )

    print("\nMethod:")
    print(
        result.get(
            "method",
            "N/A"
        )
    )

    print("\nData Source:")
    print(
        result.get(
            "data_source",
            "N/A"
        )
    )

    print("\nRelated Products:")

    for relationship in result.get(
        "relationships",
        []
    ):

        print(
            f"- {relationship['product']} | "
            f"Correlation: "
            f"{relationship['correlation']} | "
            f"Strength: "
            f"{relationship['relationship_strength']} | "
            f"Type: "
            f"{relationship['relationship_type']}"
        )

    print("\n==============================================")
    print("PRODUCT RELATIONSHIP ENGINE READY")
    print("==============================================")
