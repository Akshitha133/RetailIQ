from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Query,
    Form
)
from typing import Optional

from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy.orm import Session

import bcrypt
import pandas as pd

from io import StringIO
from product_relationships import (
    get_product_relationships,
    get_product_demand_profile,
)


# ============================================================
# DATABASE
# ============================================================

from database import (
    engine,
    Base,
    get_db
)

import models


# ============================================================
# EXISTING SCHEMAS
# ============================================================

from schemas import (
    RegisterRequest,
    LoginRequest,
    DecisionRequest,
    BusinessRecommendationRequest,
    InventoryUpdateRequest
)


# ============================================================
# DECISION ENGINE
# ============================================================

from decision_engine import (
    generate_business_decision,
    analyze_review,
    forecast_df
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="RetailIQ API",
    description=(
        "Explainable AI-Driven Business Decision "
        "Intelligence Platform for Independent Retailers"
    ),
    version="2.0.0"
)


# ============================================================
# CREATE DATABASE TABLES
# ============================================================

Base.metadata.create_all(
    bind=engine
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "RetailIQ API is running",
        "version": "2.0.0",
        "status": "healthy"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "RetailIQ API"
    }


# ============================================================
# REGISTER
# ============================================================

@app.post("/register")
def register(
    data: RegisterRequest,
    db: Session = Depends(get_db)
):

    # --------------------------------------------------------
    # Check existing email
    # --------------------------------------------------------

    existing_user = (
        db.query(models.User)
        .filter(
            models.User.email == data.email
        )
        .first()
    )

    if existing_user:

        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    # --------------------------------------------------------
    # Hash password
    # --------------------------------------------------------

    password_hash = bcrypt.hashpw(
        data.password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    # --------------------------------------------------------
    # Create user
    # --------------------------------------------------------

    user = models.User(
        full_name=data.full_name,
        email=data.email,
        password_hash=password_hash
    )

    db.add(user)

    db.flush()

    # --------------------------------------------------------
    # Create business
    # --------------------------------------------------------

    business = models.Business(
        business_name=data.business_name,
        category=data.category,
        location=data.location,
        owner_id=user.id
    )

    db.add(business)

    db.commit()

    db.refresh(user)
    db.refresh(business)

    return {

        "message":
            "Account created successfully",

        "user_id":
            user.id,

        "business_id":
            business.id
    }


# ============================================================
# LOGIN
# ============================================================

@app.post("/login")
def login(
    data: LoginRequest,
    db: Session = Depends(get_db)
):

    # --------------------------------------------------------
    # Find user
    # --------------------------------------------------------

    user = (
        db.query(models.User)
        .filter(
            models.User.email == data.email
        )
        .first()
    )

    if not user:

        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    # --------------------------------------------------------
    # Verify password
    # --------------------------------------------------------

    password_correct = bcrypt.checkpw(
        data.password.encode("utf-8"),
        user.password_hash.encode("utf-8")
    )

    if not password_correct:

        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    return {

        "message":
            "Login successful",

        "user_id":
            user.id,

        "full_name":
            user.full_name,

        "business_id":
            (
                user.business.id
                if user.business
                else None
            )
    }


# ============================================================
# SALES CSV UPLOAD
# ============================================================

@app.post("/sales/upload")
async def upload_sales_csv(
    business_id: Optional[int] = Query(None),
    business_id_form: Optional[int] = Form(None, alias="business_id"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # --------------------------------------------------------
    # 1. Resolve and validate business_id
    # --------------------------------------------------------
    resolved_business_id = business_id if business_id is not None else business_id_form
    if resolved_business_id is None:
        raise HTTPException(
            status_code=400,
            detail="business_id is required either as a query parameter or form field."
        )

    business = (
        db.query(models.Business)
        .filter(models.Business.id == resolved_business_id)
        .first()
    )

    if not business:
        raise HTTPException(
            status_code=404,
            detail=f"Business with ID {resolved_business_id} not found."
        )

    # --------------------------------------------------------
    # 2. Check CSV file extension
    # --------------------------------------------------------
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid CSV file (.csv extension required)."
        )

    try:
        # Read file contents safely
        contents = await file.read()
        if not contents or len(contents.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded CSV file is empty."
            )

        try:
            csv_text = contents.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                csv_text = contents.decode("latin-1")
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="CSV file must use UTF-8 encoding."
                )

        df = pd.read_csv(StringIO(csv_text))

        if df.empty or len(df.columns) == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded CSV does not contain any data rows or columns."
            )

        # ----------------------------------------------------
        # 3. Robust column mapping and alias support
        # ----------------------------------------------------
        df.columns = [str(c).strip() for c in df.columns]

        mapping_rules = {
            "Date": [
                "date", "sale date", "sale_date", "saledate",
                "transaction date", "transaction_date", "timestamp"
            ],
            "Product": [
                "product", "product id", "product_id", "productid",
                "product name", "product_name", "item", "item name", "item_name", "sku"
            ],
            "Quantity": [
                "quantity", "units sold", "units_sold", "unitssold",
                "qty", "units", "quantity sold", "quantity_sold", "volume"
            ],
            "Price": [
                "price", "unit price", "unit_price", "unitprice",
                "sales price", "sales_price", "rate", "cost", "amount"
            ]
        }

        col_map = {}
        normalized_cols = {
            c.lower().replace("_", " ").replace("-", " "): c
            for c in df.columns
        }

        for target, aliases in mapping_rules.items():
            # Exact match
            if target in df.columns:
                col_map[target] = target
                continue
            # Case-insensitive match on target
            found = False
            for c in df.columns:
                if c.lower() == target.lower():
                    col_map[target] = c
                    found = True
                    break
            if found:
                continue
            # Alias match
            for alias in aliases:
                norm_alias = alias.lower().replace("_", " ").replace("-", " ")
                if norm_alias in normalized_cols:
                    col_map[target] = normalized_cols[norm_alias]
                    found = True
                    break
            if found:
                continue

        missing_columns = [
            target
            for target in ["Date", "Product", "Quantity", "Price"]
            if target not in col_map
        ]

        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Missing columns: {missing_columns}. "
                    "Expected columns: Date (or Sale Date), Product (or Product ID), "
                    "Quantity (or Units Sold), and Price (or Unit Price). "
                    f"Found columns in uploaded file: {list(df.columns)}"
                )
            )

        # Rename to standard internal column names
        rename_dict = {col_map[k]: k for k in ["Date", "Product", "Quantity", "Price"]}
        df = df.rename(columns=rename_dict)

        # ----------------------------------------------------
        # 4. Clean, validate, and convert numeric fields
        # ----------------------------------------------------
        df = df.dropna(subset=["Date", "Product", "Quantity", "Price"])

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="The CSV does not contain valid sales records after dropping empty values."
            )

        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

        df = df.dropna(subset=["Quantity", "Price"])

        df["Date"] = df["Date"].astype(str).str.strip()
        df["Product"] = df["Product"].astype(str).str.strip()

        # Enforce positive quantities, non-negative prices, and non-empty strings
        df = df[
            (df["Quantity"] > 0) &
            (df["Price"] >= 0) &
            (df["Date"] != "") &
            (df["Product"] != "")
        ]

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="Quantity and Price must contain valid positive numbers, and Date and Product cannot be empty."
            )

        df["Quantity"] = df["Quantity"].round().astype(int)
        df["Price"] = df["Price"].astype(float).round(2)

        # ----------------------------------------------------
        # 5. Deduplicate and safely batch insert
        # ----------------------------------------------------
        initial_file_count = len(df)
        df = df.drop_duplicates(subset=["Date", "Product", "Quantity", "Price"])
        in_file_duplicates = initial_file_count - len(df)

        # Query existing records for this business to avoid duplicate insertion
        existing_records = set(
            db.query(
                models.SalesRecord.sale_date,
                models.SalesRecord.product,
                models.SalesRecord.quantity,
                models.SalesRecord.price
            )
            .filter(models.SalesRecord.business_id == resolved_business_id)
            .all()
        )

        records_to_insert = []
        db_duplicates = 0

        for row in df[["Date", "Product", "Quantity", "Price"]].itertuples(index=False):
            tup = (str(row[0]), str(row[1]), int(row[2]), float(row[3]))
            if tup in existing_records:
                db_duplicates += 1
                continue
            records_to_insert.append({
                "business_id": resolved_business_id,
                "sale_date": tup[0],
                "product": tup[1],
                "quantity": tup[2],
                "price": tup[3],
            })
            existing_records.add(tup)

        if not records_to_insert:
            return {
                "message": "All records in the CSV already exist for this business.",
                "business_id": resolved_business_id,
                "records_added": 0,
                "duplicate_rows_removed": in_file_duplicates + db_duplicates,
                "total_rows_evaluated": initial_file_count
            }

        # Bulk insert in chunks of 5000 for high performance
        chunk_size = 5000
        for i in range(0, len(records_to_insert), chunk_size):
            chunk = records_to_insert[i : i + chunk_size]
            db.bulk_insert_mappings(models.SalesRecord, chunk)

        db.commit()

        return {
            "message": "Sales CSV uploaded and processed successfully.",
            "business_id": resolved_business_id,
            "records_added": len(records_to_insert),
            "duplicate_rows_removed": in_file_duplicates + db_duplicates,
            "total_rows_evaluated": initial_file_count
        }

    except HTTPException:
        db.rollback()
        raise

    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing CSV: {str(error)}"
        )


# ============================================================
# BUSINESS DECISION INTELLIGENCE
# ============================================================

@app.post("/decision")
def get_business_decision(
    request: DecisionRequest
):

    # --------------------------------------------------------
    # Get optional fields safely
    # --------------------------------------------------------

    product = request.product

    review = getattr(
        request,
        "review",
        None
    )

    store_id = getattr(
        request,
        "store_id",
        None
    )

    # --------------------------------------------------------
    # Run complete AI pipeline
    # --------------------------------------------------------

    result = generate_business_decision(

        product=product,

        review_text=review,

        store_id=store_id
    )

    # --------------------------------------------------------
    # Handle error
    # --------------------------------------------------------

    if "error" in result:

        raise HTTPException(
            status_code=404,
            detail=result["error"]
        )

    return result


# ============================================================
# CUSTOMER REVIEW ANALYSIS
# ============================================================

@app.post("/review/analyze")
def analyze_customer_review(
    request: DecisionRequest
):

    review = getattr(
        request,
        "review",
        None
    )

    if not review:

        raise HTTPException(
            status_code=400,
            detail=(
                "Please provide a customer review."
            )
        )

    result = analyze_review(
        review
    )

    if result is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Review sentiment model "
                "is unavailable."
            )
        )

    return {

        "review":
            review,

        "sentiment":
            result.get(
                "sentiment"
            ),

        "confidence":
            result.get(
                "confidence"
            )
    }


# ============================================================
# AVAILABLE PRODUCTS
# ============================================================

@app.get("/products")
def get_products():

    products = (
        forecast_df["Product ID"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    products.sort()

    return {

        "products":
            products,

        "count":
            len(products)
    }


# ============================================================
# SALES SUMMARY
# ============================================================

@app.get("/sales/summary/{business_id}")
def get_sales_summary(
    business_id: int,
    db: Session = Depends(get_db)
):

    records = (
        db.query(
            models.SalesRecord
        )
        .filter(
            models.SalesRecord.business_id
            == business_id
        )
        .all()
    )

    if not records:

        return {

            "business_id":
                business_id,

            "total_records":
                0,

            "total_quantity":
                0,

            "total_sales":
                0,

            "products":
                []
        }

    total_quantity = sum(
        record.quantity
        for record in records
    )

    total_sales = round(
        sum(
            record.quantity
            * record.price
            for record in records
        ),
        2
    )

    products = sorted(
        list(
            set(
                record.product
                for record in records
            )
        )
    )

    return {

        "business_id":
            business_id,

        "total_records":
            len(records),

        "total_quantity":
            total_quantity,

        "total_sales":
            total_sales,

        "products":
            products
    }


# ============================================================
# CLEAR SALES
# ============================================================

@app.delete("/sales/clear/{business_id}")
def clear_sales(
    business_id: int,
    db: Session = Depends(get_db)
):

    deleted = (
        db.query(
            models.SalesRecord
        )
        .filter(
            models.SalesRecord.business_id
            == business_id
        )
        .delete(
            synchronize_session=False
        )
    )

    db.commit()

    return {

        "business_id":
            business_id,

        "records_deleted":
            deleted
    }


# ============================================================
# SALES ANALYTICS
# ============================================================

@app.get("/sales/analytics/{business_id}")
def get_sales_analytics(
    business_id: int,
    db: Session = Depends(get_db)
):

    records = (
        db.query(
            models.SalesRecord
        )
        .filter(
            models.SalesRecord.business_id
            == business_id
        )
        .order_by(
            models.SalesRecord.sale_date
        )
        .all()
    )

    if not records:

        return {

            "business_id":
                business_id,

            "products":
                []
        }

    product_data = {}

    for record in records:

        if record.product not in product_data:

            product_data[
                record.product
            ] = {

                "total_quantity":
                    0,

                "total_sales":
                    0,

                "daily_sales":
                    []
            }

        product_data[
            record.product
        ][
            "total_quantity"
        ] += record.quantity

        product_data[
            record.product
        ][
            "total_sales"
        ] += (
            record.quantity
            * record.price
        )

        product_data[
            record.product
        ][
            "daily_sales"
        ].append({

            "date":
                record.sale_date,

            "quantity":
                record.quantity
        })

    analytics = []

    for product, data in product_data.items():

        daily_sales = data[
            "daily_sales"
        ]

        quantities = [
            item["quantity"]
            for item in daily_sales
        ]

        average_daily_quantity = (
            sum(quantities)
            / len(quantities)
        )

        if len(quantities) >= 2:

            first_quantity = quantities[0]

            last_quantity = quantities[-1]

            if last_quantity > first_quantity:

                demand_trend = "Increasing"

            elif last_quantity < first_quantity:

                demand_trend = "Decreasing"

            else:

                demand_trend = "Stable"

        else:

            demand_trend = (
                "Insufficient Data"
            )

        analytics.append({

            "product":
                product,

            "total_quantity":
                data[
                    "total_quantity"
                ],

            "total_sales":
                round(
                    data[
                        "total_sales"
                    ],
                    2
                ),

            "average_daily_quantity":
                round(
                    average_daily_quantity,
                    2
                ),

            "latest_quantity":
                quantities[-1],

            "demand_trend":
                demand_trend
        })

    analytics.sort(
        key=lambda x:
            x["total_quantity"],
        reverse=True
    )

    return {

        "business_id":
            business_id,

        "products":
            analytics
    }


# ============================================================
# INVENTORY ANALYSIS
# ============================================================

@app.get("/inventory/analysis/{business_id}")
def get_inventory_analysis(
    business_id: int,
    db: Session = Depends(get_db)
):

    records = (
        db.query(
            models.SalesRecord
        )
        .filter(
            models.SalesRecord.business_id
            == business_id
        )
        .order_by(
            models.SalesRecord.sale_date
        )
        .all()
    )

    if not records:

        return {

            "business_id":
                business_id,

            "products":
                []
        }

    product_data = {}

    for record in records:

        if record.product not in product_data:

            product_data[
                record.product
            ] = []

        product_data[
            record.product
        ].append({

            "date":
                record.sale_date,

            "quantity":
                record.quantity
        })

    results = []

    for product, sales in product_data.items():

        quantities = [
            item["quantity"]
            for item in sales
        ]

        total_quantity = sum(
            quantities
        )

        average_quantity = (
            total_quantity
            / len(quantities)
        )

        latest_quantity = (
            quantities[-1]
        )

        if (
            latest_quantity
            >= average_quantity * 1.25
        ):

            risk = "High"

            action = (
                "Restock soon because "
                "recent demand is "
                "significantly higher."
            )

        elif (
            latest_quantity
            >= average_quantity * 0.90
        ):

            risk = "Medium"

            action = (
                "Monitor stock because "
                "demand is close to the "
                "recent average."
            )

        else:

            risk = "Low"

            action = (
                "Current demand is relatively "
                "low. Normal stock monitoring "
                "is sufficient."
            )

        results.append({

            "product":
                product,

            "average_daily_demand":
                round(
                    average_quantity,
                    2
                ),

            "latest_demand":
                latest_quantity,

            "inventory_risk":
                risk,

            "recommended_action":
                action
        })

    risk_order = {

        "High": 1,
        "Medium": 2,
        "Low": 3
    }

    results.sort(
        key=lambda x:
            risk_order[
                x["inventory_risk"]
            ]
    )

    return {

        "business_id":
            business_id,

        "products":
            results
    }


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

@app.get("/business/insights/{business_id}")
def get_business_insights(
    business_id: int,
    db: Session = Depends(get_db)
):

    records = (
        db.query(
            models.SalesRecord
        )
        .filter(
            models.SalesRecord.business_id
            == business_id
        )
        .order_by(
            models.SalesRecord.sale_date
        )
        .all()
    )

    if not records:

        return {

            "business_id":
                business_id,

            "business_summary":
                {},

            "insights":
                []
        }

    product_data = {}

    for record in records:

        if record.product not in product_data:

            product_data[
                record.product
            ] = {

                "quantities":
                    [],

                "sales":
                    0
            }

        product_data[
            record.product
        ][
            "quantities"
        ].append(
            record.quantity
        )

        product_data[
            record.product
        ][
            "sales"
        ] += (
            record.quantity
            * record.price
        )

    insights = []

    for product, data in product_data.items():

        quantities = data[
            "quantities"
        ]

        total_quantity = sum(
            quantities
        )

        average_demand = (
            total_quantity
            / len(quantities)
        )

        latest_demand = (
            quantities[-1]
        )

        if latest_demand > average_demand:

            demand_status = "Increasing"

        elif latest_demand < average_demand:

            demand_status = "Decreasing"

        else:

            demand_status = "Stable"

        if (
            latest_demand
            >= average_demand * 1.25
        ):

            risk = "High"

            recommendation = (
                f"{product}: Demand is high. "
                "Consider restocking soon."
            )

        elif (
            latest_demand
            >= average_demand * 0.90
        ):

            risk = "Medium"

            recommendation = (
                f"{product}: Demand is close "
                "to the recent average. "
                "Monitor stock regularly."
            )

        else:

            risk = "Low"

            recommendation = (
                f"{product}: Demand is relatively "
                "low. Maintain normal stock levels."
            )

        insights.append({

            "product":
                product,

            "total_quantity":
                total_quantity,

            "total_sales":
                data["sales"],

            "average_demand":
                round(
                    average_demand,
                    2
                ),

            "latest_demand":
                latest_demand,

            "demand_status":
                demand_status,

            "inventory_risk":
                risk,

            "recommendation":
                recommendation
        })

    total_quantity = sum(
        record.quantity
        for record in records
    )

    total_sales = sum(
        record.quantity
        * record.price
        for record in records
    )

    insights.sort(
        key=lambda x:
            x["total_sales"],
        reverse=True
    )

    return {

        "business_id":
            business_id,

        "business_summary": {

            "total_records":
                len(records),

            "total_quantity":
                total_quantity,

            "total_sales":
                total_sales,

            "product_count":
                len(product_data)
        },

        "insights":
            insights
    }


# ============================================================
# AI BUSINESS RECOMMENDATION
# ============================================================

@app.post("/business/recommendation")
def get_business_recommendation(
    request: BusinessRecommendationRequest
):

    product = request.product

    review = getattr(
        request,
        "review",
        None
    )

    store_id = getattr(
        request,
        "store_id",
        None
    )

    result = generate_business_decision(

        product=product,

        review_text=review,

        store_id=store_id
    )

    if "error" in result:

        raise HTTPException(
            status_code=404,
            detail=result["error"]
        )

    result["business_id"] = (
        request.business_id
        if hasattr(
            request,
            "business_id"
        )
        else None
    )

    return result


# ============================================================
# INVENTORY UPDATE
# ============================================================

@app.post("/inventory/update")
def update_inventory(
    request: InventoryUpdateRequest,
    db: Session = Depends(get_db)
):

    if request.stock_quantity < 0:

        raise HTTPException(
            status_code=400,
            detail=(
                "Stock quantity cannot "
                "be negative."
            )
        )

    business = (
        db.query(models.Business)
        .filter(
            models.Business.id
            == request.business_id
        )
        .first()
    )

    if not business:

        raise HTTPException(
            status_code=404,
            detail="Business not found."
        )

    existing_inventory = (
        db.query(
            models.InventoryRecord
        )
        .filter(
            models.InventoryRecord.business_id
            == request.business_id
        )
        .filter(
            models.InventoryRecord.product
            == request.product
        )
        .first()
    )

    if existing_inventory:

        existing_inventory.stock_quantity = (
            request.stock_quantity
        )

    else:

        existing_inventory = (
            models.InventoryRecord(

                business_id=
                    request.business_id,

                product=
                    request.product,

                stock_quantity=
                    request.stock_quantity
            )
        )

        db.add(
            existing_inventory
        )

    db.commit()

    db.refresh(
        existing_inventory
    )

    return {

        "message":
            "Inventory updated successfully.",

        "business_id":
            request.business_id,

        "product":
            request.product,

        "stock_quantity":
            existing_inventory.stock_quantity
    }


# ============================================================
# INVENTORY STATUS
# ============================================================

@app.get(
    "/inventory/status/{business_id}/{product}"
)
def get_inventory_status(
    business_id: int,
    product: str,
    db: Session = Depends(get_db)
):

    inventory = (
        db.query(
            models.InventoryRecord
        )
        .filter(
            models.InventoryRecord.business_id
            == business_id
        )
        .filter(
            models.InventoryRecord.product
            == product
        )
        .first()
    )

    if not inventory:

        raise HTTPException(
            status_code=404,
            detail=(
                "Inventory record not found."
            )
        )

    sales_records = (
        db.query(
            models.SalesRecord
        )
        .filter(
            models.SalesRecord.business_id
            == business_id
        )
        .filter(
            models.SalesRecord.product
            == product
        )
        .order_by(
            models.SalesRecord.sale_date
        )
        .all()
    )

    if not sales_records:

        raise HTTPException(
            status_code=404,
            detail=(
                "Sales data not found "
                "for this product."
            )
        )

    quantities = [
        record.quantity
        for record in sales_records
    ]

    average_daily_demand = (
        sum(quantities)
        / len(quantities)
    )

    stock_quantity = (
        inventory.stock_quantity
    )

    if average_daily_demand > 0:

        estimated_days_remaining = (
            stock_quantity
            / average_daily_demand
        )

    else:

        estimated_days_remaining = None

    if estimated_days_remaining is None:

        inventory_status = "Unknown"

        recommendation = (
            "Not enough demand data "
            "to estimate stock requirements."
        )

    elif estimated_days_remaining <= 1:

        inventory_status = "Critical"

        recommendation = (
            f"{product} stock is very low. "
            "Consider restocking immediately."
        )

    elif estimated_days_remaining <= 3:

        inventory_status = "Low"

        recommendation = (
            f"{product} stock may run low soon. "
            "Plan a restock."
        )

    elif estimated_days_remaining <= 7:

        inventory_status = "Moderate"

        recommendation = (
            f"{product} has moderate stock "
            "coverage. Continue monitoring demand."
        )

    else:

        inventory_status = "Healthy"

        recommendation = (
            f"{product} has sufficient stock "
            "based on recent demand."
        )

    return {

        "business_id":
            business_id,

        "product":
            product,

        "stock_quantity":
            stock_quantity,

        "average_daily_demand":
            round(
                average_daily_demand,
                2
            ),

        "estimated_days_remaining":
            (
                round(
                    estimated_days_remaining,
                    2
                )
                if estimated_days_remaining
                is not None
                else None
            ),

        "inventory_status":
            inventory_status,

        "recommendation":
            recommendation
    }

# ============================================================
# PRODUCT RELATIONSHIPS AND DEMAND PROFILE
# ============================================================

@app.get("/product-relationships/{product_id}")
def product_relationships_endpoint(
    product_id: str,
    limit: int = 5,
):
    if limit < 1 or limit > 20:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 20.",
        )

    result = get_product_relationships(
        product_id=product_id,
        limit=limit,
    )

    if "error" in result:
        raise HTTPException(
            status_code=404,
            detail=result["error"],
        )

    return result


@app.get("/product-demand-profile/{product_id}")
def product_demand_profile_endpoint(
    product_id: str,
):
    result = get_product_demand_profile(product_id)

    if "error" in result:
        raise HTTPException(
            status_code=404,
            detail=result["error"],
        )

    return result
