# RetailIQ — AI-Powered Retail Business Intelligence

RetailIQ is an AI-driven business intelligence platform designed to help independent retailers make smarter decisions using sales forecasting, inventory risk prediction, customer sentiment analysis, and explainable machine learning.

## Project Overview

Small retailers often struggle to forecast product demand, manage inventory efficiently, and understand customer feedback. RetailIQ combines machine learning and business analytics to turn retail data into actionable insights.

## Key Features

- **Sales Forecasting:** Uses XGBoost to estimate product demand.
- **Inventory Risk Prediction:** Uses Random Forest to classify inventory risk.
- **Explainable AI:** Uses SHAP-based explanations to help users understand model predictions.
- **Customer Sentiment Analysis:** Analyzes customer reviews to identify positive and negative feedback.
- **Sales Analytics Dashboard:** Summarizes sales records, quantities, revenue, and product trends.
- **Inventory Tracking:** Records stock quantities and estimates remaining stock coverage using demand data.
- **Product Relationship Analysis:** Examines relationships between products' daily sales patterns.

## Technology Stack

**Frontend**
- React
- Vite
- JavaScript
- CSS

**Backend**
- Python
- FastAPI
- SQLAlchemy

**Machine Learning and Analytics**
- XGBoost
- Random Forest
- TF-IDF and Logistic Regression
- SHAP
- Pandas and NumPy

## Project Architecture

1. The React frontend provides the dashboard and analytics interface.
2. The FastAPI backend handles API requests and business logic.
3. Machine learning models generate forecasts, inventory-risk predictions, and review sentiment results.
4. SHAP explanations provide insight into the factors influencing supported model predictions.
5. Database components store application records.

## Project Structure

```text
RetailIQ/
├── backend/
│   ├── main.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   ├── decision_engine.py
│   ├── create_forecasting_features.py
│   ├── create_inventory_features.py
│   ├── create_review_model.py
│   ├── explain_forecasting.py
│   ├── explain_inventory.py
│   ├── train_forecasting_model.py
│   └── train_inventory_model.py
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── data/
└── README.md
```

## Getting Started

### Prerequisites

- Python 3
- Node.js and npm
- Git

### 1. Clone the repository

```bash
git clone https://github.com/Akshitha133/RetailIQ.git
cd RetailIQ
```

### 2. Set up the backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

Install the Python dependencies listed in the project's dependency file, if provided. If no dependency file exists yet, create one containing the packages required by the backend before following this step.

Start the API:

```bash
uvicorn main:app --reload
```

Open the API documentation at:

http://127.0.0.1:8000/docs

### 3. Set up the frontend

Open a second terminal:

```bash
cd RetailIQ/frontend
npm install
npm run dev
```

Open the local frontend URL shown in your terminal, usually:

http://localhost:5173

## Machine Learning

### Sales Forecasting
XGBoost predicts product demand using historical sales and engineered features such as lagged demand and rolling statistics.

### Inventory Risk Prediction
A Random Forest model estimates inventory risk using demand and inventory-related features.

### Customer Sentiment Analysis
Text features generated using TF-IDF are used to classify customer reviews.

### Explainable AI
SHAP-based visualizations and feature-importance outputs help explain the contribution of input features to supported model predictions.

## Important Notes

- Forecasts depend on the historical dataset and trained models available in the local environment.
- Dataset files, trained model artifacts, and environment variables may need to be configured separately when setting up the project.
- Predictions are decision-support estimates and should be validated against actual business conditions.

## Future Improvements

- Deploy the application online.
- Add secure retailer authentication and business-specific access controls.
- Integrate live sales and inventory data.
- Improve forecasting accuracy through further feature engineering and evaluation.
- Add automated alerts for low stock and high inventory risk.

## Author

**Akshitha**

GitHub: https://github.com/Akshitha133

## License

No license has been specified yet.