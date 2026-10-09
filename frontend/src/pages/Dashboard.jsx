
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Dashboard.css";

const API_URL = "http://127.0.0.1:8000";

const STORES = ["S001", "S002", "S003", "S004", "S005"];
const PRODUCTS = Array.from(
  { length: 20 },
  (_, index) => `P${String(index + 1).padStart(4, "0")}`
);

function Dashboard() {
  const navigate = useNavigate();

  const [store, setStore] = useState("S001");
  const [product, setProduct] = useState("P0001");

  const [decision, setDecision] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [reviewText, setReviewText] = useState("");
  const [reviewResult, setReviewResult] = useState(null);
  const [reviewLoading, setReviewLoading] = useState(false);
  const [reviewError, setReviewError] = useState("");

  // Product relationships
  const [relationships, setRelationships] = useState(null);
  const [demandProfile, setDemandProfile] = useState(null);
  const [relationshipLoading, setRelationshipLoading] = useState(false);
  const [relationshipError, setRelationshipError] = useState("");

  const fullName = localStorage.getItem("full_name") || "Retailer";

  const formatNumber = (value, decimals = 2) => {
    if (value === null || value === undefined || value === "") return "—";
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(decimals) : "—";
  };

  const probabilityToPercent = (value) => {
    if (value === null || value === undefined || value === "") return "—";
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    return `${(number * 100).toFixed(2)}%`;
  };

  const formatFeatureName = (name) =>
    String(name || "Unknown feature")
      .replaceAll("_", " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());

  const getRiskClass = (risk) => {
    const value = String(risk || "").toLowerCase();
    if (value === "high") return "risk-high";
    if (value === "medium") return "risk-medium";
    if (value === "low") return "risk-low";
    return "";
  };

  const getHealthClass = (score) => {
    const number = Number(score);
    if (!Number.isFinite(number)) return "";
    if (number >= 75) return "health-good";
    if (number >= 50) return "health-medium";
    return "health-poor";
  };

  const getPriorityClass = (priority) => {
    const value = String(priority || "").toLowerCase();
    if (value === "high") return "priority-high";
    if (value === "medium") return "priority-medium";
    return "priority-low";
  };

  const handleLogout = () => {
    localStorage.removeItem("user_id");
    localStorage.removeItem("full_name");
    localStorage.removeItem("business_id");
    navigate("/");
  };

  // ---------------------------------------------------------
  // BUSINESS ANALYSIS
  // ---------------------------------------------------------

  const analyzeBusiness = async () => {
    setLoading(true);
    setError("");
    setDecision(null);

    try {
      const response = await fetch(`${API_URL}/decision`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          product,
          store_id: store,
          review: "",
        }),
      });

      const data = await response.json();

      if (!response.ok || data?.error) {
        throw new Error(
          data?.detail || data?.error || "Unable to analyze the business."
        );
      }

      setDecision(data);
    } catch (err) {
      console.error(err);
      setError(err.message || "Could not connect to the RetailIQ backend.");
    } finally {
      setLoading(false);
    }
  };

  // ---------------------------------------------------------
  // CUSTOMER FEEDBACK ANALYSIS
  // ---------------------------------------------------------

  const analyzeReview = async () => {
    if (!reviewText.trim()) {
      setReviewError("Please enter customer feedback first.");
      return;
    }

    setReviewLoading(true);
    setReviewError("");
    setReviewResult(null);

    try {
      const savedBusinessId = localStorage.getItem("business_id");

      const response = await fetch(`${API_URL}/review/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          business_id: savedBusinessId ? Number(savedBusinessId) : 0,
          product,
          review: reviewText.trim(),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data?.detail || data?.error || "Unable to analyze customer feedback."
        );
      }

      setReviewResult(data);
    } catch (err) {
      console.error(err);
      setReviewError(err.message || "Could not analyze customer feedback.");
    } finally {
      setReviewLoading(false);
    }
  };

  // ---------------------------------------------------------
  // PRODUCT RELATIONSHIPS AND DEMAND PROFILE
  // ---------------------------------------------------------

  const analyzeProductRelationships = async () => {
    setRelationshipLoading(true);
    setRelationshipError("");
    setRelationships(null);
    setDemandProfile(null);

    try {
      const encodedProduct = encodeURIComponent(product);

      const [relationshipResponse, profileResponse] = await Promise.all([
        fetch(
          `${API_URL}/product-relationships/${encodedProduct}?limit=5`
        ),
        fetch(`${API_URL}/product-demand-profile/${encodedProduct}`),
      ]);

      const [relationshipData, profileData] = await Promise.all([
        relationshipResponse.json(),
        profileResponse.json(),
      ]);

      if (!relationshipResponse.ok || relationshipData?.error) {
        throw new Error(
          relationshipData?.detail ||
            relationshipData?.error ||
            "Unable to load related products."
        );
      }

      if (!profileResponse.ok || profileData?.error) {
        throw new Error(
          profileData?.detail ||
            profileData?.error ||
            "Unable to load the product demand profile."
        );
      }

      setRelationships(relationshipData);
      setDemandProfile(profileData);
    } catch (err) {
      console.error(err);
      setRelationshipError(
        err.message || "Could not load product relationships."
      );
    } finally {
      setRelationshipLoading(false);
    }
  };

  // ---------------------------------------------------------
  // DATA REFERENCES
  // ---------------------------------------------------------

  const forecast = decision?.sales_forecast || {};
  const inventory = decision?.inventory_risk || {};
  const forecastExplanation = decision?.explainability?.forecast || {};
  const inventoryExplanation = decision?.explainability?.inventory || {};
  const healthScore = decision?.business_health_score;
  const recommendations = decision?.recommendations || [];
  const inventoryProbabilities = inventory?.probabilities || {};
  const inventoryRisk = inventory?.risk || "—";

  const inventoryShapProbability =
    inventoryExplanation?.probabilities?.[
      inventoryExplanation?.predicted_risk
    ];

  // ---------------------------------------------------------
  // REUSABLE DISPLAY HELPERS
  // ---------------------------------------------------------

  const Metric = ({ label, value, description, className = "" }) => (
    <div className="metric-card">
      <span className="metric-label">{label}</span>
      <strong className={`metric-value ${className}`}>{value ?? "—"}</strong>
      {description && (
        <span className="metric-description">{description}</span>
      )}
    </div>
  );

  const renderProbabilityRows = () => {
    const rows = [
      { label: "High", value: inventoryProbabilities.High },
      { label: "Medium", value: inventoryProbabilities.Medium },
      { label: "Low", value: inventoryProbabilities.Low },
    ];

    if (rows.every((row) => row.value === undefined)) {
      return <div className="empty-state">Probability information unavailable.</div>;
    }

    return (
      <div className="probability-list">
        {rows.map((row) => {
          const numericValue = Number(row.value || 0);

          return (
            <div className="probability-row" key={row.label}>
              <div className="probability-label">
                <span>{row.label}</span>
                <strong>{probabilityToPercent(row.value)}</strong>
              </div>
              <div className="probability-bar">
                <div
                  className={`probability-fill ${getRiskClass(row.label)}`}
                  style={{
                    width: `${Math.max(0, Math.min(100, numericValue * 100))}%`,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  const renderShapFeatures = (features, inventoryMode = false) => {
    if (!features?.length) {
      return (
        <div className="empty-state">
          Explainability information is unavailable.
        </div>
      );
    }

    return (
      <div className="shap-list">
        {features.map((item, index) => {
          const shapValue = Number(item?.shap_value ?? 0);
          const positive = shapValue >= 0;

          return (
            <div
              className="shap-row"
              key={`${item?.feature || "feature"}-${index}`}
            >
              <div className="shap-main">
                <div className="shap-feature">
                  {formatFeatureName(item?.feature)}
                </div>
                <div className="shap-value">
                  Value: <strong>{formatNumber(item?.value)}</strong>
                </div>
              </div>

              <div
                className={`shap-impact ${
                  positive ? "shap-positive" : "shap-negative"
                }`}
              >
                <strong>
                  {positive ? "+" : ""}
                  {formatNumber(shapValue, 3)}
                </strong>
                <span>
                  {item?.effect ||
                    (inventoryMode
                      ? positive
                        ? "supports predicted risk"
                        : "reduces predicted risk"
                      : positive
                        ? "increases forecast"
                        : "decreases forecast")}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  // ---------------------------------------------------------
  // DASHBOARD
  // ---------------------------------------------------------

  return (
    <div className="retailiq-dashboard">
      <header className="dashboard-header">
        <div className="brand-section">
          <div className="brand-icon">R</div>
          <div>
            <h1>RetailIQ</h1>
            <p>Business Decision Intelligence</p>
          </div>
        </div>

        <div className="header-right">
          <span className="welcome-text">Welcome, {fullName}</span>
          <button className="logout-button" onClick={handleLogout}>
            Logout
          </button>
        </div>
      </header>

      <main className="dashboard-main">
        <section className="hero-section">
          <div>
            <h2>AI-Powered Retail Intelligence</h2>
            <p>
              Analyze demand, inventory risk, customer feedback, and business
              performance using explainable machine learning.
            </p>
          </div>
          <div className="technology-badge">
            XGBoost + Random Forest + SHAP
          </div>
        </section>

        {/* BUSINESS ANALYSIS */}
        <section className="dashboard-card analysis-card">
          <div className="section-heading">
            <h2>Business Analysis</h2>
            <p>Select a store and product from the real retail dataset.</p>
          </div>

          <div className="analysis-controls">
            <div className="control-group">
              <label htmlFor="store">Store</label>
              <select
                id="store"
                value={store}
                onChange={(event) => setStore(event.target.value)}
              >
                {STORES.map((storeId) => (
                  <option value={storeId} key={storeId}>
                    {storeId}
                  </option>
                ))}
              </select>
            </div>

            <div className="control-group">
              <label htmlFor="product">Product</label>
              <select
                id="product"
                value={product}
                onChange={(event) => setProduct(event.target.value)}
              >
                {PRODUCTS.map((productId) => (
                  <option value={productId} key={productId}>
                    {productId}
                  </option>
                ))}
              </select>
            </div>

            <button
              className="primary-button analyze-button"
              onClick={analyzeBusiness}
              disabled={loading}
            >
              {loading ? "Analyzing..." : "Analyze Business"}
            </button>
          </div>

          {error && <div className="error-message">{error}</div>}
        </section>

        {loading && (
          <section className="dashboard-card loading-card">
            <div className="loading-spinner" />
            <h3>Analyzing {product} at {store}</h3>
            <p>Running forecasting, inventory analysis, and SHAP explanations.</p>
          </section>
        )}

        {decision && !loading && (
          <>
            {/* SUMMARY */}
            <section className="dashboard-card">
              <div className="section-heading">
                <h2>Business Intelligence Summary</h2>
                <p>
                  Analysis for <strong>{decision.product || product}</strong> at{" "}
                  <strong>{decision.store || store}</strong>
                </p>
              </div>

              <div className="summary-grid">
                <Metric
                  label="Forecast Demand"
                  value={formatNumber(forecast.forecast_demand)}
                  description="Predicted units"
                />
                <Metric
                  label="Recent 7-Day Average"
                  value={formatNumber(forecast.recent_7_day_average)}
                  description="Historical demand"
                />
                <Metric
                  label="Inventory Risk"
                  value={inventoryRisk}
                  description="Random Forest prediction"
                  className={getRiskClass(inventoryRisk)}
                />
                <Metric
                  label="Business Health"
                  value={healthScore ?? "—"}
                  description="Overall business score"
                  className={getHealthClass(healthScore)}
                />
              </div>
            </section>

            {/* FORECAST */}
            <section className="dashboard-card">
              <div className="card-header-row">
                <div>
                  <h2>📈 Demand Forecast</h2>
                  <p>XGBoost-based future demand prediction.</p>
                </div>
                <span className="model-badge">XGBoost</span>
              </div>

              <div className="forecast-grid">
                <div className="forecast-stat primary-stat">
                  <span>Predicted Demand</span>
                  <strong>{formatNumber(forecast.forecast_demand)}</strong>
                  <small>units</small>
                </div>
                <div className="forecast-stat">
                  <span>Recent 7-Day Average</span>
                  <strong>{formatNumber(forecast.recent_7_day_average)}</strong>
                  <small>units</small>
                </div>
                <div className="forecast-stat">
                  <span>Demand Change</span>
                  <strong>
                    {formatNumber(forecast.demand_change_7)}
                  </strong>
                  <small>change over recent period</small>
                </div>
                <div className="forecast-stat">
                  <span>Forecast Date</span>
                  <strong className="date-value">{forecast.date || "—"}</strong>
                  <small>latest forecast point</small>
                </div>
              </div>
            </section>

            {/* INVENTORY RISK */}
            <section className="dashboard-card">
              <div className="card-header-row">
                <div>
                  <h2>📦 Inventory Risk</h2>
                  <p>Random Forest prediction of future inventory risk.</p>
                </div>
                <span className="model-badge">Random Forest</span>
              </div>

              <div className="inventory-top">
                <div className={`risk-display ${getRiskClass(inventoryRisk)}`}>
                  <span>Predicted Risk</span>
                  <strong>{inventoryRisk}</strong>
                </div>
                <div className="inventory-stat">
                  <span>Current Inventory</span>
                  <strong>{formatNumber(inventory.inventory_level, 0)}</strong>
                  <small>units</small>
                </div>
                <div className="inventory-stat">
                  <span>Predicted Demand</span>
                  <strong>{formatNumber(inventory.predicted_demand)}</strong>
                  <small>units</small>
                </div>
                <div className="inventory-stat">
                  <span>Inventory Coverage</span>
                  <strong>
                    {formatNumber(inventory.inventory_coverage_days)}
                  </strong>
                  <small>days</small>
                </div>
                <div className="inventory-stat">
                  <span>High-Risk Probability</span>
                  <strong>
                    {probabilityToPercent(inventoryProbabilities.High)}
                  </strong>
                  <small>model probability</small>
                </div>
              </div>

              <div className="subsection">
                <h3>Risk Probabilities</h3>
                {renderProbabilityRows()}
              </div>
            </section>

            {/* BUSINESS HEALTH */}
            <section className="dashboard-card health-card">
              <div className="card-header-row">
                <div>
                  <h2>❤️ Business Health</h2>
                  <p>Overall health assessment from the decision engine.</p>
                </div>
                <div className={`health-score ${getHealthClass(healthScore)}`}>
                  <strong>{healthScore ?? "—"}</strong>
                  <span>/ 100</span>
                </div>
              </div>

              <div className="health-progress-container">
                <div className="health-progress">
                  <div
                    className={`health-progress-fill ${getHealthClass(healthScore)}`}
                    style={{
                      width: `${Math.max(
                        0,
                        Math.min(100, Number(healthScore || 0))
                      )}%`,
                    }}
                  />
                </div>
              </div>

              <div className="health-description">
                <strong>Business Health Score</strong>
                <p>
                  The score combines available business intelligence signals,
                  including demand outlook and inventory risk.
                </p>
              </div>
            </section>

            {/* RECOMMENDATIONS */}
            <section className="dashboard-card">
              <div className="section-heading">
                <h2>💡 Business Recommendations</h2>
                <p>Actionable recommendations generated from the analysis.</p>
              </div>

              {recommendations.length === 0 ? (
                <div className="empty-state">
                  No recommendations are currently available.
                </div>
              ) : (
                <div className="recommendation-list">
                  {recommendations.map((recommendation, index) => (
                    <div className="recommendation-card" key={index}>
                      <div className="recommendation-top">
                        <span
                          className={`priority-badge ${getPriorityClass(
                            recommendation?.priority
                          )}`}
                        >
                          {recommendation?.priority || "Normal"}
                        </span>
                        <span className="recommendation-type">
                          {recommendation?.type}
                        </span>
                      </div>
                      <p>{recommendation?.message}</p>
                    </div>
                  ))}
                </div>
              )}
            </section>

            {/* FORECAST SHAP */}
            <section className="dashboard-card">
              <div className="card-header-row">
                <div>
                  <h2>🔍 Forecast Explainability</h2>
                  <p>
                    SHAP shows which features influenced the XGBoost prediction.
                  </p>
                </div>
                <span className="model-badge">SHAP + XGBoost</span>
              </div>

              <div className="explain-summary">
                <div>
                  <span>Model Prediction</span>
                  <strong>
                    {formatNumber(
                      forecastExplanation.predicted_demand ??
                        forecast.forecast_demand
                    )}
                  </strong>
                </div>
                <div>
                  <span>Base Value</span>
                  <strong>{formatNumber(forecastExplanation.base_value)}</strong>
                </div>
                <div>
                  <span>Store</span>
                  <strong>{forecastExplanation.store || decision.store || store}</strong>
                </div>
                <div>
                  <span>Date</span>
                  <strong>
                    {forecastExplanation.date || forecast.date || "—"}
                  </strong>
                </div>
              </div>

              <div className="subsection">
                <h3>Top Influencing Features</h3>
                {renderShapFeatures(forecastExplanation.top_features)}
              </div>
            </section>

            {/* INVENTORY SHAP */}
            <section className="dashboard-card">
              <div className="card-header-row">
                <div>
                  <h2>🔍 Inventory Risk Explainability</h2>
                  <p>
                    SHAP shows which features influenced the Random Forest risk
                    prediction.
                  </p>
                </div>
                <span className="model-badge">SHAP + Random Forest</span>
              </div>

              <div className="explain-summary">
                <div>
                  <span>Predicted Risk</span>
                  <strong
                    className={getRiskClass(
                      inventoryExplanation.predicted_risk || inventoryRisk
                    )}
                  >
                    {inventoryExplanation.predicted_risk || inventoryRisk}
                  </strong>
                </div>
                <div>
                  <span>Prediction Probability</span>
                  <strong>{probabilityToPercent(inventoryShapProbability)}</strong>
                </div>
                <div>
                  <span>Store</span>
                  <strong>{inventoryExplanation.store || decision.store || store}</strong>
                </div>
                <div>
                  <span>Date</span>
                  <strong>
                    {inventoryExplanation.date || inventory.date || "—"}
                  </strong>
                </div>
              </div>

              <div className="subsection">
                <h3>Top Influencing Features</h3>
                {renderShapFeatures(inventoryExplanation.top_features, true)}
              </div>
            </section>
          </>
        )}

        {/* PRODUCT RELATIONSHIPS */}
        <section className="dashboard-card">
          <div className="section-heading">
            <h2>🛍️ Product Relationships &amp; Demand Profile</h2>
            <p>
              Explore historical daily sales relationships and demand
              statistics from the real retail dataset.
            </p>
          </div>

          <p>
            Selected product: <strong>{product}</strong>
          </p>

          <button
            className="primary-button"
            onClick={analyzeProductRelationships}
            disabled={relationshipLoading}
          >
            {relationshipLoading
              ? "Analyzing Products..."
              : `Analyze Relationships for ${product}`}
          </button>

          {relationshipError && (
            <div className="error-message">{relationshipError}</div>
          )}

          {demandProfile && (
            <>
              <div className="subsection">
                <h3>Product Demand Profile</h3>
                <div className="summary-grid">
                  <Metric
                    label="Average Daily Demand"
                    value={formatNumber(demandProfile.average_daily_demand)}
                  />
                  <Metric
                    label="Maximum Daily Demand"
                    value={formatNumber(demandProfile.maximum_daily_demand)}
                  />
                  <Metric
                    label="Minimum Daily Demand"
                    value={formatNumber(demandProfile.minimum_daily_demand)}
                  />
                  <Metric
                    label="Demand Volatility"
                    value={formatNumber(demandProfile.demand_volatility)}
                  />
                </div>
                <p className="metric-description">
                  Days represented: {demandProfile.number_of_days ?? "—"}
                </p>
              </div>
            </>
          )}

          {relationships?.relationships && (
            <div className="subsection">
              <h3>Most Related Products</h3>
              <p>
                Pearson correlation measures how daily sales patterns move
                together. It does not establish that products were purchased
                in the same transaction.
              </p>

              {relationships.relationships.length === 0 ? (
                <div className="empty-state">
                  No related products were found for this product.
                </div>
              ) : (
                <div className="recommendation-list">
                  {relationships.relationships.map((item) => (
                    <div className="recommendation-card" key={item.product}>
                      <div className="recommendation-top">
                        <strong>{item.product}</strong>
                        <span className="model-badge">
                          {item.relationship_strength}
                        </span>
                      </div>
                      <p>
                        Correlation:{" "}
                        <strong>{formatNumber(item.correlation, 4)}</strong>
                      </p>
                      <p>Relationship type: {item.relationship_type}</p>
                    </div>
                  ))}
                </div>
              )}

              <p className="metric-description">
                Method: {relationships.method || "Historical daily sales correlation"}
              </p>
              <p className="metric-description">
                Data source: {relationships.data_source || "Retail dataset"}
              </p>
            </div>
          )}
        </section>

        {/* CUSTOMER FEEDBACK */}
        <section className="dashboard-card">
          <div className="card-header-row">
            <div>
              <h2>💬 Customer Feedback Intelligence</h2>
              <p>Analyze customer feedback using the trained sentiment model.</p>
            </div>
            <span className="model-badge">TF-IDF + Logistic Regression</span>
          </div>

          <div className="feedback-context">
            <span>Current Analysis</span>
            <strong>{product} • {store}</strong>
          </div>

          <div className="review-form">
            <label htmlFor="review">Customer Feedback</label>
            <textarea
              id="review"
              value={reviewText}
              onChange={(event) => setReviewText(event.target.value)}
              placeholder="Enter a customer review or feedback..."
              rows={5}
            />

            <button
              className="primary-button"
              onClick={analyzeReview}
              disabled={reviewLoading}
            >
              {reviewLoading
                ? "Analyzing Feedback..."
                : "Analyze Customer Feedback"}
            </button>
          </div>

          {reviewError && <div className="error-message">{reviewError}</div>}

          {reviewResult && (
            <div className="review-result">
              <div className="review-result-header">
                <h3>Analysis Result</h3>
              </div>
              <div className="review-result-grid">
                <div>
                  <span>Sentiment</span>
                  <strong
                    className={
                      String(reviewResult.sentiment || "").toLowerCase() ===
                      "positive"
                        ? "sentiment-positive"
                        : String(reviewResult.sentiment || "").toLowerCase() ===
                            "negative"
                          ? "sentiment-negative"
                          : ""
                    }
                  >
                    {reviewResult.sentiment || "—"}
                  </strong>
                </div>
                <div>
                  <span>Confidence</span>
                  <strong>{probabilityToPercent(reviewResult.confidence)}</strong>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* MODEL INFORMATION */}
        <section className="dashboard-card information-card">
          <div className="section-heading">
            <h2>About This Analysis</h2>
            <p>
              RetailIQ uses the real retail inventory dataset and trained
              machine-learning models to support business decisions.
            </p>
          </div>

          <div className="information-grid">
            <div>
              <span>Data Source</span>
              <strong>Real Retail Inventory Dataset</strong>
            </div>
            <div>
              <span>Demand Model</span>
              <strong>XGBoost</strong>
            </div>
            <div>
              <span>Inventory Model</span>
              <strong>Random Forest</strong>
            </div>
            <div>
              <span>Explainability</span>
              <strong>SHAP</strong>
            </div>
            <div>
              <span>Sentiment Model</span>
              <strong>TF-IDF + Logistic Regression</strong>
            </div>
            <div>
              <span>Product Relationships</span>
              <strong>Pearson Correlation</strong>
            </div>
          </div>
        </section>

        {!decision && !loading && !error && (
          <section className="dashboard-card empty-dashboard">
            <div className="empty-icon">📊</div>
            <h2>Ready to analyze your business</h2>
            <p>
              Select a store and product above, then click{" "}
              <strong>Analyze Business</strong>.
            </p>
          </section>
        )}
      </main>
    </div>
  );
}

export default Dashboard;
