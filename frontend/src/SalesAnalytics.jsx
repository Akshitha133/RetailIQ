
import React, { useState } from "react";

const API_URL = "http://127.0.0.1:8000";

export default function SalesAnalytics() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);

  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const [summary, setSummary] = useState(null);
  const [analytics, setAnalytics] = useState(null);

  const businessId = localStorage.getItem("business_id");

  async function loadAnalytics() {
    if (!businessId) {
      setError("Business ID is missing. Please log in again.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const [summaryResponse, analyticsResponse] = await Promise.all([
        fetch(
          `${API_URL}/sales/summary/${encodeURIComponent(businessId)}`
        ),
        fetch(
          `${API_URL}/sales/analytics/${encodeURIComponent(businessId)}`
        ),
      ]);

      const summaryData = await summaryResponse.json();
      const analyticsData = await analyticsResponse.json();

      if (!summaryResponse.ok) {
        throw new Error(
          summaryData.detail || "Unable to load sales summary."
        );
      }

      if (!analyticsResponse.ok) {
        throw new Error(
          analyticsData.detail || "Unable to load sales analytics."
        );
      }

      setSummary(summaryData);
      setAnalytics(analyticsData);
      setMessage("");
    } catch (err) {
      setError(err.message || "Unable to load sales analytics.");
    } finally {
      setLoading(false);
    }
  }

  async function uploadSales(event) {
    event.preventDefault();

    setMessage("");
    setError("");

    if (!businessId) {
      setError("Business ID is missing. Please log in again.");
      return;
    }

    if (!file) {
      setError("Please select a CSV file first.");
      return;
    }

    if (!file.name.toLowerCase().endsWith(".csv")) {
      setError("Please select a valid CSV file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setUploading(true);

    try {
      const response = await fetch(
        `${API_URL}/sales/upload?business_id=${encodeURIComponent(businessId)}`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        const detail =
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail || "Sales upload failed.");

        throw new Error(detail);
      }

      let successMessage = data.message || "Upload completed.";

      if (typeof data.records_added === "number") {
        successMessage += ` Records added: ${data.records_added}.`;
      }

      if (typeof data.duplicate_rows_removed === "number") {
        successMessage +=
          ` Duplicate rows removed: ${data.duplicate_rows_removed}.`;
      }

      setMessage(successMessage);
      setFile(null);

      const fileInput = document.getElementById("sales-csv");
      if (fileInput) {
        fileInput.value = "";
      }

      await loadAnalytics();
    } catch (err) {
      setError(err.message || "Unable to upload sales data.");
    } finally {
      setUploading(false);
    }
  }

  function formatValue(value) {
    if (value === null || value === undefined || value === "") {
      return "—";
    }

    if (typeof value === "number") {
      return value.toLocaleString("en-IN", {
        maximumFractionDigits: 2,
      });
    }

    if (typeof value === "boolean") {
      return value ? "Yes" : "No";
    }

    return String(value);
  }

  function renderAnalyticsTable() {
    if (!analytics) {
      return null;
    }

    const products = Array.isArray(analytics.products)
      ? analytics.products
      : [];

    if (products.length === 0) {
      return (
        <p className="sales-empty">
          No product-level analytics are available yet.
        </p>
      );
    }

    const columns = Object.keys(products[0]);

    return (
      <div className="sales-table-wrapper">
        <table className="sales-table">
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column}>
                  {column
                    .replace(/_/g, " ")
                    .replace(/\b\w/g, (character) =>
                      character.toUpperCase()
                    )}
                </th>
              ))}
            </tr>
          </thead>

          <tbody>
            {products.map((product, index) => (
              <tr key={`${product.product || "product"}-${index}`}>
                {columns.map((column) => (
                  <td key={column}>
                    {formatValue(product[column])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  return (
    <main className="sales-page">
      <header className="sales-header">
        <p className="sales-eyebrow">
          RETAILIQ BUSINESS INTELLIGENCE
        </p>

        <h1>Sales Upload &amp; Analytics</h1>

        <p className="sales-subtitle">
          Upload sales records and explore the performance of your
          registered business.
        </p>
      </header>

      {!businessId && (
        <div className="sales-alert sales-warning">
          Business ID is missing. Please log in again.
        </div>
      )}

      {message && (
        <div className="sales-alert sales-success" role="status">
          {message}
        </div>
      )}

      {error && (
        <div className="sales-alert sales-error" role="alert">
          {error}
        </div>
      )}

      <section className="sales-card">
        <div className="sales-section-heading">
          <div>
            <h2>Upload Sales CSV</h2>
            <p>
              Select a CSV file with the required column names.
            </p>
          </div>
        </div>

        <form
          onSubmit={uploadSales}
          className="sales-upload-form"
        >
          <label
            htmlFor="sales-csv"
            className="sales-file-label"
          >
            Choose CSV file
          </label>

          <input
            id="sales-csv"
            type="file"
            accept=".csv,text/csv"
            onChange={(event) => {
              setFile(event.target.files?.[0] || null);
              setMessage("");
              setError("");
            }}
            disabled={uploading}
          />

          {file && (
            <p className="sales-file-name">
              Selected file: <strong>{file.name}</strong>
            </p>
          )}

          <p className="sales-help">
            Required columns: <code>Date</code>,{" "}
            <code>Product</code>, <code>Quantity</code>, and{" "}
            <code>Price</code>.
          </p>

          <button
            type="submit"
            className="sales-button sales-primary-button"
            disabled={uploading || !businessId}
          >
            {uploading ? "Uploading..." : "Upload Sales CSV"}
          </button>
        </form>
      </section>

      <section className="sales-card">
        <div className="sales-section-heading">
          <div>
            <h2>Sales Performance</h2>
            <p>
              Review total sales, quantities, products, and
              product-level trends.
            </p>
          </div>

          <button
            type="button"
            className="sales-button sales-secondary-button"
            onClick={loadAnalytics}
            disabled={loading || !businessId}
          >
            {loading ? "Loading..." : "Refresh Analytics"}
          </button>
        </div>

        {loading && (
          <p className="sales-loading">
            Loading sales analytics...
          </p>
        )}

        {summary && (
          <>
            <h3 className="sales-subheading">
              Business Summary
            </h3>

            <div className="sales-metrics">
              <div className="sales-metric">
                <span>Total Records</span>
                <strong>
                  {formatValue(summary.total_records)}
                </strong>
              </div>

              <div className="sales-metric">
                <span>Total Quantity Sold</span>
                <strong>
                  {formatValue(summary.total_quantity)}
                </strong>
              </div>

              <div className="sales-metric">
                <span>Total Sales</span>
                <strong>
                  ₹{formatValue(summary.total_sales)}
                </strong>
              </div>

              <div className="sales-metric">
                <span>Products</span>
                <strong>
                  {Array.isArray(summary.products)
                    ? summary.products.length
                    : 0}
                </strong>
              </div>
            </div>

            {Array.isArray(summary.products) &&
              summary.products.length > 0 && (
                <div className="sales-product-list">
                  <h3>Products in Sales Records</h3>

                  <div className="sales-product-tags">
                    {summary.products.map((product) => (
                      <span
                        className="sales-product-tag"
                        key={product}
                      >
                        {product}
                      </span>
                    ))}
                  </div>
                </div>
              )}
          </>
        )}

        {analytics && (
          <>
            <h3 className="sales-subheading">
              Product-Level Analytics
            </h3>

            {renderAnalyticsTable()}
          </>
        )}

        {!summary && !analytics && !loading && (
          <p className="sales-empty">
            Click <strong>Refresh Analytics</strong> to load your
            business's sales data.
          </p>
        )}
      </section>

      <footer className="sales-footer">
        RetailIQ · Sales analytics for independent retailers
      </footer>
    </main>
  );
}
