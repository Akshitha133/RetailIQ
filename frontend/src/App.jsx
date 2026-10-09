
import { useState } from "react";
import {
  BrowserRouter,
  Routes,
  Route,
  Link,
  useNavigate,
  Navigate,
} from "react-router-dom";

import Register from "./pages/Register";
import Dashboard from "./pages/Dashboard";
import SalesAnalytics from "./SalesAnalytics";

import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function Login() {
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  const handleLogin = async (event) => {
    event.preventDefault();

    if (!email.trim() || !password) {
      alert("Please enter your email and password.");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email: email.trim(),
          password,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        alert(data.detail || "Login failed.");
        return;
      }

      localStorage.setItem("user_id", data.user_id ?? "");
      localStorage.setItem("full_name", data.full_name ?? "");
      localStorage.setItem("business_id", data.business_id ?? "");

      alert(`Welcome back, ${data.full_name || "Retailer"}!`);
      navigate("/dashboard");
    } catch (error) {
      console.error("Login error:", error);
      alert(
        "Cannot connect to the server. Make sure the FastAPI backend is running."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="login-container">
        <div className="brand-section">
          <h1>RetailIQ</h1>
          <p>AI-Powered Business Decision Intelligence</p>
        </div>

        <div className="login-card">
          <h2>Welcome Back</h2>
          <p className="subtitle">
            Sign in to manage and understand your business
          </p>

          <form onSubmit={handleLogin}>
            <label htmlFor="login-email">Email</label>
            <input
              id="login-email"
              type="email"
              placeholder="Enter your email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              autoComplete="email"
              required
            />

            <label htmlFor="login-password">Password</label>
            <input
              id="login-password"
              type="password"
              placeholder="Enter your password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete="current-password"
              required
            />

            <button type="submit" disabled={loading}>
              {loading ? "Logging in..." : "Login"}
            </button>
          </form>

          <p className="register-text">
            Don't have an account?{" "}
            <Link to="/register" className="login-link">
              Register
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}

function ProtectedRoute({ children }) {
  const userId = localStorage.getItem("user_id");
  const businessId = localStorage.getItem("business_id");

  if (!userId || !businessId) {
    return <Navigate to="/" replace />;
  }

  return children;
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Login />} />

        <Route path="/register" element={<Register />} />

        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }
        />

        <Route
          path="/sales"
          element={
            <ProtectedRoute>
              <SalesAnalytics />
            </ProtectedRoute>
          }
        />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
