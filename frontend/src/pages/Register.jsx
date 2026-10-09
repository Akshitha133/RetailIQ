import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

function Register() {
  const navigate = useNavigate();

  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [businessName, setBusinessName] = useState("");
  const [category, setCategory] = useState("");
  const [location, setLocation] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Basic validation
    if (
      !fullName ||
      !email ||
      !password ||
      !businessName ||
      !category
    ) {
      alert("Please fill in all required fields.");
      return;
    }

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/register",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            full_name: fullName,
            email: email,
            password: password,
            business_name: businessName,
            category: category,
            location: location,
          }),
        }
      );

      const data = await response.json();

      // Backend returned an error
      if (!response.ok) {
        alert(data.detail || "Registration failed.");
        return;
      }

      // Registration successful
      alert(
        `Account created successfully!\nUser ID: ${data.user_id}\nBusiness ID: ${data.business_id}`
      );

      // Go back to Login page
      navigate("/");
    } catch (error) {
      console.error("Registration error:", error);

      alert(
        "Cannot connect to the server. Make sure the FastAPI backend is running."
      );
    }
  };

  return (
    <div className="register-container">

      {/* Left side - Branding */}
      <div className="register-brand">
        <h1>RetailIQ</h1>

        <p>
          Turn your business data into understandable
          insights and better decisions.
        </p>
      </div>

      {/* Right side - Register Form */}
      <div className="register-card">

        <h2>Create Account</h2>

        <p className="register-subtitle">
          Register your business to get started
        </p>

        <form onSubmit={handleSubmit}>

          {/* Full Name */}
          <div className="form-group">
            <label>Full Name</label>

            <input
              type="text"
              placeholder="Enter your full name"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
            />
          </div>

          {/* Email */}
          <div className="form-group">
            <label>Email</label>

            <input
              type="email"
              placeholder="Enter your email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>

          {/* Password */}
          <div className="form-group">
            <label>Password</label>

            <input
              type="password"
              placeholder="Create a password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          {/* Business Name */}
          <div className="form-group">
            <label>Business Name</label>

            <input
              type="text"
              placeholder="Enter your business name"
              value={businessName}
              onChange={(e) => setBusinessName(e.target.value)}
            />
          </div>

          {/* Business Category */}
          <div className="form-group">
            <label>Business Category</label>

            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
            >
              <option value="">Select business category</option>
              <option value="Grocery">Grocery</option>
              <option value="Retail">Retail</option>
              <option value="Restaurant">Restaurant</option>
              <option value="Clothing">Clothing</option>
              <option value="Electronics">Electronics</option>
              <option value="Pharmacy">Pharmacy</option>
              <option value="Other">Other</option>
            </select>
          </div>

          {/* Business Location */}
          <div className="form-group">
            <label>Business Location</label>

            <input
              type="text"
              placeholder="Optional - e.g. Hyderabad"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
            />
          </div>

          {/* Submit */}
          <button
            type="submit"
            className="register-button"
          >
            Create Account
          </button>

        </form>

        {/* Login Link */}
        <p className="login-link">
          Already have an account?{" "}
          <Link to="/">
            Login
          </Link>
        </p>

      </div>
    </div>
  );
}

export default Register;