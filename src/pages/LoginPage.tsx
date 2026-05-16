import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useUser } from "../context/UserContext";
import type { User } from "../context/UserContext";
import { Activity } from "lucide-react";

const MOCK_USERS: User[] = [
  { id: "1", name: "Aanya Sharma", role: "patient", email: "patient@test.com" },
  { id: "2", name: "Dr. Rohan Mehta", role: "therapist", email: "therapist@test.com" }
];

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();
  const { setCurrentUser } = useUser();

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    // Simple mock authentication matching by email
    const user = MOCK_USERS.find((u) => u.email === email);

    if (user && password) { // Accepting any password for the mock
      setCurrentUser(user);
      if (user.role === "patient") {
        navigate("/patient/dashboard");
      } else {
        navigate("/therapist/dashboard");
      }
    } else {
      setError("Invalid email or password");
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-gray-50 items-center justify-center p-4 font-sans">
      <div className="w-full max-w-md bg-white border border-gray-200 rounded-2xl p-8 sm:p-10">
        <div className="flex flex-col items-center mb-8">
          <div className="w-12 h-12 bg-teal-50 text-[var(--primary)] rounded-full flex items-center justify-center mb-4">
            <Activity className="w-6 h-6" />
          </div>
          <h1 className="text-2xl font-bold text-[var(--foreground)]">Welcome back</h1>
          <p className="text-gray-500 text-sm mt-1">Sign in to your MindBridge account</p>
        </div>

        <form onSubmit={handleLogin} className="flex flex-col gap-5">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Email address</label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--primary)] focus:bg-white transition-all text-[var(--foreground)]"
              placeholder="patient@test.com"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--primary)] focus:bg-white transition-all text-[var(--foreground)]"
              placeholder="••••••••"
            />
          </div>

          {error && (
            <p className="text-red-500 text-sm bg-red-50 p-3 rounded-lg border border-red-100">{error}</p>
          )}

          <button
            type="submit"
            className="w-full py-3 bg-[var(--primary)] text-white font-semibold rounded-xl hover:bg-teal-700 transition-colors mt-2"
          >
            Login
          </button>
        </form>

        <p className="text-center text-sm text-gray-500 mt-8">
          Don't have an account?{" "}
          <Link to="/register" className="text-[var(--primary)] font-semibold hover:underline">
            Register
          </Link>
        </p>
      </div>
    </div>
  );
}
