import { useState, useEffect } from "react";
import { useNavigate, useSearchParams, Link } from "react-router-dom";
import { Activity, User as UserIcon, Stethoscope } from "lucide-react";

export default function RegisterPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [role, setRole] = useState<"patient" | "therapist" | "">("");
  const [error, setError] = useState("");

  useEffect(() => {
    const roleParam = searchParams.get("role");
    if (roleParam === "patient" || roleParam === "therapist") {
      setRole(roleParam);
    }
  }, [searchParams]);

  const handleRegister = (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (!name || !email || !password || !confirmPassword || !role) {
      setError("Please fill out all fields and select a role.");
      return;
    }

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    // In a real app, we'd register the user here. For the prototype, we just redirect.
    if (role === "patient") {
      navigate("/patient/intake");
    } else {
      navigate("/therapist/dashboard");
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-gray-50 items-center justify-center p-4 font-sans">
      <div className="w-full max-w-lg bg-white border border-gray-200 rounded-2xl p-8 sm:p-10 my-8">
        <div className="flex flex-col items-center mb-8">
          <div className="w-12 h-12 bg-teal-50 text-[var(--primary)] rounded-full flex items-center justify-center mb-4">
            <Activity className="w-6 h-6" />
          </div>
          <h1 className="text-2xl font-bold text-[var(--foreground)]">Create an account</h1>
          <p className="text-gray-500 text-sm mt-1">Join MindBridge today</p>
        </div>

        <form onSubmit={handleRegister} className="flex flex-col gap-5">
          
          <div className="grid grid-cols-2 gap-4 mb-2">
            <button
              type="button"
              onClick={() => setRole("patient")}
              className={`flex flex-col items-center justify-center p-4 border rounded-xl transition-all ${
                role === "patient" 
                  ? "border-[var(--primary)] bg-teal-50 text-[var(--primary)]" 
                  : "border-gray-200 text-gray-500 hover:border-gray-300 hover:bg-gray-50"
              }`}
            >
              <UserIcon className="w-6 h-6 mb-2" />
              <span className="font-medium text-sm">I am a Patient</span>
            </button>
            <button
              type="button"
              onClick={() => setRole("therapist")}
              className={`flex flex-col items-center justify-center p-4 border rounded-xl transition-all ${
                role === "therapist" 
                  ? "border-[var(--primary)] bg-teal-50 text-[var(--primary)]" 
                  : "border-gray-200 text-gray-500 hover:border-gray-300 hover:bg-gray-50"
              }`}
            >
              <Stethoscope className="w-6 h-6 mb-2" />
              <span className="font-medium text-sm">I am a Therapist</span>
            </button>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Full Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--primary)] focus:bg-white transition-all text-[var(--foreground)]"
              placeholder="Aanya Sharma"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Email address</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--primary)] focus:bg-white transition-all text-[var(--foreground)]"
              placeholder="name@example.com"
            />
          </div>

          <div className="grid sm:grid-cols-2 gap-5">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--primary)] focus:bg-white transition-all text-[var(--foreground)]"
                placeholder="••••••••"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Confirm Password</label>
              <input
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--primary)] focus:bg-white transition-all text-[var(--foreground)]"
                placeholder="••••••••"
              />
            </div>
          </div>

          {error && (
            <p className="text-red-500 text-sm bg-red-50 p-3 rounded-lg border border-red-100">{error}</p>
          )}

          <button
            type="submit"
            className="w-full py-3 bg-[var(--primary)] text-white font-semibold rounded-xl hover:bg-teal-700 transition-colors mt-2"
          >
            Create Account
          </button>
        </form>

        <p className="text-center text-sm text-gray-500 mt-8">
          Already have an account?{" "}
          <Link to="/login" className="text-[var(--primary)] font-semibold hover:underline">
            Login
          </Link>
        </p>
      </div>
    </div>
  );
}
