import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { UserProvider } from "./context/UserContext";

// Pages
import LandingPage from "./pages/LandingPage";
import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";
import IntakeFormPage from "./pages/IntakeFormPage";
import TherapistMatchPage from "./pages/TherapistMatchPage";
import PatientDashboard from "./pages/PatientDashboard";
import TherapistDashboard from "./pages/TherapistDashboard";
import SessionPage from "./pages/SessionPage";

function App() {
  return (
    <UserProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          
          {/* Patient Routes */}
          <Route path="/patient/intake" element={<IntakeFormPage />} />
          <Route path="/patient/match" element={<TherapistMatchPage />} />
          <Route path="/patient/dashboard" element={<PatientDashboard />} />
          <Route path="/patient/session" element={<SessionPage />} />
          
          {/* Therapist Routes */}
          <Route path="/therapist/dashboard" element={<TherapistDashboard />} />
          <Route path="/therapist/session" element={<SessionPage />} />
          
          {/* Fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </UserProvider>
  );
}

export default App;
