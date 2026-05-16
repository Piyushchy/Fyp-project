import { useUser } from "../context/UserContext";

export default function TherapistDashboard() {
  const { currentUser, logout } = useUser();

  return (
    <div className="flex-1 flex flex-col items-center justify-center bg-[var(--background)]">
      <div className="text-center">
        <h1 className="text-4xl font-bold mb-4">Therapist Dashboard</h1>
        <p className="mb-4">Welcome, {currentUser?.name || "Therapist"}</p>
        <button onClick={logout} className="px-4 py-2 bg-red-100 text-red-600 rounded-xl">Logout</button>
      </div>
    </div>
  );
}
