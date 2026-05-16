import { useUser } from "../context/UserContext";

export default function SessionPage() {
  const { currentUser } = useUser();

  return (
    <div className="flex-1 flex flex-col items-center justify-center bg-[var(--background)]">
      <div className="text-center">
        <h1 className="text-4xl font-bold mb-4">Live Session</h1>
        <p className="mb-4 text-xl">Current Role View: <span className="font-bold text-[var(--primary)]">{currentUser?.role || "Guest"}</span></p>
      </div>
    </div>
  );
}
