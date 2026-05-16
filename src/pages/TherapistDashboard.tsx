import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useUser } from "../context/UserContext";
import {
  Activity, LayoutDashboard, Calendar, Users, BarChart3,
  Settings, LogOut, Search, X, Check, Clock, ChevronRight,
  UserCircle2
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Patient {
  id: string;
  name: string;
  initials: string;
  color: string;
  age: number;
  concern: string;
  lastSession: string;
  emotion: string;
  emotionKey: string;
  sessions: { date: string; duration: string; emotion: string }[];
}

// ─── Mock Data ────────────────────────────────────────────────────────────────

const MOCK_PATIENTS: Patient[] = [
  {
    id: "1", name: "Aanya Sharma", initials: "AS", color: "bg-emerald-100 text-emerald-700",
    age: 26, concern: "Anxiety", lastSession: "5 May", emotion: "😊 Happy", emotionKey: "happy",
    sessions: [
      { date: "5 May", duration: "45 min", emotion: "😊 Happy" },
      { date: "28 Apr", duration: "50 min", emotion: "😟 Sad" },
    ],
  },
  {
    id: "2", name: "Rahul Iyer", initials: "RI", color: "bg-blue-100 text-blue-700",
    age: 31, concern: "Depression", lastSession: "3 May", emotion: "😟 Sad", emotionKey: "sad",
    sessions: [
      { date: "3 May", duration: "60 min", emotion: "😟 Sad" },
      { date: "20 Apr", duration: "45 min", emotion: "😐 Neutral" },
    ],
  },
  {
    id: "3", name: "Meera Joshi", initials: "MJ", color: "bg-red-100 text-red-700",
    age: 24, concern: "Anger Management", lastSession: "28 Apr", emotion: "😠 Angry", emotionKey: "angry",
    sessions: [
      { date: "28 Apr", duration: "40 min", emotion: "😠 Angry" },
    ],
  },
  {
    id: "4", name: "Priya Kapoor", initials: "PK", color: "bg-purple-100 text-purple-700",
    age: 29, concern: "Trauma & Fear", lastSession: "20 Apr", emotion: "😨 Fear", emotionKey: "fear",
    sessions: [
      { date: "20 Apr", duration: "55 min", emotion: "😨 Fear" },
      { date: "10 Apr", duration: "45 min", emotion: "😟 Sad" },
    ],
  },
];

const EMOTION_BADGE: Record<string, string> = {
  happy: "bg-emerald-50 text-emerald-700 border-emerald-100",
  sad:   "bg-blue-50 text-blue-700 border-blue-100",
  angry: "bg-red-50 text-red-700 border-red-100",
  fear:  "bg-purple-50 text-purple-700 border-purple-100",
};

const PENDING_REQUESTS = [
  { id: "r1", name: "Aanya Sharma", initials: "AS", concern: "Anxiety",    date: "18 May, 3 PM"  },
  { id: "r2", name: "Rahul Iyer",   initials: "RI", concern: "Depression", date: "20 May, 5 PM"  },
];

const UPCOMING_SESSIONS = [
  { name: "Aanya Sharma", initials: "AS", time: "17 May, 3 PM"  },
  { name: "Meera Joshi",  initials: "MJ", time: "19 May, 10 AM" },
];

const STAT_CARDS = [
  { label: "Total Patients",        value: "12",     icon: Users,     color: "text-teal-600 bg-teal-50"   },
  { label: "Sessions This Month",   value: "8",      icon: Calendar,  color: "text-blue-600 bg-blue-50"   },
  { label: "Avg Session Duration",  value: "47 min", icon: Clock,     color: "text-amber-600 bg-amber-50" },
  { label: "Most Flagged Emotion",  value: "😟 Sad", icon: BarChart3, color: "text-purple-600 bg-purple-50" },
];

// ─── Component ────────────────────────────────────────────────────────────────

export default function TherapistDashboard() {
  const { currentUser, logout } = useUser();
  const navigate = useNavigate();

  const [searchQuery,      setSearchQuery]      = useState("");
  const [selectedPatient,  setSelectedPatient]  = useState<Patient | null>(null);
  const [notes,            setNotes]            = useState("");
  const [notesSaved,       setNotesSaved]       = useState(false);
  const [dismissed,        setDismissed]        = useState<string[]>([]);

  const name = currentUser?.name || "Dr. Rohan Mehta";

  const handleLogout = () => { logout(); navigate("/login"); };

  const handleSaveNotes = () => {
    setNotesSaved(true);
    setTimeout(() => setNotesSaved(false), 2000);
  };

  const filteredPatients = MOCK_PATIENTS.filter(p =>
    p.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const pendingVisible = PENDING_REQUESTS.filter(r => !dismissed.includes(r.id));

  return (
    <div className="flex h-screen bg-gray-50 font-sans overflow-hidden">

      {/* ─── Sidebar ─── */}
      <aside className="w-[220px] bg-[var(--primary)] text-teal-50 flex-col justify-between hidden md:flex shrink-0 h-full">
        <div>
          <div className="flex items-center gap-2 p-6 font-bold text-2xl text-white mb-6">
            <Activity className="w-8 h-8" /><span>MindBridge</span>
          </div>
          <nav className="flex flex-col gap-1 px-4">
            {[
              { icon: LayoutDashboard, label: "Dashboard",    active: true },
              { icon: Calendar,        label: "Appointments" },
              { icon: Users,           label: "My Patients"  },
              { icon: BarChart3,       label: "Analytics"    },
              { icon: Settings,        label: "Settings"     },
            ].map(({ icon: Icon, label, active }) => (
              <button
                key={label}
                className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-colors text-sm font-medium w-full text-left
                  ${active ? "bg-teal-700/50 text-white" : "hover:bg-teal-700/30"}`}
              >
                <Icon className="w-5 h-5" /> {label}
              </button>
            ))}
          </nav>
        </div>

        <div className="p-4 mb-4">
          <div className="flex items-center gap-3 px-3 py-3 bg-teal-800/40 rounded-xl">
            <div className="w-8 h-8 rounded-full bg-teal-100 text-teal-800 flex items-center justify-center font-bold text-sm shrink-0">
              {name.replace("Dr. ", "").charAt(0)}
            </div>
            <p className="text-sm font-medium text-white truncate flex-1">{name}</p>
            <button onClick={handleLogout} className="text-teal-200 hover:text-white" title="Logout">
              <LogOut className="w-5 h-5" />
            </button>
          </div>
        </div>
      </aside>

      {/* ─── Main Content ─── */}
      <main className="flex-1 flex flex-col overflow-y-auto overflow-x-hidden">

        {/* Mobile Header */}
        <div className="md:hidden flex items-center justify-between p-4 bg-[var(--primary)] text-white">
          <div className="flex items-center gap-2 font-bold text-xl">
            <Activity className="w-6 h-6" /><span>MindBridge</span>
          </div>
          <button onClick={handleLogout}><LogOut className="w-5 h-5" /></button>
        </div>

        <div className="max-w-6xl mx-auto w-full p-6 md:p-10 space-y-10">

          {/* Greeting */}
          <header>
            <h1 className="text-3xl font-bold text-[var(--foreground)] mb-1">
              Welcome back, {name.split(" ").slice(-1)[0]} 👋
            </h1>
            <p className="text-gray-500">Here's an overview of your practice today.</p>
          </header>

          {/* Stat Cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {STAT_CARDS.map(({ label, value, icon: Icon, color }) => (
              <div key={label} className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm">
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center mb-4 ${color}`}>
                  <Icon className="w-5 h-5" />
                </div>
                <p className="text-2xl font-bold text-[var(--foreground)] mb-1">{value}</p>
                <p className="text-sm text-gray-500">{label}</p>
              </div>
            ))}
          </div>

          {/* Section 1: Pending Requests */}
          <section>
            <h2 className="text-xl font-bold text-[var(--foreground)] mb-4">
              Appointment Requests{" "}
              <span className="text-sm font-normal text-gray-400">({pendingVisible.length})</span>
            </h2>

            {pendingVisible.length === 0 ? (
              <div className="bg-white border border-gray-200 rounded-2xl p-8 text-center text-gray-400 shadow-sm">
                No pending appointment requests.
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {pendingVisible.map(req => (
                  <div key={req.id} className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm">
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-10 h-10 rounded-full bg-teal-100 text-teal-700 flex items-center justify-center font-bold shrink-0">
                        {req.initials}
                      </div>
                      <div>
                        <p className="font-bold text-[var(--foreground)]">{req.name}</p>
                        <p className="text-sm text-gray-500">Primary concern: {req.concern}</p>
                      </div>
                    </div>
                    <p className="text-sm text-gray-600 mb-5 flex items-center gap-2">
                      <Calendar className="w-4 h-4 text-gray-400" /> Requested: {req.date}
                    </p>
                    <div className="flex flex-wrap gap-2">
                      <button
                        onClick={() => setDismissed(d => [...d, req.id])}
                        className="flex items-center gap-1 px-4 py-2 bg-[var(--primary)] text-white text-sm font-medium rounded-xl hover:bg-teal-700 transition-colors"
                      >
                        <Check className="w-4 h-4" /> Confirm
                      </button>
                      <button className="px-4 py-2 border-2 border-[var(--primary)] text-[var(--primary)] text-sm font-medium rounded-xl hover:bg-teal-50 transition-colors">
                        Propose New Time
                      </button>
                      <button
                        onClick={() => setDismissed(d => [...d, req.id])}
                        className="px-4 py-2 text-red-500 text-sm font-medium rounded-xl hover:bg-red-50 transition-colors"
                      >
                        Decline
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>

          {/* Section 2: Upcoming Confirmed Sessions */}
          <section>
            <h2 className="text-xl font-bold text-[var(--foreground)] mb-4">Upcoming Confirmed Sessions</h2>
            <div className="bg-white border border-gray-200 rounded-2xl shadow-sm overflow-hidden">
              {UPCOMING_SESSIONS.map((s, i) => (
                <div
                  key={s.name}
                  className={`flex items-center justify-between px-6 py-4 hover:bg-gray-50 transition-colors ${i < UPCOMING_SESSIONS.length - 1 ? "border-b border-gray-100" : ""}`}
                >
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-full bg-teal-100 text-teal-700 flex items-center justify-center font-bold text-sm shrink-0">
                      {s.initials}
                    </div>
                    <div>
                      <p className="font-semibold text-[var(--foreground)]">{s.name}</p>
                      <p className="text-sm text-gray-500 flex items-center gap-1">
                        <Clock className="w-3 h-3" /> {s.time}
                      </p>
                    </div>
                  </div>
                  <button onClick={() => navigate("/therapist/session")} className="px-5 py-2 bg-[var(--primary)] text-white text-sm font-medium rounded-xl hover:bg-teal-700 transition-colors">
                    Start Session
                  </button>
                </div>
              ))}
            </div>
          </section>

          {/* Section 3: Patient List */}
          <section className="pb-10">
            <h2 className="text-xl font-bold text-[var(--foreground)] mb-4">Patient List</h2>

            <div className="mb-4 relative">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search patients by name..."
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                className="w-full pl-11 pr-4 py-3 bg-white border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--primary)] text-sm"
              />
            </div>

            <div className="bg-white border border-gray-200 rounded-2xl shadow-sm overflow-hidden">
              {filteredPatients.length === 0 ? (
                <div className="px-6 py-8 text-center text-gray-400 text-sm">
                  No patients match "{searchQuery}"
                </div>
              ) : (
                filteredPatients.map((patient, i) => (
                  <div
                    key={patient.id}
                    className={`flex items-center justify-between px-6 py-4 hover:bg-gray-50 transition-colors ${i < filteredPatients.length - 1 ? "border-b border-gray-100" : ""}`}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold shrink-0 ${patient.color}`}>
                        {patient.initials}
                      </div>
                      <div>
                        <p className="font-semibold text-[var(--foreground)]">{patient.name}</p>
                        <p className="text-sm text-gray-500">Last session: {patient.lastSession}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className={`hidden sm:inline-flex px-2.5 py-1 text-xs font-semibold rounded-full border ${EMOTION_BADGE[patient.emotionKey]}`}>
                        {patient.emotion}
                      </span>
                      <button
                        onClick={() => navigate(`/therapist/patient/${patient.id}`)}
                        className="text-[var(--primary)] text-sm font-medium flex items-center gap-1 hover:underline"
                      >
                        View Analytics <ChevronRight className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>

        </div>
      </main>

    </div>
  );
}
