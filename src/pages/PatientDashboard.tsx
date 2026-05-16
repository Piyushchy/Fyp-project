import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useUser } from "../context/UserContext";
import {
  Activity,
  LayoutDashboard,
  Video,
  User as UserIcon,
  Smile,
  Settings,
  LogOut,
  Calendar,
  Clock
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const PAST_SESSIONS = [
  { date: "5 May", therapist: "Dr. Priya Nair", duration: "45 min", emotion: "😊 Happy", action: "Leave Feedback" },
  { date: "28 Apr", therapist: "Dr. Priya Nair", duration: "50 min", emotion: "😟 Sad", action: "View Summary" },
];

const MOOD_DATA = [
  { day: "Mon", score: 4 }, // Green
  { day: "Tue", score: 5 }, // Green
  { day: "Wed", score: 3 }, // Amber
  { day: "Thu", score: 2 }, // Red
  { day: "Fri", score: 4 }, // Green
  { day: "Sat", score: 5 }, // Green
  { day: "Sun", score: 4 }, // Green
];

export default function PatientDashboard() {
  const { currentUser, logout } = useUser();
  const navigate = useNavigate();
  const [moodLogged, setMoodLogged] = useState(false);
  const [selectedMood, setSelectedMood] = useState<number | null>(null);

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  const handleMoodSelect = (index: number) => {
    setSelectedMood(index);
    setMoodLogged(true);
  };

  const name = currentUser?.name || "Aanya";

  const getBarColor = (score: number) => {
    if (score >= 4) return "bg-emerald-400";
    if (score === 3) return "bg-amber-400";
    return "bg-rose-400";
  };

  return (
    <div className="flex h-screen bg-gray-50 font-sans overflow-hidden">
      
      {/* Sidebar */}
      <aside className="w-[220px] bg-[var(--primary)] text-teal-50 flex flex-col justify-between hidden md:flex shrink-0 h-full">
        <div>
          <div className="flex items-center gap-2 p-6 font-bold text-2xl text-white mb-6">
            <Activity className="w-8 h-8" />
            <span>MindBridge</span>
          </div>

          <nav className="flex flex-col gap-2 px-4">
            <button onClick={() => navigate("/patient/dashboard")} className="flex items-center gap-3 px-4 py-3 bg-teal-700/50 text-white rounded-xl font-medium transition-colors">
              <LayoutDashboard className="w-5 h-5" /> Dashboard
            </button>
            <button onClick={() => navigate("/patient/session")} className="flex items-center gap-3 px-4 py-3 hover:bg-teal-700/30 rounded-xl transition-colors">
              <Video className="w-5 h-5" /> My Sessions
            </button>
            <button onClick={() => navigate("/patient/match")} className="flex items-center gap-3 px-4 py-3 hover:bg-teal-700/30 rounded-xl transition-colors">
              <UserIcon className="w-5 h-5" /> My Therapist
            </button>
            <button className="flex items-center gap-3 px-4 py-3 hover:bg-teal-700/30 rounded-xl transition-colors">
              <Smile className="w-5 h-5" /> Mood Log
            </button>
            <button className="flex items-center gap-3 px-4 py-3 hover:bg-teal-700/30 rounded-xl transition-colors">
              <Settings className="w-5 h-5" /> Settings
            </button>
          </nav>
        </div>

        <div className="p-4 mb-4">
          <div className="flex items-center gap-3 px-4 py-3 bg-teal-800/40 rounded-xl">
            <div className="w-8 h-8 rounded-full bg-teal-100 text-teal-800 flex items-center justify-center font-bold text-sm">
              {name.charAt(0)}
            </div>
            <div className="flex-1 overflow-hidden">
              <p className="text-sm font-medium text-white truncate">{name}</p>
            </div>
            <button onClick={handleLogout} className="text-teal-200 hover:text-white" title="Logout">
              <LogOut className="w-5 h-5" />
            </button>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-y-auto overflow-x-hidden">
        {/* Mobile Header */}
        <div className="md:hidden flex items-center justify-between p-4 bg-[var(--primary)] text-white">
          <div className="flex items-center gap-2 font-bold text-xl">
            <Activity className="w-6 h-6" />
            <span>MindBridge</span>
          </div>
          <button onClick={handleLogout} className="p-2">
            <LogOut className="w-5 h-5" />
          </button>
        </div>

        <div className="max-w-5xl mx-auto w-full p-6 md:p-10">
          
          {/* Greeting */}
          <header className="mb-10">
            <h1 className="text-3xl font-bold text-[var(--foreground)] mb-2">Good morning, {name.split(' ')[0]} 👋</h1>
            <p className="text-gray-600 text-lg">Your next session is in 2 days.</p>
          </header>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            {/* Left Column (Wider) */}
            <div className="lg:col-span-2 space-y-8">
              
              {/* Section 1: Upcoming Sessions */}
              <section>
                <h2 className="text-xl font-bold text-[var(--foreground)] mb-4">Upcoming Sessions</h2>
                <div className="flex overflow-x-auto pb-4 gap-4 snap-x hide-scrollbar">
                  
                  {/* Card 1 */}
                  <div className="min-w-[300px] sm:min-w-[320px] bg-white border border-gray-200 rounded-2xl p-5 snap-start shadow-sm flex flex-col justify-between">
                    <div>
                      <div className="flex items-start justify-between mb-4">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center font-bold">PN</div>
                          <h3 className="font-bold text-[var(--foreground)]">Dr. Priya Nair</h3>
                        </div>
                        <span className="px-2 py-1 bg-emerald-50 text-emerald-700 text-xs font-semibold rounded-md border border-emerald-100">Confirmed</span>
                      </div>
                      <div className="text-gray-600 text-sm space-y-2 mb-6">
                        <p className="flex items-center gap-2"><Calendar className="w-4 h-4" /> Tomorrow</p>
                        <p className="flex items-center gap-2"><Clock className="w-4 h-4" /> 3:00 PM (45 min)</p>
                      </div>
                    </div>
                    <button onClick={() => navigate("/patient/session")} className="w-full py-2.5 bg-[var(--primary)] text-white font-medium rounded-xl hover:bg-teal-700 transition-colors">
                      Join Session
                    </button>
                  </div>

                  {/* Card 2 */}
                  <div className="min-w-[300px] sm:min-w-[320px] bg-white border border-gray-200 rounded-2xl p-5 snap-start shadow-sm flex flex-col justify-between">
                    <div>
                      <div className="flex items-start justify-between mb-4">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 rounded-full bg-blue-100 text-blue-700 flex items-center justify-center font-bold">AD</div>
                          <h3 className="font-bold text-[var(--foreground)]">Dr. Arjun Desai</h3>
                        </div>
                        <span className="px-2 py-1 bg-amber-50 text-amber-700 text-xs font-semibold rounded-md border border-amber-100">Pending</span>
                      </div>
                      <div className="text-gray-600 text-sm space-y-2 mb-6">
                        <p className="flex items-center gap-2"><Calendar className="w-4 h-4" /> 20 May</p>
                        <p className="flex items-center gap-2"><Clock className="w-4 h-4" /> 5:00 PM</p>
                      </div>
                    </div>
                    <div className="w-full py-2.5 bg-gray-50 text-gray-500 font-medium rounded-xl border border-gray-200 text-center text-sm">
                      Awaiting confirmation
                    </div>
                  </div>

                </div>
              </section>

              {/* Section 3: Past Sessions */}
              <section>
                <h2 className="text-xl font-bold text-[var(--foreground)] mb-4">Past Sessions</h2>
                <div className="bg-white border border-gray-200 rounded-2xl overflow-hidden shadow-sm">
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left">
                      <thead className="bg-gray-50 text-gray-600 border-b border-gray-200">
                        <tr>
                          <th className="px-6 py-4 font-medium">Date</th>
                          <th className="px-6 py-4 font-medium">Therapist</th>
                          <th className="px-6 py-4 font-medium">Duration</th>
                          <th className="px-6 py-4 font-medium">Dominant Emotion</th>
                          <th className="px-6 py-4 font-medium text-right">Action</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-100">
                        {PAST_SESSIONS.map((session, i) => (
                          <tr key={i} className="hover:bg-gray-50 transition-colors">
                            <td className="px-6 py-4 text-gray-900 font-medium">{session.date}</td>
                            <td className="px-6 py-4 text-gray-600">{session.therapist}</td>
                            <td className="px-6 py-4 text-gray-600">{session.duration}</td>
                            <td className="px-6 py-4">{session.emotion}</td>
                            <td className="px-6 py-4 text-right">
                              <button className="text-[var(--primary)] font-medium hover:underline">
                                {session.action}
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </section>

            </div>

            {/* Right Column (Narrower) */}
            <div className="space-y-8">
              
              {/* Section 2: Daily Mood Check-in */}
              <section className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm">
                <h2 className="text-lg font-bold text-[var(--foreground)] mb-4">How are you feeling today?</h2>
                
                <div className="flex justify-between mb-4">
                  {["😊", "😌", "😐", "😟", "😔"].map((emoji, index) => (
                    <button
                      key={index}
                      onClick={() => handleMoodSelect(index)}
                      className={`text-3xl sm:text-4xl hover:scale-110 transition-transform ${
                        selectedMood === index ? "scale-125 opacity-100 drop-shadow-md" : "opacity-70 hover:opacity-100"
                      }`}
                    >
                      {emoji}
                    </button>
                  ))}
                </div>

                <AnimatePresence>
                  {moodLogged && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      className="bg-teal-50 border border-teal-100 text-teal-800 text-sm p-3 rounded-xl font-medium text-center"
                    >
                      Mood logged. Take care today!
                    </motion.div>
                  )}
                </AnimatePresence>
              </section>

              {/* Section 4: Mood This Week */}
              <section className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm">
                <h2 className="text-lg font-bold text-[var(--foreground)] mb-6">Mood This Week</h2>
                
                <div className="flex items-end justify-between h-40 gap-2">
                  {MOOD_DATA.map((data, i) => (
                    <div key={i} className="flex flex-col items-center flex-1 group">
                      <div className="w-full bg-gray-100 rounded-t-sm h-full flex items-end relative overflow-hidden">
                        <div 
                          className={`w-full rounded-t-sm transition-all duration-500 ease-out ${getBarColor(data.score)}`}
                          style={{ height: `${(data.score / 5) * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-500 mt-2 font-medium">{data.day}</span>
                    </div>
                  ))}
                </div>
              </section>

            </div>

          </div>
        </div>
      </main>
    </div>
  );
}
