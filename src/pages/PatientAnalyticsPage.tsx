import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  LineChart, Line, BarChart, Bar, RadarChart, Radar,
  PolarGrid, PolarAngleAxis, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from "recharts";
import { ArrowLeft, AlertTriangle, TrendingUp, Brain, Activity } from "lucide-react";

// ── Mock per-patient analytics data ──────────────────────────────────────────

const PATIENT_DATA: Record<string, {
  name: string; age: number; concern: string; initials: string; color: string;
  overallEmotion: string; criticality: number; criticalityLabel: string;
  symptoms: string[]; summary: string;
  timeline: { time: string; Happy: number; Calm: number; Sad: number; Anxious: number; Angry: number }[];
  sessions: { session: string; happy: number; calm: number; sad: number; anxious: number }[];
  radar: { emotion: string; score: number }[];
}> = {
  "1": {
    name: "Aanya Sharma", age: 26, concern: "Anxiety", initials: "AS", color: "bg-emerald-100 text-emerald-700",
    overallEmotion: "😟 Predominantly Anxious", criticality: 62, criticalityLabel: "Moderate",
    symptoms: ["Generalized Anxiety Disorder (likely)", "Mild Depressive Episodes", "Sleep Disturbance", "Rumination"],
    summary: "Aanya shows consistent anxiety spikes in the first half of sessions, gradually calming. Sadness clusters around mid-session. Positive progress noted over last 3 sessions.",
    timeline: [
      { time: "3:00", Happy: 60, Calm: 30, Sad: 5,  Anxious: 5,  Angry: 0 },
      { time: "3:05", Happy: 40, Calm: 20, Sad: 15, Anxious: 25, Angry: 0 },
      { time: "3:10", Happy: 25, Calm: 15, Sad: 30, Anxious: 30, Angry: 0 },
      { time: "3:15", Happy: 20, Calm: 10, Sad: 35, Anxious: 35, Angry: 0 },
      { time: "3:20", Happy: 30, Calm: 25, Sad: 25, Anxious: 20, Angry: 0 },
      { time: "3:25", Happy: 50, Calm: 35, Sad: 10, Anxious: 5,  Angry: 0 },
      { time: "3:30", Happy: 65, Calm: 25, Sad: 5,  Anxious: 5,  Angry: 0 },
    ],
    sessions: [
      { session: "5 May",  happy: 55, calm: 25, sad: 15, anxious: 5  },
      { session: "28 Apr", happy: 30, calm: 20, sad: 30, anxious: 20 },
      { session: "14 Apr", happy: 20, calm: 15, sad: 35, anxious: 30 },
    ],
    radar: [
      { emotion: "Happy",   score: 55 },
      { emotion: "Calm",    score: 30 },
      { emotion: "Sad",     score: 45 },
      { emotion: "Anxious", score: 65 },
      { emotion: "Angry",   score: 10 },
    ],
  },
  "2": {
    name: "Rahul Iyer", age: 31, concern: "Depression", initials: "RI", color: "bg-blue-100 text-blue-700",
    overallEmotion: "😔 Predominantly Sad", criticality: 78, criticalityLabel: "High",
    symptoms: ["Major Depressive Disorder (possible)", "Anhedonia", "Low Energy", "Social Withdrawal"],
    summary: "Rahul displays persistently low positive affect. Sadness dominates sessions with minimal fluctuation. Requires close monitoring and possible medication review discussion.",
    timeline: [
      { time: "3:00", Happy: 20, Calm: 30, Sad: 35, Anxious: 15, Angry: 0 },
      { time: "3:05", Happy: 10, Calm: 20, Sad: 50, Anxious: 20, Angry: 0 },
      { time: "3:10", Happy: 15, Calm: 15, Sad: 55, Anxious: 15, Angry: 0 },
      { time: "3:15", Happy: 10, Calm: 10, Sad: 60, Anxious: 20, Angry: 0 },
      { time: "3:20", Happy: 20, Calm: 20, Sad: 45, Anxious: 15, Angry: 0 },
      { time: "3:25", Happy: 25, Calm: 30, Sad: 35, Anxious: 10, Angry: 0 },
    ],
    sessions: [
      { session: "3 May",  happy: 15, calm: 20, sad: 50, anxious: 15 },
      { session: "20 Apr", happy: 10, calm: 15, sad: 55, anxious: 20 },
      { session: "6 Apr",  happy: 5,  calm: 10, sad: 65, anxious: 20 },
    ],
    radar: [
      { emotion: "Happy",   score: 15 },
      { emotion: "Calm",    score: 20 },
      { emotion: "Sad",     score: 70 },
      { emotion: "Anxious", score: 35 },
      { emotion: "Angry",   score: 20 },
    ],
  },
  "3": {
    name: "Meera Joshi", age: 24, concern: "Anger Management", initials: "MJ", color: "bg-red-100 text-red-700",
    overallEmotion: "😠 Predominantly Angry", criticality: 55, criticalityLabel: "Moderate",
    symptoms: ["Intermittent Explosive Disorder (mild)", "Frustration Intolerance", "Stress Reactivity"],
    summary: "Meera shows anger spikes particularly in the first 15 minutes of sessions. Sessions consistently trend toward calm resolution. Good engagement with coping techniques.",
    timeline: [
      { time: "3:00", Happy: 10, Calm: 10, Sad: 5,  Anxious: 20, Angry: 55 },
      { time: "3:05", Happy: 10, Calm: 5,  Sad: 10, Anxious: 25, Angry: 50 },
      { time: "3:10", Happy: 15, Calm: 15, Sad: 10, Anxious: 20, Angry: 40 },
      { time: "3:15", Happy: 20, Calm: 30, Sad: 10, Anxious: 15, Angry: 25 },
      { time: "3:20", Happy: 35, Calm: 40, Sad: 5,  Anxious: 10, Angry: 10 },
      { time: "3:25", Happy: 50, Calm: 35, Sad: 5,  Anxious: 5,  Angry: 5  },
    ],
    sessions: [
      { session: "28 Apr", happy: 25, calm: 30, sad: 10, anxious: 15 },
      { session: "14 Apr", happy: 10, calm: 15, sad: 15, anxious: 20 },
    ],
    radar: [
      { emotion: "Happy",   score: 25 },
      { emotion: "Calm",    score: 30 },
      { emotion: "Sad",     score: 15 },
      { emotion: "Anxious", score: 30 },
      { emotion: "Angry",   score: 70 },
    ],
  },
  "4": {
    name: "Priya Kapoor", age: 29, concern: "Trauma & Fear", initials: "PK", color: "bg-purple-100 text-purple-700",
    overallEmotion: "😨 Predominantly Fearful", criticality: 85, criticalityLabel: "Critical",
    symptoms: ["PTSD (possible)", "Panic Episodes", "Hypervigilance", "Avoidance Behaviour"],
    summary: "Priya exhibits strong fear responses with trauma-linked triggers. Criticality is high — consider trauma-focused CBT. Sessions show minimal improvement over 3 months.",
    timeline: [
      { time: "3:00", Happy: 10, Calm: 15, Sad: 20, Anxious: 20, Angry: 35 },
      { time: "3:05", Happy: 5,  Calm: 5,  Sad: 25, Anxious: 30, Angry: 35 },
      { time: "3:10", Happy: 5,  Calm: 5,  Sad: 30, Anxious: 35, Angry: 25 },
      { time: "3:15", Happy: 10, Calm: 10, Sad: 25, Anxious: 35, Angry: 20 },
      { time: "3:20", Happy: 15, Calm: 20, Sad: 20, Anxious: 30, Angry: 15 },
      { time: "3:25", Happy: 25, Calm: 30, Sad: 15, Anxious: 20, Angry: 10 },
    ],
    sessions: [
      { session: "20 Apr", happy: 10, calm: 15, sad: 25, anxious: 50 },
      { session: "10 Apr", happy: 5,  calm: 10, sad: 30, anxious: 55 },
      { session: "27 Mar", happy: 5,  calm: 5,  sad: 35, anxious: 55 },
    ],
    radar: [
      { emotion: "Happy",   score: 10 },
      { emotion: "Calm",    score: 15 },
      { emotion: "Sad",     score: 40 },
      { emotion: "Anxious", score: 80 },
      { emotion: "Angry",   score: 45 },
    ],
  },
};

const CRITICALITY_COLOR = {
  Low:      { bar: "bg-emerald-400", text: "text-emerald-700", bg: "bg-emerald-50 border-emerald-200" },
  Moderate: { bar: "bg-amber-400",   text: "text-amber-700",   bg: "bg-amber-50 border-amber-200"   },
  High:     { bar: "bg-orange-500",  text: "text-orange-700",  bg: "bg-orange-50 border-orange-200" },
  Critical: { bar: "bg-red-500",     text: "text-red-700",     bg: "bg-red-50 border-red-200"       },
};

const LINE_COLORS = { Happy: "#10b981", Calm: "#0d9488", Sad: "#60a5fa", Anxious: "#f59e0b", Angry: "#ef4444" };

// ─────────────────────────────────────────────────────────────────────────────

export default function PatientAnalyticsPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const patient = PATIENT_DATA[id ?? "1"] ?? PATIENT_DATA["1"];
  const [activeSession, setActiveSession] = useState(0);
  const crit = CRITICALITY_COLOR[patient.criticalityLabel as keyof typeof CRITICALITY_COLOR];

  return (
    <div className="min-h-screen bg-gray-50 font-sans">
      {/* Header */}
      <div className="bg-[var(--primary)] text-white px-6 py-4 flex items-center gap-4">
        <button onClick={() => navigate("/therapist/dashboard")}
          className="p-2 rounded-xl hover:bg-teal-700 transition-colors">
          <ArrowLeft className="w-5 h-5" />
        </button>
        <Activity className="w-6 h-6" />
        <div>
          <h1 className="font-bold text-lg">Patient Analytics — {patient.name}</h1>
          <p className="text-teal-100 text-sm">Age {patient.age} · {patient.concern}</p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto p-6 space-y-8">

        {/* Row 1: Summary + Criticality + Overall */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

          {/* Overall Emotion */}
          <div className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm flex flex-col gap-3">
            <p className="text-xs text-gray-400 uppercase tracking-wide font-medium">Overall Emotion</p>
            <p className="text-2xl font-bold text-[var(--foreground)]">{patient.overallEmotion}</p>
            <p className="text-sm text-gray-500 leading-relaxed">{patient.summary}</p>
          </div>

          {/* Criticality Score */}
          <div className={`border rounded-2xl p-5 shadow-sm ${crit.bg}`}>
            <p className="text-xs uppercase tracking-wide font-medium text-gray-500 mb-3">Criticality Score</p>
            <div className="flex items-end gap-2 mb-3">
              <span className={`text-4xl font-extrabold ${crit.text}`}>{patient.criticality}</span>
              <span className="text-gray-400 text-sm mb-1">/100</span>
            </div>
            <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden mb-2">
              <div className={`h-full rounded-full transition-all ${crit.bar}`} style={{ width: `${patient.criticality}%` }} />
            </div>
            <div className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold border ${crit.bg} ${crit.text}`}>
              <AlertTriangle className="w-3.5 h-3.5" />
              {patient.criticalityLabel} Risk
            </div>
          </div>

          {/* Predicted Symptoms */}
          <div className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm">
            <p className="text-xs text-gray-400 uppercase tracking-wide font-medium mb-3 flex items-center gap-1.5">
              <Brain className="w-3.5 h-3.5" /> Predicted Symptoms
            </p>
            <ul className="space-y-2">
              {patient.symptoms.map((s, i) => (
                <li key={i} className="flex items-start gap-2 text-sm">
                  <span className="mt-1 w-2 h-2 rounded-full bg-[var(--primary)] shrink-0" />
                  <span className="text-gray-700">{s}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Row 2: Emotion Timeline (Line Chart) */}
        <div className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm">
          <div className="flex items-center gap-2 mb-5">
            <TrendingUp className="w-5 h-5 text-[var(--primary)]" />
            <h2 className="text-lg font-bold text-[var(--foreground)]">Emotion Timeline — Last Session</h2>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={patient.timeline} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="time" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} unit="%" />
              <Tooltip />
              <Legend />
              {Object.entries(LINE_COLORS).map(([key, color]) => (
                <Line key={key} type="monotone" dataKey={key} stroke={color} strokeWidth={2}
                  dot={false} activeDot={{ r: 5 }} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Row 3: Radar + Session Bar side by side */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

          {/* Radar Chart — Emotion Distribution */}
          <div className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm">
            <h2 className="text-lg font-bold text-[var(--foreground)] mb-5">Emotion Distribution</h2>
            <ResponsiveContainer width="100%" height={260}>
              <RadarChart data={patient.radar}>
                <PolarGrid stroke="#e5e7eb" />
                <PolarAngleAxis dataKey="emotion" tick={{ fontSize: 12 }} />
                <Radar name={patient.name} dataKey="score" stroke="#0d9488" fill="#0d9488" fillOpacity={0.25} />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          {/* Bar Chart — Session Comparison */}
          <div className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm">
            <h2 className="text-lg font-bold text-[var(--foreground)] mb-1">Session Comparison</h2>
            <div className="flex gap-2 mb-4 flex-wrap">
              {patient.sessions.map((s, i) => (
                <button key={i} onClick={() => setActiveSession(i)}
                  className={`px-3 py-1 rounded-full text-xs font-medium border transition-all ${
                    activeSession === i
                      ? "bg-[var(--primary)] text-white border-[var(--primary)]"
                      : "border-gray-200 text-gray-600 hover:border-gray-300"
                  }`}>
                  {s.session}
                </button>
              ))}
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={[patient.sessions[activeSession]]}
                margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="session" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} unit="%" />
                <Tooltip />
                <Legend />
                <Bar dataKey="happy"   fill="#10b981" radius={[4,4,0,0]} name="Happy" />
                <Bar dataKey="calm"    fill="#0d9488" radius={[4,4,0,0]} name="Calm" />
                <Bar dataKey="sad"     fill="#60a5fa" radius={[4,4,0,0]} name="Sad" />
                <Bar dataKey="anxious" fill="#f59e0b" radius={[4,4,0,0]} name="Anxious" />
              </BarChart>
            </ResponsiveContainer>
          </div>

        </div>

        {/* Row 4: Start Session CTA */}
        <div className="bg-[var(--primary)] rounded-2xl p-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div>
            <p className="text-white font-bold text-lg">Ready to start a session?</p>
            <p className="text-teal-100 text-sm">Continue working with {patient.name} based on these insights.</p>
          </div>
          <button onClick={() => navigate("/therapist/session")}
            className="px-6 py-3 bg-white text-[var(--primary)] font-bold rounded-xl hover:bg-teal-50 transition-colors whitespace-nowrap">
            Start Session →
          </button>
        </div>

      </div>
    </div>
  );
}
