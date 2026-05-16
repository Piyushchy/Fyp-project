import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { useUser } from "../context/UserContext";
import {
  Mic, MicOff, Video, VideoOff, Monitor, PhoneOff,
  Send, MessageSquare, Info, Brain, X
} from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";

// ─── Mock Data ────────────────────────────────────────────────────────────────

const MOCK_MESSAGES = [
  { from: "therapist", name: "Dr. Priya Nair", text: "Hello Aanya, how have you been this week?" },
  { from: "patient",   name: "You",            text: "A bit anxious lately, but better than last week." },
  { from: "therapist", name: "Dr. Priya Nair", text: "That's progress. Let's talk about what's been triggering it." },
];

const EMOTION_TIMELINE = [
  { time: "3:02 PM", emotion: "😊 Happy" },
  { time: "3:05 PM", emotion: "😌 Calm"  },
  { time: "3:08 PM", emotion: "😟 Sad"   },
  { time: "3:11 PM", emotion: "😟 Sad"   },
];

type TabKey = "chat" | "info" | "emotions";

// ─── Component ────────────────────────────────────────────────────────────────

export default function SessionPage() {
  const { currentUser, logout } = useUser();
  const navigate = useNavigate();
  const role = currentUser?.role ?? "patient";

  const [micOn,    setMicOn]    = useState(true);
  const [camOn,    setCamOn]    = useState(true);
  const [sharing,  setSharing]  = useState(false);
  const [tab,      setTab]      = useState<TabKey>("chat");
  const [message,  setMessage]  = useState("");
  const [messages, setMessages] = useState(MOCK_MESSAGES);
  const [showEnd,  setShowEnd]  = useState(false);

  // Duration timer
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setSeconds(s => s + 1), 1000);
    return () => clearInterval(id);
  }, []);
  const duration = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;

  const chatBottom = useRef<HTMLDivElement>(null);
  useEffect(() => { chatBottom.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  const handleSend = () => {
    if (!message.trim()) return;
    setMessages(m => [...m, { from: "patient", name: "You", text: message.trim() }]);
    setMessage("");
  };

  const handleEndConfirm = () => {
    logout();
    navigate(role === "therapist" ? "/therapist/dashboard" : "/patient/dashboard");
  };

  return (
    <div className="flex flex-col h-screen bg-[#111] font-sans overflow-hidden">

      {/* ─── Main Area ─── */}
      <div className="flex flex-1 overflow-hidden">

        {/* ─── Video Area (70%) ─── */}
        <div className="flex-[7] relative bg-[#111] flex items-center justify-center min-w-0">

          {/* Remote video placeholder */}
          <div className="flex flex-col items-center justify-center gap-4 select-none">
            <div className="w-24 h-24 rounded-full bg-emerald-900 text-emerald-300 flex items-center justify-center text-4xl font-bold">
              PN
            </div>
            <p className="text-white text-lg font-semibold">Dr. Priya Nair</p>
            <p className="text-gray-500 text-sm">Video connected</p>
          </div>

          {/* Self-preview (bottom-right) */}
          <div className="absolute bottom-4 right-4 w-40 h-24 bg-[#222] rounded-xl border border-gray-700 flex flex-col items-center justify-center gap-1">
            {camOn ? (
              <>
                <div className="w-8 h-8 rounded-full bg-teal-900 text-teal-300 flex items-center justify-center font-bold text-sm">
                  {(currentUser?.name ?? "Y").charAt(0)}
                </div>
                <p className="text-gray-400 text-xs">You</p>
              </>
            ) : (
              <p className="text-gray-600 text-xs">Camera off</p>
            )}
          </div>

          {/* Session duration badge */}
          <div className="absolute top-4 left-4 bg-black/60 text-white text-sm px-3 py-1.5 rounded-lg font-mono backdrop-blur-sm">
            {duration}
          </div>
        </div>

        {/* ─── Right Panel (30%) ─── */}
        <div className="flex-[3] min-w-[280px] max-w-[380px] bg-white flex flex-col border-l border-gray-200">

          {/* Tabs */}
          <div className="flex border-b border-gray-200 shrink-0">
            {(["chat", "info", "emotions"] as TabKey[]).map(t => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`flex-1 py-3 text-sm font-semibold capitalize transition-colors ${
                  tab === t
                    ? "text-[var(--primary)] border-b-2 border-[var(--primary)]"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                {t === "chat" && <MessageSquare className="w-4 h-4 inline mr-1" />}
                {t === "info" && <Info className="w-4 h-4 inline mr-1" />}
                {t === "emotions" && <Brain className="w-4 h-4 inline mr-1" />}
                {t}
              </button>
            ))}
          </div>

          {/* Tab: Chat */}
          {tab === "chat" && (
            <div className="flex flex-col flex-1 overflow-hidden">
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((msg, i) => (
                  <div key={i} className={`flex flex-col ${msg.from === "patient" ? "items-end" : "items-start"}`}>
                    <p className="text-xs text-gray-400 mb-1">{msg.name}</p>
                    <div className={`max-w-[85%] px-4 py-2.5 rounded-2xl text-sm ${
                      msg.from === "patient"
                        ? "bg-[var(--primary)] text-white rounded-tr-sm"
                        : "bg-gray-100 text-gray-800 rounded-tl-sm"
                    }`}>
                      {msg.text}
                    </div>
                  </div>
                ))}
                <div ref={chatBottom} />
              </div>

              <div className="p-3 border-t border-gray-100 flex gap-2 shrink-0">
                <input
                  type="text"
                  value={message}
                  onChange={e => setMessage(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && handleSend()}
                  placeholder="Type a message..."
                  className="flex-1 px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--primary)] text-sm"
                />
                <button
                  onClick={handleSend}
                  className="p-2.5 bg-[var(--primary)] text-white rounded-xl hover:bg-teal-700 transition-colors"
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}

          {/* Tab: Info */}
          {tab === "info" && (
            <div className="flex-1 overflow-y-auto p-5 space-y-5">
              <div>
                <p className="text-xs text-gray-400 uppercase tracking-wide mb-1">Patient</p>
                <p className="font-semibold text-[var(--foreground)]">Aanya Sharma</p>
              </div>
              <div>
                <p className="text-xs text-gray-400 uppercase tracking-wide mb-1">Therapist</p>
                <p className="font-semibold text-[var(--foreground)]">Dr. Priya Nair</p>
              </div>
              <div>
                <p className="text-xs text-gray-400 uppercase tracking-wide mb-1">Session Started</p>
                <p className="font-semibold text-[var(--foreground)]">3:00 PM</p>
              </div>
              <div>
                <p className="text-xs text-gray-400 uppercase tracking-wide mb-1">Live Duration</p>
                <p className="font-mono text-2xl font-bold text-[var(--primary)]">{duration}</p>
              </div>
              <div>
                <p className="text-xs text-gray-400 uppercase tracking-wide mb-1">Appointment Date</p>
                <p className="font-semibold text-[var(--foreground)]">17 May 2026</p>
              </div>
            </div>
          )}

          {/* Tab: Emotions */}
          {tab === "emotions" && (
            <div className="flex-1 overflow-y-auto p-5">
              {role === "therapist" ? (
                <div className="space-y-6">
                  {/* Current Emotion */}
                  <div className="text-center p-6 bg-blue-50 border border-blue-100 rounded-2xl">
                    <p className="text-5xl mb-3">😟</p>
                    <p className="text-xl font-bold text-[var(--foreground)]">Sad</p>
                    <p className="text-blue-600 font-semibold text-sm mt-1">78% confidence</p>
                  </div>

                  {/* Timeline */}
                  <div>
                    <h3 className="text-sm font-semibold text-gray-600 mb-3 uppercase tracking-wide">Timeline</h3>
                    <div className="space-y-2">
                      {EMOTION_TIMELINE.map((e, i) => (
                        <div key={i} className="flex items-center justify-between p-3 bg-gray-50 rounded-xl text-sm">
                          <span className="text-gray-500 font-mono">{e.time}</span>
                          <span className="font-medium">{e.emotion}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-center gap-4 text-gray-500 py-10">
                  <Brain className="w-12 h-12 text-gray-300" />
                  <p className="font-medium">Your session is being analysed</p>
                  <p className="text-sm text-gray-400">Emotion insights will be shared in your session summary.</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ─── Bottom Control Bar ─── */}
      <div className="bg-[#1a1a1a] px-6 py-4 flex items-center justify-center gap-4 shrink-0">
        {/* Mic */}
        <button
          onClick={() => setMicOn(v => !v)}
          title={micOn ? "Mute" : "Unmute"}
          className={`relative w-12 h-12 rounded-full flex items-center justify-center transition-colors ${
            micOn ? "bg-[#333] hover:bg-[#444] text-white" : "bg-red-600 hover:bg-red-700 text-white"
          }`}
        >
          {micOn ? <Mic className="w-5 h-5" /> : <MicOff className="w-5 h-5" />}
        </button>

        {/* Camera */}
        <button
          onClick={() => setCamOn(v => !v)}
          title={camOn ? "Turn off camera" : "Turn on camera"}
          className={`w-12 h-12 rounded-full flex items-center justify-center transition-colors ${
            camOn ? "bg-[#333] hover:bg-[#444] text-white" : "bg-red-600 hover:bg-red-700 text-white"
          }`}
        >
          {camOn ? <Video className="w-5 h-5" /> : <VideoOff className="w-5 h-5" />}
        </button>

        {/* Screen Share */}
        <button
          onClick={() => setSharing(v => !v)}
          title={sharing ? "Stop sharing" : "Share screen"}
          className={`w-12 h-12 rounded-full flex items-center justify-center transition-colors ${
            sharing ? "bg-[var(--primary)] text-white" : "bg-[#333] hover:bg-[#444] text-white"
          }`}
        >
          <Monitor className="w-5 h-5" />
        </button>

        {/* End Session */}
        <button
          onClick={() => setShowEnd(true)}
          title="End session"
          className="px-6 h-12 bg-red-600 hover:bg-red-700 text-white font-semibold rounded-full flex items-center gap-2 transition-colors"
        >
          <PhoneOff className="w-5 h-5" />
          <span className="hidden sm:inline">End Session</span>
        </button>
      </div>

      {/* ─── End Session Modal ─── */}
      <AnimatePresence>
        {showEnd && (
          <div className="fixed inset-0 z-50 flex items-center justify-center px-4">
            <motion.div
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="absolute inset-0 bg-black/60"
              onClick={() => setShowEnd(false)}
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }}
              className="relative bg-white rounded-2xl p-8 w-full max-w-sm text-center shadow-2xl"
            >
              <button onClick={() => setShowEnd(false)} className="absolute top-4 right-4 p-1 text-gray-400 hover:text-gray-600">
                <X className="w-5 h-5" />
              </button>
              <div className="w-14 h-14 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <PhoneOff className="w-7 h-7 text-red-600" />
              </div>
              <h2 className="text-xl font-bold text-[var(--foreground)] mb-2">End Session?</h2>
              <p className="text-gray-500 text-sm mb-6">Are you sure you want to end the session? This cannot be undone.</p>
              <div className="flex gap-3">
                <button
                  onClick={() => setShowEnd(false)}
                  className="flex-1 py-3 border-2 border-gray-200 text-gray-700 font-semibold rounded-xl hover:bg-gray-50 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleEndConfirm}
                  className="flex-1 py-3 bg-red-600 text-white font-semibold rounded-xl hover:bg-red-700 transition-colors"
                >
                  End Session
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
