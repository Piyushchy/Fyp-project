import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Activity, ChevronRight } from "lucide-react";

const CONCERNS = ["Anxiety", "Depression", "Stress", "Trauma", "Relationships", "Grief", "Sleep", "Anger", "Other"];

export default function IntakeFormPage() {
  const navigate = useNavigate();
  const [step, setStep] = useState(1);
  const [concerns, setConcerns] = useState<string[]>([]);
  const [moodScore, setMoodScore] = useState(3);
  const [experience, setExperience] = useState("");
  const [goals, setGoals] = useState("");

  const toggleConcern = (c: string) =>
    setConcerns(prev => prev.includes(c) ? prev.filter(x => x !== c) : [...prev, c]);

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-4 font-sans">
      <div className="w-full max-w-xl bg-white border border-gray-200 rounded-3xl p-8 my-8">

        <div className="flex items-center gap-2 text-[var(--primary)] font-bold text-xl mb-8">
          <Activity className="w-6 h-6" /><span>MindBridge</span>
        </div>

        {/* Progress */}
        <div className="mb-8">
          <div className="flex justify-between text-xs text-gray-400 mb-2">
            <span>Step {step} of 2</span>
            <span>{step === 1 ? "About You" : "Your Goals"}</span>
          </div>
          <div className="w-full h-1.5 bg-gray-100 rounded-full overflow-hidden">
            <div className="h-full bg-[var(--primary)] rounded-full transition-all duration-500"
              style={{ width: step === 1 ? "50%" : "100%" }} />
          </div>
        </div>

        {step === 1 && (
          <div className="space-y-6">
            <div>
              <h1 className="text-2xl font-bold text-[var(--foreground)] mb-1">Tell us about yourself</h1>
              <p className="text-gray-500 text-sm">This helps us find the right therapist for you.</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Primary concerns <span className="text-gray-400">(select all that apply)</span>
              </label>
              <div className="flex flex-wrap gap-2">
                {CONCERNS.map(c => (
                  <button key={c} type="button" onClick={() => toggleConcern(c)}
                    className={`px-3 py-1.5 text-sm rounded-full border transition-all font-medium ${
                      concerns.includes(c)
                        ? "border-[var(--primary)] bg-teal-50 text-[var(--primary)]"
                        : "border-gray-200 text-gray-600 hover:border-gray-300"
                    }`}>
                    {c}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Current mood: <span className="font-bold text-[var(--primary)]">{moodScore}/5</span>
              </label>
              <input type="range" min={1} max={5} step={1} value={moodScore}
                onChange={e => setMoodScore(Number(e.target.value))}
                className="w-full accent-teal-600" />
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>Very Low</span><span>Neutral</span><span>Great</span>
              </div>
            </div>

            <button onClick={() => setStep(2)}
              className="w-full py-3 bg-[var(--primary)] text-white font-semibold rounded-xl hover:bg-teal-700 transition-colors flex items-center justify-center gap-2">
              Continue <ChevronRight className="w-5 h-5" />
            </button>
          </div>
        )}

        {step === 2 && (
          <div className="space-y-6">
            <div>
              <h1 className="text-2xl font-bold text-[var(--foreground)] mb-1">Your therapy goals</h1>
              <p className="text-gray-500 text-sm">Help us understand what you're hoping to achieve.</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Had therapy before?</label>
              <div className="grid grid-cols-3 gap-3">
                {["Yes", "No", "Not sure"].map(opt => (
                  <button key={opt} type="button" onClick={() => setExperience(opt)}
                    className={`py-3 text-sm rounded-xl border transition-all font-medium ${
                      experience === opt
                        ? "border-[var(--primary)] bg-teal-50 text-[var(--primary)]"
                        : "border-gray-200 text-gray-600 hover:border-gray-300"
                    }`}>
                    {opt}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">What would you like to achieve?</label>
              <textarea value={goals} onChange={e => setGoals(e.target.value)} rows={4}
                placeholder="E.g. I want to manage my anxiety and feel more in control..."
                className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--primary)] text-sm resize-none" />
            </div>

            <div className="flex gap-3">
              <button onClick={() => setStep(1)}
                className="flex-1 py-3 border-2 border-gray-200 text-gray-600 font-semibold rounded-xl hover:bg-gray-50 transition-colors">
                Back
              </button>
              <button onClick={() => navigate("/patient/match")}
                className="flex-1 py-3 bg-[var(--primary)] text-white font-semibold rounded-xl hover:bg-teal-700 transition-colors">
                Find My Therapist →
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
