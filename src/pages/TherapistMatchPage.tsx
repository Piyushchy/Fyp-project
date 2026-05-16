import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Star, Clock, CalendarDays, CheckCircle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Therapist {
  id: string;
  name: string;
  initials: string;
  color: string;
  specializations: string[];
  experience: string;
  languages: string;
  rating: number;
  availableDays: string;
}

const THERAPISTS: Therapist[] = [
  {
    id: "1",
    name: "Dr. Priya Nair",
    initials: "PN",
    color: "bg-emerald-100 text-emerald-700",
    specializations: ["Anxiety", "Depression"],
    experience: "7 years experience",
    languages: "English, Hindi",
    rating: 4.2,
    availableDays: "Mon, Wed, Fri",
  },
  {
    id: "2",
    name: "Dr. Arjun Desai",
    initials: "AD",
    color: "bg-blue-100 text-blue-700",
    specializations: ["Stress", "Trauma", "Grief"],
    experience: "5 years experience",
    languages: "English, Marathi",
    rating: 4.5,
    availableDays: "Tue, Thu, Sat",
  },
  {
    id: "3",
    name: "Dr. Sneha Kulkarni",
    initials: "SK",
    color: "bg-purple-100 text-purple-700",
    specializations: ["Depression", "Relationships"],
    experience: "10 years experience",
    languages: "English, Hindi",
    rating: 4.8,
    availableDays: "Mon–Fri",
  },
  {
    id: "4",
    name: "Dr. Kabir Verma",
    initials: "KV",
    color: "bg-amber-100 text-amber-700",
    specializations: ["Anxiety", "Sleep", "Stress"],
    experience: "3 years experience",
    languages: "English",
    rating: 3.9,
    availableDays: "Wed, Fri, Sun",
  },
];

export default function TherapistMatchPage() {
  const navigate = useNavigate();
  const [selectedTherapist, setSelectedTherapist] = useState<Therapist | null>(null);
  const [date, setDate] = useState("");
  const [timeSlot, setTimeSlot] = useState("");
  const [showToast, setShowToast] = useState(false);

  const handleBookingConfirm = () => {
    if (!date || !timeSlot) return; // Simple validation
    setSelectedTherapist(null);
    setShowToast(true);
    setTimeout(() => {
      navigate("/patient/dashboard");
    }, 2000);
  };

  const renderStars = (rating: number) => {
    return (
      <div className="flex items-center gap-1 text-amber-400">
        {[1, 2, 3, 4, 5].map((star) => (
          <Star
            key={star}
            className={`w-4 h-4 ${star <= Math.round(rating) ? "fill-current" : "text-gray-300"}`}
          />
        ))}
        <span className="text-gray-600 text-sm ml-1 font-medium">{rating}</span>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 font-sans">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-10 text-center md:text-left">
          <h1 className="text-3xl font-bold text-[var(--foreground)] mb-2">
            We found 4 therapists for you
          </h1>
          <p className="text-gray-600 text-lg">
            Based on your responses, these therapists are a great fit.
          </p>
        </div>

        {/* Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {THERAPISTS.map((therapist) => (
            <div
              key={therapist.id}
              className="bg-white border border-gray-200 rounded-2xl p-6 flex flex-col justify-between"
            >
              <div>
                <div className="flex items-start gap-4 mb-4">
                  <div className={`w-14 h-14 rounded-full flex items-center justify-center font-bold text-xl ${therapist.color}`}>
                    {therapist.initials}
                  </div>
                  <div className="flex-1">
                    <h2 className="text-xl font-bold text-[var(--foreground)]">{therapist.name}</h2>
                    <div className="mt-1 mb-2">{renderStars(therapist.rating)}</div>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2 mb-4">
                  {therapist.specializations.map((spec) => (
                    <span
                      key={spec}
                      className="px-3 py-1 bg-teal-50 text-teal-700 text-xs font-semibold rounded-full border border-teal-100"
                    >
                      {spec}
                    </span>
                  ))}
                </div>

                <div className="space-y-2 text-sm text-gray-600 mb-6">
                  <p><strong>Experience:</strong> {therapist.experience}</p>
                  <p><strong>Languages:</strong> {therapist.languages}</p>
                  <p className="flex items-center gap-1">
                    <CalendarDays className="w-4 h-4" /> 
                    <span>Available: {therapist.availableDays}</span>
                  </p>
                </div>
              </div>

              <button
                onClick={() => {
                  setSelectedTherapist(therapist);
                  setDate("");
                  setTimeSlot("");
                }}
                className="w-full py-3 border-2 border-[var(--primary)] text-[var(--primary)] font-semibold rounded-xl hover:bg-teal-50 transition-colors"
              >
                Book Appointment
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Booking Modal */}
      <AnimatePresence>
        {selectedTherapist && (
          <div className="fixed inset-0 z-50 flex items-center justify-center px-4">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 bg-black/40 backdrop-blur-sm"
              onClick={() => setSelectedTherapist(null)}
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="relative bg-white rounded-3xl p-6 sm:p-8 w-full max-w-md shadow-2xl"
            >
              <h2 className="text-2xl font-bold mb-1">Book a session</h2>
              <p className="text-gray-500 mb-6">with {selectedTherapist.name}</p>

              <div className="space-y-5">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Select Date</label>
                  <input
                    type="date"
                    value={date}
                    onChange={(e) => setDate(e.target.value)}
                    className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--primary)] focus:bg-white transition-all text-[var(--foreground)]"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Available Time Slots</label>
                  <div className="grid grid-cols-2 gap-3">
                    {["10:00 AM", "12:00 PM", "3:00 PM", "5:00 PM"].map((time) => (
                      <label
                        key={time}
                        className={`flex items-center justify-center p-3 border rounded-xl cursor-pointer transition-all ${
                          timeSlot === time
                            ? "border-[var(--primary)] bg-teal-50 text-[var(--primary)] font-semibold"
                            : "border-gray-200 text-gray-600 hover:border-gray-300"
                        }`}
                      >
                        <input
                          type="radio"
                          name="timeSlot"
                          value={time}
                          checked={timeSlot === time}
                          onChange={(e) => setTimeSlot(e.target.value)}
                          className="hidden"
                        />
                        <Clock className="w-4 h-4 mr-2" />
                        <span className="text-sm">{time}</span>
                      </label>
                    ))}
                  </div>
                </div>

                <div className="pt-2">
                  <button
                    onClick={handleBookingConfirm}
                    disabled={!date || !timeSlot}
                    className="w-full py-3 bg-[var(--primary)] text-white font-semibold rounded-xl hover:bg-teal-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Confirm Booking
                  </button>
                  <button
                    onClick={() => setSelectedTherapist(null)}
                    className="w-full py-3 mt-2 text-gray-500 font-medium hover:bg-gray-50 rounded-xl transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Success Toast */}
      <AnimatePresence>
        {showToast && (
          <motion.div
            initial={{ opacity: 0, y: 50, x: "-50%" }}
            animate={{ opacity: 1, y: 0, x: "-50%" }}
            exit={{ opacity: 0, y: 50, x: "-50%" }}
            className="fixed bottom-8 left-1/2 -translate-x-1/2 bg-gray-900 text-white px-6 py-4 rounded-2xl shadow-xl flex items-center gap-3 z-50 whitespace-nowrap"
          >
            <CheckCircle className="text-teal-400 w-6 h-6" />
            <span className="font-medium">Appointment requested! You'll be notified once confirmed.</span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
