import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { 
  Activity, 
  ClipboardType, 
  Users, 
  Video, 
  Smile, 
  ShieldCheck, 
  Award 
} from "lucide-react";

export default function LandingPage() {
  return (
    <div className="flex flex-col min-h-screen bg-[var(--background)] font-sans text-[var(--foreground)]">
      {/* 1. Navbar */}
      <nav className="flex items-center justify-between px-6 py-4 md:px-12 border-b border-gray-100">
        <Link to="/" className="flex items-center gap-2 text-[var(--primary)] font-bold text-2xl tracking-tight">
          <Activity className="w-8 h-8" />
          <span>MindBridge</span>
        </Link>
        <div className="flex items-center gap-4">
          <Link to="/login" className="text-gray-600 hover:text-[var(--primary)] font-medium transition-colors">
            Login
          </Link>
          <Link to="/register" className="px-5 py-2 bg-[var(--primary)] text-white font-medium rounded-full hover:bg-teal-700 transition-colors shadow-sm">
            Get Started
          </Link>
        </div>
      </nav>

      <main className="flex-1 flex flex-col items-center">
        {/* 2. Hero Section */}
        <section className="w-full max-w-5xl px-6 py-24 md:py-32 flex flex-col items-center text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-50 text-teal-700 text-sm font-semibold mb-8 border border-teal-100"
          >
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-teal-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-teal-500"></span>
            </span>
            Clinical Emotion AI Available Now
          </motion.div>

          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="text-4xl md:text-6xl font-extrabold tracking-tight mb-6 leading-tight max-w-4xl"
          >
            Therapy that understands you — <br className="hidden md:block"/>
            <span className="text-[var(--primary)]">emotionally and clinically</span>
          </motion.h1>
          
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-lg md:text-xl text-gray-600 mb-10 max-w-2xl"
          >
            MindBridge connects patients with licensed therapists through AI-powered emotion-aware video sessions.
          </motion.p>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto"
          >
            <Link 
              to="/register?role=patient" 
              className="px-8 py-4 bg-[var(--primary)] text-white font-semibold rounded-full hover:bg-teal-700 transition-all hover:shadow-lg hover:-translate-y-0.5 text-center"
            >
              I'm a Patient
            </Link>
            <Link 
              to="/register?role=therapist" 
              className="px-8 py-4 bg-white border-2 border-[var(--primary)] text-[var(--primary)] font-semibold rounded-full hover:bg-teal-50 transition-all text-center"
            >
              I'm a Therapist
            </Link>
          </motion.div>
        </section>

        {/* 3. How it Works Section */}
        <section className="w-full bg-gray-50 py-24 px-6 border-y border-gray-100">
          <div className="max-w-6xl mx-auto">
            <div className="text-center mb-16">
              <h2 className="text-3xl md:text-4xl font-bold mb-4">How it works</h2>
              <p className="text-gray-600 max-w-xl mx-auto">Getting started with MindBridge is seamless, secure, and intuitive.</p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {/* Step 1 */}
              <div className="flex flex-col items-center text-center p-8 bg-white rounded-3xl shadow-sm border border-gray-100">
                <div className="w-16 h-16 bg-teal-50 text-[var(--primary)] rounded-full flex items-center justify-center mb-6 relative">
                  <ClipboardType className="w-8 h-8" />
                  <div className="absolute -top-3 -right-3 w-8 h-8 bg-[var(--primary)] text-white rounded-full flex items-center justify-center font-bold text-sm border-4 border-white">1</div>
                </div>
                <h3 className="text-xl font-bold mb-2">Fill a short intake form</h3>
                <p className="text-gray-600 text-sm">Tell us about your goals and what brings you here today.</p>
              </div>

              {/* Step 2 */}
              <div className="flex flex-col items-center text-center p-8 bg-white rounded-3xl shadow-sm border border-gray-100 relative">
                <div className="hidden md:block absolute top-1/2 -left-4 w-8 h-[2px] bg-gray-200"></div>
                <div className="hidden md:block absolute top-1/2 -right-4 w-8 h-[2px] bg-gray-200"></div>
                <div className="w-16 h-16 bg-teal-50 text-[var(--primary)] rounded-full flex items-center justify-center mb-6 relative">
                  <Users className="w-8 h-8" />
                  <div className="absolute -top-3 -right-3 w-8 h-8 bg-[var(--primary)] text-white rounded-full flex items-center justify-center font-bold text-sm border-4 border-white">2</div>
                </div>
                <h3 className="text-xl font-bold mb-2">Get matched with a therapist</h3>
                <p className="text-gray-600 text-sm">Our system finds the perfect licensed professional for you.</p>
              </div>

              {/* Step 3 */}
              <div className="flex flex-col items-center text-center p-8 bg-white rounded-3xl shadow-sm border border-gray-100">
                <div className="w-16 h-16 bg-teal-50 text-[var(--primary)] rounded-full flex items-center justify-center mb-6 relative">
                  <Video className="w-8 h-8" />
                  <div className="absolute -top-3 -right-3 w-8 h-8 bg-[var(--primary)] text-white rounded-full flex items-center justify-center font-bold text-sm border-4 border-white">3</div>
                </div>
                <h3 className="text-xl font-bold mb-2">Join a secure video session</h3>
                <p className="text-gray-600 text-sm">Start your journey with AI-enhanced emotion tracking.</p>
              </div>
            </div>
          </div>
        </section>

        {/* 4. Features Section */}
        <section className="w-full max-w-6xl mx-auto py-24 px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Why choose MindBridge?</h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
            <div className="flex flex-col items-center md:items-start text-center md:text-left">
              <div className="w-12 h-12 bg-teal-50 text-[var(--primary)] rounded-2xl flex items-center justify-center mb-6">
                <Smile className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold mb-3">Multimodal emotion detection</h3>
              <p className="text-gray-600">AI reads facial, voice, and text cues in real-time to provide deeper clinical insights during sessions.</p>
            </div>

            <div className="flex flex-col items-center md:items-start text-center md:text-left">
              <div className="w-12 h-12 bg-teal-50 text-[var(--primary)] rounded-2xl flex items-center justify-center mb-6">
                <ShieldCheck className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold mb-3">Private & secure</h3>
              <p className="text-gray-600">Your privacy is our priority. All sessions are end-to-end encrypted and fully HIPAA compliant.</p>
            </div>

            <div className="flex flex-col items-center md:items-start text-center md:text-left">
              <div className="w-12 h-12 bg-teal-50 text-[var(--primary)] rounded-2xl flex items-center justify-center mb-6">
                <Award className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold mb-3">Licensed therapists</h3>
              <p className="text-gray-600">Connect with highly vetted, verified professionals dedicated to your emotional well-being.</p>
            </div>
          </div>
        </section>
      </main>

      {/* 5. Footer */}
      <footer className="w-full py-8 text-center border-t border-gray-100 text-gray-500 text-sm mt-auto">
        C-48 © 2026
      </footer>
    </div>
  );
}
