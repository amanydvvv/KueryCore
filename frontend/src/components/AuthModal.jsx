import { useEffect, useRef, useState } from 'react';
import { loginUser, signupUser, requestPasswordReset, warmUpBackend } from '../services/api';
import Interactive3DSpheres from './auth/Interactive3DSpheres';

const ANIMATED_SUBTITLES = [
  'Grounded Document RAG Engine.',
  'Mathematical Citation Proofs.',
  'Automated PII Guardrails.',
  'Sub-350ms Hybrid Search.',
];

// Modes: 'signin' | 'signup' | 'forgot'
export default function AuthModal({ onAuthSuccess, onAuthError }) {
  const [mode, setMode] = useState('signin'); // 'signin' | 'signup' | 'forgot'
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [successMsg, setSuccessMsg] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingSeconds, setLoadingSeconds] = useState(0);
  const submitInFlight = useRef(false);

  // Pre-warm backend when AuthModal mounts
  useEffect(() => {
    warmUpBackend();
  }, []);

  // Track elapsed loading seconds to display cold-start status
  useEffect(() => {
    let interval;
    if (loading) {
      setLoadingSeconds(0);
      interval = setInterval(() => {
        setLoadingSeconds((prev) => prev + 1);
      }, 1000);
    } else {
      setLoadingSeconds(0);
    }
    return () => clearInterval(interval);
  }, [loading]);

  // Animated typewriter state
  const [subtitleIndex, setSubtitleIndex] = useState(0);
  const [displayedText, setDisplayedText] = useState('');
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    const currentTarget = ANIMATED_SUBTITLES[subtitleIndex];
    const speed = isDeleting ? 30 : 65;

    const timeout = setTimeout(() => {
      if (!isDeleting) {
        if (displayedText.length < currentTarget.length) {
          setDisplayedText(currentTarget.slice(0, displayedText.length + 1));
        } else {
          setTimeout(() => setIsDeleting(true), 2000);
        }
      } else {
        if (displayedText.length > 0) {
          setDisplayedText(currentTarget.slice(0, displayedText.length - 1));
        } else {
          setIsDeleting(false);
          setSubtitleIndex((prev) => (prev + 1) % ANIMATED_SUBTITLES.length);
        }
      }
    }, speed);

    return () => clearTimeout(timeout);
  }, [displayedText, isDeleting, subtitleIndex]);

  const switchMode = (newMode) => {
    setMode(newMode);
    setError('');
    setSuccessMsg('');
  };

  // ─────────────────────────────────────────────
  // Sign-in / Sign-up submit
  // ─────────────────────────────────────────────
  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    if (submitInFlight.current) return;
    submitInFlight.current = true;
    setError('');
    setSuccessMsg('');
    setLoading(true);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 75_000);

    try {
      const authFn = mode === 'signin' ? loginUser : signupUser;
      const data = await authFn(email, password, controller.signal);
      clearTimeout(timeoutId);
      onAuthSuccess(data);
    } catch (err) {
      clearTimeout(timeoutId);
      const isTimeout = err.name === 'AbortError' || /aborted/i.test(err.message);
      setError(
        isTimeout
          ? 'Server took longer than expected to wake up. Please click again to retry.'
          : err.message || 'Authentication failed. Please verify credentials.'
      );
      if (onAuthError) onAuthError();
    } finally {
      setLoading(false);
      submitInFlight.current = false;
    }
  };

  // ─────────────────────────────────────────────
  // Forgot-password submit
  // ─────────────────────────────────────────────
  const handleForgotSubmit = async (e) => {
    if (e) e.preventDefault();
    if (submitInFlight.current) return;
    submitInFlight.current = true;
    setError('');
    setSuccessMsg('');
    setLoading(true);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15_000);

    try {
      const result = await requestPasswordReset(email, controller.signal);
      clearTimeout(timeoutId);
      // Always show the server's generic message — prevents enumeration on the client too
      setSuccessMsg(result.message || 'If an account exists with this email, reset instructions have been sent.');
      setEmail('');
    } catch (err) {
      clearTimeout(timeoutId);
      const isTimeout = err.name === 'AbortError' || /aborted/i.test(err.message);
      setError(
        isTimeout
          ? 'Request timed out. Please try again.'
          : err.message || 'Something went wrong. Please try again.'
      );
    } finally {
      setLoading(false);
      submitInFlight.current = false;
    }
  };

  // ─────────────────────────────────────────────
  // Demo / guest login
  // ─────────────────────────────────────────────
  const handleDemoLogin = async () => {
    if (submitInFlight.current) return;
    submitInFlight.current = true;
    setError('');
    setSuccessMsg('');
    setLoading(true);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 75_000);

    try {
      const guestId = Math.random().toString(36).substring(2, 8);
      const guestEmail = `guest_${guestId}@kuerycore.ai`;
      const guestPass = `Guest_${guestId}_${Date.now()}!`;

      const signupData = await signupUser(guestEmail, guestPass, controller.signal);
      clearTimeout(timeoutId);
      onAuthSuccess(signupData);
    } catch (err) {
      clearTimeout(timeoutId);
      const isTimeout = err.name === 'AbortError' || /aborted/i.test(err.message);
      setError(
        isTimeout
          ? 'Server took longer than expected to wake up. Please click again to retry.'
          : err.message || 'Demo initialization failed. Please try standard sign up.'
      );
      if (onAuthError) onAuthError();
    } finally {
      setLoading(false);
      submitInFlight.current = false;
    }
  };

  // ─────────────────────────────────────────────
  // Heading text per mode
  // ─────────────────────────────────────────────
  const headingText = mode === 'signin'
    ? 'Welcome Back.'
    : mode === 'signup'
    ? 'Get Started.'
    : 'Reset Password.';

  const isForgot = mode === 'forgot';

  return (
    <div className="relative min-h-screen w-screen flex flex-col items-center justify-center p-6 overflow-hidden select-none font-sans text-slate-100 bg-[#050d08]">
      
      {/* ── 3D REAL-TIME INTERACTIVE WEBGL LIVE WALLPAPER SPHERES ── */}
      <Interactive3DSpheres />

      {/* ── MAIN AUTHENTICATION CONTAINER ── */}
      <div className="relative z-10 w-full max-w-[390px] sm:max-w-[410px] flex flex-col items-center">
        
        {/* Large Bold Brand Heading */}
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight text-center mb-2">
          {headingText}
        </h1>

        {/* Animated Typewriter Subtitle */}
        <div className="h-6 flex items-center justify-center gap-1 mb-5 text-xs font-medium text-slate-400 font-mono">
          <span>{displayedText}</span>
          <span className="w-0.5 h-3.5 bg-emerald-400 animate-pulse inline-block" />
        </div>

        {/* Glass Card */}
        <div
          className="w-full rounded-2xl p-6 sm:p-7 flex flex-col items-center relative"
          style={{
            background: 'linear-gradient(180deg, rgba(13, 29, 21, 0.9) 0%, rgba(9, 20, 16, 0.98) 100%)',
            backdropFilter: 'blur(28px)',
            WebkitBackdropFilter: 'blur(28px)',
            border: '1px solid rgba(0, 255, 170, 0.2)',
            boxShadow: '0 24px 70px rgba(0, 0, 0, 0.85), 0 0 60px rgba(0, 214, 143, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.12)',
          }}
        >
          {/* Mode switcher — only shown on signin / signup */}
          {!isForgot && (
            <div
              className="w-full p-1 rounded-xl flex items-center mb-5"
              style={{
                background: 'linear-gradient(180deg, #222824 0%, #111613 100%)',
                border: '1px solid rgba(255, 255, 255, 0.12)',
                boxShadow: 'inset 0 1px 0 rgba(255, 255, 255, 0.18), 0 4px 12px rgba(0, 0, 0, 0.4)',
              }}
            >
              <button
                type="button"
                onClick={() => switchMode('signin')}
                className={`flex-1 py-2 text-xs font-bold rounded-lg transition-all cursor-pointer ${
                  mode === 'signin'
                    ? 'bg-emerald-500/25 text-white border border-emerald-400/60 shadow-[0_0_15px_rgba(0,255,170,0.3)]'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                Sign In
              </button>
              <button
                type="button"
                onClick={() => switchMode('signup')}
                className={`flex-1 py-2 text-xs font-bold rounded-lg transition-all cursor-pointer ${
                  mode === 'signup'
                    ? 'bg-emerald-500/25 text-white border border-emerald-400/60 shadow-[0_0_15px_rgba(0,255,170,0.3)]'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                Sign Up
              </button>
            </div>
          )}

          {/* Cold-start progressive indicator */}
          {loading && loadingSeconds >= 4 && (
            <div className="w-full mb-4 px-4 py-3 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center gap-3 animate-in fade-in">
              <span className="relative flex h-3 w-3 shrink-0">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
              </span>
              <div className="flex flex-col text-left">
                <span className="text-xs font-semibold text-emerald-300">
                  {loadingSeconds < 20 ? 'Waking up cloud server...' : 'Almost ready, finalizing connection...'}
                </span>
                <span className="text-[10px] text-slate-400">
                  Free tier takes ~30s on cold start ({loadingSeconds}s elapsed)
                </span>
              </div>
            </div>
          )}

          {/* Error banner */}
          {error && (
            <div className="w-full mb-4 px-4 py-2.5 rounded-xl bg-red-500/10 border border-red-500/25 text-red-300 text-xs text-center animate-in fade-in duration-150">
              {error}
            </div>
          )}

          {/* Success banner (forgot-password confirmation) */}
          {successMsg && (
            <div className="w-full mb-4 px-4 py-2.5 rounded-xl bg-emerald-500/10 border border-emerald-500/25 text-emerald-300 text-xs text-center animate-in fade-in duration-150">
              {successMsg}
            </div>
          )}

          {/* ── FORGOT PASSWORD FORM ── */}
          {isForgot ? (
            <>
              {!successMsg && (
                <form onSubmit={handleForgotSubmit} className="w-full flex flex-col gap-3">
                  <p className="text-xs text-slate-400 text-center mb-1">
                    Enter your email and we'll send you a reset link.
                  </p>
                  <input
                    id="forgot-email"
                    type="email"
                    required
                    autoComplete="email"
                    placeholder="E-mail address..."
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full bg-[#060e09] border border-white/[0.09] hover:border-emerald-400/40 focus:border-emerald-400 focus:ring-2 focus:ring-emerald-400/40 rounded-xl px-4 py-3 text-xs sm:text-sm text-white placeholder:text-slate-500 focus:outline-none transition-all duration-150 shadow-inner"
                  />
                  <button
                    type="submit"
                    disabled={loading}
                    id="forgot-submit-btn"
                    className="w-full mt-1 py-3.5 px-4 rounded-xl font-bold text-xs sm:text-sm tracking-wide transition-all duration-150 cursor-pointer active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    style={{
                      background: 'linear-gradient(135deg, #00d68f 0%, #00ffaa 100%)',
                      color: '#020804',
                      boxShadow: '0 0 35px rgba(0, 214, 143, 0.65), 0 4px 16px rgba(0, 255, 170, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.4)',
                    }}
                  >
                    <span>{loading ? 'Sending...' : 'Send Reset Link'}</span>
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#020804" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="5" y1="12" x2="19" y2="12" />
                      <polyline points="12 5 19 12 12 19" />
                    </svg>
                  </button>
                </form>
              )}

              {/* Back to sign-in link */}
              <div className="mt-5 text-center">
                <button
                  type="button"
                  onClick={() => switchMode('signin')}
                  className="text-xs text-emerald-400 hover:text-emerald-300 transition-colors cursor-pointer font-bold"
                >
                  ← Back to Sign In
                </button>
              </div>
            </>
          ) : (
            /* ── SIGN-IN / SIGN-UP FORM ── */
            <>
              <form onSubmit={handleSubmit} className="w-full flex flex-col gap-3">
                <div className="w-full">
                  <input
                    id="auth-email"
                    type="email"
                    required
                    autoComplete="email"
                    placeholder="E-mail address..."
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full bg-[#060e09] border border-white/[0.09] hover:border-emerald-400/40 focus:border-emerald-400 focus:ring-2 focus:ring-emerald-400/40 rounded-xl px-4 py-3 text-xs sm:text-sm text-white placeholder:text-slate-500 focus:outline-none transition-all duration-150 shadow-inner"
                  />
                </div>

                <div className="w-full">
                  <input
                    id="auth-password"
                    type="password"
                    required
                    autoComplete={mode === 'signin' ? 'current-password' : 'new-password'}
                    placeholder="Password..."
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full bg-[#060e09] border border-white/[0.09] hover:border-emerald-400/40 focus:border-emerald-400 focus:ring-2 focus:ring-emerald-400/40 rounded-xl px-4 py-3 text-xs sm:text-sm text-white placeholder:text-slate-500 focus:outline-none transition-all duration-150 shadow-inner"
                  />
                </div>

                {/* Submit CTA */}
                <button
                  type="submit"
                  disabled={loading}
                  id="auth-submit-btn"
                  className="w-full mt-2 py-3.5 px-4 rounded-xl font-bold text-xs sm:text-sm tracking-wide transition-all duration-150 cursor-pointer active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  style={{
                    background: 'linear-gradient(135deg, #00d68f 0%, #00ffaa 100%)',
                    color: '#020804',
                    boxShadow: '0 0 35px rgba(0, 214, 143, 0.65), 0 4px 16px rgba(0, 255, 170, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.4)',
                  }}
                >
                  <span>{loading ? 'Processing...' : mode === 'signin' ? 'Sign In' : 'Create Account'}</span>
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#020804" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="5" y1="12" x2="19" y2="12" />
                    <polyline points="12 5 19 12 12 19" />
                  </svg>
                </button>
              </form>

              {/* Footer links */}
              <div className="mt-4 flex flex-col items-center gap-1.5 text-xs text-slate-400">
                <div className="flex items-center gap-1.5">
                  <span>{mode === 'signin' ? "Don't have an account?" : 'Already have an account?'}</span>
                  <button
                    type="button"
                    onClick={() => switchMode(mode === 'signin' ? 'signup' : 'signin')}
                    className="font-bold text-emerald-400 hover:text-emerald-300 transition-colors cursor-pointer"
                  >
                    {mode === 'signin' ? 'Sign up' : 'Sign in'}
                  </button>
                </div>

                {mode === 'signin' && (
                  <button
                    type="button"
                    id="forgot-password-link"
                    onClick={() => switchMode('forgot')}
                    className="text-[11px] text-slate-500 hover:text-slate-300 transition-colors cursor-pointer"
                  >
                    Forgot password?
                  </button>
                )}
              </div>

              {/* Demo section */}
              <div className="w-full my-4 flex items-center gap-3">
                <div className="flex-1 h-px bg-white/[0.08]" />
                <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500">or explore demo</span>
                <div className="flex-1 h-px bg-white/[0.08]" />
              </div>

              <div className="w-full">
                <button
                  type="button"
                  id="demo-access-btn"
                  onClick={handleDemoLogin}
                  disabled={loading}
                  className="w-full py-3 px-4 rounded-xl bg-[#060e09] hover:bg-[#0c2016] border border-emerald-400/30 hover:border-emerald-400/70 transition-all duration-150 flex items-center justify-between text-xs font-bold text-slate-200 cursor-pointer shadow-[0_0_20px_rgba(0,214,143,0.15)] active:scale-[0.98] group"
                >
                  <div className="flex items-center gap-2.5">
                    <div className="w-6 h-6 rounded-lg bg-emerald-500/20 border border-emerald-400/50 flex items-center justify-center text-emerald-400 text-xs shadow-[0_0_10px_rgba(0,255,170,0.4)]">
                      ⚡
                    </div>
                    <div className="flex flex-col items-start">
                      <span className="text-white text-xs font-bold">Instant Demo Access</span>
                      <span className="text-[10px] text-slate-400 font-normal">Isolated guest session • No account needed</span>
                    </div>
                  </div>
                  <span className="text-emerald-400 group-hover:translate-x-0.5 transition-transform text-sm font-bold">→</span>
                </button>
              </div>
            </>
          )}

        </div>
      </div>
    </div>
  );
}