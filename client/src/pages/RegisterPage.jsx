import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Shield, Eye, EyeOff, ArrowRight, AlertTriangle, RefreshCw, Sparkles } from '../components/Icons';
import { api } from '../services/api';

export default function RegisterPage({ setCurrentUser }) {
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPw, setShowPw] = useState(false);
  const [showCpw, setShowCpw] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!fullName.trim() || !email.trim() || !password || !confirmPassword) {
      setError('Please fill in all fields.');
      return;
    }
    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }
    if (password.length < 8) {
      setError('Password must be at least 8 characters.');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const data = await api.register(fullName, email, password, confirmPassword);
      if (data.token) {
        localStorage.setItem('trustguard_token', data.token);
        setCurrentUser(data.user || { full_name: fullName, email });
        navigate('/dashboard');
      } else {
        navigate('/login');
      }
    } catch (err) {
      const msg = err.message || '';
      setError(
        msg && msg !== 'Not Found'
          ? msg
          : 'Registration service is connecting. Please try again in a moment.'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        backgroundColor: '#F8F6FA',
      }}
    >
      {/* ========================================================================= */}
      {/* LEFT COLUMN: HERO (Royal Plum & Amethyst Palette)                         */}
      {/* ========================================================================= */}
      <div
        style={{
          flex: '1 1 50%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          padding: '4rem 3.5rem',
          position: 'relative',
          overflow: 'hidden',
          background: 'linear-gradient(135deg, #231533 0%, #1A0F26 50%, #2A1740 100%)',
          borderRight: '1px solid rgba(127, 93, 153, 0.2)',
        }}
        className="hidden-mobile"
      >
        {/* Ambient Glowing Orbs */}
        <div
          style={{
            position: 'absolute',
            bottom: '-15%',
            right: '-10%',
            width: '550px',
            height: '550px',
            background: 'radial-gradient(circle, rgba(127, 93, 153, 0.3) 0%, transparent 70%)',
            borderRadius: '50%',
            pointerEvents: 'none',
          }}
        />
        <div
          style={{
            position: 'absolute',
            top: '-20%',
            left: '-10%',
            width: '500px',
            height: '500px',
            background: 'radial-gradient(circle, rgba(158, 125, 183, 0.22) 0%, transparent 70%)',
            borderRadius: '50%',
            pointerEvents: 'none',
          }}
        />

        <div style={{ position: 'relative', zIndex: 1, maxWidth: '440px', width: '100%' }}>
          
          {/* Brand Shield & Eyebrow */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.9rem',
              marginBottom: '2.5rem',
            }}
          >
            <div
              style={{
                width: '50px',
                height: '50px',
                borderRadius: '14px',
                background: 'linear-gradient(135deg, #7F5D99 0%, #5E3977 100%)',
                border: '1px solid rgba(229, 221, 236, 0.35)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 8px 24px rgba(127, 93, 153, 0.45)',
              }}
            >
              <Shield size={26} color="#FFFFFF" />
            </div>

            <div>
              <h1
                style={{
                  fontSize: '1.35rem',
                  fontWeight: '800',
                  color: '#FFFFFF',
                  margin: 0,
                  letterSpacing: '0.04em',
                }}
              >
                TRUSTGUARD <span style={{ color: '#C8A9E6' }}>AI</span>
              </h1>
              <p
                style={{
                  fontSize: '0.68rem',
                  color: '#B3A5C1',
                  textTransform: 'uppercase',
                  letterSpacing: '0.12em',
                  margin: '0.15rem 0 0 0',
                  fontWeight: '700',
                }}
              >
                Intelligent Media & Identity Security
              </p>
            </div>
          </div>

          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.35rem 0.85rem',
              borderRadius: '9999px',
              backgroundColor: 'rgba(127, 93, 153, 0.25)',
              border: '1px solid rgba(158, 125, 183, 0.35)',
              color: '#DFD0EE',
              fontSize: '0.75rem',
              fontWeight: '700',
              marginBottom: '1.25rem',
            }}
          >
            <Sparkles size={14} color="#C8A9E6" />
            Deploy Secure Workspace
          </div>

          <h2
            style={{
              fontSize: '2.1rem',
              fontWeight: '800',
              color: '#FFFFFF',
              lineHeight: '1.25',
              marginBottom: '1.15rem',
              letterSpacing: '-0.02em',
            }}
          >
            Join the Enterprise Verification Network
          </h2>

          <p
            style={{
              fontSize: '0.92rem',
              color: '#D8C6E6',
              lineHeight: '1.7',
              marginBottom: '2.25rem',
            }}
          >
            Create your protected workspace and safeguard your digital presence with real-time heuristic verification and deep learning models.
          </p>

          {/* Value Props Checklist */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {[
              'Instant access to all 9 multi-modal security modules',
              'Persistent scan telemetry and verification history',
              'Cryptographically signed PDF audit report downloads',
              'Self-sovereign data privacy with zero model retention',
            ].map((feat, i) => (
              <div
                key={i}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.85rem',
                }}
              >
                <div
                  style={{
                    width: '24px',
                    height: '24px',
                    borderRadius: '50%',
                    backgroundColor: 'rgba(61, 167, 121, 0.2)',
                    border: '1px solid rgba(61, 167, 121, 0.4)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                  }}
                >
                  <svg
                    width="13"
                    height="13"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="#48C790"
                    strokeWidth="3"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                </div>

                <span
                  style={{
                    fontSize: '0.86rem',
                    color: '#EFEBF4',
                    fontWeight: '500',
                  }}
                >
                  {feat}
                </span>
              </div>
            ))}
          </div>

        </div>
      </div>

      {/* ========================================================================= */}
      {/* RIGHT COLUMN: REGISTER FORM CARD                                          */}
      {/* ========================================================================= */}
      <div
        style={{
          flex: '1 1 50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '2.5rem',
          backgroundColor: '#F8F6FA',
        }}
      >
        <div
          style={{
            width: '100%',
            maxWidth: '460px',
            backgroundColor: '#FFFFFF',
            borderRadius: '20px',
            border: '1px solid #E5DDEC',
            padding: '2.5rem 2.25rem',
            boxShadow: '0 12px 35px -4px rgba(35, 21, 51, 0.08), 0 4px 12px -2px rgba(35, 21, 51, 0.03)',
            position: 'relative',
          }}
        >
          {/* Top subtle violet gradient line */}
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              height: '4px',
              borderRadius: '20px 20px 0 0',
              background: 'linear-gradient(90deg, #7F5D99 0%, #9E7DB7 50%, #EDE4F5 100%)',
            }}
          />

          {/* Mobile Logo Only */}
          <div
            style={{
              display: 'none',
              marginBottom: '1.75rem',
              textAlign: 'center',
            }}
            className="show-mobile"
          >
            <div
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.65rem',
                padding: '0.5rem 1rem',
                borderRadius: '12px',
                backgroundColor: '#EDE4F5',
              }}
            >
              <Shield size={24} color="#7F5D99" />
              <span style={{ fontSize: '1.15rem', fontWeight: '800', color: '#221730' }}>
                TRUSTGUARD <span style={{ color: '#7F5D99' }}>AI</span>
              </span>
            </div>
          </div>

          <div style={{ marginBottom: '1.75rem' }}>
            <h2
              style={{
                fontSize: '1.65rem',
                fontWeight: '800',
                color: '#221730',
                margin: 0,
                letterSpacing: '-0.02em',
              }}
            >
              Create Your Account
            </h2>
            <p
              style={{
                fontSize: '0.85rem',
                color: '#6D5E7B',
                margin: '0.35rem 0 0 0',
              }}
            >
              Start protecting your digital identity with TrustGuard AI.
            </p>
          </div>

          {error && (
            <div
              style={{
                backgroundColor: '#FDEFF2',
                border: '1px solid #F8BDC8',
                color: '#A8324C',
                padding: '0.85rem 1rem',
                borderRadius: '10px',
                fontSize: '0.825rem',
                marginBottom: '1.25rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.65rem',
                fontWeight: '600',
              }}
            >
              <AlertTriangle size={18} color="#D9536F" />
              <span>{error}</span>
            </div>
          )}

          <form
            onSubmit={handleSubmit}
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '1.15rem',
            }}
          >
            <div>
              <label
                style={{
                  display: 'block',
                  fontSize: '0.825rem',
                  fontWeight: '700',
                  color: '#221730',
                  marginBottom: '0.45rem',
                }}
              >
                Full Name
              </label>
              <input
                type="text"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="e.g. Harinie V K"
                className="cyber-input"
                required
                style={{
                  backgroundColor: '#FDFCFE',
                  borderColor: fullName ? '#D8C6E6' : '#E5DDEC',
                }}
              />
            </div>

            <div>
              <label
                style={{
                  display: 'block',
                  fontSize: '0.825rem',
                  fontWeight: '700',
                  color: '#221730',
                  marginBottom: '0.45rem',
                }}
              >
                Email Address
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="name@company.com"
                className="cyber-input"
                autoComplete="email"
                required
                style={{
                  backgroundColor: '#FDFCFE',
                  borderColor: email ? '#D8C6E6' : '#E5DDEC',
                }}
              />
            </div>

            <div>
              <label
                style={{
                  display: 'block',
                  fontSize: '0.825rem',
                  fontWeight: '700',
                  color: '#221730',
                  marginBottom: '0.45rem',
                }}
              >
                Password
              </label>
              <div style={{ position: 'relative' }}>
                <input
                  type={showPw ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Minimum 8 characters"
                  className="cyber-input"
                  style={{
                    paddingRight: '2.75rem',
                    backgroundColor: '#FDFCFE',
                    borderColor: password ? '#D8C6E6' : '#E5DDEC',
                  }}
                  autoComplete="new-password"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPw(!showPw)}
                  style={{
                    position: 'absolute',
                    right: '0.75rem',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    background: 'none',
                    border: 'none',
                    color: '#9E7DB7',
                    cursor: 'pointer',
                    padding: '0.25rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                  title={showPw ? 'Hide password' : 'Show password'}
                >
                  {showPw ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
            </div>

            <div>
              <label
                style={{
                  display: 'block',
                  fontSize: '0.825rem',
                  fontWeight: '700',
                  color: '#221730',
                  marginBottom: '0.45rem',
                }}
              >
                Confirm Password
              </label>
              <div style={{ position: 'relative' }}>
                <input
                  type={showCpw ? 'text' : 'password'}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="Re-enter your password"
                  className="cyber-input"
                  style={{
                    paddingRight: '2.75rem',
                    backgroundColor: '#FDFCFE',
                    borderColor: confirmPassword ? '#D8C6E6' : '#E5DDEC',
                  }}
                  autoComplete="new-password"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowCpw(!showCpw)}
                  style={{
                    position: 'absolute',
                    right: '0.75rem',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    background: 'none',
                    border: 'none',
                    color: '#9E7DB7',
                    cursor: 'pointer',
                    padding: '0.25rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                  title={showCpw ? 'Hide password' : 'Show password'}
                >
                  {showCpw ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="cyber-button-primary"
              style={{
                width: '100%',
                padding: '0.85rem',
                fontSize: '0.92rem',
                fontWeight: '700',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.65rem',
                marginTop: '0.5rem',
                boxShadow: '0 8px 24px -3px rgba(127, 93, 153, 0.45)',
              }}
            >
              {loading ? (
                <>
                  <RefreshCw size={18} className="animate-spin" />
                  <span>Creating Workspace...</span>
                </>
              ) : (
                <>
                  <span>Create Account</span>
                  <ArrowRight size={18} />
                </>
              )}
            </button>
          </form>

          {/* Footer Navigation */}
          <div
            style={{
              textAlign: 'center',
              marginTop: '1.75rem',
              paddingTop: '1.25rem',
              borderTop: '1px solid #F1ECF5',
            }}
          >
            <p
              style={{
                fontSize: '0.85rem',
                color: '#6D5E7B',
                margin: 0,
              }}
            >
              Already have an account?{' '}
              <Link
                to="/login"
                style={{
                  color: '#7F5D99',
                  fontWeight: '800',
                  textDecoration: 'none',
                }}
              >
                Sign In
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
