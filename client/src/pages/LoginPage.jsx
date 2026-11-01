import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import {
  Shield,
  Lock,
  Eye,
  EyeOff,
  ArrowRight,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  Sparkles
} from '../components/Icons';
import { api } from '../services/api';

export default function LoginPage({ setCurrentUser }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPw, setShowPw] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!email.trim() || !password.trim()) {
      setError('Please enter both email and password.');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const data = await api.login(email, password);

      if (data.token) {
        localStorage.setItem('trustguard_token', data.token);
      }

      setCurrentUser(data.user || { email });
      navigate('/dashboard');
    } catch (err) {
      setError(
        err instanceof TypeError
          ? 'Cannot reach the TrustGuard AI API. Please try again.'
          : err.message || 'Authentication failed. Please check your credentials.'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleDemoLogin = (demoEmail, demoPw) => {
    setEmail(demoEmail);
    setPassword(demoPw);
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
      {/* LEFT COLUMN: BRAND HERO (Royal Plum & Amethyst Palette)                   */}
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
            top: '-15%',
            left: '-10%',
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
            bottom: '-20%',
            right: '-10%',
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
            Multi-Modal Neural Protection
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
            Autonomous Threat Intelligence & Verification
          </h2>

          <p
            style={{
              fontSize: '0.92rem',
              color: '#D8C6E6',
              lineHeight: '1.7',
              marginBottom: '2.25rem',
            }}
          >
            Detect visual deepfakes, analyze phishing vectors, inspect news articles, and protect your digital identity with enterprise-grade multi-layer AI security.
          </p>

          {/* Value Props Checklist */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {[
              'MobileNetV2 V4 calibrated deepfake detection engine',
              'Multi-layer threat heuristics across 9 active modules',
              'Audited Kaggle benchmarks with zero-retention privacy',
              'Instant cryptographic PDF security audit reports',
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
      {/* RIGHT COLUMN: LOGIN FORM CARD (Clean, Attractive Workspace)               */}
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
            maxWidth: '440px',
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

          <div style={{ marginBottom: '2rem' }}>
            <h2
              style={{
                fontSize: '1.65rem',
                fontWeight: '800',
                color: '#221730',
                margin: 0,
                letterSpacing: '-0.02em',
              }}
            >
              Welcome Back
            </h2>
            <p
              style={{
                fontSize: '0.85rem',
                color: '#6D5E7B',
                margin: '0.35rem 0 0 0',
              }}
            >
              Sign in to access your enterprise TrustGuard AI security portal.
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
                marginBottom: '1.5rem',
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
              gap: '1.25rem',
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
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: '0.45rem',
                }}
              >
                <label
                  style={{
                    fontSize: '0.825rem',
                    fontWeight: '700',
                    color: '#221730',
                    margin: 0,
                  }}
                >
                  Password
                </label>
                <span
                  style={{
                    color: '#7F5D99',
                    cursor: 'pointer',
                    fontSize: '0.78rem',
                    fontWeight: '700',
                  }}
                >
                  Forgot Password?
                </span>
              </div>

              <div style={{ position: 'relative' }}>
                <input
                  type={showPw ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter account password"
                  className="cyber-input"
                  style={{
                    paddingRight: '2.75rem',
                    backgroundColor: '#FDFCFE',
                    borderColor: password ? '#D8C6E6' : '#E5DDEC',
                  }}
                  autoComplete="current-password"
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

            {/* Quick Demo Credentials Preset */}
            <div
              style={{
                padding: '0.75rem 0.95rem',
                borderRadius: '10px',
                backgroundColor: '#F8F5FA',
                border: '1px solid #EBE4F2',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                fontSize: '0.78rem',
              }}
            >
              <span style={{ color: '#6D5E7B', fontWeight: '600' }}>Demo Quick-Fill:</span>
              <button
                type="button"
                onClick={() => handleDemoLogin('vkharinie@gmail.com', 'Password123!')}
                style={{
                  background: '#EDE4F5',
                  border: '1px solid #D8C6E6',
                  color: '#7F5D99',
                  fontWeight: '700',
                  fontSize: '0.72rem',
                  padding: '0.25rem 0.65rem',
                  borderRadius: '6px',
                  cursor: 'pointer',
                }}
              >
                Fill Credentials
              </button>
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
                  <span>Authenticating...</span>
                </>
              ) : (
                <>
                  <span>Sign In to Dashboard</span>
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
              Don't have an account yet?{' '}
              <Link
                to="/register"
                style={{
                  color: '#7F5D99',
                  fontWeight: '800',
                  textDecoration: 'none',
                }}
              >
                Create Account
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
