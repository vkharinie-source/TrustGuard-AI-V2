```jsx
import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import {
  Shield,
  Lock,
  Eye,
  EyeOff,
  ArrowRight,
  AlertTriangle
} from '../components/Icons';
import { api } from '../services/api';

export const Login = ({ setCurrentUser }) => {
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
          : err.message ||
            'Authentication failed. Please check your credentials.'
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
        backgroundColor: '#090D16',
      }}
    >
      {/* Left Brand Panel */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          padding: '3rem',
          position: 'relative',
          overflow: 'hidden',
          background:
            'linear-gradient(135deg, #0D1322 0%, #131B2E 50%, #0D1322 100%)',
        }}
        className="hidden-mobile"
      >
        <div
          style={{
            position: 'absolute',
            top: '-20%',
            left: '-10%',
            width: '500px',
            height: '500px',
            background:
              'radial-gradient(circle, rgba(99,102,241,0.1) 0%, transparent 70%)',
            borderRadius: '50%',
            pointerEvents: 'none',
          }}
        />

        <div
          style={{
            position: 'relative',
            zIndex: 1,
            maxWidth: '420px',
          }}
        >
          {/* Logo */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
              marginBottom: '2rem',
            }}
          >
            <div
              style={{
                width: '44px',
                height: '44px',
                borderRadius: '12px',
                background:
                  'linear-gradient(135deg, #4F46E5, #6366F1)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 0 20px rgba(99,102,241,0.4)',
              }}
            >
              <Shield size={24} color="#FFFFFF" />
            </div>

            <div>
              <h1
                style={{
                  fontSize: '1.2rem',
                  fontWeight: '800',
                  color: '#F9FAFB',
                  margin: 0,
                }}
              >
                TRUSTGUARD{' '}
                <span style={{ color: '#818CF8' }}>AI</span>
              </h1>

              <p
                style={{
                  fontSize: '0.65rem',
                  color: '#6B7280',
                  textTransform: 'uppercase',
                  letterSpacing: '0.1em',
                  margin: 0,
                }}
              >
                Digital Security Platform
              </p>
            </div>
          </div>

          {/* Heading */}
          <h2
            style={{
              fontSize: '1.8rem',
              fontWeight: '800',
              color: '#F9FAFB',
              lineHeight: '1.3',
              marginBottom: '1rem',
            }}
          >
            AI-Powered Threat Intelligence
          </h2>

          <p
            style={{
              fontSize: '0.9rem',
              color: '#9CA3AF',
              lineHeight: '1.7',
              marginBottom: '2rem',
            }}
          >
            Detect deepfakes, analyze phishing attempts, verify news articles
            and protect your digital identity with enterprise-grade AI
            security.
          </p>

          {/* Features */}
          {[
            'MobileNetV2 V3 deep learning detection',
            'Multi-layer threat analysis across 9 modules',
            'Real-time security intelligence reports',
          ].map((feat, i) => (
            <div
              key={i}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.65rem',
                marginBottom: '0.85rem',
              }}
            >
              <div
                style={{
                  width: '22px',
                  height: '22px',
                  borderRadius: '50%',
                  backgroundColor: 'rgba(16,185,129,0.15)',
                  border: '1px solid rgba(16,185,129,0.3)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                }}
              >
                <svg
                  width="12"
                  height="12"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="#34D399"
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <polyline points="20 6 9 17 4 12" />
                </svg>
              </div>

              <span
                style={{
                  fontSize: '0.825rem',
                  color: '#D1D5DB',
                }}
              >
                {feat}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Right Form Panel */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '2rem',
        }}
      >
        <div
          style={{
            width: '100%',
            maxWidth: '420px',
          }}
        >
          {/* Mobile Brand */}
          <div
            style={{
              display: 'none',
              marginBottom: '2rem',
              textAlign: 'center',
            }}
            className="show-mobile"
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.6rem',
                marginBottom: '0.5rem',
              }}
            >
              <Shield size={28} color="#818CF8" />

              <span
                style={{
                  fontSize: '1.1rem',
                  fontWeight: '800',
                  color: '#F9FAFB',
                }}
              >
                TRUSTGUARD{' '}
                <span style={{ color: '#818CF8' }}>AI</span>
              </span>
            </div>
          </div>

          {/* Welcome */}
          <h2
            style={{
              fontSize: '1.5rem',
              fontWeight: '800',
              color: '#F9FAFB',
              marginBottom: '0.4rem',
            }}
          >
            Welcome Back
          </h2>

          <p
            style={{
              fontSize: '0.875rem',
              color: '#9CA3AF',
              marginBottom: '1.75rem',
            }}
          >
            Sign in to continue to your TrustGuard AI security workspace.
          </p>

          {/* Error */}
          {error && (
            <div
              style={{
                backgroundColor: 'rgba(239,68,68,0.12)',
                border: '1px solid rgba(239,68,68,0.3)',
                color: '#F87171',
                padding: '0.75rem 1rem',
                borderRadius: '8px',
                fontSize: '0.825rem',
                marginBottom: '1.25rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
              }}
            >
              <AlertTriangle size={16} />
              {error}
            </div>
          )}

          {/* Login Form */}
          <form
            onSubmit={handleSubmit}
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '1.15rem',
            }}
          >
            {/* Email */}
            <div>
              <label
                style={{
                  display: 'block',
                  fontSize: '0.8rem',
                  fontWeight: '600',
                  color: '#D1D5DB',
                  marginBottom: '0.4rem',
                }}
              >
                Email Address
              </label>

              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="your@email.com"
                className="cyber-input"
                autoComplete="email"
                required
              />
            </div>

            {/* Password */}
            <div>
              <label
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  fontSize: '0.8rem',
                  fontWeight: '600',
                  color: '#D1D5DB',
                  marginBottom: '0.4rem',
                }}
              >
                Password

                <span
                  style={{
                    color: '#818CF8',
                    cursor: 'pointer',
                    fontSize: '0.75rem',
                    fontWeight: '500',
                  }}
                >
                  Forgot Password?
                </span>
              </label>

              <div
                style={{
                  position: 'relative',
                }}
              >
                <input
                  type={showPw ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  className="cyber-input"
                  style={{
                    paddingRight: '2.75rem',
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
                    color: '#6B7280',
                    cursor: 'pointer',
                    padding: '0.25rem',
                  }}
                >
                  {showPw ? (
                    <EyeOff size={18} />
                  ) : (
                    <Eye size={18} />
                  )}
                </button>
              </div>
            </div>

            {/* Submit */}
            <button
              type="submit"
              className="cyber-button-primary"
              disabled={loading}
              style={{
                width: '100%',
                marginTop: '0.5rem',
                padding: '0.85rem',
              }}
            >
              {loading ? (
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  style={{
                    animation: 'spin 0.8s linear infinite',
                  }}
                >
                  <path
                    d="M21 12a9 9 0 11-6.219-8.56"
                    strokeLinecap="round"
                  />
                </svg>
              ) : (
                <>
                  Sign In
                  <ArrowRight size={16} />
                </>
              )}
            </button>
          </form>

          {/* Register */}
          <p
            style={{
              textAlign: 'center',
              marginTop: '1.5rem',
              fontSize: '0.85rem',
              color: '#6B7280',
            }}
          >
            Don't have an account?{' '}
            <Link
              to="/register"
              style={{
                color: '#818CF8',
                fontWeight: '600',
              }}
            >
              Create Account
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
};

export default Login;
```
