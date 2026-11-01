import React from 'react';
import {
  Settings as SettingsIcon,
  Shield,
  User,
  Lock,
  Activity,
  Info,
} from '../components/Icons';

export default function SettingsView({ currentUser }) {
  const user = currentUser || {};

  return (
    <div
      style={{
        maxWidth: '840px',
        margin: '0 auto',
        display: 'flex',
        flexDirection: 'column',
        gap: '1.5rem',
      }}
    >
      {/* Header card */}
      <div
        style={{
          backgroundColor: '#FFFFFF',
          border: '1px solid #E1E7DD',
          borderRadius: '12px',
          padding: '1.5rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.85rem',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.03)',
        }}
      >
        <div
          style={{
            width: '42px',
            height: '42px',
            borderRadius: '10px',
            backgroundColor: '#EDF3E8',
            border: '1px solid #D4E2CC',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
          }}
        >
          <SettingsIcon size={22} color="#36435E" />
        </div>

        <div>
          <h3
            style={{
              fontSize: '1.1rem',
              fontWeight: '700',
              color: '#202734',
              margin: 0,
            }}
          >
            Settings
          </h3>

          <p
            style={{
              fontSize: '0.825rem',
              color: '#697586',
              margin: '0.15rem 0 0 0',
            }}
          >
            Platform configuration, security compliance, and account preferences.
          </p>
        </div>
      </div>

      {/* Account Profile */}
      <div
        style={{
          backgroundColor: '#FFFFFF',
          border: '1px solid #E1E7DD',
          borderRadius: '12px',
          padding: '1.75rem',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.03)',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            marginBottom: '1.25rem',
          }}
        >
          <User size={18} color="#36435E" />

          <h4
            style={{
              fontSize: '0.9rem',
              fontWeight: '700',
              color: '#202734',
              margin: 0,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
            }}
          >
            Account Profile
          </h4>
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
            gap: '1.25rem',
            marginBottom: '1.25rem',
          }}
        >
          <div>
            <label
              style={{
                display: 'block',
                fontSize: '0.8rem',
                fontWeight: '600',
                color: '#697586',
                marginBottom: '0.4rem',
              }}
            >
              Full Name
            </label>

            <div
              className="cyber-input"
              style={{
                backgroundColor: '#F7F9F4',
                color: '#202734',
                border: '1px solid #E1E7DD',
                fontWeight: '500',
                cursor: 'not-allowed',
              }}
            >
              {user.full_name || 'Authenticated User'}
            </div>
          </div>

          <div>
            <label
              style={{
                display: 'block',
                fontSize: '0.8rem',
                fontWeight: '600',
                color: '#697586',
                marginBottom: '0.4rem',
              }}
            >
              Email Address
            </label>

            <div
              className="cyber-input"
              style={{
                backgroundColor: '#F7F9F4',
                color: '#202734',
                border: '1px solid #E1E7DD',
                fontWeight: '500',
                cursor: 'not-allowed',
              }}
            >
              {user.email || 'user@trustguard.ai'}
            </div>
          </div>
        </div>

        <div
          style={{
            padding: '0.85rem 1rem',
            borderRadius: '8px',
            backgroundColor: '#EDF3E8',
            border: '1px solid #D4E2CC',
            fontSize: '0.825rem',
            color: '#36435E',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}
        >
          <Info size={16} color="#36435E" />
          <span>Profile records are managed securely through the TrustGuard AI authentication service.</span>
        </div>
      </div>

      {/* Security */}
      <div
        style={{
          backgroundColor: '#FFFFFF',
          border: '1px solid #E1E7DD',
          borderRadius: '12px',
          padding: '1.75rem',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.03)',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            marginBottom: '1.25rem',
          }}
        >
          <Lock size={18} color="#36435E" />

          <h4
            style={{
              fontSize: '0.9rem',
              fontWeight: '700',
              color: '#202734',
              margin: 0,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
            }}
          >
            Security Controls
          </h4>
        </div>

        {[
          {
            label: 'Secure Authentication',
            value: 'PBKDF2-SHA256 (100,000 rounds)',
            status: 'Active',
          },
          {
            label: 'Session Token',
            value: 'URL-safe Bearer Token',
            status: 'Active',
          },
          {
            label: 'Data Transmission',
            value: 'CORS-protected REST API',
            status: 'Secured',
          },
          {
            label: 'Password Storage',
            value: 'Salted hash, never plaintext',
            status: 'Secured',
          },
        ].map((item, idx) => (
          <div
            key={idx}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '0.85rem 0',
              borderBottom: idx < 3 ? '1px solid #E1E7DD' : 'none',
              flexWrap: 'wrap',
              gap: '0.5rem',
            }}
          >
            <div>
              <p
                style={{
                  fontSize: '0.875rem',
                  fontWeight: '600',
                  color: '#202734',
                  margin: 0,
                }}
              >
                {item.label}
              </p>

              <p
                style={{
                  fontSize: '0.78rem',
                  color: '#697586',
                  margin: '0.15rem 0 0 0',
                }}
              >
                {item.value}
              </p>
            </div>

            <span
              style={{
                fontSize: '0.72rem',
                fontWeight: '700',
                padding: '0.2rem 0.65rem',
                borderRadius: '9999px',
                backgroundColor: '#EDF3E8',
                color: '#3D683D',
                border: '1px solid #BCD5BC',
              }}
            >
              {item.status}
            </span>
          </div>
        ))}
      </div>

      {/* Detection Modules */}
      <div
        style={{
          backgroundColor: '#FFFFFF',
          border: '1px solid #E1E7DD',
          borderRadius: '12px',
          padding: '1.75rem',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.03)',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            marginBottom: '1.25rem',
          }}
        >
          <Activity size={18} color="#36435E" />

          <h4
            style={{
              fontSize: '0.9rem',
              fontWeight: '700',
              color: '#202734',
              margin: 0,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
            }}
          >
            Detection Modules & Models
          </h4>
        </div>

        {[
          { name: 'DeepFake Image', model: 'MobileNetV2 V3' },
          {
            name: 'Website Analysis',
            model: 'Rule-based + ML Classifier',
          },
          {
            name: 'Email / Phishing',
            model: 'Pattern Heuristics Engine',
          },
          {
            name: 'Message / SMS',
            model: 'NLP Scam Classifier',
          },
          {
            name: 'Phone Call',
            model: 'Transcript Scam Analyzer',
          },
          {
            name: 'Job / Internship',
            model: 'Recruitment Fraud Detector',
          },
          {
            name: 'News Verification',
            model: 'Misinformation Classifier',
          },
          {
            name: 'App Audit',
            model: 'Package Metadata Inspector',
          },
        ].map((mod, idx) => (
          <div
            key={idx}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '0.75rem 0',
              borderBottom: idx < 7 ? '1px solid #E1E7DD' : 'none',
              flexWrap: 'wrap',
              gap: '0.5rem',
            }}
          >
            <div>
              <p
                style={{
                  fontSize: '0.875rem',
                  fontWeight: '600',
                  color: '#202734',
                  margin: 0,
                }}
              >
                {mod.name}
              </p>

              <p
                style={{
                  fontSize: '0.78rem',
                  color: '#697586',
                  margin: '0.15rem 0 0 0',
                }}
              >
                {mod.model}
              </p>
            </div>

            <span
              style={{
                fontSize: '0.72rem',
                fontWeight: '700',
                padding: '0.2rem 0.65rem',
                borderRadius: '9999px',
                backgroundColor: '#EDF3E8',
                color: '#3D683D',
                border: '1px solid #BCD5BC',
              }}
            >
              Online
            </span>
          </div>
        ))}
      </div>

      {/* Platform Information */}
      <div
        style={{
          backgroundColor: '#FFFFFF',
          border: '1px solid #E1E7DD',
          borderRadius: '12px',
          padding: '1.75rem',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.03)',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            marginBottom: '1.25rem',
          }}
        >
          <Shield size={18} color="#36435E" />

          <h4
            style={{
              fontSize: '0.9rem',
              fontWeight: '700',
              color: '#202734',
              margin: 0,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
            }}
          >
            Platform Information
          </h4>
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
            gap: '1rem',
          }}
        >
          {[
            { label: 'Platform', value: 'TrustGuard AI' },
            { label: 'Version', value: '2.0.0 Enterprise' },
            { label: 'Backend', value: 'FastAPI + Python' },
            {
              label: 'API Endpoint',
              value: 'http://127.0.0.1:8000',
            },
          ].map((info, idx) => (
            <div
              key={idx}
              style={{
                backgroundColor: '#F7F9F4',
                padding: '0.85rem 1rem',
                borderRadius: '8px',
                border: '1px solid #E1E7DD',
              }}
            >
              <p
                style={{
                  fontSize: '0.72rem',
                  fontWeight: '700',
                  color: '#697586',
                  textTransform: 'uppercase',
                  margin: 0,
                  letterSpacing: '0.04em',
                }}
              >
                {info.label}
              </p>

              <p
                style={{
                  fontSize: '0.875rem',
                  fontWeight: '600',
                  color: '#202734',
                  margin: '0.25rem 0 0 0',
                  wordBreak: 'break-word',
                }}
              >
                {info.value}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
