import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Shield, Lock, ImageIcon, Video, Globe, Mail, MessageSquare, PhoneCall, FileCheck, GraduationCap, Briefcase, Newspaper, CheckCircle, ArrowRight } from '../components/Icons';

const screens = [
  {
    badge: 'TRUSTGUARD AI',
    title: 'Protect Your Digital World',
    subtitle: 'One intelligent workspace to analyze suspicious digital content before it becomes a threat.',
    features: [
      { icon: Shield, label: 'AI-Powered Security', desc: 'Deep learning models trained on real-world threats' },
      { icon: CheckCircle, label: 'Multi-Layer Threat Detection', desc: 'Analyze images, text, URLs, emails and more' },
      { icon: Globe, label: 'Centralized Analysis', desc: 'One platform for all your digital security needs' },
    ],
  },
  {
    badge: 'AI-POWERED IMAGE SECURITY',
    title: 'Detect Manipulated Images',
    subtitle: 'Analyze images using the trained MobileNetV2 V3 detection model and receive a clear authenticity assessment.',
    features: [
      { icon: ImageIcon, label: 'AI-powered image analysis', desc: 'Trained MobileNetV2 V3 deep learning model' },
      { icon: CheckCircle, label: 'Fast authenticity assessment', desc: 'Get results in seconds with confidence scores' },
      { icon: Shield, label: 'Clear detection results', desc: 'Detailed reports with recommendations' },
    ],
  },
  {
    badge: 'MULTI-LAYER DIGITAL PROTECTION',
    title: 'Analyze More Than Images',
    subtitle: 'TrustGuard AI protects you across every digital channel.',
    cards: [
      { icon: ImageIcon, name: 'DeepFake Images', color: '#818CF8' },
      { icon: Video, name: 'Video Analysis', color: '#83DED5' },
      { icon: Globe, name: 'Website Security', color: '#60A5FA' },
      { icon: Mail, name: 'Email Security', color: '#F472B6' },
      { icon: MessageSquare, name: 'Message Security', color: '#34D399' },
      { icon: PhoneCall, name: 'Phone Call Analysis', color: '#FBBF24' },
      { icon: FileCheck, name: 'Application Analysis', color: '#A78BFA' },
      { icon: GraduationCap, name: 'Internship Analysis', color: '#F87171' },
      { icon: Briefcase, name: 'Job Offer Analysis', color: '#38BDF8' },
      { icon: Newspaper, name: 'News Verification', color: '#FB923C' },
    ],
  },
  {
    badge: 'YOUR SECURITY WORKSPACE',
    title: 'Analyze. Understand. Verify.',
    lines: ['Ask.', 'Listen.', 'Download.'],
    subtitle: 'Review analysis reports, ask questions about suspicious content, listen to AI explanations and download professional reports.',
    isFinal: true,
  },
];

export default function Onboarding({ onComplete }) {
  const [step, setStep] = useState(0);
  const navigate = useNavigate();
  const screen = screens[step];

  const complete = () => {
    localStorage.setItem('trustguard_onboarding_completed', 'true');
    onComplete?.();
  };

  const skip = () => { complete(); navigate('/login'); };
  const goSignIn = () => { complete(); navigate('/login'); };
  const goCreateAccount = () => { complete(); navigate('/register'); };

  return (
    <div className="onboarding-page">
      <div className="onboarding-layout">
        <aside className="onboarding-rail">
          <div className="onboarding-brand">
            <span><Shield size={21} /></span>
            <div><strong>TRUSTGUARD AI</strong><small>DIGITAL SECURITY</small></div>
          </div>
          <div className="onboarding-rail-copy">
            <span>YOUR FIRST LOOK</span>
            <h2>Security starts with a clearer view.</h2>
            <p>Get oriented, then step into your own analysis workspace.</p>
          </div>
          <nav className="onboarding-step-list" aria-label="Onboarding progress">
            {['Introduction', 'Image analysis', 'Protection layers', 'Your workspace'].map((label, index) => (
              <button
                type="button"
                key={label}
                className={step === index ? 'is-current' : step > index ? 'is-complete' : ''}
                onClick={() => setStep(index)}
                aria-current={step === index ? 'step' : undefined}
              >
                <span>{String(index + 1).padStart(2, '0')}</span>{label}
              </button>
            ))}
          </nav>
          <div className="onboarding-rail-note"><Lock size={15} /> Your account stays yours. Sign in to continue.</div>
        </aside>

        <main className="onboarding-content">
          <div className="onboarding-step-count">STEP {String(step + 1).padStart(2, '0')} <span>/</span> 04</div>
          {/* Badge */}
          <div style={{
            fontSize: '0.7rem', fontWeight: '700', letterSpacing: '0.12em',
            color: '#818CF8', marginBottom: '1.5rem',
            padding: '0.35rem 1rem', borderRadius: '9999px',
            backgroundColor: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.25)',
          }}>
            {screen.badge}
          </div>

          {/* Title */}
          <h1 style={{
            fontSize: '2.1rem', fontWeight: '800',
            color: '#F9FAFB', lineHeight: '1.25', marginBottom: '0.5rem',
          }}>
            {screen.title}
          </h1>

          {/* Extra title lines */}
          {screen.lines && screen.lines.map((line, i) => (
            <span key={i} style={{
              fontSize: '1.45rem', fontWeight: '700',
              color: '#818CF8', display: 'block', lineHeight: '1.4',
            }}>
              {line}
            </span>
          ))}

          {/* Subtitle */}
          <p style={{
            fontSize: '0.95rem', color: '#9CA3AF', maxWidth: '500px',
            lineHeight: '1.7', marginTop: '1rem', marginBottom: '2rem',
          }}>
            {screen.subtitle}
          </p>

          {/* Feature list */}
          {screen.features && (
            <div style={{
              display: 'flex', flexDirection: 'column', gap: '1rem', width: '100%',
              maxWidth: '480px', marginBottom: '2rem',
            }}>
              {screen.features.map((f, i) => {
                const Icon = f.icon;
                return (
                  <div key={i} style={{
                    display: 'flex', alignItems: 'center', gap: '1rem',
                    backgroundColor: '#131B2E', border: '1px solid rgba(255,255,255,0.07)',
                    borderRadius: '12px', padding: '1rem 1.25rem', textAlign: 'left',
                  }}>
                    <div style={{
                      width: '42px', height: '42px', borderRadius: '10px',
                      backgroundColor: 'rgba(99,102,241,0.12)',
                      border: '1px solid rgba(99,102,241,0.25)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                    }}>
                      <Icon size={20} color="#818CF8" />
                    </div>
                    <div>
                      <p style={{ fontSize: '0.875rem', fontWeight: '600', color: '#F9FAFB', margin: 0 }}>{f.label}</p>
                      <p style={{ fontSize: '0.78rem', color: '#6B7280', margin: '0.15rem 0 0 0' }}>{f.desc}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Module cards grid */}
          {screen.cards && (
            <div style={{
              display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem',
              width: '100%', maxWidth: '520px', marginBottom: '2rem',
            }}>
              {screen.cards.map((c, i) => {
                const Icon = c.icon;
                return (
                  <div key={i} style={{
                    backgroundColor: '#131B2E', border: '1px solid rgba(255,255,255,0.07)',
                    borderRadius: '10px', padding: '0.9rem 0.5rem',
                    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem',
                  }}>
                    <Icon size={22} color={c.color} />
                    <span style={{ fontSize: '0.7rem', fontWeight: '600', color: '#E5E7EB', textAlign: 'center' }}>
                      {c.name}
                    </span>
                  </div>
                );
              })}
            </div>
          )}

          {/* Buttons */}
          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', justifyContent: 'center' }}>
            {step === 0 && (
              <>
                <button className="cyber-button-primary" onClick={() => setStep(1)} style={{ minWidth: '160px' }}>
                  Get Started <ArrowRight size={16} />
                </button>
                <button className="cyber-button-secondary" onClick={skip}>
                  Skip
                </button>
              </>
            )}

            {step > 0 && !screen.isFinal && (
              <>
                <button className="cyber-button-secondary" onClick={() => setStep(step - 1)}>
                  Back
                </button>
                <button className="cyber-button-primary" onClick={() => setStep(step + 1)}>
                  Continue <ArrowRight size={16} />
                </button>
              </>
            )}

            {screen.isFinal && (
              <>
                <button className="cyber-button-secondary" onClick={() => setStep(step - 1)}>
                  Back
                </button>
                <button className="cyber-button-primary" onClick={goCreateAccount} style={{ minWidth: '170px' }}>
                  Create Account
                </button>
                <button className="cyber-button-secondary" onClick={goSignIn}>
                  Sign In
                </button>
              </>
            )}
          </div>

          {/* Progress dots */}
          <div className="onboarding-progress-dots" aria-label={`Step ${step + 1} of 4`}>
            {screens.map((_, i) => (
              <button
                type="button"
                key={i}
                aria-label={`Go to step ${i + 1}`}
                aria-current={i === step ? 'step' : undefined}
                onClick={() => setStep(i)}
                style={{
                  width: i === step ? '24px' : '8px', height: '8px',
                  borderRadius: '9999px', cursor: 'pointer',
                  backgroundColor: i === step ? '#6366F1' : 'rgba(255,255,255,0.15)',
                  transition: 'all 0.3s ease',
                  border: 0,
                  padding: 0,
                }}
              />
            ))}
          </div>
        </main>
      </div>
    </div>
  );
}
