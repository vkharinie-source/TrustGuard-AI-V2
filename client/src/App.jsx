import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Topbar from './components/Topbar';
import Login from './pages/LoginPage';
import Register from './pages/RegisterPage';
import Onboarding from './pages/Onboarding';
import Landing from './pages/Landing';
import Dashboard from './pages/Dashboard';
import DeepFake from './pages/DeepFake';
import VideoAnalysis from './pages/VideoAnalysis';
import Website from './pages/Website';
import Call from './pages/Call';
import Email from './pages/Email';
import Message from './pages/Message';
import Application from './pages/Application';
import Internship from './pages/Internship';
import Job from './pages/Job';
import News from './pages/News';
import HistoryPage from './pages/HistoryPage';
import SettingsPage from './pages/SettingsView';
import { api } from './services/api';
import './index.css';

const PAGE_META = {
  '/dashboard': {
    title: 'SECURITY DASHBOARD',
    description: 'Monitor and analyze suspicious digital content across your TrustGuard AI security modules.',
  },
  '/deepfake': {
    title: 'DEEPFAKE IMAGE DETECTION',
    description: 'Analyze image uploads with MobileNetV2 V3 to detect AI-generated manipulation.',
  },
  '/video': {
    title: 'VIDEO ANALYSIS',
    description: 'Inspect representative video frames for possible visual manipulation.',
  },
  '/website': {
    title: 'WEBSITE THREAT SCANNER',
    description: 'Audit website URLs and domains for phishing, malicious scripts and fraud patterns.',
  },
  '/call': {
    title: 'PHONE CALL SCAM DETECTOR',
    description: 'Inspect caller profiles and transcripts for voice cloning and banking scam tactics.',
  },
  '/email': {
    title: 'PHISHING EMAIL INSPECTOR',
    description: 'Analyze email headers, senders and body text for spear-phishing attacks.',
  },
  '/message': {
    title: 'SMS & MESSAGE SCANNER',
    description: 'Detect financial scam urgency tactics and OTP fraud in text messages.',
  },
  '/application': {
    title: 'APPLICATION METADATA AUDITOR',
    description: 'Audit mobile app packages and unverified APKs for malicious permission risks.',
  },
  '/internship': {
    title: 'FAKE INTERNSHIP DETECTOR',
    description: 'Verify internship offer letters and prevent upfront registration fee scams.',
  },
  '/job': {
    title: 'FAKE JOB OFFER DETECTOR',
    description: 'Verify corporate recruitment contracts and flag advance check fraud.',
  },
  '/news': {
    title: 'NEWS & ARTICLE VERIFIER',
    description: 'Verify viral headlines and articles for misinformation and deceptive content.',
  },
  '/history': {
    title: 'SCAN HISTORY',
    description: 'Persistent audit log of all threat inspections and confidence records.',
  },
  '/settings': {
    title: 'PLATFORM SETTINGS',
    description: 'Account configuration and TrustGuard AI platform preferences.',
  },
};

function AppShell({ children, currentUser, onSignOut, systemStatus }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();
  const meta = PAGE_META[location.pathname] || PAGE_META['/dashboard'];

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: '#F8F6FA', color: '#221730' }}>
      {/* Sidebar */}
      <Sidebar
        mobileOpen={mobileOpen}
        setMobileOpen={setMobileOpen}
        currentUser={currentUser}
        onSignOut={onSignOut}
      />

      {/* Main Content Area */}
      <div
        style={{
          marginLeft: '260px',
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          minHeight: '100vh',
          overflow: 'hidden',
        }}
        className="main-content"
      >
        {/* Topbar */}
        <Topbar
          title={meta.title}
          description={meta.description}
          setMobileOpen={setMobileOpen}
          systemStatus={systemStatus}
        />

        {/* Page Content */}
        <main style={{ flex: 1, padding: '2rem', overflowY: 'auto' }}>
          {children}
        </main>
      </div>
    </div>
  );
}

export default function App() {
  const [currentUser, setCurrentUser] = useState(null);
  const [systemStatus, setSystemStatus] = useState('healthy');
  const [onboardingComplete, setOnboardingComplete] = useState(
    () => localStorage.getItem('trustguard_onboarding_completed') === 'true'
  );

  // Restore saved user
  useEffect(() => {
    const savedUser = localStorage.getItem('trustguard_user');
    const savedToken = localStorage.getItem('trustguard_token');
    if (savedUser && savedToken) {
      try {
        setCurrentUser(JSON.parse(savedUser));
      } catch {
        // ignore parse error
      }
    }
  }, []);

  // Persist user
  useEffect(() => {
    if (currentUser) {
      localStorage.setItem('trustguard_user', JSON.stringify(currentUser));
    }
  }, [currentUser]);

  // Check system health
  useEffect(() => {
    const checkHealth = async () => {
      const health = await api.checkHealth();
      setSystemStatus(health.status === 'healthy' ? 'healthy' : 'offline');
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleSignOut = () => {
    localStorage.removeItem('trustguard_token');
    localStorage.removeItem('trustguard_user');
    setCurrentUser(null);
  };

  const handleSetUser = (user) => {
    setCurrentUser(user);
  };

  const isAuthenticated = !!currentUser && !!localStorage.getItem('trustguard_token');

  const protectedRoute = (element) => {
    if (!onboardingComplete) return <Navigate to="/onboarding" replace />;
    return isAuthenticated ? element : <Navigate to="/login" replace />;
  };

  return (
    <Router>
      <Routes>
        {/* Public Routes */}
        <Route path="/onboarding" element={
          onboardingComplete ? <Navigate to={isAuthenticated ? '/dashboard' : '/login'} replace /> :
            <Onboarding onComplete={() => setOnboardingComplete(true)} />
        } />
        <Route path="/login" element={<Login setCurrentUser={handleSetUser} />} />
        <Route path="/register" element={<Register setCurrentUser={handleSetUser} />} />

        {/* Protected Routes */}
        <Route path="/" element={isAuthenticated ? protectedRoute(
          <AppShell currentUser={currentUser} onSignOut={handleSignOut} systemStatus={systemStatus}>
            <Dashboard />
          </AppShell>
        ) : <Landing onboardingComplete={onboardingComplete} />} />
        <Route path="/dashboard" element={protectedRoute(
          <AppShell currentUser={currentUser} onSignOut={handleSignOut} systemStatus={systemStatus}>
            <Dashboard />
          </AppShell>
        )} />
        <Route path="/deepfake" element={protectedRoute(
          <AppShell currentUser={currentUser} onSignOut={handleSignOut} systemStatus={systemStatus}>
            <DeepFake />
          </AppShell>
        )} />
        <Route path="/video" element={protectedRoute(
          <AppShell currentUser={currentUser} onSignOut={handleSignOut} systemStatus={systemStatus}>
            <VideoAnalysis />
          </AppShell>
        )} />
        <Route path="/website" element={protectedRoute(
          <AppShell currentUser={currentUser} onSignOut={handleSignOut} systemStatus={systemStatus}>
            <Website />
          </AppShell>
        )} />
        <Route path="/call" element={protectedRoute(
          <AppShell currentUser={currentUser} onSignOut={handleSignOut} systemStatus={systemStatus}>
            <Call />
          </AppShell>
        )} />
        <Route path="/email" element={protectedRoute(
          <AppShell currentUser={currentUser} onSignOut={handleSignOut} systemStatus={systemStatus}>
            <Email />
          </AppShell>
        )} />
        <Route path="/message" element={protectedRoute(
          <AppShell currentUser={currentUser} onSignOut={handleSignOut} systemStatus={systemStatus}>
            <Message />
          </AppShell>
        )} />
        <Route path="/application" element={protectedRoute(
          <AppShell currentUser={currentUser} onSignOut={handleSignOut} systemStatus={systemStatus}>
            <Application />
          </AppShell>
        )} />
        <Route path="/internship" element={protectedRoute(
          <AppShell currentUser={currentUser} onSignOut={handleSignOut} systemStatus={systemStatus}>
            <Internship />
          </AppShell>
        )} />
        <Route path="/job" element={protectedRoute(
          <AppShell currentUser={currentUser} onSignOut={handleSignOut} systemStatus={systemStatus}>
            <Job />
          </AppShell>
        )} />
        <Route path="/news" element={protectedRoute(
          <AppShell currentUser={currentUser} onSignOut={handleSignOut} systemStatus={systemStatus}>
            <News />
          </AppShell>
        )} />
        <Route path="/history" element={protectedRoute(
          <AppShell currentUser={currentUser} onSignOut={handleSignOut} systemStatus={systemStatus}>
            <HistoryPage />
          </AppShell>
        )} />
        <Route path="/settings" element={protectedRoute(
          <AppShell currentUser={currentUser} onSignOut={handleSignOut} systemStatus={systemStatus}>
            <SettingsPage currentUser={currentUser} />
          </AppShell>
        )} />

        {/* Catch-all */}
        <Route path="*" element={<Navigate to={isAuthenticated ? '/dashboard' : '/'} replace />} />
      </Routes>
    </Router>
  );
}
