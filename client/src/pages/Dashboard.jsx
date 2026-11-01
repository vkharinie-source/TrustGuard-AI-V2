import React, { useEffect, useState } from 'react';
import StatCard from '../components/StatCard';
import QuickCard from '../components/QuickCard';
import DashboardCharts from '../components/DashboardCharts';
import {
  Shield,
  Activity,
  CheckCircle,
  AlertTriangle,
  XCircle,
  ImageIcon,
  Globe,
  PhoneCall,
  Mail,
  MessageSquare,
  Newspaper,
  Briefcase,
  GraduationCap,
  FileCheck,
  RefreshCw
} from '../components/Icons';
import { api } from '../services/api';

export default function Dashboard() {
  const [stats, setStats] = useState({
    total: 0,
    safe: 0,
    threats_detected: 0,
    suspicious: 0,
    fake: 0,
  });
  const [recentScans, setRecentScans] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchDashboardData = async () => {
    setLoading(true);
    try {
      const data = await api.getDashboardStats();
      if (data && data.stats) {
        setStats(data.stats);
      }
      if (data && data.recent) {
        setRecentScans(data.recent);
      } else {
        const historyData = await api.getHistory();
        setRecentScans(historyData.slice(0, 5));
      }
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const quickModules = [
    {
      title: 'DeepFake Image',
      description: 'Analyze face uploads using MobileNetV2 neural vision to detect AI manipulation.',
      icon: ImageIcon,
      path: '/deepfake',
      badge: 'Vision AI',
    },
    {
      title: 'Website Scanner',
      description: 'Audit URLs for phishing indicators, deceptive domains, and SSL risk factors.',
      icon: Globe,
      path: '/website',
      badge: 'URL Audit',
    },
    {
      title: 'Phone Calls',
      description: 'Inspect voice transcripts and caller ID profiles for voice cloning scams.',
      icon: PhoneCall,
      path: '/call',
      badge: 'Voice AI',
    },
    {
      title: 'Email Security',
      description: 'Scan email headers, senders, and message bodies for phishing attacks.',
      icon: Mail,
      path: '/email',
      badge: 'Phishing',
    },
    {
      title: 'SMS & Messages',
      description: 'Analyze text messages and chat content for financial scam urgency tactics.',
      icon: MessageSquare,
      path: '/message',
      badge: 'Text Scam',
    },
    {
      title: 'News Verification',
      description: 'Verify news articles and viral headlines for misinformation and clickbait.',
      icon: Newspaper,
      path: '/news',
      badge: 'Fact Check',
    },
    {
      title: 'Job Offers',
      description: 'Verify corporate recruitment letters and employment contract legitimacy.',
      icon: Briefcase,
      path: '/job',
      badge: 'Career',
    },
    {
      title: 'Internships',
      description: 'Inspect internship listings to prevent upfront fee payment fraud.',
      icon: GraduationCap,
      path: '/internship',
      badge: 'Student',
    },
    {
      title: 'Applications',
      description: 'Audit mobile application permissions and package metadata for malware.',
      icon: FileCheck,
      path: '/application',
      badge: 'App Audit',
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Hero Intelligence Status Banner */}
      <div
        style={{
          background: 'linear-gradient(135deg, rgba(127, 93, 153, 0.1) 0%, rgba(158, 125, 183, 0.05) 100%)',
          border: '1px solid rgba(127, 93, 153, 0.22)',
          borderRadius: '16px',
          padding: '1.4rem 1.75rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '1.25rem',
          boxShadow: '0 2px 10px rgba(127, 93, 153, 0.05)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div
            style={{
              width: '46px',
              height: '46px',
              borderRadius: '12px',
              background: 'linear-gradient(135deg, #7F5D99 0%, #9E7DB7 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 4px 14px rgba(127, 93, 153, 0.35)',
              flexShrink: 0,
            }}
          >
            <Shield size={24} color="#FFFFFF" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.2rem' }}>
              <h2 style={{ fontSize: '1.15rem', fontWeight: '800', color: '#221730', margin: 0, letterSpacing: '-0.01em' }}>
                TrustGuard AI Threat Intelligence Engine
              </h2>
              <span
                style={{
                  fontSize: '0.68rem',
                  fontWeight: '700',
                  padding: '0.15rem 0.55rem',
                  borderRadius: '9999px',
                  backgroundColor: '#EAF7F1',
                  color: '#1D6E4B',
                  border: '1px solid #B7E7D1',
                }}
              >
                9 Modules Online
              </span>
            </div>
            <p style={{ fontSize: '0.825rem', color: '#6D5E7B', margin: 0 }}>
              MobileNetV2 neural vision & multi-layer threat heuristics actively monitoring digital inputs.
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <button
            onClick={fetchDashboardData}
            className="cyber-button-secondary"
            style={{ padding: '0.55rem 1rem', fontSize: '0.8rem' }}
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            Refresh Telemetry
          </button>
        </div>
      </div>

      {/* Overview Stats Bar */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', marginBottom: '0.25rem' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#7F5D99' }} />
              <span style={{ fontSize: '0.72rem', fontWeight: '800', color: '#7F5D99', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                REAL-TIME TELEMETRY
              </span>
            </div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: '800', color: '#221730', margin: 0, letterSpacing: '-0.01em' }}>
              Security Overview & Metric Counters
            </h3>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1.1rem' }}>
          <StatCard
            title="Total Scans"
            value={stats.total}
            subtitle="Inspections completed"
            icon={Activity}
            color="indigo"
          />
          <StatCard
            title="Safe Content"
            value={stats.safe}
            subtitle="Verified authentic items"
            icon={CheckCircle}
            color="emerald"
          />
          <StatCard
            title="Threats Detected"
            value={stats.threats_detected || (stats.fake + stats.suspicious)}
            subtitle="Malicious or scam items"
            icon={XCircle}
            color="rose"
          />
          <StatCard
            title="Suspicious / Warnings"
            value={stats.suspicious || 0}
            subtitle="Flagged for manual review"
            icon={AlertTriangle}
            color="amber"
          />
        </div>
      </div>

      {/* Interactive Visual Analytics & Trends */}
      <DashboardCharts stats={stats} recentScans={recentScans} />

      {/* Quick Detection Modules */}
      <div>
        <div style={{ marginBottom: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', marginBottom: '0.25rem' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#7F5D99' }} />
            <span style={{ fontSize: '0.72rem', fontWeight: '800', color: '#7F5D99', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              MULTI-MODAL VERIFICATION
            </span>
          </div>
          <h3 style={{ fontSize: '1.15rem', fontWeight: '800', color: '#221730', margin: 0, letterSpacing: '-0.01em' }}>
            AI Detection & Security Services
          </h3>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(285px, 1fr))', gap: '1.25rem' }}>
          {quickModules.map((mod) => (
            <QuickCard
              key={mod.path}
              title={mod.title}
              description={mod.description}
              icon={mod.icon}
              path={mod.path}
              badge={mod.badge}
            />
          ))}
        </div>
      </div>

      {/* Recent Activity Table */}
      <div
        style={{
          backgroundColor: '#FFFFFF',
          border: '1px solid #E5DDEC',
          borderRadius: '16px',
          padding: '1.75rem',
          boxShadow: '0 2px 12px rgba(35, 21, 51, 0.04)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', marginBottom: '0.25rem' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#7F5D99' }} />
              <span style={{ fontSize: '0.72rem', fontWeight: '800', color: '#7F5D99', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                AUDIT LOG
              </span>
            </div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: '800', color: '#221730', margin: 0 }}>
              Recent Scan History
            </h3>
            <p style={{ fontSize: '0.825rem', color: '#6D5E7B', margin: '0.2rem 0 0 0' }}>
              Latest threat inspection logs and evaluation confidence records across your security modules.
            </p>
          </div>
        </div>

        {loading ? (
          <div style={{ padding: '3rem', textAlign: 'center', color: '#6D5E7B', fontSize: '0.9rem' }}>
            <RefreshCw size={24} color="#7F5D99" className="animate-spin" style={{ marginBottom: '0.5rem' }} />
            <p style={{ margin: 0, fontWeight: '500' }}>Loading telemetry...</p>
          </div>
        ) : recentScans.length === 0 ? (
          <div style={{ padding: '3rem', textAlign: 'center', color: '#6D5E7B', fontSize: '0.875rem' }}>
            <Shield size={36} color="#CBB9D8" style={{ marginBottom: '0.75rem' }} />
            <p style={{ margin: 0, fontWeight: '600', color: '#221730' }}>No recent scan history available</p>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.8rem' }}>Select a module above to perform your first threat inspection.</p>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #E5DDEC', backgroundColor: '#FBF9FC' }}>
                  <th style={{ padding: '0.85rem 1rem', fontSize: '0.72rem', fontWeight: '700', color: '#6D5E7B', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                    Module / Target
                  </th>
                  <th style={{ padding: '0.85rem 1rem', fontSize: '0.72rem', fontWeight: '700', color: '#6D5E7B', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                    Result
                  </th>
                  <th style={{ padding: '0.85rem 1rem', fontSize: '0.72rem', fontWeight: '700', color: '#6D5E7B', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                    Confidence
                  </th>
                  <th style={{ padding: '0.85rem 1rem', fontSize: '0.72rem', fontWeight: '700', color: '#6D5E7B', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                    Timestamp
                  </th>
                </tr>
              </thead>
              <tbody>
                {recentScans.map((item, idx) => {
                  const pred = String(item.prediction || item.result || '').toUpperCase();
                  const isSafe = pred === 'REAL' || pred === 'SAFE';
                  const isDanger = pred === 'FAKE' || pred === 'SCAM' || pred === 'PHISHING';
                  const badgeStyle = isSafe
                    ? { bg: '#EAF7F1', color: '#1D6E4B', border: '#B7E7D1' }
                    : isDanger
                    ? { bg: '#FDEFF2', color: '#A32943', border: '#F8BDC8' }
                    : { bg: '#FEF7ED', color: '#8F5410', border: '#F8D8A7' };

                  return (
                    <tr
                      key={item.id || idx}
                      style={{ borderBottom: '1px solid #E5DDEC', transition: 'background-color 0.15s ease' }}
                    >
                      <td style={{ padding: '0.95rem 1rem' }}>
                        <span
                          style={{
                            fontSize: '0.72rem',
                            fontWeight: '700',
                            padding: '0.18rem 0.55rem',
                            borderRadius: '6px',
                            backgroundColor: '#EDE4F5',
                            color: '#5D3A7A',
                            border: '1px solid #D8C6E6',
                            display: 'inline-block',
                            marginBottom: '0.25rem',
                          }}
                        >
                          {item.category || item.analysis_type || 'Scan'}
                        </span>
                        <div style={{ fontSize: '0.825rem', color: '#221730', fontWeight: '500', maxWidth: '320px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {item.filename || item.input_value || 'Direct Input'}
                        </div>
                      </td>

                      <td style={{ padding: '0.95rem 1rem' }}>
                        <span
                          style={{
                            fontSize: '0.72rem',
                            fontWeight: '700',
                            padding: '0.25rem 0.65rem',
                            borderRadius: '9999px',
                            backgroundColor: badgeStyle.bg,
                            color: badgeStyle.color,
                            border: `1px solid ${badgeStyle.border}`,
                          }}
                        >
                          {pred}
                        </span>
                      </td>

                      <td style={{ padding: '0.95rem 1rem', fontSize: '0.85rem', color: '#221730', fontWeight: '700' }}>
                        {typeof item.confidence === 'number' ? `${item.confidence.toFixed(2)}%` : item.confidence || 'N/A'}
                      </td>

                      <td style={{ padding: '0.95rem 1rem', fontSize: '0.78rem', color: '#6D5E7B' }}>
                        {item.timestamp ? new Date(item.timestamp).toLocaleString() : 'Just now'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
