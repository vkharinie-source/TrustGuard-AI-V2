import React from 'react';
import { Menu, Shield, Activity } from './Icons';

export default function Topbar({ title, description, setMobileOpen, systemStatus = 'healthy' }) {
  const isHealthy = systemStatus === 'healthy';

  return (
    <header
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '1.15rem 2rem',
        backgroundColor: '#FFFFFF',
        borderBottom: '1px solid #E5DDEC',
        boxShadow: '0 2px 8px rgba(35, 21, 51, 0.03)',
        position: 'sticky',
        top: 0,
        zIndex: 30,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <button
          onClick={() => setMobileOpen && setMobileOpen(true)}
          style={{
            background: '#EDE4F5',
            border: '1px solid #D8C6E6',
            color: '#7F5D99',
            padding: '0.5rem',
            borderRadius: '9px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.15s ease',
          }}
          className="mobile-menu-trigger"
        >
          <Menu size={20} />
        </button>

        <div>
          <h2
            style={{
              fontSize: '1.2rem',
              fontWeight: '800',
              color: '#221730',
              letterSpacing: '-0.01em',
              margin: 0,
            }}
          >
            {title || 'SECURITY DASHBOARD'}
          </h2>
          <p
            style={{
              fontSize: '0.8rem',
              color: '#6D5E7B',
              margin: '0.15rem 0 0 0',
            }}
          >
            {description || 'Monitor and analyze suspicious digital content across your TrustGuard AI security modules.'}
          </p>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        {/* System Online Status Badge */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.4rem 0.85rem',
            borderRadius: '9999px',
            backgroundColor: isHealthy ? '#EAF7F1' : '#FDEFF2',
            border: `1px solid ${isHealthy ? '#B7E7D1' : '#F8BDC8'}`,
            boxShadow: '0 1px 3px rgba(0,0,0,0.02)',
          }}
        >
          <span
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: isHealthy ? '#3DA779' : '#D9536F',
              boxShadow: isHealthy ? '0 0 8px rgba(61, 167, 121, 0.7)' : '0 0 8px rgba(217, 83, 111, 0.7)',
            }}
          />
          <span
            style={{
              fontSize: '0.75rem',
              fontWeight: '700',
              color: isHealthy ? '#1D6E4B' : '#A32943',
              letterSpacing: '0.02em',
            }}
          >
            {isHealthy ? 'System Online' : 'Service Offline'}
          </span>
        </div>
      </div>
    </header>
  );
}
