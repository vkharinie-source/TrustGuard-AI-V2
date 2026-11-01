import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Shield,
  LayoutDashboard,
  ImageIcon,
  Globe,
  PhoneCall,
  Mail,
  MessageSquare,
  Briefcase,
  GraduationCap,
  FileCheck,
  Newspaper,
  Video,
  History,
  Settings,
  LogOut,
  User,
  X
} from './Icons';

export default function Sidebar({ mobileOpen, setMobileOpen, currentUser, onSignOut }) {
  const navigate = useNavigate();
  const location = useLocation();

  const mainNav = [
    { name: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
  ];

  const aiSecurityNav = [
    { name: 'DeepFake Image', path: '/deepfake', icon: ImageIcon },
    { name: 'Video Analysis', path: '/video', icon: Video },
    { name: 'Website', path: '/website', icon: Globe },
    { name: 'Phone Calls', path: '/call', icon: PhoneCall },
    { name: 'Emails', path: '/email', icon: Mail },
    { name: 'Messages', path: '/message', icon: MessageSquare },
    { name: 'Applications', path: '/application', icon: FileCheck },
    { name: 'Internships', path: '/internship', icon: GraduationCap },
    { name: 'Job Offers', path: '/job', icon: Briefcase },
    { name: 'News Verification', path: '/news', icon: Newspaper },
  ];

  const managementNav = [
    { name: 'History', path: '/history', icon: History },
    { name: 'Settings', path: '/settings', icon: Settings },
  ];

  const handleNavClick = (path) => {
    navigate(path);
    if (setMobileOpen) setMobileOpen(false);
  };

  const renderNavItem = (item) => {
    const isActive = location.pathname === item.path || (item.path === '/dashboard' && location.pathname === '/');
    const IconComponent = item.icon;

    return (
      <button
        key={item.path}
        onClick={() => handleNavClick(item.path)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          width: '100%',
          padding: '0.65rem 0.95rem',
          borderRadius: '9px',
          fontSize: '0.85rem',
          fontWeight: isActive ? '700' : '500',
          color: isActive ? '#FFFFFF' : '#B3A5C1',
          backgroundColor: isActive ? 'rgba(127, 93, 153, 0.28)' : 'transparent',
          borderLeft: isActive ? '3px solid #7F5D99' : '3px solid transparent',
          cursor: 'pointer',
          borderTop: 'none',
          borderRight: 'none',
          borderBottom: 'none',
          outline: 'none',
          textAlign: 'left',
          transition: 'all 0.15s ease',
          boxShadow: isActive ? 'inset 0 0 12px rgba(127, 93, 153, 0.15)' : 'none',
        }}
        onMouseEnter={(e) => {
          if (!isActive) {
            e.currentTarget.style.backgroundColor = 'rgba(127, 93, 153, 0.16)';
            e.currentTarget.style.color = '#F7F4FA';
          }
        }}
        onMouseLeave={(e) => {
          if (!isActive) {
            e.currentTarget.style.backgroundColor = 'transparent';
            e.currentTarget.style.color = '#B3A5C1';
          }
        }}
      >
        <IconComponent
          size={18}
          color={isActive ? '#EDE4F5' : '#8E7E9E'}
          style={{ flexShrink: 0, transition: 'color 0.15s ease' }}
        />
        <span style={{ flexGrow: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', letterSpacing: '0.01em' }}>
          {item.name}
        </span>
      </button>
    );
  };

  const sidebarContent = (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        backgroundColor: '#231533',
        borderRight: '1px solid rgba(255, 255, 255, 0.06)',
        width: '260px',
        padding: '1.25rem 1rem',
        boxSizing: 'border-box',
      }}
    >
      {/* Brand Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '1.75rem',
          paddingLeft: '0.4rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div
            style={{
              width: '38px',
              height: '38px',
              borderRadius: '11px',
              background: 'linear-gradient(135deg, #7F5D99 0%, #9E7DB7 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 4px 14px rgba(127, 93, 153, 0.45)',
            }}
          >
            <Shield size={20} color="#FFFFFF" />
          </div>
          <div>
            <h1
              style={{
                fontSize: '1.05rem',
                fontWeight: '800',
                color: '#FFFFFF',
                letterSpacing: '0.02em',
                lineHeight: '1.2',
              }}
            >
              TRUSTGUARD <span style={{ color: '#D8C6E6' }}>AI</span>
            </h1>
            <p style={{ fontSize: '0.68rem', color: '#B3A5C1', textTransform: 'uppercase', letterSpacing: '0.08em', margin: 0 }}>
              Digital Security
            </p>
          </div>
        </div>
        {setMobileOpen && (
          <button
            onClick={() => setMobileOpen(false)}
            style={{
              background: 'none',
              border: 'none',
              color: '#B3A5C1',
              cursor: 'pointer',
              display: 'none',
            }}
            className="mobile-close-btn"
          >
            <X size={20} />
          </button>
        )}
      </div>

      {/* Navigation Sections */}
      <div
        style={{
          flexGrow: 1,
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.25rem',
          paddingRight: '0.25rem',
        }}
      >
        {/* Main */}
        <div>
          <p
            style={{
              fontSize: '0.65rem',
              fontWeight: '700',
              color: '#A9B5C7',
              textTransform: 'uppercase',
              letterSpacing: '0.1em',
              marginBottom: '0.5rem',
              paddingLeft: '0.5rem',
            }}
          >
            MAIN
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            {mainNav.map(renderNavItem)}
          </div>
        </div>

        {/* AI Security */}
        <div>
          <p
            style={{
              fontSize: '0.65rem',
              fontWeight: '700',
              color: '#A9B5C7',
              textTransform: 'uppercase',
              letterSpacing: '0.1em',
              marginBottom: '0.5rem',
              paddingLeft: '0.5rem',
            }}
          >
            AI SECURITY
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            {aiSecurityNav.map(renderNavItem)}
          </div>
        </div>

        {/* Management */}
        <div>
          <p
            style={{
              fontSize: '0.65rem',
              fontWeight: '700',
              color: '#A9B5C7',
              textTransform: 'uppercase',
              letterSpacing: '0.1em',
              marginBottom: '0.5rem',
              paddingLeft: '0.5rem',
            }}
          >
            MANAGEMENT
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            {managementNav.map(renderNavItem)}
          </div>
        </div>
      </div>

      {/* Footer / User Profile */}
      <div
        style={{
          marginTop: 'auto',
          paddingTop: '1rem',
          borderTop: '1px solid rgba(255, 255, 255, 0.08)',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0.6rem 0.65rem',
            borderRadius: '10px',
            backgroundColor: 'rgba(127, 93, 153, 0.12)',
            border: '1px solid rgba(127, 93, 153, 0.2)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
            <div
              style={{
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                backgroundColor: '#332047',
                border: '1.5px solid #7F5D99',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <User size={16} color="#D8C6E6" />
            </div>
            <div style={{ overflow: 'hidden' }}>
              <p
                style={{
                  fontSize: '0.8rem',
                  fontWeight: '600',
                  color: '#F7F4FA',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  margin: 0,
                }}
              >
                {currentUser?.full_name || 'harinie vk'}
              </p>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', marginTop: '0.15rem' }}>
                <span
                  style={{
                    width: '6px',
                    height: '6px',
                    borderRadius: '50%',
                    backgroundColor: '#3DA779',
                    boxShadow: '0 0 6px rgba(61, 167, 121, 0.6)',
                  }}
                />
                <span style={{ fontSize: '0.65rem', color: '#B7E7D1', fontWeight: '600' }}>
                  Authenticated
                </span>
              </div>
            </div>
          </div>
          <button
            onClick={onSignOut}
            title="Sign Out"
            style={{
              background: 'none',
              border: 'none',
              color: '#B3A5C1',
              cursor: 'pointer',
              padding: '0.35rem',
              borderRadius: '6px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = '#D9536F';
              e.currentTarget.style.backgroundColor = 'rgba(217, 83, 111, 0.15)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = '#B3A5C1';
              e.currentTarget.style.backgroundColor = 'transparent';
            }}
          >
            <LogOut size={16} />
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop Sidebar */}
      <aside
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          bottom: 0,
          width: '260px',
          zIndex: 40,
        }}
        className="hidden-mobile"
      >
        {sidebarContent}
      </aside>

      {/* Mobile Drawer Backdrop */}
      {mobileOpen && (
        <div
          onClick={() => setMobileOpen(false)}
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            backdropFilter: 'blur(4px)',
            zIndex: 50,
          }}
        />
      )}

      {/* Mobile Slide-Out Sidebar */}
      <div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          bottom: 0,
          width: '260px',
          zIndex: 51,
          transform: mobileOpen ? 'translateX(0)' : 'translateX(-100%)',
          transition: 'transform 0.3s ease-in-out',
        }}
      >
        {sidebarContent}
      </div>
    </>
  );
}
