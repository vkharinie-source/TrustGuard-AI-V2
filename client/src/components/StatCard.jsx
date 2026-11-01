import React from 'react';

export default function StatCard({ title, value, subtitle, icon: IconComponent, color = 'indigo' }) {
  const colorMap = {
    indigo: {
      bg: '#EDE4F5',
      border: '#D8C6E6',
      text: '#7F5D99',
      accent: '#7F5D99',
    },
    emerald: {
      bg: '#EAF7F1',
      border: '#B7E7D1',
      text: '#3DA779',
      accent: '#3DA779',
    },
    rose: {
      bg: '#FDEFF2',
      border: '#F8BDC8',
      text: '#D9536F',
      accent: '#D9536F',
    },
    amber: {
      bg: '#FEF7ED',
      border: '#F8D8A7',
      text: '#D98928',
      accent: '#D98928',
    },
  };

  const scheme = colorMap[color] || colorMap.indigo;

  return (
    <div
      style={{
        backgroundColor: '#FFFFFF',
        border: '1px solid #E5DDEC',
        borderRadius: '14px',
        padding: '1.4rem 1.5rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        boxShadow: '0 2px 10px rgba(35, 21, 51, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02)',
        position: 'relative',
        overflow: 'hidden',
        transition: 'all 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
        cursor: 'default',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = 'translateY(-3px)';
        e.currentTarget.style.borderColor = scheme.border;
        e.currentTarget.style.boxShadow = '0 10px 24px -4px rgba(127, 93, 153, 0.12), 0 3px 8px -2px rgba(0, 0, 0, 0.04)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = 'translateY(0)';
        e.currentTarget.style.borderColor = '#E5DDEC';
        e.currentTarget.style.boxShadow = '0 2px 10px rgba(35, 21, 51, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02)';
      }}
    >
      {/* Top subtle color indicator line */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '3px',
          backgroundColor: scheme.accent,
          opacity: 0.85,
        }}
      />

      <div>
        <p
          style={{
            fontSize: '0.72rem',
            fontWeight: '700',
            color: '#6D5E7B',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            margin: 0,
          }}
        >
          {title}
        </p>
        <h3
          style={{
            fontSize: '2rem',
            fontWeight: '800',
            color: '#221730',
            margin: '0.35rem 0 0.2rem 0',
            letterSpacing: '-0.02em',
            lineHeight: 1.1,
          }}
        >
          {value !== undefined && value !== null ? value : 0}
        </h3>
        {subtitle && (
          <p style={{ fontSize: '0.78rem', color: '#6D5E7B', margin: 0, fontWeight: '500' }}>
            {subtitle}
          </p>
        )}
      </div>

      {IconComponent && (
        <div
          style={{
            width: '48px',
            height: '48px',
            borderRadius: '12px',
            backgroundColor: scheme.bg,
            border: `1px solid ${scheme.border}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
            boxShadow: `0 4px 12px ${scheme.bg}`,
          }}
        >
          <IconComponent size={22} color={scheme.text} />
        </div>
      )}
    </div>
  );
}
