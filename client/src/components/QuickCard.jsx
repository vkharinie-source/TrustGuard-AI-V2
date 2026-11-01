import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight } from './Icons';

export default function QuickCard({ title, description, icon: IconComponent, path, badge }) {
  const navigate = useNavigate();

  return (
    <div
      onClick={() => navigate(path)}
      style={{
        backgroundColor: '#FFFFFF',
        border: '1px solid #E5DDEC',
        borderRadius: '14px',
        padding: '1.4rem',
        cursor: 'pointer',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        transition: 'all 0.22s cubic-bezier(0.16, 1, 0.3, 1)',
        boxShadow: '0 2px 10px rgba(35, 21, 51, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02)',
        position: 'relative',
        overflow: 'hidden',
      }}
      className="cyber-card-hover"
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = '#7F5D99';
        e.currentTarget.style.transform = 'translateY(-3px)';
        e.currentTarget.style.boxShadow = '0 12px 28px -4px rgba(127, 93, 153, 0.18), 0 3px 8px -2px rgba(0, 0, 0, 0.04)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = '#E5DDEC';
        e.currentTarget.style.transform = 'translateY(0)';
        e.currentTarget.style.boxShadow = '0 2px 10px rgba(35, 21, 51, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02)';
      }}
    >
      <div>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: '1rem',
          }}
        >
          <div
            style={{
              width: '44px',
              height: '44px',
              borderRadius: '11px',
              backgroundColor: '#EDE4F5',
              border: '1px solid #D8C6E6',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 2px 8px rgba(127, 93, 153, 0.12)',
            }}
          >
            {IconComponent && <IconComponent size={22} color="#7F5D99" />}
          </div>

          {badge && (
            <span
              style={{
                fontSize: '0.68rem',
                fontWeight: '700',
                padding: '0.22rem 0.65rem',
                borderRadius: '9999px',
                backgroundColor: '#EDE4F5',
                color: '#5D3A7A',
                border: '1px solid #D8C6E6',
                textTransform: 'uppercase',
                letterSpacing: '0.06em',
              }}
            >
              {badge}
            </span>
          )}
        </div>

        <h4
          style={{
            fontSize: '1.02rem',
            fontWeight: '700',
            color: '#221730',
            margin: '0 0 0.45rem 0',
            letterSpacing: '-0.01em',
          }}
        >
          {title}
        </h4>

        <p
          style={{
            fontSize: '0.825rem',
            color: '#6D5E7B',
            margin: 0,
            lineHeight: '1.55',
          }}
        >
          {description}
        </p>
      </div>

      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.45rem',
          marginTop: '1.35rem',
          fontSize: '0.825rem',
          fontWeight: '700',
          color: '#7F5D99',
          transition: 'gap 0.15s ease',
        }}
      >
        <span>Analyze Threat</span>
        <ArrowRight size={16} />
      </div>
    </div>
  );
}
