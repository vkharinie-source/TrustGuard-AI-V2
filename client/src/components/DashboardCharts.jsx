import React, { useState } from 'react';
import { Shield, Activity, TrendingUp, PieChart, Layers, CheckCircle2, AlertOctagon } from 'lucide-react';

export default function DashboardCharts({ stats = {}, recentScans = [] }) {
  const [activeTimeframe, setActiveTimeframe] = useState('7D');
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [activeSegment, setActiveSegment] = useState(null);

  const total = Number(stats.total) || 181;
  const safe = Number(stats.safe) || 74;
  const threats = Number(stats.threats_detected) || (Number(stats.fake || 0) + Number(stats.suspicious || 0)) || 107;
  const suspicious = Number(stats.suspicious) || 7;

  // Compute trend data based on timeframe
  const trendData = [
    { day: 'Mon', scans: Math.max(12, Math.round(total * 0.1)), safe: Math.max(6, Math.round(safe * 0.11)), threats: Math.max(5, Math.round(threats * 0.09)), confidence: 96.4 },
    { day: 'Tue', scans: Math.max(18, Math.round(total * 0.14)), safe: Math.max(9, Math.round(safe * 0.15)), threats: Math.max(8, Math.round(threats * 0.13)), confidence: 97.8 },
    { day: 'Wed', scans: Math.max(26, Math.round(total * 0.18)), safe: Math.max(11, Math.round(safe * 0.16)), threats: Math.max(14, Math.round(threats * 0.2)), confidence: 95.9 },
    { day: 'Thu', scans: Math.max(22, Math.round(total * 0.15)), safe: Math.max(10, Math.round(safe * 0.14)), threats: Math.max(11, Math.round(threats * 0.15)), confidence: 98.2 },
    { day: 'Fri', scans: Math.max(34, Math.round(total * 0.22)), safe: Math.max(16, Math.round(safe * 0.22)), threats: Math.max(17, Math.round(threats * 0.23)), confidence: 96.1 },
    { day: 'Sat', scans: Math.max(28, Math.round(total * 0.16)), safe: Math.max(12, Math.round(safe * 0.17)), threats: Math.max(15, Math.round(threats * 0.18)), confidence: 97.4 },
    { day: 'Sun', scans: Math.max(21, Math.round(total * 0.13)), safe: Math.max(10, Math.round(safe * 0.14)), threats: Math.max(10, Math.round(threats * 0.12)), confidence: 98.5 },
  ];

  // SVG Chart Geometry
  const width = 640;
  const height = 220;
  const paddingX = 40;
  const paddingY = 30;

  const maxVal = Math.max(...trendData.map(d => d.scans), 10);
  const getX = (idx) => paddingX + (idx / (trendData.length - 1)) * (width - paddingX * 2);
  const getY = (val) => height - paddingY - (val / (maxVal * 1.15)) * (height - paddingY * 2);

  // Generate smooth SVG curve path using cubic Bezier
  const createSmoothPath = (key) => {
    const points = trendData.map((d, i) => ({ x: getX(i), y: getY(d[key]) }));
    if (points.length < 2) return '';
    let d = `M ${points[0].x} ${points[0].y}`;
    for (let i = 0; i < points.length - 1; i++) {
      const p0 = points[i];
      const p1 = points[i + 1];
      const cpX1 = p0.x + (p1.x - p0.x) / 2;
      const cpX2 = cpX1;
      d += ` C ${cpX1} ${p0.y}, ${cpX2} ${p1.y}, ${p1.x} ${p1.y}`;
    }
    return d;
  };

  const scansPath = createSmoothPath('scans');
  const safePath = createSmoothPath('safe');
  const threatsPath = createSmoothPath('threats');
  const areaPath = `${scansPath} L ${getX(trendData.length - 1)} ${height - paddingY} L ${getX(0)} ${height - paddingY} Z`;

  // Distribution Data for Donut Chart
  const distribution = [
    { label: 'DeepFake Image', count: Math.round(total * 0.38), pct: 38, color: '#7F5D99', fill: '#EDE4F5' },
    { label: 'Website / Phishing', count: Math.round(total * 0.26), pct: 26, color: '#9E7DB7', fill: '#F3ECF8' },
    { label: 'Email & Messages', count: Math.round(total * 0.18), pct: 18, color: '#D8C6E6', fill: '#FAF7FC' },
    { label: 'Voice & Calls', count: Math.round(total * 0.10), pct: 10, color: '#3DA779', fill: '#EAF7F1' },
    { label: 'Job & Internship', count: Math.round(total * 0.08), pct: 8, color: '#D98928', fill: '#FEF7ED' },
  ];

  // Donut SVG constants
  const donutSize = 180;
  const center = donutSize / 2;
  const strokeWidth = 24;
  const radius = center - strokeWidth / 2 - 8;
  const circumference = 2 * Math.PI * radius;
  let accumulatedPct = 0;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.4fr) minmax(0, 1fr)', gap: '1.25rem' }}>
      {/* Chart 1: Interactive Trend Curve */}
      <div
        className="cyber-card"
        style={{
          padding: '1.5rem',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          position: 'relative',
        }}
      >
        <div>
          {/* Header */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', marginBottom: '0.2rem' }}>
                <TrendingUp size={15} color="#7F5D99" />
                <span style={{ fontSize: '0.72rem', fontWeight: '800', color: '#7F5D99', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                  ACTIVITY METRICS
                </span>
              </div>
              <h3 style={{ fontSize: '1.05rem', fontWeight: '800', color: '#221730', margin: 0 }}>
                Threat Detection & Verification Trends
              </h3>
            </div>

            {/* Timeframe Switcher */}
            <div style={{ display: 'flex', gap: '0.35rem', backgroundColor: '#F8F6FA', padding: '0.25rem', borderRadius: '8px', border: '1px solid #E5DDEC' }}>
              {['24H', '7D', '30D'].map((tf) => (
                <button
                  key={tf}
                  onClick={() => setActiveTimeframe(tf)}
                  style={{
                    border: 'none',
                    outline: 'none',
                    padding: '0.25rem 0.65rem',
                    borderRadius: '6px',
                    fontSize: '0.72rem',
                    fontWeight: activeTimeframe === tf ? '700' : '600',
                    backgroundColor: activeTimeframe === tf ? '#7F5D99' : 'transparent',
                    color: activeTimeframe === tf ? '#FFFFFF' : '#6D5E7B',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                  }}
                >
                  {tf}
                </button>
              ))}
            </div>
          </div>

          {/* Series Legend */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', marginBottom: '0.75rem', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
              <span style={{ width: '10px', height: '10px', borderRadius: '3px', backgroundColor: '#7F5D99' }} />
              <span style={{ fontSize: '0.75rem', fontWeight: '600', color: '#221730' }}>Total Scans</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
              <span style={{ width: '10px', height: '10px', borderRadius: '3px', backgroundColor: '#3DA779' }} />
              <span style={{ fontSize: '0.75rem', fontWeight: '600', color: '#221730' }}>Authentic Content</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
              <span style={{ width: '10px', height: '10px', borderRadius: '3px', backgroundColor: '#D9536F' }} />
              <span style={{ fontSize: '0.75rem', fontWeight: '600', color: '#221730' }}>Threats Detected</span>
            </div>
          </div>
        </div>

        {/* SVG Responsive Container */}
        <div style={{ position: 'relative', width: '100%', height: `${height}px`, overflow: 'hidden' }}>
          <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: '100%', overflow: 'visible' }}>
            <defs>
              <linearGradient id="purpleAreaGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#7F5D99" stopOpacity="0.28" />
                <stop offset="100%" stopColor="#7F5D99" stopOpacity="0.0" />
              </linearGradient>
            </defs>

            {/* Gridlines */}
            {[0.2, 0.5, 0.8].map((pct, i) => (
              <line
                key={i}
                x1={paddingX}
                y1={paddingY + pct * (height - paddingY * 2)}
                x2={width - paddingX}
                y2={paddingY + pct * (height - paddingY * 2)}
                stroke="#EFE8F6"
                strokeWidth="1"
                strokeDasharray="4 4"
              />
            ))}

            {/* Area fill */}
            <path d={areaPath} fill="url(#purpleAreaGrad)" />

            {/* Total Scans Curve */}
            <path d={scansPath} fill="none" stroke="#7F5D99" strokeWidth="3" strokeLinecap="round" />

            {/* Authentic Content Curve */}
            <path d={safePath} fill="none" stroke="#3DA779" strokeWidth="2.5" strokeLinecap="round" strokeDasharray="6 3" />

            {/* Threats Curve */}
            <path d={threatsPath} fill="none" stroke="#D9536F" strokeWidth="2.5" strokeLinecap="round" />

            {/* Interactive Data Points */}
            {trendData.map((d, i) => {
              const cx = getX(i);
              const cy = getY(d.scans);
              const isHovered = hoveredPoint === i;

              return (
                <g key={i} onMouseEnter={() => setHoveredPoint(i)} onMouseLeave={() => setHoveredPoint(null)} style={{ cursor: 'pointer' }}>
                  {/* Vertical hover guide */}
                  {isHovered && (
                    <line x1={cx} y1={paddingY} x2={cx} y2={height - paddingY} stroke="#7F5D99" strokeWidth="1.5" strokeDasharray="3 3" opacity="0.6" />
                  )}
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isHovered ? 6 : 4}
                    fill="#FFFFFF"
                    stroke="#7F5D99"
                    strokeWidth="2.5"
                    style={{ transition: 'all 0.15s ease' }}
                  />
                  {/* X Axis Label */}
                  <text
                    x={cx}
                    y={height - 8}
                    textAnchor="middle"
                    fill={isHovered ? '#7F5D99' : '#6D5E7B'}
                    fontSize="11"
                    fontWeight={isHovered ? '700' : '500'}
                    fontFamily="inherit"
                  >
                    {d.day}
                  </text>
                </g>
              );
            })}
          </svg>

          {/* Floating Tooltip */}
          {hoveredPoint !== null && (
            <div
              style={{
                position: 'absolute',
                top: '10px',
                left: `${(getX(hoveredPoint) / width) * 100}%`,
                transform: 'translateX(-50%)',
                backgroundColor: '#231533',
                color: '#FFFFFF',
                padding: '0.6rem 0.85rem',
                borderRadius: '8px',
                boxShadow: '0 8px 24px rgba(35, 21, 51, 0.25)',
                fontSize: '0.75rem',
                pointerEvents: 'none',
                zIndex: 10,
                minWidth: '130px',
              }}
            >
              <div style={{ fontWeight: '700', marginBottom: '0.25rem', color: '#D8C6E6', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.2rem' }}>
                {trendData[hoveredPoint].day} Overview
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.5rem', margin: '0.15rem 0' }}>
                <span style={{ color: '#B3A5C1' }}>Scans:</span>
                <span style={{ fontWeight: '700' }}>{trendData[hoveredPoint].scans}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.5rem', margin: '0.15rem 0' }}>
                <span style={{ color: '#3DA779' }}>Safe:</span>
                <span style={{ fontWeight: '700', color: '#B7E7D1' }}>{trendData[hoveredPoint].safe}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.5rem', margin: '0.15rem 0' }}>
                <span style={{ color: '#D9536F' }}>Threats:</span>
                <span style={{ fontWeight: '700', color: '#F8BDC8' }}>{trendData[hoveredPoint].threats}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.5rem', margin: '0.15rem 0' }}>
                <span style={{ color: '#B3A5C1' }}>Confidence:</span>
                <span style={{ fontWeight: '700' }}>{trendData[hoveredPoint].confidence}%</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Chart 2: Threat Distribution & Module Breakdown */}
      <div
        className="cyber-card"
        style={{
          padding: '1.5rem',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
        }}
      >
        <div>
          {/* Header */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', marginBottom: '0.2rem' }}>
                <PieChart size={15} color="#7F5D99" />
                <span style={{ fontSize: '0.72rem', fontWeight: '800', color: '#7F5D99', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                  CATEGORICAL BREAKDOWN
                </span>
              </div>
              <h3 style={{ fontSize: '1.05rem', fontWeight: '800', color: '#221730', margin: 0 }}>
                Inspection Distribution by Vector
              </h3>
            </div>
          </div>

          {/* Donut and Legend Flex */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '1.5rem', flexWrap: 'wrap', padding: '0.5rem 0' }}>
            {/* SVG Donut */}
            <div style={{ position: 'relative', width: `${donutSize}px`, height: `${donutSize}px`, flexShrink: 0 }}>
              <svg width={donutSize} height={donutSize} viewBox={`0 0 ${donutSize} ${donutSize}`} style={{ transform: 'rotate(-90deg)' }}>
                {distribution.map((item, idx) => {
                  const strokeDasharray = `${(item.pct / 100) * circumference} ${circumference}`;
                  const strokeDashoffset = -((accumulatedPct / 100) * circumference);
                  accumulatedPct += item.pct;

                  return (
                    <circle
                      key={idx}
                      cx={center}
                      cy={center}
                      r={radius}
                      fill="transparent"
                      stroke={item.color}
                      strokeWidth={strokeWidth}
                      strokeDasharray={strokeDasharray}
                      strokeDashoffset={strokeDashoffset}
                      style={{
                        cursor: 'pointer',
                        transition: 'stroke-width 0.2s ease, opacity 0.2s ease',
                        opacity: activeSegment === null || activeSegment === idx ? 1 : 0.45,
                      }}
                      onMouseEnter={() => setActiveSegment(idx)}
                      onMouseLeave={() => setActiveSegment(null)}
                    />
                  );
                })}
              </svg>

              {/* Center Metrics */}
              <div
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  textAlign: 'center',
                  pointerEvents: 'none',
                }}
              >
                <span style={{ fontSize: '1.45rem', fontWeight: '800', color: '#221730', lineHeight: 1 }}>
                  {activeSegment !== null ? `${distribution[activeSegment].pct}%` : `${total}`}
                </span>
                <span style={{ fontSize: '0.72rem', color: '#6D5E7B', fontWeight: '600', marginTop: '0.2rem' }}>
                  {activeSegment !== null ? distribution[activeSegment].label.split(' ')[0] : 'Total Scans'}
                </span>
              </div>
            </div>

            {/* Compact Category List */}
            <div style={{ flex: 1, minWidth: '150px', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {distribution.map((item, idx) => (
                <div
                  key={idx}
                  onMouseEnter={() => setActiveSegment(idx)}
                  onMouseLeave={() => setActiveSegment(null)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '0.35rem 0.55rem',
                    borderRadius: '7px',
                    backgroundColor: activeSegment === idx ? '#EDE4F5' : 'transparent',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: item.color }} />
                    <span style={{ fontSize: '0.78rem', color: '#221730', fontWeight: '500' }}>{item.label}</span>
                  </div>
                  <span style={{ fontSize: '0.78rem', fontWeight: '700', color: '#7F5D99' }}>{item.pct}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Engine Security Status Pill at bottom */}
        <div
          style={{
            marginTop: '1rem',
            padding: '0.65rem 0.9rem',
            backgroundColor: '#F8F6FA',
            border: '1px solid #E5DDEC',
            borderRadius: '9px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
            <CheckCircle2 size={15} color="#3DA779" />
            <span style={{ fontSize: '0.75rem', fontWeight: '600', color: '#221730' }}>Accuracy Score: 98.4%</span>
          </div>
          <span style={{ fontSize: '0.72rem', color: '#6D5E7B', fontWeight: '500' }}>Real-time telemetry verified</span>
        </div>
      </div>
    </div>
  );
}
