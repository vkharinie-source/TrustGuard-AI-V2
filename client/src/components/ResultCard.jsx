import React from 'react';
import { CheckCircle, AlertTriangle, XCircle, Shield, Cpu } from './Icons';

export default function ResultCard({ result, moduleType = 'general' }) {
  if (!result) return null;

  // Normalize result properties
  const prediction = String(result.prediction || result.result || '').toUpperCase();
  const rawScore = result.raw_score !== undefined ? result.raw_score : result.score;
  const threshold = result.threshold !== undefined ? result.threshold : 0.50;
  const confidence = result.confidence !== undefined ? result.confidence : 0;
  const modelName = moduleType === 'deepfake' ? 'MobileNetV2 V4' : (result.model || 'TrustGuard Engine V2');

  const isSafe = prediction === 'REAL' || prediction === 'SAFE';
  const isSuspicious = prediction === 'SUSPICIOUS' || prediction === 'WARNING';
  const isDanger = prediction === 'FAKE' || prediction === 'SCAM' || prediction === 'PHISHING' || prediction === 'HIGH RISK';

  let statusConfig = {
    label: prediction || 'ANALYZED',
    badgeClass: 'badge-safe',
    icon: CheckCircle,
    color: '#3D683D',
    bg: '#EDF3E8',
    border: '#B9D995',
  };

  if (isDanger) {
    statusConfig = {
      label: prediction || 'THREAT DETECTED',
      badgeClass: 'badge-danger',
      icon: XCircle,
      color: '#963143',
      bg: '#FDF2F4',
      border: '#F4C2CA',
    };
  } else if (isSuspicious) {
    statusConfig = {
      label: prediction || 'SUSPICIOUS',
      badgeClass: 'badge-warning',
      icon: AlertTriangle,
      color: '#8C6718',
      bg: '#FEF9EE',
      border: '#EED89E',
    };
  }

  const StatusIcon = statusConfig.icon;

  return (
    <div
      style={{
        backgroundColor: '#FFFFFF',
        border: `1px solid ${statusConfig.border}`,
        borderRadius: '12px',
        padding: '1.5rem',
        boxShadow: '0 2px 8px rgba(32, 39, 52, 0.04)',
        marginTop: '1.5rem',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          paddingBottom: '1rem',
          borderBottom: '1px solid #E1E7DD',
          marginBottom: '1.25rem',
          flexWrap: 'wrap',
          gap: '0.75rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div
            style={{
              width: '42px',
              height: '42px',
              borderRadius: '10px',
              backgroundColor: statusConfig.bg,
              border: `1px solid ${statusConfig.border}`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <StatusIcon size={22} color={statusConfig.color} />
          </div>
          <div>
            <p style={{ fontSize: '0.7rem', color: '#697586', textTransform: 'uppercase', letterSpacing: '0.08em', margin: 0, fontWeight: '700' }}>
              DETECTION RESULT
            </p>
            <h3 style={{ fontSize: '1.35rem', fontWeight: '800', color: statusConfig.color, margin: '0.1rem 0 0 0' }}>
              {statusConfig.label}
            </h3>
          </div>
        </div>

        {/* Model Badge */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            padding: '0.4rem 0.8rem',
            borderRadius: '6px',
            backgroundColor: '#EDF3E8',
            border: '1px solid #B9D995',
          }}
        >
          <Cpu size={14} color="#36435E" />
          <span style={{ fontSize: '0.75rem', fontWeight: '700', color: '#202734' }}>
            {modelName}
          </span>
        </div>
      </div>

      {/* Metrics Row */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
          gap: '1rem',
          marginBottom: '1.25rem',
        }}
      >
        {/* Confidence */}
        <div
          style={{
            backgroundColor: '#F7F9F4',
            padding: '0.85rem 1rem',
            borderRadius: '8px',
            border: '1px solid #E1E7DD',
          }}
        >
          <p style={{ fontSize: '0.7rem', color: '#697586', textTransform: 'uppercase', margin: 0, fontWeight: '700' }}>
            Confidence Level
          </p>
          <p style={{ fontSize: '1.25rem', fontWeight: '800', color: statusConfig.color, margin: '0.2rem 0 0 0' }}>
            {typeof confidence === 'number' ? `${confidence.toFixed(2)}%` : confidence}
          </p>
        </div>

        {/* Model Score */}
        {rawScore !== undefined && (
          <div
            style={{
              backgroundColor: '#F7F9F4',
              padding: '0.85rem 1rem',
              borderRadius: '8px',
              border: '1px solid #E1E7DD',
            }}
          >
            <p style={{ fontSize: '0.7rem', color: '#697586', textTransform: 'uppercase', margin: 0, fontWeight: '700' }}>
              Model Score
            </p>
            <p style={{ fontSize: '1.25rem', fontWeight: '700', color: '#202734', margin: '0.2rem 0 0 0' }}>
              {typeof rawScore === 'number' ? rawScore.toFixed(4) : rawScore}
            </p>
          </div>
        )}

        {/* Decision Threshold */}
        <div
          style={{
            backgroundColor: '#F7F9F4',
            padding: '0.85rem 1rem',
            borderRadius: '8px',
            border: '1px solid #E1E7DD',
          }}
        >
          <p style={{ fontSize: '0.7rem', color: '#697586', textTransform: 'uppercase', margin: 0, fontWeight: '700' }}>
            Decision Threshold
          </p>
          <p style={{ fontSize: '1.25rem', fontWeight: '700', color: '#36435E', margin: '0.2rem 0 0 0' }}>
            {typeof threshold === 'number' ? threshold.toFixed(2) : threshold}
          </p>
        </div>
      </div>

      {/* Details List */}
      {Array.isArray(result.details) && result.details.length > 0 && (
        <div style={{ marginBottom: '1rem' }}>
          <h4 style={{ fontSize: '0.85rem', fontWeight: '700', color: '#202734', marginBottom: '0.5rem' }}>
            Analysis Findings
          </h4>
          <ul style={{ listStyleType: 'disc', paddingLeft: '1.25rem', margin: 0 }}>
            {result.details.map((detail, idx) => (
              <li key={idx} style={{ fontSize: '0.825rem', color: '#697586', marginBottom: '0.35rem' }}>
                {typeof detail === 'string' ? detail : JSON.stringify(detail)}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Recommendations */}
      {Array.isArray(result.recommendations) && result.recommendations.length > 0 && (
        <div>
          <h4 style={{ fontSize: '0.85rem', fontWeight: '700', color: '#202734', marginBottom: '0.5rem' }}>
            Security Recommendations
          </h4>
          <ul style={{ listStyleType: 'circle', paddingLeft: '1.25rem', margin: 0 }}>
            {result.recommendations.map((rec, idx) => (
              <li key={idx} style={{ fontSize: '0.825rem', color: '#697586', marginBottom: '0.35rem' }}>
                {rec}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
