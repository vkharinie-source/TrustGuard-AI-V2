import React from 'react';
import { PhoneCall, AlertTriangle, CheckCircle } from '../components/Icons';
import AnalysisPage from '../components/AnalysisPage';
import { api } from '../services/api';

export default function Call() {
  return (
    <AnalysisPage
      icon={<PhoneCall size={22} color="#818CF8" />}
      title="Call Transcript Analyzer"
      subtitle="Inspect caller profiles and transcripts for voice scam tactics"
      reportTitle="Call Security Report"
      moduleType="call"
      fields={[
        { key: 'caller', label: 'Caller Information', placeholder: 'e.g. Unknown number claiming to be from your bank' },
        { key: 'transcript', label: 'Call Transcript', placeholder: 'Enter or paste the call transcript here...', multiline: true, rows: 7 },
      ]}
      onSubmit={async (vals) => api.analyzeCall(vals.caller, vals.transcript)}
      buildPDFData={(result, vals) => ({
        inputInfo: { Caller: vals.caller, 'Transcript Length': `${vals.transcript.length} characters` },
        findings: [
          `Verdict: ${String(result.prediction || result.result || '').toUpperCase()}`,
          result.confidence !== undefined ? `Confidence: ${Number(result.confidence).toFixed(2)}%` : null,
          `Caller: ${vals.caller}`,
        ].filter(Boolean),
        details: Object.entries(result.analysis || result.checks || {}).map(([k, v]) => ({
          label: k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
          flagged: v === true || v === 'flagged',
        })),
        recommendations: [
          'Never share OTP, PIN or passwords during phone calls.',
          'Verify the caller by calling back the official number.',
          'Report suspicious calls to your service provider.',
        ],
        technicalInfo: { 'Analysis Engine': 'Transcript Scam Analyzer', Caller: vals.caller },
      })}
      renderChecks={(result) => {
        const checks = result.analysis || result.checks || {};
        if (Object.keys(checks).length === 0) return null;
        return (
          <div style={{ marginBottom: '1.25rem' }}>
            <h4 style={{ fontSize: '0.72rem', fontWeight: '700', color: '#818CF8', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.65rem' }}>Scam Indicators</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.5rem' }}>
              {Object.entries(checks).map(([k, v]) => {
                const flagged = v === true || v === 'flagged' || v === 'suspicious';
                return (
                  <div key={k} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', backgroundColor: '#0D1424', padding: '0.55rem 0.75rem', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.04)' }}>
                    {flagged ? <AlertTriangle size={14} color="#FBBF24" /> : <CheckCircle size={14} color="#34D399" />}
                    <span style={{ fontSize: '0.78rem', color: '#D1D5DB', flex: 1 }}>{k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</span>
                    <span style={{ fontSize: '0.7rem', fontWeight: '600', color: flagged ? '#FBBF24' : '#34D399' }}>{flagged ? 'Flagged' : 'Clear'}</span>
                  </div>
                );
              })}
            </div>
          </div>
        );
      }}
    />
  );
}
