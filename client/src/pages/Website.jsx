import React, { useState } from 'react';
import { Globe, AlertTriangle, CheckCircle, XCircle, RefreshCw } from '../components/Icons';
import ReportActions from '../components/ReportActions';
import { api } from '../services/api';
import { generatePDFReport } from '../utils/pdfReport';

const WEBSITE_SIGNALS = [
  'HTTPS scheme',
  'Raw IP address',
  'Suspicious top-level domain',
  'Known URL shortener',
  'Security-related URL terms',
  'Subdomain depth',
  'Unusually long URL',
  'At-sign and encoded characters',
  'Brand/domain patterns',
  'Sensitive query parameters',
];

export default function Website() {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleAnalyze = async () => {
    if (!url.trim()) { setError('Please enter a URL.'); return; }
    setLoading(true); setError(''); setResult(null);
    try {
      setLoadingMsg('Resolving domain...');
      await new Promise(r => setTimeout(r, 300));
      setLoadingMsg('Analyzing URL structure...');
      await new Promise(r => setTimeout(r, 300));
      setLoadingMsg('Running security checks...');
      const data = await api.analyzeWebsite(url);
      setLoadingMsg('Generating report...');
      await new Promise(r => setTimeout(r, 200));
      setResult(data);
    } catch (err) { setError(err.message || 'Website analysis failed.'); }
    finally { setLoading(false); setLoadingMsg(''); }
  };

  const handleReset = () => { setUrl(''); setResult(null); setError(''); };

  const handlePDF = () => {
    if (!result) return;
    const pred = String(result.prediction || result.result || '').toUpperCase();
    const checks = result.checks || result.analysis || {};
    const details = Array.isArray(result.details)
      ? result.details
      : Object.entries(checks).map(([k, v]) => ({
        label: k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        flagged: v === false || v === 'flagged' || v === 'suspicious',
      }));
    const mlProbStr = result.ml_phishing_probability !== undefined && result.ml_phishing_probability !== null
      ? `${(result.ml_phishing_probability * 100).toFixed(2)}%`
      : 'N/A';
    generatePDFReport({
      title: 'Website Security Report',
      moduleType: 'Website Threat Scanner',
      result: { ...result, prediction: pred },
      inputInfo: { 'URL': url, 'Domain': new URL(url.startsWith('http') ? url : `https://${url}`).hostname || url, 'Protocol': url.startsWith('https') ? 'HTTPS' : 'HTTP' },
      findings: [
        `Overall verdict: ${pred}`,
        result.confidence !== undefined ? `Confidence: ${Number(result.confidence).toFixed(2)}%` : null,
        result.risk_score !== undefined ? `Risk score: ${result.risk_score}/100` : null,
        `AI Phishing Probability: ${mlProbStr}`,
        `Indicators Evaluated: ${result.indicators_evaluated || 14} (${result.indicators_flagged || 0} flagged)`,
      ].filter(Boolean),
      details,
      recommendations: Array.isArray(result.recommendations) ? result.recommendations : [],
      technicalInfo: {
        'URL': url,
        'Domain': result.domain,
        'Protocol': result.protocol,
        'Risk Score': `${result.risk_score}/100`,
        'AI Model Probability': mlProbStr,
        'Analysis Engine': 'TrustGuard V5 Lexical + NLP Machine Learning Pipeline',
      },
    });
  };

  const pred = result ? String(result.prediction || result.result || '').toUpperCase() : '';
  const isSafe = pred === 'SAFE' || pred === 'REAL';
  const isSuspicious = pred === 'SUSPICIOUS';
  const checks = result ? (result.checks || result.analysis || {}) : {};
  const details = result && Array.isArray(result.details) ? result.details : [];
  const mlProbStr = result && result.ml_phishing_probability !== undefined && result.ml_phishing_probability !== null
    ? `${(result.ml_phishing_probability * 100).toFixed(2)}%`
    : (isSafe ? '0.00%' : 'N/A');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Input */}
      <div className="cyber-card" style={{ padding: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem', marginBottom: '1.5rem' }}>
          <div style={{
            width: '44px',
            height: '44px',
            borderRadius: '12px',
            backgroundColor: '#EDE4F5',
            border: '1px solid #D8C6E6',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 2px 8px rgba(127, 93, 153, 0.12)',
          }}>
            <Globe size={22} color="#7F5D99" />
          </div>
          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: '800', color: '#221730', margin: 0 }}>Enter Website URL</h3>
            <p style={{ fontSize: '0.825rem', color: '#6D5E7B', margin: '0.15rem 0 0 0' }}>Paste a URL to scan for phishing, spoofing, and malicious scripts</p>
          </div>
        </div>
        {/* Quick Test Presets */}
        <div style={{ marginBottom: '1.25rem', padding: '0.85rem 1.15rem', borderRadius: '12px', backgroundColor: '#F8F5FA', border: '1px solid #EBE4F2', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: '700', color: '#221730' }}>Quick Test Scenarios:</span>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            <button
              onClick={() => { setUrl('http://paypal.login-verify-account-security.xyz/signin'); setResult(null); setError(''); }}
              type="button"
              style={{
                padding: '0.35rem 0.8rem', fontSize: '0.75rem', fontWeight: '700', borderRadius: '8px', cursor: 'pointer',
                border: '1px solid #F8BDC8', backgroundColor: '#FDEFF2', color: '#A8324C',
              }}
            >
              ⚡ Phishing Typosquat URL
            </button>
            <button
              onClick={() => { setUrl('http://192.168.1.100:8080/secure-banking-login'); setResult(null); setError(''); }}
              type="button"
              style={{
                padding: '0.35rem 0.8rem', fontSize: '0.75rem', fontWeight: '700', borderRadius: '8px', cursor: 'pointer',
                border: '1px solid #F8D8A7', backgroundColor: '#FEF7ED', color: '#8F5410',
              }}
            >
              ⚡ Raw IP Phish URL
            </button>
            <button
              onClick={() => { setUrl('https://www.google.com'); setResult(null); setError(''); }}
              type="button"
              style={{
                padding: '0.35rem 0.8rem', fontSize: '0.75rem', fontWeight: '700', borderRadius: '8px', cursor: 'pointer',
                border: '1px solid #B7E7D1', backgroundColor: '#EAF7F1', color: '#247352',
              }}
            >
              ⚡ Legitimate Domain
            </button>
            <button
              onClick={handleReset}
              type="button"
              style={{
                padding: '0.35rem 0.7rem', fontSize: '0.75rem', fontWeight: '600', borderRadius: '8px', cursor: 'pointer',
                border: '1px solid #E5DDEC', backgroundColor: '#FFFFFF', color: '#6D5E7B',
              }}
            >
              Reset
            </button>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '0.85rem', flexWrap: 'wrap' }}>
          <input
            type="url"
            value={url}
            onChange={e => setUrl(e.target.value)}
            placeholder="https://example.com"
            className="cyber-input"
            style={{ flex: 1, minWidth: '260px' }}
            onKeyDown={e => e.key === 'Enter' && handleAnalyze()}
          />
          <button className="cyber-button-primary" onClick={handleAnalyze} disabled={loading} style={{ padding: '0.7rem 1.75rem', fontSize: '0.88rem', fontWeight: '700' }}>
            {loading ? <><RefreshCw size={16} className="animate-spin" /> Scanning...</> : 'Scan URL Security'}
          </button>
        </div>
      </div>

      {!result && !loading && (
        <section className="analysis-context" aria-label="Website security checks">
          <div className="analysis-context-heading">
            <span className="analysis-context-icon"><Globe size={18} /></span>
            <div>
              <span className="analysis-context-kicker">URL CHECK COVERAGE</span>
              <h3 style={{ margin: 0, marginTop: '2px' }}>Supported Threat Signals</h3>
            </div>
          </div>
          <div className="website-signal-grid">
            {WEBSITE_SIGNALS.map((signal, index) => (
              <div className="website-signal" key={signal}>
                <span>{String(index + 1).padStart(2, '0')}</span>
                {signal}
              </div>
            ))}
          </div>
          <p className="analysis-context-note">
            These are heuristic indicators evaluated against high-confidence phishing features. A SAFE result confirms known threats were absent at the time of evaluation.
          </p>
        </section>
      )}

      {loading && (
        <div className="cyber-card" style={{ padding: '2.5rem', textAlign: 'center' }}>
          <RefreshCw size={32} color="#7F5D99" className="animate-spin" style={{ marginBottom: '1rem' }} />
          <p style={{ fontSize: '0.95rem', color: '#221730', fontWeight: '700' }}>{loadingMsg}</p>
        </div>
      )}

      {error && (
        <div style={{ backgroundColor: '#FDEFF2', border: '1px solid #F8BDC8', color: '#A32943', padding: '0.85rem 1rem', borderRadius: '8px', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: '600' }}>
          <AlertTriangle size={16} color="#D9536F" /> {error}
        </div>
      )}

      {result && !loading && (
        <div className="cyber-card" style={{ padding: '2rem' }}>
          <h3 style={{ fontSize: '0.75rem', fontWeight: '800', textTransform: 'uppercase', letterSpacing: '0.08em', color: '#7F5D99', marginBottom: '1.25rem' }}>
            Website Security Report
          </h3>

          {/* Result Header */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: '1rem', padding: '1.25rem', borderRadius: '12px', marginBottom: '1.25rem',
            backgroundColor: isSafe ? '#EAF7F1' : isSuspicious ? '#FEF7ED' : '#FDEFF2',
            border: `1px solid ${isSafe ? '#B7E7D1' : isSuspicious ? '#F8D8A7' : '#F8BDC8'}`,
          }}>
            {isSafe ? <CheckCircle size={32} color="#3DA779" /> : isSuspicious ? <AlertTriangle size={32} color="#D98928" /> : <XCircle size={32} color="#D9536F" />}
            <div>
              <p style={{ fontSize: '1.5rem', fontWeight: '800', color: isSafe ? '#1D6E4B' : isSuspicious ? '#8F5410' : '#A32943', margin: 0 }}>{pred}</p>
              <p style={{ fontSize: '0.8rem', color: '#6D5E7B', margin: '0.15rem 0 0 0', fontWeight: '600' }}>Website Security Result</p>
            </div>
            <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
              {result.confidence !== undefined && <p style={{ fontSize: '1.25rem', fontWeight: '800', color: '#221730', margin: 0 }}>{Number(result.confidence).toFixed(2)}%</p>}
              <p style={{ fontSize: '0.72rem', color: '#6D5E7B', margin: 0, fontWeight: '600' }}>Verdict Confidence</p>
            </div>
          </div>

          {/* 4-Metric Breakdown Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem', marginBottom: '1.5rem' }}>
            <div style={{ backgroundColor: '#F8F6FA', padding: '0.85rem 1rem', borderRadius: '10px', border: '1px solid #E5DDEC' }}>
              <p style={{ fontSize: '0.68rem', fontWeight: '700', color: '#6D5E7B', textTransform: 'uppercase', margin: 0 }}>Verdict Confidence</p>
              <p style={{ fontSize: '1.15rem', fontWeight: '800', color: isSafe ? '#1D6E4B' : isSuspicious ? '#8F5410' : '#A32943', margin: '0.2rem 0 0 0' }}>
                {result.confidence !== undefined ? `${Number(result.confidence).toFixed(2)}%` : 'N/A'}
              </p>
              <p style={{ fontSize: '0.68rem', color: '#6D5E7B', margin: '0.15rem 0 0 0' }}>Overall assessment certainty</p>
            </div>

            <div style={{ backgroundColor: '#F8F6FA', padding: '0.85rem 1rem', borderRadius: '10px', border: '1px solid #E5DDEC' }}>
              <p style={{ fontSize: '0.68rem', fontWeight: '700', color: '#6D5E7B', textTransform: 'uppercase', margin: 0 }}>Composite Risk Score</p>
              <p style={{ fontSize: '1.15rem', fontWeight: '800', color: result.risk_score >= 40 ? '#A32943' : result.risk_score >= 20 ? '#8F5410' : '#1D6E4B', margin: '0.2rem 0 0 0' }}>
                {result.risk_score !== undefined ? `${result.risk_score} / 100` : '0 / 100'}
              </p>
              <p style={{ fontSize: '0.68rem', color: '#6D5E7B', margin: '0.15rem 0 0 0' }}>Multi-vector threat index</p>
            </div>

            <div style={{ backgroundColor: '#F8F6FA', padding: '0.85rem 1rem', borderRadius: '10px', border: '1px solid #E5DDEC' }}>
              <p style={{ fontSize: '0.68rem', fontWeight: '700', color: '#6D5E7B', textTransform: 'uppercase', margin: 0 }}>AI Phishing Probability</p>
              <p style={{ fontSize: '1.15rem', fontWeight: '800', color: '#221730', margin: '0.2rem 0 0 0' }}>
                {mlProbStr}
              </p>
              <p style={{ fontSize: '0.68rem', color: '#6D5E7B', margin: '0.15rem 0 0 0' }}>V5 machine learning prediction</p>
            </div>

            <div style={{ backgroundColor: '#F8F6FA', padding: '0.85rem 1rem', borderRadius: '10px', border: '1px solid #E5DDEC' }}>
              <p style={{ fontSize: '0.68rem', fontWeight: '700', color: '#6D5E7B', textTransform: 'uppercase', margin: 0 }}>Security Checks</p>
              <p style={{ fontSize: '1.15rem', fontWeight: '800', color: (result.indicators_flagged || 0) > 0 ? '#A32943' : '#1D6E4B', margin: '0.2rem 0 0 0' }}>
                {result.indicators_flagged !== undefined ? `${result.indicators_flagged} Flagged` : '0 Flagged'}
              </p>
              <p style={{ fontSize: '0.68rem', color: '#6D5E7B', margin: '0.15rem 0 0 0' }}>Out of {result.indicators_evaluated || 14} checks evaluated</p>
            </div>
          </div>

          {/* Evaluated Security Checks Grid */}
          {Object.keys(checks).length > 0 && (
            <div style={{ marginBottom: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.65rem' }}>
                <h4 style={{ fontSize: '0.75rem', fontWeight: '800', color: '#7F5D99', textTransform: 'uppercase', letterSpacing: '0.06em', margin: 0 }}>
                  Evaluated Security Indicator Checks ({Object.keys(checks).length})
                </h4>
                <span style={{ fontSize: '0.72rem', color: '#6D5E7B', fontWeight: '600' }}>
                  {result.indicators_flagged || 0} flagged / {Object.keys(checks).length} clear
                </span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '0.65rem' }}>
                {Object.entries(checks).map(([key, val]) => {
                  const passed = val === true || val === 'clear' || val === 'safe';
                  return (
                    <div key={key} style={{
                      display: 'flex', alignItems: 'center', gap: '0.65rem',
                      backgroundColor: '#F8F6FA', padding: '0.65rem 0.85rem', borderRadius: '8px',
                      border: `1px solid ${passed ? '#E5DDEC' : '#F8BDC8'}`,
                    }}>
                      {passed ? <CheckCircle size={15} color="#3DA779" /> : <AlertTriangle size={15} color="#D98928" />}
                      <span style={{ fontSize: '0.78rem', color: '#221730', flex: 1, fontWeight: '500' }}>
                        {key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                      </span>
                      <span style={{
                        fontSize: '0.7rem', fontWeight: '700',
                        color: passed ? '#1D6E4B' : '#A32943',
                        backgroundColor: passed ? '#EAF7F1' : '#FDEFF2',
                        padding: '0.15rem 0.45rem', borderRadius: '4px',
                      }}>
                        {passed ? 'Clear' : 'Flagged'}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Flagged Specific Threat Details */}
          {details.length > 0 && (
            <div style={{ marginBottom: '1.25rem' }}>
              <h4 style={{ fontSize: '0.75rem', fontWeight: '800', color: '#7F5D99', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.65rem' }}>
                Flagged Specific Findings ({details.length})
              </h4>
              <ul style={{ display: 'grid', gap: '0.5rem', margin: 0, padding: 0, listStyle: 'none' }}>
                {details.map((detail, index) => (
                  <li key={index} className="website-finding">
                    <AlertTriangle size={15} color="#D98928" />
                    {typeof detail === 'string' ? detail : JSON.stringify(detail)}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Key Findings */}
          <div style={{ marginBottom: '1.25rem' }}>
            <h4 style={{ fontSize: '0.75rem', fontWeight: '800', color: '#7F5D99', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.65rem' }}>Key Findings</h4>
            <div style={{ backgroundColor: '#F8F6FA', padding: '1rem 1.25rem', borderRadius: '8px', border: '1px solid #E5DDEC', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <p style={{ fontSize: '0.85rem', color: '#221730', lineHeight: '1.65', margin: 0 }}>
                The website detector classified this URL as <strong>{pred}</strong>. It evaluated {result.indicators_evaluated || 14} security indicator categories ({result.indicators_flagged || 0} flagged) with a composite threat risk score of {result.risk_score !== undefined ? `${result.risk_score}/100` : '0/100'} and an AI model phishing probability of {mlProbStr}.
              </p>
              <p style={{ fontSize: '0.75rem', color: '#6D5E7B', lineHeight: '1.5', margin: 0, borderTop: '1px solid #EBE4F2', paddingTop: '0.5rem' }}>
                <em>Note on Security Scoring:</em> Composite risk score (0–100) measures defense-in-depth indicator accumulation, while AI Phishing Probability is the raw statistical likelihood from the trained lexical/NLP model. A zero risk score confirms no known phishing structural patterns were present in the URL, but does not certify that external server content or dynamic pages are permanently benign.
              </p>
            </div>
          </div>

          {/* Recommendations */}
          <div style={{ marginBottom: '0.5rem' }}>
            <h4 style={{ fontSize: '0.75rem', fontWeight: '800', color: '#7F5D99', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.65rem' }}>Recommendations</h4>
            <div style={{ backgroundColor: '#F8F6FA', padding: '1rem 1.25rem', borderRadius: '8px', border: '1px solid #E5DDEC' }}>
              {Array.isArray(result.recommendations) && result.recommendations.length > 0 ? (
                <ul style={{ margin: 0, paddingLeft: '1.25rem', color: '#221730', fontSize: '0.85rem', lineHeight: '1.7' }}>
                  {result.recommendations.map((recommendation, index) => <li key={index}>{recommendation}</li>)}
                </ul>
              ) : <p style={{ fontSize: '0.85rem', color: '#6D5E7B', lineHeight: '1.65', margin: 0 }}>No critical flags identified. Continue practicing standard cyber hygiene when sharing credentials.</p>}
            </div>
          </div>

          {/* Technical Details */}
          <div style={{ marginBottom: '0.5rem', marginTop: '1.25rem' }}>
            <h4 style={{ fontSize: '0.75rem', fontWeight: '800', color: '#7F5D99', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.65rem' }}>Technical Details</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.65rem' }}>
              {[
                { label: 'URL', value: url },
                { label: 'Domain', value: result.domain || new URL(url.startsWith('http') ? url : `https://${url}`).hostname },
                { label: 'Protocol', value: result.protocol || (url.startsWith('https') ? 'HTTPS' : 'HTTP') },
                { label: 'Risk Score', value: result.risk_score !== undefined ? `${result.risk_score}/100` : 'N/A' },
                { label: 'AI Phishing Probability', value: mlProbStr },
                { label: 'Model Artifact', value: result.model_loaded ? 'V5 Production Pipeline' : 'Loaded' },
              ].map((t, i) => (
                <div key={i} style={{ backgroundColor: '#F8F6FA', padding: '0.75rem 1rem', borderRadius: '8px', border: '1px solid #E5DDEC' }}>
                  <p style={{ fontSize: '0.68rem', fontWeight: '700', color: '#6D5E7B', textTransform: 'uppercase', margin: 0 }}>{t.label}</p>
                  <p style={{ fontSize: '0.85rem', fontWeight: '700', color: '#221730', margin: '0.25rem 0 0 0', wordBreak: 'break-all' }}>{t.value}</p>
                </div>
              ))}
            </div>
          </div>

          <ReportActions result={result} moduleType="website" onDownloadPDF={handlePDF} onAnalyzeAnother={handleReset} />
        </div>
      )}
    </div>
  );
}
