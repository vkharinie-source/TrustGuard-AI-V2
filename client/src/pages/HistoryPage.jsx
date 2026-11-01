import React, { useState, useEffect } from 'react';
import { History as HistoryIcon, RefreshCw, Trash2, AlertTriangle, CheckCircle, XCircle, Search } from '../components/Icons';
import { api } from '../services/api';

const RESULT_FILTERS = ['ALL', 'REAL', 'FAKE', 'SAFE', 'SUSPICIOUS'];

export default function History() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('ALL');

  const fetchHistory = async () => {
    setLoading(true);
    setError('');
    try {
      const data = await api.getHistory();
      setItems(Array.isArray(data) ? data : []);
    } catch (err) {
      setError('Failed to load history. Please verify the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = async () => {
    if (!window.confirm('Are you sure you want to clear all scan history? This action cannot be undone.')) return;
    try {
      await api.clearHistory();
      setItems([]);
    } catch (err) {
      setError('Failed to clear history.');
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const filtered = items.filter((item) => {
    const q = search.toLowerCase();
    const pred = String(item.prediction || item.result || '').toLowerCase();
    const filename = String(item.filename || item.input_value || '').toLowerCase();
    const cat = String(item.category || item.analysis_type || '').toLowerCase();
    const matchesSearch = !q || pred.includes(q) || filename.includes(q) || cat.includes(q);
    return matchesSearch && (filter === 'ALL' || pred === filter);
  });

  const getBadgeStyle = (item) => {
    const pred = String(item.prediction || item.result || '').toUpperCase();
    if (pred === 'REAL' || pred === 'SAFE') {
      return { bg: '#EDF3E8', color: '#3D683D', border: '#BCD5BC' };
    }
    if (pred === 'SUSPICIOUS' || pred === 'WARNING' || pred === 'INCONCLUSIVE') {
      return { bg: '#FEF9EE', color: '#8C6718', border: '#E8D099' };
    }
    return { bg: '#FDF2F4', color: '#963143', border: '#F2BDC6' };
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Header */}
      <div
        style={{
          backgroundColor: '#FFFFFF',
          border: '1px solid #E1E7DD',
          borderRadius: '12px',
          padding: '1.5rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '1rem',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.03)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
          <div
            style={{
              width: '42px',
              height: '42px',
              borderRadius: '10px',
              backgroundColor: '#EDF3E8',
              border: '1px solid #D4E2CC',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <HistoryIcon size={22} color="#36435E" />
          </div>
          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: '700', color: '#202734', margin: 0 }}>
              Scan History
            </h3>
            <p style={{ fontSize: '0.825rem', color: '#697586', margin: '0.15rem 0 0 0' }}>
              {items.length} total inspection records across security modules
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
          <button
            onClick={fetchHistory}
            className="cyber-button-secondary"
            style={{ padding: '0.5rem 1rem', fontSize: '0.825rem' }}
          >
            <RefreshCw size={15} />
            Refresh
          </button>
          {items.length > 0 && (
            <button
              onClick={handleClear}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.4rem',
                padding: '0.5rem 1rem',
                fontSize: '0.825rem',
                fontWeight: '600',
                backgroundColor: '#FDF2F4',
                border: '1px solid #F2BDC6',
                color: '#963143',
                borderRadius: '8px',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              <Trash2 size={15} />
              Clear History
            </button>
          )}
        </div>
      </div>

      {/* Search and result filters */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap' }}>
        <div style={{ position: 'relative', flex: '1 1 260px' }}>
          <Search
            size={18}
            color="#697586"
            style={{ position: 'absolute', left: '0.875rem', top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none' }}
          />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by module, result, or content..."
            className="cyber-input"
            style={{ paddingLeft: '2.5rem' }}
          />
        </div>
        <div role="group" aria-label="Filter history by result" style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
          {RESULT_FILTERS.map((value) => (
            <button
              key={value}
              type="button"
              aria-pressed={filter === value}
              onClick={() => setFilter(value)}
              style={{
                padding: '0.45rem 0.85rem',
                borderRadius: '8px',
                border: `1px solid ${filter === value ? '#98C26E' : '#E1E7DD'}`,
                backgroundColor: filter === value ? '#B9D995' : '#FFFFFF',
                color: filter === value ? '#202734' : '#697586',
                fontSize: '0.75rem',
                fontWeight: '700',
                cursor: 'pointer',
                fontFamily: 'inherit',
                transition: 'all 0.15s ease',
              }}
            >
              {value}
            </button>
          ))}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div
          style={{
            backgroundColor: '#FDF2F4',
            border: '1px solid #F2BDC6',
            color: '#963143',
            padding: '0.85rem 1rem',
            borderRadius: '8px',
            fontSize: '0.85rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <AlertTriangle size={16} />
            {error}
          </div>
        </div>
      )}

      {/* Content */}
      <div
        style={{
          backgroundColor: '#FFFFFF',
          border: '1px solid #E1E7DD',
          borderRadius: '12px',
          overflow: 'hidden',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.03)',
        }}
      >
        {loading ? (
          <div style={{ padding: '3.5rem', textAlign: 'center', color: '#697586' }}>
            <RefreshCw size={28} color="#36435E" style={{ marginBottom: '0.75rem', animation: 'spin 1s linear infinite' }} />
            <p style={{ margin: 0, fontWeight: '500' }}>Loading inspection records...</p>
          </div>
        ) : filtered.length === 0 ? (
          <div style={{ padding: '3.5rem', textAlign: 'center' }}>
            <HistoryIcon size={42} color="#A9B5C7" style={{ marginBottom: '0.75rem' }} />
            <h4 style={{ color: '#202734', margin: 0, fontSize: '1rem', fontWeight: '600' }}>
              {search || filter !== 'ALL' ? 'No records match these filters.' : 'No inspection history found.'}
            </h4>
            <p style={{ color: '#697586', fontSize: '0.85rem', marginTop: '0.35rem' }}>
              {search || filter !== 'ALL' ? 'Adjust the search keyword or select another result filter.' : 'Use any detection module to begin auditing content for threats.'}
            </p>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #E1E7DD', backgroundColor: '#F7F9F4' }}>
                  {['ID', 'Module', 'Target / Content', 'Result', 'Confidence', 'Risk Score', 'Timestamp'].map((h) => (
                    <th
                      key={h}
                      style={{
                        padding: '0.85rem 1rem',
                        fontSize: '0.72rem',
                        fontWeight: '700',
                        color: '#697586',
                        textTransform: 'uppercase',
                        letterSpacing: '0.05em',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((item, idx) => {
                  const pred = String(item.prediction || item.result || '').toUpperCase();
                  const badge = getBadgeStyle(item);
                  return (
                    <tr
                      key={item.id || idx}
                      style={{ borderBottom: '1px solid #E1E7DD' }}
                    >
                      <td style={{ padding: '0.85rem 1rem', fontSize: '0.78rem', color: '#697586', fontWeight: '600' }}>
                        #{item.id || idx + 1}
                      </td>
                      <td style={{ padding: '0.85rem 1rem' }}>
                        <span
                          style={{
                            fontSize: '0.75rem',
                            fontWeight: '600',
                            padding: '0.2rem 0.6rem',
                            borderRadius: '6px',
                            backgroundColor: '#EDF3E8',
                            color: '#36435E',
                            border: '1px solid #D4E2CC',
                          }}
                        >
                          {item.category || item.analysis_type || 'General'}
                        </span>
                      </td>
                      <td style={{ padding: '0.85rem 1rem', maxWidth: '300px' }}>
                        <div
                          style={{
                            fontSize: '0.85rem',
                            color: '#202734',
                            fontWeight: '500',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}
                          title={item.filename || item.input_value}
                        >
                          {item.filename || item.input_value || '—'}
                        </div>
                      </td>
                      <td style={{ padding: '0.85rem 1rem' }}>
                        <span
                          style={{
                            fontSize: '0.72rem',
                            fontWeight: '700',
                            padding: '0.25rem 0.65rem',
                            borderRadius: '9999px',
                            backgroundColor: badge.bg,
                            color: badge.color,
                            border: `1px solid ${badge.border}`,
                          }}
                        >
                          {pred}
                        </span>
                      </td>
                      <td style={{ padding: '0.85rem 1rem', fontSize: '0.85rem', color: '#202734', fontWeight: '600' }}>
                        {typeof item.confidence === 'number' ? `${item.confidence.toFixed(2)}%` : (item.confidence || 'N/A')}
                      </td>
                      <td style={{ padding: '0.85rem 1rem', fontSize: '0.825rem', color: '#697586' }}>
                        {item.risk_score !== undefined ? `${item.risk_score}` : '—'}
                      </td>
                      <td style={{ padding: '0.85rem 1rem', fontSize: '0.78rem', color: '#697586', whiteSpace: 'nowrap' }}>
                        {item.timestamp ? new Date(item.timestamp).toLocaleString() : '—'}
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
