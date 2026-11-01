import React, { useState } from 'react';
import { RefreshCw } from '../components/Icons';
import { tts, buildSpokenSummary } from '../utils/voice';

export default function ReportActions({ result, moduleType, onDownloadPDF, onAnalyzeAnother }) {
  const [speaking, setSpeaking] = useState(false);
  const [paused, setPaused] = useState(false);

  const handleSpeak = () => {
    if (!tts.isSupported()) {
      alert('Text-to-speech is not supported in this browser.');
      return;
    }
    const text = buildSpokenSummary(result, moduleType);
    setSpeaking(true);
    setPaused(false);
    tts.speak(text, () => {
      setSpeaking(false);
      setPaused(false);
    });
  };

  const handlePause = () => {
    if (tts.isPaused()) {
      tts.resume();
      setPaused(false);
    } else {
      tts.pause();
      setPaused(true);
    }
  };

  const handleStop = () => {
    tts.stop();
    setSpeaking(false);
    setPaused(false);
  };

  const btnStyle = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '0.45rem',
    padding: '0.6rem 1.15rem',
    fontSize: '0.825rem',
    fontWeight: '700',
    borderRadius: '8px',
    border: '1px solid #E1E7DD',
    cursor: 'pointer',
    transition: 'all 0.15s ease',
    fontFamily: 'inherit',
  };

  return (
    <div
      style={{
        display: 'flex',
        gap: '0.75rem',
        flexWrap: 'wrap',
        marginTop: '1.25rem',
        paddingTop: '1.25rem',
        borderTop: '1px solid #E1E7DD',
      }}
    >
      {/* Download PDF */}
      <button
        onClick={onDownloadPDF}
        style={{
          ...btnStyle,
          backgroundColor: '#B9D995',
          color: '#202734',
          border: '1px solid #A8CD82',
          boxShadow: '0 2px 8px rgba(185, 217, 149, 0.35)',
        }}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
        Download PDF
      </button>

      {/* Voice Controls */}
      {!speaking ? (
        <button
          onClick={handleSpeak}
          style={{
            ...btnStyle,
            backgroundColor: '#EDF3E8',
            color: '#36435E',
            border: '1px solid #B9D995',
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 010 14.14"/><path d="M15.54 8.46a5 5 0 010 7.07"/></svg>
          Read Result Aloud
        </button>
      ) : (
        <>
          <button
            onClick={handlePause}
            style={{
              ...btnStyle,
              backgroundColor: '#FEF9EE',
              color: '#8C6718',
              border: '1px solid #EED89E',
            }}
          >
            {paused ? (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
            )}
            {paused ? 'Resume' : 'Pause'}
          </button>
          <button
            onClick={handleStop}
            style={{
              ...btnStyle,
              backgroundColor: '#FDF2F4',
              color: '#963143',
              border: '1px solid #F4C2CA',
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>
            Stop
          </button>
        </>
      )}

      {/* Analyze Another */}
      <button
        onClick={onAnalyzeAnother}
        style={{
          ...btnStyle,
          backgroundColor: '#EDF3E8',
          color: '#263142',
          border: '1px solid #E1E7DD',
        }}
      >
        <RefreshCw size={15} />
        Analyze Another
      </button>
    </div>
  );
}
