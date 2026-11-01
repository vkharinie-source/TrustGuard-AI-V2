// ====================================================
// TrustGuard AI — Voice Utility
// Browser Web Speech API (TTS + STT)
// No external service required
// ====================================================

// ---- TTS (Text-to-Speech) ----

let currentUtterance = null;

export const tts = {
  isSupported: () => 'speechSynthesis' in window,

  speak: (text, onEnd) => {
    if (!tts.isSupported()) return;
    tts.stop();
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = 'en-US';
    utter.rate = 0.95;
    utter.pitch = 1;
    utter.volume = 1;
    if (onEnd) utter.onend = onEnd;
    currentUtterance = utter;
    window.speechSynthesis.speak(utter);
  },

  pause: () => {
    if (window.speechSynthesis.speaking) window.speechSynthesis.pause();
  },

  resume: () => {
    if (window.speechSynthesis.paused) window.speechSynthesis.resume();
  },

  stop: () => {
    window.speechSynthesis.cancel();
    currentUtterance = null;
  },

  isSpeaking: () => window.speechSynthesis.speaking,
  isPaused: () => window.speechSynthesis.paused,
};

// ---- STT (Speech-to-Text) ----

let recognition = null;

export const stt = {
  isSupported: () =>
    'SpeechRecognition' in window || 'webkitSpeechRecognition' in window,

  start: (onResult, onError, onEnd) => {
    if (!stt.isSupported()) {
      if (onError) onError('Speech recognition is not supported in this browser.');
      return null;
    }

    stt.stop();

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.continuous = false;

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      if (onResult) onResult(transcript);
    };

    recognition.onerror = (event) => {
      if (onError) onError(event.error || 'Speech recognition error.');
    };

    recognition.onend = () => {
      if (onEnd) onEnd();
    };

    recognition.start();
    return recognition;
  },

  stop: () => {
    if (recognition) {
      try { recognition.stop(); } catch {}
      recognition = null;
    }
  },
};

// ---- Build spoken summary from analysis result ----
export const buildSpokenSummary = (result, moduleType) => {
  if (!result) return 'No analysis result available.';

  const pred = String(result.prediction || result.result || 'unknown').toUpperCase();
  const conf = result.confidence !== undefined
    ? `${Number(result.confidence).toFixed(1)} percent confidence`
    : '';

  const moduleNames = {
    deepfake: 'DeepFake Image',
    website: 'Website',
    email: 'Email',
    message: 'Message',
    call: 'Phone Call',
    application: 'Application',
    internship: 'Internship Offer',
    job: 'Job Offer',
    news: 'News Article',
    general: 'Content',
  };

  const name = moduleNames[moduleType] || moduleNames.general;
  let spoken = `TrustGuard AI ${name} Analysis. `;
  spoken += `The result is: ${pred}. `;
  if (conf) spoken += `${conf}. `;

  if (pred === 'REAL' || pred === 'SAFE') {
    spoken += 'The content appears to be authentic and safe. ';
  } else if (pred === 'FAKE' || pred === 'SCAM' || pred === 'PHISHING') {
    spoken += 'Warning: This content has been flagged as potentially fraudulent or fake. Exercise extreme caution. ';
  } else if (pred === 'SUSPICIOUS') {
    spoken += 'This content contains suspicious indicators. Proceed with caution. ';
  }

  const recs = Array.isArray(result.recommendations) ? result.recommendations : [];
  if (recs.length > 0) {
    spoken += 'Recommendations: ' + recs.slice(0, 2).join('. ') + '. ';
  }

  spoken += 'Analysis complete.';
  return spoken;
};
