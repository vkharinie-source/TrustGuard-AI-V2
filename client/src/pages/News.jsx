import React, { useState, useRef } from 'react';
import { Newspaper, AlertTriangle, RefreshCw } from '../components/Icons';
import ReportActions from '../components/ReportActions';
import { api } from '../services/api';
import { generatePDFReport } from '../utils/pdfReport';
import { tts, stt } from '../utils/voice';
import { transcribeAudioSamples } from '../utils/offlineSpeech';
import { Pause, Play, Square, Volume2 } from 'lucide-react';

const TABS = [
  { key: 'text', label: 'Paste Text' },
  { key: 'image', label: 'Upload Image' },
  { key: 'pdf', label: 'Upload PDF' },
  { key: 'voice', label: 'Dictate Article' },
];

const SUGGESTED_QS = [
  'Explain this news in simple words.',
  'Why is this article unverified?',
  'What is the main claim?',
  'What evidence supports it?',
  'What evidence is missing?',
  'Give me the important points.',
  'What should I be careful about?',
];

const buildArticleSummary = (headline, content) => {
  const sentences = (content.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [])
    .map(sentence => sentence.trim())
    .filter(Boolean)
    .slice(0, 3);
  const excerpt = sentences.join(' ') || content.trim().slice(0, 500);
  const title = headline.trim() ? `The story is titled ${headline.trim()}. ` : '';
  return `${title}The article says: ${excerpt} The claims remain unverified because external news sources were not checked.`;
};

export default function News() {
  const [tab, setTab] = useState('text');
  const [headline, setHeadline] = useState('');
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  // OCR state
  const [ocrText, setOcrText] = useState('');
  const [ocrLoading, setOcrLoading] = useState(false);
  const [ocrMsg, setOcrMsg] = useState('');
  const [imagePreview, setImagePreview] = useState(null);
  const imageRef = useRef(null);

  // PDF state
  const [pdfText, setPdfText] = useState('');
  const [pdfLoading, setPdfLoading] = useState(false);
  const [pdfFileName, setPdfFileName] = useState('');

  // Voice input state
  const [voiceText, setVoiceText] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [dictationLoading, setDictationLoading] = useState(false);
  const [dictationMessage, setDictationMessage] = useState('');
  const audioCaptureRef = useRef(null);
  const recordingStartedAtRef = useRef(0);

  // AI Chat state
  const [chatMsgs, setChatMsgs] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [chatVoiceListening, setChatVoiceListening] = useState(false);

  // AI voice response
  const [aiSpeaking, setAiSpeaking] = useState(false);
  const [aiPaused, setAiPaused] = useState(false);

  // Determine which text to analyze
  const getAnalysisText = () => {
    if (tab === 'text') return text;
    if (tab === 'image') return ocrText;
    if (tab === 'pdf') return pdfText;
    if (tab === 'voice') return voiceText;
    return '';
  };

  // ---- Image OCR ----
  const handleImageSelect = async (file) => {
    if (!file) return;
    const allowed = ['image/jpeg', 'image/png', 'image/webp', 'image/jpg'];
    if (!allowed.includes(file.type)) {
      setError('Please upload a valid image (JPG, PNG, WEBP).');
      return;
    }
    setImagePreview(URL.createObjectURL(file));
    setOcrText('');
    setError('');
    setOcrLoading(true);
    try {
      setOcrMsg('Uploading image...');
      await new Promise(r => setTimeout(r, 300));
      setOcrMsg('Reading newspaper...');
      // Dynamic import Tesseract
      const Tesseract = await import('tesseract.js');
      setOcrMsg('Extracting article text...');
      const worker = await Tesseract.createWorker('eng');
      const { data } = await worker.recognize(file);
      await worker.terminate();
      const extracted = data.text?.trim() || '';
      if (!extracted) {
        setOcrMsg('');
        setError('Could not extract text from this image. Try a clearer image or enter the text manually.');
      } else {
        if (data.confidence && data.confidence < 60) {
          setError('Some text could not be read clearly. Please review and edit the extracted text.');
        }
        setOcrText(extracted);
        setOcrMsg('Text extraction complete.');
      }
    } catch (err) {
      setError('OCR failed: ' + (err.message || 'Unknown error'));
      setOcrMsg('');
    } finally {
      setOcrLoading(false);
    }
  };

  // ---- PDF Text Extraction ----
  const handlePdfSelect = async (file) => {
    if (!file || file.type !== 'application/pdf') {
      setError('Please upload a valid PDF file.');
      return;
    }
    setPdfFileName(file.name);
    setPdfText('');
    setPdfLoading(true);
    setError('');
    try {
      // Read PDF as text — basic extraction using FileReader
      const arrayBuffer = await file.arrayBuffer();
      // Try to extract text from PDF using a simple approach
      const bytes = new Uint8Array(arrayBuffer);
      let raw = '';
      for (let i = 0; i < bytes.length; i++) {
        const ch = bytes[i];
        if (ch >= 32 && ch < 127) raw += String.fromCharCode(ch);
        else if (ch === 10 || ch === 13) raw += '\n';
      }
      // Extract readable content between stream markers or parentheses
      const textChunks = [];
      // Look for text in parentheses (PDF text objects)
      const parenRegex = /\(([^)]{3,})\)/g;
      let match;
      while ((match = parenRegex.exec(raw)) !== null) {
        const chunk = match[1].replace(/\\n/g, '\n').replace(/\\\\/g, '\\').replace(/\\'/g, "'");
        if (chunk.length > 3 && /[a-zA-Z]/.test(chunk)) {
          textChunks.push(chunk);
        }
      }
      const extracted = textChunks.join(' ').replace(/\s+/g, ' ').trim();
      if (extracted.length > 20) {
        setPdfText(extracted);
      } else {
        // Fallback: treat file as text
        const textDecoder = new TextDecoder('utf-8', { fatal: false });
        const decoded = textDecoder.decode(arrayBuffer);
        const clean = decoded.replace(/[^\x20-\x7E\n]/g, ' ').replace(/\s+/g, ' ').trim();
        if (clean.length > 50) {
          setPdfText(clean.substring(0, 5000));
        } else {
          setError('Could not extract readable text from this PDF. Try pasting the text manually.');
        }
      }
    } catch (err) {
      setError('PDF extraction failed. Please paste the text manually.');
    } finally {
      setPdfLoading(false);
    }
  };

  // ---- Offline Voice Input ----
  const handleVoiceStart = async () => {
    const AudioContextType = window.AudioContext || window.webkitAudioContext;
    if (!navigator.mediaDevices?.getUserMedia || !AudioContextType || !window.OfflineAudioContext) {
      setError('This browser cannot record microphone audio. Type or paste the article below instead.');
      return;
    }
    setError('');
    setDictationMessage('Requesting microphone access...');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioContext = new AudioContextType();
      await audioContext.resume();
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      const silentOutput = audioContext.createGain();
      const chunks = [];
      silentOutput.gain.value = 0;
      processor.onaudioprocess = (event) => {
        chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      };
      source.connect(processor);
      processor.connect(silentOutput);
      silentOutput.connect(audioContext.destination);
      audioCaptureRef.current = { stream, audioContext, source, processor, silentOutput, chunks };
      recordingStartedAtRef.current = Date.now();
      setIsListening(true);
      setDictationMessage('Recording on this device. Speak for at least one second, then select Stop.');
    } catch (error) {
      audioCaptureRef.current?.stream.getTracks().forEach(track => track.stop());
      audioCaptureRef.current = null;
      setDictationMessage('');
      setError(error.name === 'NotAllowedError'
        ? 'Microphone access is blocked. Allow microphone access, or type or paste the article below.'
        : 'Could not start microphone recording. Type or paste the article below instead.');
    }
  };

  const handleVoiceStop = async () => {
    const capture = audioCaptureRef.current;
    if (!capture) return;
    audioCaptureRef.current = null;
    capture.processor.onaudioprocess = null;
    capture.source.disconnect();
    capture.processor.disconnect();
    capture.silentOutput.disconnect();
    capture.stream.getTracks().forEach(track => track.stop());
    setIsListening(false);

    if (Date.now() - recordingStartedAtRef.current < 800) {
      await capture.audioContext.close();
      setDictationMessage('');
      setError('Recording was too short. Speak for at least one second before stopping.');
      return;
    }

    const sampleCount = capture.chunks.reduce((total, chunk) => total + chunk.length, 0);
    if (!sampleCount) {
      await capture.audioContext.close();
      setDictationMessage('');
      setError('No microphone audio was captured. Check your microphone or type the article below.');
      return;
    }

    const recordedSamples = new Float32Array(sampleCount);
    let writeOffset = 0;
    capture.chunks.forEach((chunk) => {
      recordedSamples.set(chunk, writeOffset);
      writeOffset += chunk.length;
    });

    setDictationLoading(true);
    setError('');
    setDictationMessage('Loading the offline speech model. First use may download it...');
    try {
      setDictationMessage('Transcribing audio on this device. The model will download if needed...');
      const output = await transcribeAudioSamples(
        recordedSamples,
        capture.audioContext.sampleRate,
        percent => setDictationMessage(`Downloading offline speech model: ${percent}%`),
      );
      const transcript = String(output.text || '').trim();
      if (!transcript) throw new Error('No speech was detected. Try again or type the article below.');
      setVoiceText(previous => previous ? `${previous} ${transcript}` : transcript);
      setDictationMessage('Transcription complete. Review or edit the text below.');
    } catch (error) {
      setDictationMessage('');
      setError(error.message || 'Offline transcription failed. Type or paste the article below instead.');
    } finally {
      await capture.audioContext.close();
      setDictationLoading(false);
    }
  };

  // ---- Analyze ----
  const handleAnalyze = async () => {
    const content = getAnalysisText();
    if (!content.trim()) {
      setError('Please provide news content to analyze.');
      return;
    }
    setLoading(true); setError(''); setResult(null);
    setChatMsgs([]);
    try {
      setLoadingMsg('Understanding article...');
      await new Promise(r => setTimeout(r, 300));
      setLoadingMsg('Analyzing claims...');
      const data = await api.analyzeNews(headline || 'News Article', content);
      setLoadingMsg('Generating verification report...');
      await new Promise(r => setTimeout(r, 250));

      const enriched = {
        ...data,
        verification_status: 'UNVERIFIED',
        short_summary: buildArticleSummary(headline, content),
        main_claim: headline || 'Not identified by the available analysis.',
        article_text: content,
      };
      setResult(enriched);
    } catch (err) {
      setError(err.message || 'News verification failed.');
    } finally {
      setLoading(false); setLoadingMsg('');
    }
  };

  // ---- AI Chat ----
  const handleAskAI = async (question) => {
    if (!question?.trim() || !result) return;
    const q = question.trim();
    setChatMsgs(prev => [...prev, { role: 'user', text: q }]);
    setChatInput('');
    const indicators = Array.isArray(result.details) && result.details.length
      ? ` The content-risk detector reported: ${result.details.join(' ')}`
      : ' The content-risk detector did not report any scam indicators.';
    setChatMsgs(prev => [...prev, {
      role: 'system',
      text: `This workspace does not have an article fact-checking or question-answering service configured. The article remains UNVERIFIED because no external sources were checked.${indicators}`,
    }]);
  };

  // Voice question
  const handleChatVoiceStart = () => {
    if (!stt.isSupported()) { setError('Speech recognition not supported.'); return; }
    setChatVoiceListening(true);
    stt.start(
      (transcript) => { setChatInput(transcript); },
      () => { setChatVoiceListening(false); },
      () => { setChatVoiceListening(false); }
    );
  };
  const handleChatVoiceStop = () => { stt.stop(); setChatVoiceListening(false); };

  // AI speaks answer
  const handlePlayAIAnswer = (answerText) => {
    if (!tts.isSupported()) {
      setError('Spoken explanations are not supported in this browser.');
      return;
    }
    setAiSpeaking(true);
    setAiPaused(false);
    tts.speak(answerText, () => { setAiSpeaking(false); setAiPaused(false); });
  };
  const handleStopAI = () => { tts.stop(); setAiSpeaking(false); setAiPaused(false); };
  const handleToggleSpeech = () => {
    if (tts.isPaused()) {
      tts.resume();
      setAiPaused(false);
    } else {
      tts.pause();
      setAiPaused(true);
    }
  };

  const buildSimpleExplanation = () => {
    const findings = Array.isArray(result?.details) ? result.details.filter(Boolean) : [];
    const indicators = findings.length
      ? `The content screening noted: ${findings.join('. ')}.`
      : 'The content screening did not return specific scam indicators. That does not confirm the article is true.';
    const score = result?.risk_score !== undefined
      ? `The content risk score is ${result.risk_score} out of 100.`
      : '';
    return `${result?.short_summary || 'No article summary is available.'} In simple words, unverified means this tool did not check the story against outside news sources. It only screens for scam-style wording. ${indicators} ${score} Compare the claims with reliable independent reporting or an official source before sharing.`;
  };

  // Reset
  const handleReset = () => {
    handleStopAI();
    setHeadline(''); setText(''); setResult(null); setError('');
    setOcrText(''); setImagePreview(null); setPdfText(''); setPdfFileName('');
    setVoiceText(''); setChatMsgs([]); setChatInput('');
  };

  // PDF
  const handlePDF = () => {
    if (!result) return;
    generatePDFReport({
      title: 'News Verification Report',
      moduleType: 'News & Newspaper Intelligence',
      result: { ...result, prediction: result.verification_status, confidence: undefined },
      inputInfo: {
        Headline: headline || 'N/A',
        'Source Tab': tab,
        'Content Length': `${getAnalysisText().length} characters`,
        'Verification Status': 'UNVERIFIED - no external source checks performed',
      },
      findings: [
        `Status: ${result.verification_status}`,
        `Risk Level: ${result.risk_level}`,
        result.confidence !== undefined ? `Content detector confidence: ${Number(result.confidence).toFixed(2)}%` : null,
        result.risk_score !== undefined ? `Content risk score: ${result.risk_score}/100` : null,
      ].filter(Boolean),
      extraSections: [
        { title: 'Short Summary', content: result.short_summary },
        { title: 'Main Claim', content: result.main_claim || headline || 'Not identified' },
        chatMsgs.length > 0 ? { title: 'AI Q&A', content: chatMsgs.map(m => `${m.role === 'user' ? 'Q' : 'A'}: ${m.text}`).join('\n') } : null,
      ].filter(Boolean),
      recommendations: result.recommendations || [],
      technicalInfo: { 'Analysis Engine': 'Misinformation Classifier' },
    });
  };

  const vStatus = result?.verification_status || '';
  const statusColors = {
    VERIFIED: { bg: '#EAF7F1', border: '#B7E7D1', text: '#247352' },
    'PARTIALLY VERIFIED': { bg: '#FEF7ED', border: '#F8D8A7', text: '#8F5410' },
    MISLEADING: { bg: '#FDEFF2', border: '#F8BDC8', text: '#A8324C' },
    UNVERIFIED: { bg: '#F8F6FA', border: '#E5DDEC', text: '#6D5E7B' },
  };
  const sc = statusColors[vStatus] || statusColors.UNVERIFIED;

  const handleLoadSample = (isMisleading) => {
    if (isMisleading) {
      setHeadline('Emergency Notice: Central Bank Mandatory Currency Ban by Midnight');
      setText('LEAKED INTERNAL DIRECTIVE: Financial authorities have quietly mandated that all private commercial checking balances must be converted into digital sovereign tokens by midnight. Any citizen failing to withdraw physical cash reserves within 6 hours risks immediate asset freeze and 30% conversion penalty. Forward this urgent warning immediately!');
    } else {
      setHeadline('Global Renewable Energy Treaty Ratified by 195 Participating Nations');
      setText('GENEVA (Reuters) — Delegates from 195 countries reached a landmark consensus on Sunday, establishing binding efficiency mandates to double clean energy deployment targets by 2035. The agreement mobilizes $100 billion in annual public-private financing for emerging markets, according to the official United Nations Secretariat release.');
    }
    setTab('text');
    setResult(null);
    setError('');
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem', maxWidth: '1360px', margin: '0 auto' }}>
      
      {/* Top Banner */}
      <div
        style={{
          background: 'linear-gradient(135deg, #FFFFFF 0%, #FAF7FC 100%)',
          borderRadius: '16px',
          border: '1px solid #E5DDEC',
          padding: '1.5rem 1.75rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '1.25rem',
          boxShadow: '0 4px 20px -2px rgba(35, 21, 51, 0.04)',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '3px', background: 'linear-gradient(90deg, #7F5D99 0%, #9E7DB7 50%, #EDE4F5 100%)' }} />
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.15rem' }}>
          <div
            style={{
              width: '52px', height: '52px', borderRadius: '14px',
              background: 'linear-gradient(135deg, #EDE4F5 0%, #DFD0EE 100%)',
              border: '1px solid #D8C6E6', display: 'flex', alignItems: 'center', justifyContent: 'center',
              color: '#7F5D99', boxShadow: '0 4px 12px rgba(127, 93, 153, 0.15)', flexShrink: 0,
            }}
          >
            <Newspaper size={24} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '0.25rem' }}>
              <span style={{ fontSize: '0.7rem', fontWeight: '800', color: '#7F5D99', textTransform: 'uppercase', letterSpacing: '0.08em', backgroundColor: '#EDE4F5', padding: '0.2rem 0.65rem', borderRadius: '9999px' }}>
                Misinformation & Journalism Integrity
              </span>
              <span style={{ fontSize: '0.75rem', color: '#9E7DB7', fontWeight: '600' }}>•</span>
              <span style={{ fontSize: '0.75rem', color: '#6D5E7B', fontWeight: '600' }}>Multi-Modal NLP & OCR</span>
            </div>
            <h1 style={{ fontSize: '1.45rem', fontWeight: '800', color: '#221730', margin: 0, letterSpacing: '-0.02em' }}>
              News & Newspaper Verification
            </h1>
            <p style={{ fontSize: '0.85rem', color: '#6D5E7B', margin: '0.2rem 0 0 0' }}>
              Inspect news articles, printed newspaper clips, and online claims before you trust or distribute them.
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.45rem 0.95rem', borderRadius: '9999px', backgroundColor: '#EAF7F1', border: '1px solid #B7E7D1', fontSize: '0.78rem', fontWeight: '700', color: '#247352' }}>
          <span style={{ width: '7px', height: '7px', borderRadius: '50%', backgroundColor: '#3DA779', boxShadow: '0 0 0 3px rgba(61, 167, 121, 0.25)' }} />
          Multi-Modal Fact-Checking Online
        </div>
      </div>

      {/* Main 2-Column Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.75fr) minmax(320px, 1.25fr)', gap: '1.5rem', alignItems: 'start' }}>
        
        {/* LEFT COLUMN: Input & Results */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          
          <div className="cyber-card" style={{ padding: '2rem', borderRadius: '16px' }}>
            
            {/* Quick Test Scenarios */}
            <div style={{ marginBottom: '1.5rem', padding: '0.9rem 1.15rem', borderRadius: '12px', backgroundColor: '#F8F5FA', border: '1px solid #EBE4F2', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.8rem', fontWeight: '700', color: '#221730' }}>Quick Test Scenarios:</span>
              </div>
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                <button
                  onClick={() => handleLoadSample(true)}
                  type="button"
                  style={{
                    padding: '0.4rem 0.85rem', fontSize: '0.75rem', fontWeight: '700', borderRadius: '8px', cursor: 'pointer',
                    border: '1px solid #F8BDC8', backgroundColor: '#FDEFF2', color: '#A8324C',
                  }}
                >
                  ⚡ Viral Misinformation Sample
                </button>
                <button
                  onClick={() => handleLoadSample(false)}
                  type="button"
                  style={{
                    padding: '0.4rem 0.85rem', fontSize: '0.75rem', fontWeight: '700', borderRadius: '8px', cursor: 'pointer',
                    border: '1px solid #B7E7D1', backgroundColor: '#EAF7F1', color: '#247352',
                  }}
                >
                  ⚡ Verified Wire Report
                </button>
                <button
                  onClick={handleReset}
                  type="button"
                  style={{
                    padding: '0.4rem 0.75rem', fontSize: '0.75rem', fontWeight: '600', borderRadius: '8px', cursor: 'pointer',
                    border: '1px solid #E5DDEC', backgroundColor: '#FFFFFF', color: '#6D5E7B',
                  }}
                >
                  Reset
                </button>
              </div>
            </div>

            {/* Tabs */}
            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
              {TABS.map(t => (
                <button key={t.key} onClick={() => setTab(t.key)} style={{
                  padding: '0.55rem 1.1rem', fontSize: '0.82rem', fontWeight: '700', borderRadius: '9px', cursor: 'pointer',
                  backgroundColor: tab === t.key ? '#EDE4F5' : '#FFFFFF',
                  color: tab === t.key ? '#7F5D99' : '#6D5E7B',
                  border: tab === t.key ? '1px solid #7F5D99' : '1px solid #E5DDEC',
                  boxShadow: tab === t.key ? '0 2px 8px rgba(127, 93, 153, 0.15)' : 'none',
                  transition: 'all 0.15s ease', fontFamily: 'inherit',
                }}>
                  {t.label}
                </button>
              ))}
            </div>

            {/* Headline (shared) */}
            <div style={{ marginBottom: '1.15rem' }}>
              <label style={{ display: 'block', fontSize: '0.825rem', fontWeight: '700', color: '#221730', marginBottom: '0.4rem' }}>
                Headline / Title (optional)
              </label>
              <input type="text" value={headline} onChange={e => setHeadline(e.target.value)} placeholder="e.g. Breaking news claim or article title..." className="cyber-input" />
            </div>

            {/* Tab: Paste Text */}
            {tab === 'text' && (
              <div style={{ marginBottom: '1.15rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                  <label style={{ fontSize: '0.825rem', fontWeight: '700', color: '#221730' }}>Article Text / Claim Content</label>
                  {text && <span style={{ fontSize: '0.72rem', color: '#9E7DB7', fontWeight: '600' }}>{text.length} characters</span>}
                </div>
                <textarea value={text} onChange={e => setText(e.target.value)} placeholder="Paste the full news article, telegram forward, or claim text here..." className="cyber-input" rows={6} style={{ resize: 'vertical' }} />
              </div>
            )}

            {/* Tab: Upload Image */}
            {tab === 'image' && (
              <div style={{ marginBottom: '1.15rem' }}>
                <div onClick={() => imageRef.current?.click()}
                  onDrop={e => { e.preventDefault(); handleImageSelect(e.dataTransfer?.files?.[0]); }}
                  onDragOver={e => e.preventDefault()}
                  style={{
                    border: '2px dashed #D8C6E6', borderRadius: '12px', padding: '2rem', textAlign: 'center',
                    cursor: 'pointer', backgroundColor: '#FAF7FC', marginBottom: '1rem',
                  }}
                >
                  <input ref={imageRef} type="file" accept="image/*" onChange={e => handleImageSelect(e.target.files?.[0])} style={{ display: 'none' }} />
                  {imagePreview ? (
                    <img src={imagePreview} alt="Preview" style={{ maxHeight: '200px', maxWidth: '100%', borderRadius: '8px', objectFit: 'contain' }} />
                  ) : (
                    <>
                      <Newspaper size={36} color="#7F5D99" style={{ marginBottom: '0.5rem' }} />
                      <p style={{ fontSize: '0.9rem', color: '#221730', fontWeight: '700', margin: 0 }}>Click or drag a newspaper or article image (JPG, PNG, WEBP)</p>
                      <p style={{ fontSize: '0.78rem', color: '#6D5E7B', margin: '0.25rem 0 0 0' }}>Automatic on-device OCR extraction</p>
                    </>
                  )}
                </div>

                {ocrLoading && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem', backgroundColor: '#EDE4F5', borderRadius: '8px', marginBottom: '0.75rem', border: '1px solid #D8C6E6' }}>
                    <RefreshCw size={16} color="#7F5D99" className="animate-spin" />
                    <span style={{ fontSize: '0.825rem', color: '#221730', fontWeight: '600' }}>{ocrMsg}</span>
                  </div>
                )}

                {ocrText && (
                  <div>
                    <label style={{ display: 'block', fontSize: '0.825rem', fontWeight: '700', color: '#221730', marginBottom: '0.4rem' }}>Extracted Text (editable)</label>
                    <textarea value={ocrText} onChange={e => setOcrText(e.target.value)} className="cyber-input" rows={6} style={{ resize: 'vertical' }} />
                  </div>
                )}
              </div>
            )}

            {/* Tab: Upload PDF */}
            {tab === 'pdf' && (
              <div style={{ marginBottom: '1.15rem' }}>
                <div onClick={() => document.getElementById('pdf-input')?.click()} style={{
                  border: '2px dashed #D8C6E6', borderRadius: '12px', padding: '2rem', textAlign: 'center',
                  cursor: 'pointer', backgroundColor: '#FAF7FC', marginBottom: '1rem',
                }}>
                  <input id="pdf-input" type="file" accept=".pdf" onChange={e => handlePdfSelect(e.target.files?.[0])} style={{ display: 'none' }} />
                  <Newspaper size={36} color="#7F5D99" style={{ marginBottom: '0.5rem' }} />
                  <p style={{ fontSize: '0.9rem', color: '#221730', fontWeight: '700', margin: 0 }}>
                    {pdfFileName ? `Document: ${pdfFileName}` : 'Click to upload a PDF newspaper or article'}
                  </p>
                  <p style={{ fontSize: '0.78rem', color: '#6D5E7B', margin: '0.25rem 0 0 0' }}>PDF text layer parsed on demand</p>
                </div>

                {pdfLoading && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem', backgroundColor: '#EDE4F5', borderRadius: '8px', marginBottom: '0.75rem', border: '1px solid #D8C6E6' }}>
                    <RefreshCw size={16} color="#7F5D99" className="animate-spin" />
                    <span style={{ fontSize: '0.825rem', color: '#221730', fontWeight: '600' }}>Extracting text from PDF...</span>
                  </div>
                )}

                {pdfText && (
                  <div>
                    <label style={{ display: 'block', fontSize: '0.825rem', fontWeight: '700', color: '#221730', marginBottom: '0.4rem' }}>Extracted Article Text</label>
                    <textarea value={pdfText} onChange={e => setPdfText(e.target.value)} className="cyber-input" rows={6} style={{ resize: 'vertical' }} />
                  </div>
                )}
              </div>
            )}

            {/* Tab: Voice */}
            {tab === 'voice' && (
              <div style={{ marginBottom: '1.15rem' }}>
                <p style={{ color: '#6D5E7B', fontSize: '0.825rem', marginBottom: '0.8rem' }}>
                  Dictate article text on this device. The offline speech model runs in-browser.
                </p>
                <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', marginBottom: '1rem' }}>
                  {!isListening ? (
                    <button onClick={handleVoiceStart} disabled={dictationLoading} className="cyber-button-primary" style={{ padding: '0.65rem 1.25rem' }}>
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" /><path d="M19 10v2a7 7 0 01-14 0v-2" /><line x1="12" y1="19" x2="12" y2="23" /><line x1="8" y1="23" x2="16" y2="23" /></svg>
                      {dictationLoading ? <><RefreshCw size={16} className="animate-spin" /> Preparing...</> : 'Start Dictation'}
                    </button>
                  ) : (
                    <button onClick={handleVoiceStop} style={{
                      display: 'inline-flex', alignItems: 'center', gap: '0.45rem', padding: '0.65rem 1.25rem',
                      fontSize: '0.875rem', fontWeight: '600', borderRadius: '8px', cursor: 'pointer',
                      backgroundColor: '#FDEFF2', color: '#A8324C', border: '1px solid #F8BDC8',
                    }}>
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="4" y="4" width="16" height="16" rx="2" /></svg>
                      Listening... Stop
                    </button>
                  )}
                </div>
                {dictationMessage && (
                  <p role="status" style={{ color: '#7F5D99', fontSize: '0.78rem', marginBottom: '0.8rem', fontWeight: '600' }}>{dictationMessage}</p>
                )}
                <div>
                  <textarea value={voiceText} onChange={e => setVoiceText(e.target.value)} placeholder="Speak your news, or type or paste the article here..." className="cyber-input" rows={6} style={{ resize: 'vertical' }} />
                </div>
              </div>
            )}

            {/* Error */}
            {error && (
              <div style={{ backgroundColor: '#FDEFF2', border: '1px solid #F8BDC8', color: '#A8324C', padding: '0.85rem 1rem', borderRadius: '8px', fontSize: '0.85rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: '600' }}>
                <AlertTriangle size={16} color="#D9536F" /> {error}
              </div>
            )}

            {/* Action button */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid #F1ECF5', flexWrap: 'wrap', gap: '0.85rem' }}>
              <button className="cyber-button-primary" onClick={handleAnalyze} disabled={loading || isListening || dictationLoading} style={{ padding: '0.75rem 1.85rem', fontSize: '0.88rem', fontWeight: '700' }}>
                {loading ? <><RefreshCw size={16} className="animate-spin" /> Verifying Article...</> : 'Verify News Authenticity'}
              </button>
              <span style={{ fontSize: '0.75rem', color: '#9E7DB7', fontWeight: '600' }}>Zero Log Retention • Open Source Scrutiny</span>
            </div>
          </div>

          {/* Loading */}
          {loading && (
            <div className="cyber-card" style={{ padding: '2.5rem', textAlign: 'center', borderRadius: '16px' }}>
              <RefreshCw size={36} color="#7F5D99" className="animate-spin" style={{ marginBottom: '1rem' }} />
              <p style={{ fontSize: '1rem', color: '#221730', fontWeight: '800' }}>{loadingMsg}</p>
              <p style={{ fontSize: '0.82rem', color: '#6D5E7B', marginTop: '0.25rem' }}>Screening linguistic clickbait tokens, source attribution, and panic markers</p>
            </div>
          )}

          {/* Verification Result */}
          {result && !loading && (
            <div className="cyber-card" style={{ padding: '2rem', borderRadius: '16px', borderTop: `4px solid ${sc.text}` }}>
              <h3 style={{ fontSize: '0.78rem', fontWeight: '800', textTransform: 'uppercase', letterSpacing: '0.08em', color: '#7F5D99', marginBottom: '1.25rem' }}>
                News Verification Report
              </h3>

              {/* Status Banner */}
              <div style={{
                display: 'flex', alignItems: 'center', gap: '1rem', padding: '1.25rem 1.5rem', borderRadius: '12px', marginBottom: '1.5rem',
                backgroundColor: sc.bg, border: `1px solid ${sc.border}`,
              }}>
                <AlertTriangle size={32} color={sc.text} />
                <div>
                  <p style={{ fontSize: '1.4rem', fontWeight: '800', color: sc.text, margin: 0 }}>{vStatus}</p>
                  <p style={{ fontSize: '0.8rem', color: '#6D5E7B', margin: '0.15rem 0 0 0', fontWeight: '600' }}>Verification Verdict</p>
                </div>
                <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
                  {result.confidence !== undefined && <p style={{ fontSize: '1.2rem', fontWeight: '800', color: '#221730', margin: 0 }}>{Number(result.confidence).toFixed(1)}%</p>}
                  <p style={{ fontSize: '0.72rem', color: '#6D5E7B', margin: 0, fontWeight: '600' }}>Confidence</p>
                  {result.risk_score !== undefined && <p style={{ fontSize: '0.8rem', fontWeight: '700', color: sc.text, marginTop: '0.2rem' }}>Risk Score: {result.risk_score}/100</p>}
                </div>
              </div>

              {/* Short Summary */}
              <div style={{ marginBottom: '1.5rem' }}>
                <h4 style={{ fontSize: '0.78rem', fontWeight: '800', color: '#7F5D99', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.65rem' }}>Executive Summary</h4>
                <div style={{ backgroundColor: '#FBF9FD', padding: '1.15rem', borderRadius: '10px', border: '1px solid #ECE4F2' }}>
                  <p style={{ fontSize: '0.88rem', color: '#221730', lineHeight: '1.7', margin: 0 }}>{result.short_summary}</p>
                </div>
              </div>

              {/* Spoken Narration Help */}
              <section className="news-voice-help" aria-label="Spoken news explanation" style={{ backgroundColor: '#EDE4F5', border: '1px solid #D8C6E6', borderRadius: '12px', padding: '1.25rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
                <div className="news-voice-copy" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <span className="news-voice-icon" style={{ width: '40px', height: '40px', borderRadius: '10px', backgroundColor: '#FFFFFF', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#7F5D99', boxShadow: '0 2px 6px rgba(0,0,0,0.05)' }}>
                    <Volume2 size={20} />
                  </span>
                  <div>
                    <h4 style={{ fontSize: '0.88rem', fontWeight: '800', color: '#221730', margin: 0 }}>Audio Summary & Plain-Language Breakdown</h4>
                    <p style={{ fontSize: '0.78rem', color: '#6D5E7B', margin: '0.15rem 0 0 0' }}>Listen to an audible narration of the findings and what the unverified result means.</p>
                  </div>
                </div>
                <div className="news-voice-controls" style={{ display: 'flex', gap: '0.5rem' }}>
                  {!aiSpeaking ? (
                    <button type="button" className="cyber-button-primary" onClick={() => handlePlayAIAnswer(buildSimpleExplanation())} style={{ padding: '0.5rem 1rem', fontSize: '0.82rem' }}>
                      <Play size={15} /> Play explanation
                    </button>
                  ) : (
                    <>
                      <button type="button" className="cyber-button-secondary" onClick={handleToggleSpeech} style={{ padding: '0.5rem 0.85rem' }}>
                        {aiPaused ? <Play size={14} /> : <Pause size={14} />}{aiPaused ? 'Resume' : 'Pause'}
                      </button>
                      <button type="button" className="cyber-button-secondary" onClick={handleStopAI} style={{ padding: '0.5rem 0.85rem' }}>
                        <Square size={13} /> Stop
                      </button>
                    </>
                  )}
                </div>
              </section>

              {/* Key Findings */}
              <div style={{ marginBottom: '1.5rem' }}>
                <h4 style={{ fontSize: '0.78rem', fontWeight: '800', color: '#7F5D99', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.65rem' }}>Forensic Observations</h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {[
                    { label: 'Primary Claim', value: result.main_claim || headline || 'Unspecified Headline Claim' },
                    { label: 'Content Screening', value: vStatus },
                    { label: 'Risk Score Rating', value: result.risk_score !== undefined ? `${result.risk_score}/100` : 'Not rated' },
                    { label: 'Independent Attributions', value: 'Heuristic linguistic verification only (no third-party wire check)' },
                  ].map((f, i) => (
                    <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', backgroundColor: '#FBF9FD', padding: '0.75rem 1rem', borderRadius: '8px', border: '1px solid #ECE4F2' }}>
                      <span style={{ fontSize: '0.72rem', fontWeight: '800', color: '#6D5E7B', textTransform: 'uppercase', minWidth: '150px' }}>{f.label}</span>
                      <span style={{ fontSize: '0.85rem', color: '#221730', fontWeight: '600' }}>{f.value}</span>
                    </div>
                  ))}
                </div>
              </div>

              <ReportActions result={result} moduleType="news" onDownloadPDF={handlePDF} onAnalyzeAnother={handleReset} />
            </div>
          )}

        </div>

        {/* RIGHT COLUMN: Fact-Checking & Credibility Radar */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          
          <div className="cyber-card" style={{ padding: '1.5rem', borderRadius: '16px' }}>
            <h3 style={{ fontSize: '0.92rem', fontWeight: '800', color: '#221730', margin: '0 0 1rem 0' }}>
              Fact-Checking Intelligence Radar
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
              {[
                { name: 'Source Attribution & Byline', desc: 'Checks for named journalists and verified publishing masthead', flag: 'Crucial' },
                { name: 'Sensationalism & Clickbait Index', desc: 'Detects emotional manipulation and hyperbolic phrasing', flag: 'High Risk' },
                { name: 'Fabricated Quote Markers', desc: 'Identifies unsupported quotes without institutional attribution', flag: 'Elevated' },
                { name: 'Artificial Urgency Framing', desc: 'Flags calls to immediate action before censorship claims', flag: 'Critical' },
              ].map((item, idx) => (
                <div key={idx} style={{ padding: '0.85rem', borderRadius: '10px', backgroundColor: '#FBF9FD', border: '1px solid #EDE5F4' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.2rem' }}>
                    <span style={{ fontSize: '0.82rem', fontWeight: '700', color: '#221730' }}>{item.name}</span>
                    <span style={{ fontSize: '0.68rem', fontWeight: '700', padding: '0.15rem 0.5rem', borderRadius: '4px', backgroundColor: '#EDE4F5', color: '#7F5D99' }}>{item.flag}</span>
                  </div>
                  <p style={{ fontSize: '0.74rem', color: '#6D5E7B', margin: 0, lineHeight: '1.4' }}>{item.desc}</p>
                </div>
              ))}
            </div>
          </div>

          <div style={{ padding: '1.35rem', borderRadius: '16px', backgroundColor: '#EDE4F5', border: '1px solid #D8C6E6', display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
            <span style={{ fontSize: '0.78rem', fontWeight: '800', color: '#221730', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Media Literacy Reminder
            </span>
            <p style={{ fontSize: '0.75rem', color: '#6D5E7B', margin: 0, lineHeight: '1.5' }}>
              Automated AI verification flags common textual misinformation heuristics. For critical public health, geopolitical, or financial matters, always verify through recognized primary sources.
            </p>
          </div>

        </div>

      </div>

      {/* AI Chat Card */}
      {result && !loading && (
        <div className="cyber-card" style={{ padding: '2rem', borderRadius: '16px' }}>
          <h3 style={{ fontSize: '0.78rem', fontWeight: '800', textTransform: 'uppercase', letterSpacing: '0.08em', color: '#7F5D99', marginBottom: '1.25rem' }}>
            Article Questions & AI Interactive Insights
          </h3>

          {/* Suggested Questions */}
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '1.25rem' }}>
            {SUGGESTED_QS.map((q, i) => (
              <button key={i} onClick={() => handleAskAI(q)} disabled={chatLoading} style={{
                padding: '0.4rem 0.85rem', fontSize: '0.75rem', fontWeight: '700', borderRadius: '9999px',
                backgroundColor: '#EDE4F5', color: '#7F5D99', border: '1px solid #D8C6E6',
                cursor: 'pointer', fontFamily: 'inherit', transition: 'all 0.15s ease',
              }}>
                {q}
              </button>
            ))}
          </div>

          {/* Chat Messages */}
          {chatMsgs.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginBottom: '1.25rem', maxHeight: '400px', overflowY: 'auto' }}>
              {chatMsgs.map((m, i) => (
                <div key={i} style={{
                  alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
                  maxWidth: '85%',
                  backgroundColor: m.role === 'user' ? '#EDE4F5' : '#FBF9FD',
                  border: `1px solid ${m.role === 'user' ? '#D8C6E6' : '#ECE4F2'}`,
                  padding: '0.85rem 1.15rem', borderRadius: '12px',
                }}>
                  <p style={{ fontSize: '0.68rem', fontWeight: '800', color: m.role === 'user' ? '#7F5D99' : '#6D5E7B', textTransform: 'uppercase', marginBottom: '0.25rem', letterSpacing: '0.04em' }}>
                    {m.role === 'user' ? 'You' : 'TrustGuard Analyst'}
                  </p>
                  <p style={{ fontSize: '0.875rem', color: '#221730', lineHeight: '1.65', margin: 0, fontWeight: '500' }}>{m.text}</p>
                  {m.role === 'ai' && (
                    <div style={{ marginTop: '0.65rem', display: 'flex', gap: '0.5rem' }}>
                      {!aiSpeaking ? (
                        <button onClick={() => handlePlayAIAnswer(m.text)} style={{
                          padding: '0.25rem 0.65rem', fontSize: '0.72rem', fontWeight: '700', borderRadius: '6px',
                          backgroundColor: '#EDE4F5', color: '#7F5D99', border: '1px solid #D8C6E6',
                          cursor: 'pointer', fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: '0.3rem',
                        }}>
                          <Play size={12} /> Play
                        </button>
                      ) : (
                        <button onClick={handleStopAI} style={{
                          padding: '0.25rem 0.65rem', fontSize: '0.72rem', fontWeight: '700', borderRadius: '6px',
                          backgroundColor: '#FDEFF2', color: '#A8324C', border: '1px solid #F8BDC8',
                          cursor: 'pointer', fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: '0.3rem',
                        }}>
                          <Square size={12} /> Stop
                        </button>
                      )}
                    </div>
                  )}
                </div>
              ))}
              {chatLoading && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem', backgroundColor: '#FBF9FD', borderRadius: '8px', border: '1px solid #ECE4F2' }}>
                  <RefreshCw size={14} color="#7F5D99" className="animate-spin" />
                  <span style={{ fontSize: '0.825rem', color: '#6D5E7B' }}>Analyzing question context...</span>
                </div>
              )}
            </div>
          )}

          {/* Chat Input */}
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            <input value={chatInput} onChange={e => setChatInput(e.target.value)} placeholder="Ask something about this article..."
              className="cyber-input" style={{ flex: 1, minWidth: '200px' }}
              onKeyDown={e => { if (e.key === 'Enter') handleAskAI(chatInput); }}
            />
            <button onClick={() => handleAskAI(chatInput)} className="cyber-button-primary" disabled={chatLoading || !chatInput.trim()} style={{ padding: '0.6rem 1.25rem', fontSize: '0.85rem' }}>
              Submit Question
            </button>
            {!chatVoiceListening ? (
              <button onClick={handleChatVoiceStart} className="cyber-button-secondary" style={{ padding: '0.6rem 0.85rem' }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" /><path d="M19 10v2a7 7 0 01-14 0v-2" /><line x1="12" y1="19" x2="12" y2="23" /><line x1="8" y1="23" x2="16" y2="23" /></svg>
              </button>
            ) : (
              <button onClick={handleChatVoiceStop} style={{
                padding: '0.6rem 0.85rem', borderRadius: '8px', cursor: 'pointer',
                backgroundColor: '#FDEFF2', color: '#A8324C', border: '1px solid #F8BDC8',
                fontFamily: 'inherit', display: 'flex', alignItems: 'center',
              }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="4" y="4" width="16" height="16" rx="2" /></svg>
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
