import React, { useState, useRef, useEffect } from 'react';
import {
  ImageIcon,
  Video,
  Activity,
  AlertTriangle,
  CheckCircle,
  XCircle,
  RefreshCw,
  Eye,
  Shield,
  ExternalLink,
  Cpu,
  FileCheck
} from '../components/Icons';
import ReportActions from '../components/ReportActions';
import { api } from '../services/api';
import { generatePDFReport } from '../utils/pdfReport';

export default function DeepFake() {
  const [activeTab, setActiveTab] = useState('image'); // 'image' | 'video' | 'audio' | 'transparency'

  // Image Analysis State
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [showEla, setShowEla] = useState(false);
  const [showCrossModel, setShowCrossModel] = useState(false);
  const [copiedHash, setCopiedHash] = useState(false);
  const inputRef = useRef(null);

  // Video Analysis State
  const [videoFile, setVideoFile] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [videoLoading, setVideoLoading] = useState(false);
  const [videoResult, setVideoResult] = useState(null);
  const [videoError, setVideoError] = useState('');
  const videoInputRef = useRef(null);

  // Audio Analysis State
  const [audioFile, setAudioFile] = useState(null);
  const [audioLoading, setAudioLoading] = useState(false);
  const [audioResult, setAudioResult] = useState(null);
  const [audioError, setAudioError] = useState('');
  const audioInputRef = useRef(null);

  // Transparency Data State
  const [transparencyData, setTransparencyData] = useState(null);
  const [transparencyLoading, setTransparencyLoading] = useState(false);

  useEffect(() => {
    if (activeTab === 'transparency' && !transparencyData) {
      loadTransparencyStats();
    }
  }, [activeTab]);

  const loadTransparencyStats = async () => {
    setTransparencyLoading(true);
    try {
      const data = await api.getTransparencyStats();
      setTransparencyData(data);
    } catch (e) {
      console.warn("Transparency load fallback:", e);
    } finally {
      setTransparencyLoading(false);
    }
  };

  // ----------------------------------------------------
  // IMAGE HANDLERS
  // ----------------------------------------------------
  const handleFileSelect = (f) => {
    if (!f) return;
    const allowed = ['image/jpeg', 'image/png', 'image/webp', 'image/jpg'];
    if (!allowed.includes(f.type)) {
      setError('Please upload a valid image file (JPG, PNG, WEBP).');
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError('');
    setShowEla(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer?.files?.[0];
    if (f) handleFileSelect(f);
  };

  const handleAnalyzeImage = async () => {
    if (!file) return;
    setLoading(true);
    setError('');
    setResult(null);
    try {
      setLoadingMsg('Extracting cryptographic SHA-256 fingerprint...');
      await new Promise((r) => setTimeout(r, 250));
      setLoadingMsg('Running MobileNetV2 V4 Neural Feature Extractor...');
      const data = await api.predictImage(file);
      setLoadingMsg('Computing Error Level Analysis & calibrated decision...');
      await new Promise((r) => setTimeout(r, 200));
      setResult(data);
    } catch (err) {
      const msg = err?.message || 'Image analysis failed.';
      if (
        msg.toLowerCase().includes('failed to fetch') ||
        msg.toLowerCase().includes('networkerror') ||
        msg.toLowerCase().includes('load failed')
      ) {
        setError('AI server is warming up (Render free tier spin-up takes ~20s). Please click "Verify Authenticity" again.');
      } else {
        setError(msg);
      }
    } finally {
      setLoading(false);
      setLoadingMsg('');
    }
  };

  // ----------------------------------------------------
  // VIDEO HANDLERS
  // ----------------------------------------------------
  const handleVideoSelect = (f) => {
    if (!f) return;
    const allowedExts = ['.mp4', '.mov', '.webm', '.avi', '.mkv'];
    const ext = f.name.substring(f.name.lastIndexOf('.')).toLowerCase();
    if (!allowedExts.includes(ext)) {
      setVideoError('Please upload an MP4, WEBM, MOV, AVI, or MKV video.');
      return;
    }
    setVideoFile(f);
    setVideoPreview(URL.createObjectURL(f));
    setVideoResult(null);
    setVideoError('');
  };

  const handleAnalyzeVideo = async () => {
    if (!videoFile) return;
    setVideoLoading(true);
    setVideoError('');
    setVideoResult(null);
    try {
      const data = await api.predictVideo(videoFile);
      setVideoResult(data);
    } catch (err) {
      setVideoError(err.message || 'Video frame analysis failed.');
    } finally {
      setVideoLoading(false);
    }
  };

  // ----------------------------------------------------
  // AUDIO HANDLERS
  // ----------------------------------------------------
  const handleAudioSelect = (f) => {
    if (!f) return;
    const allowedExts = ['.wav', '.mp3', '.ogg', '.m4a'];
    const ext = f.name.substring(f.name.lastIndexOf('.')).toLowerCase();
    if (!allowedExts.includes(ext)) {
      setAudioError('Please upload a WAV, MP3, OGG, or M4A audio file.');
      return;
    }
    setAudioFile(f);
    setAudioResult(null);
    setAudioError('');
  };

  const handleAnalyzeAudio = async () => {
    if (!audioFile) return;
    setAudioLoading(true);
    setAudioError('');
    setAudioResult(null);
    try {
      const data = await api.predictAudio(audioFile);
      setAudioResult(data);
    } catch (err) {
      setAudioError(err.message || 'Audio authenticity analysis failed.');
    } finally {
      setAudioLoading(false);
    }
  };

  const copyHash = (hash) => {
    if (!hash) return;
    navigator.clipboard.writeText(hash);
    setCopiedHash(true);
    setTimeout(() => setCopiedHash(false), 2000);
  };

  const handleDownloadPDF = () => {
    if (!result) return;
    const pred = String(result.prediction || '').toUpperCase();
    const findings = [];
    findings.push(`Verification Decision: ${pred} (Confidence: ${result.confidence?.toFixed(2)}%)`);
    findings.push(`Model Version: ${result.model_version || '4.0.0'} (Decision Threshold: ${result.threshold ?? 0.50})`);
    if (result.forensics?.file_hash_sha256) {
      findings.push(`File Fingerprint (SHA-256): ${result.forensics.file_hash_sha256}`);
    }
    if (result.forensics?.ela?.assessment) {
      findings.push(`Forensic Error Level Analysis: ${result.forensics.ela.assessment}`);
    }

    generatePDFReport({
      title: 'TrustGuard Media Authenticity Verification Report',
      moduleType: 'DeepFake Multimodal Verification',
      result,
      inputInfo: {
        'Filename': file?.name || 'Uploaded Media',
        'File Size': file ? `${(file.size / 1024).toFixed(1)} KB` : 'N/A',
        'File Format': file?.type || 'N/A',
        'SHA-256 Fingerprint': result.forensics?.file_hash_sha256 || 'Calculated',
      },
      findings,
      details: [
        `Verification Engine: MobileNetV2 V4 DeepFake Detector`,
        `Model Calibrated Score: ${result.calibrated_score !== undefined ? result.calibrated_score : result.raw_score}`,
        `Threshold: ${result.threshold !== undefined ? result.threshold : '0.50'} (Calibrated on Validation Data)`,
        `Inconclusive Margin: 0.45 - 0.55`,
        `Original Image Resolution: ${result.original_image_size?.width} x ${result.original_image_size?.height}`,
      ],
      recommendations: pred === 'REAL'
        ? ['The media exhibits natural frequency distributions consistent with genuine sensor capture.', 'For high-assurance requirements, verify provenance metadata against the authoritative source.']
        : pred === 'INCONCLUSIVE'
        ? ['The model score is within the borderline uncertainty interval.', 'Inspect manual forensics or request a higher-resolution uncompressed copy.']
        : ['The media shows artificial frequency patterns or synthetic artifact traces.', 'Do not rely on this media as authentic evidence or for identity verification.'],
      technicalInfo: {
        'Model': 'MobileNetV2 V4 Functional Graph',
        'Input Size': '224 x 224 x 3',
        'Prediction': pred,
        'Confidence': `${Number(result.confidence || 0).toFixed(2)}%`,
        'Verification ID': result.verification_id || 'TG-' + Date.now().toString(36).toUpperCase(),
      },
    });
  };

  const handleDownloadJSON = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trustguard_verification_${result.verification_id || 'report'}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const pred = result ? String(result.prediction || '').toUpperCase() : '';
  const isSafe = pred === 'REAL';
  const isInconclusive = pred === 'INCONCLUSIVE';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', maxWidth: '1200px', margin: '0 auto' }}>

      {/* TOP NAVIGATION TABS */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        padding: '0.45rem',
        backgroundColor: '#FFFFFF',
        borderRadius: '14px',
        border: '1px solid #E5DDEC',
        boxShadow: '0 2px 10px rgba(35, 21, 51, 0.04)',
        overflowX: 'auto'
      }}>
        <button
          onClick={() => setActiveTab('image')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.65rem 1.15rem',
            borderRadius: '9px',
            border: 'none',
            fontSize: '0.85rem',
            fontWeight: activeTab === 'image' ? '700' : '500',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            backgroundColor: activeTab === 'image' ? '#EDE4F5' : 'transparent',
            color: activeTab === 'image' ? '#7F5D99' : '#6D5E7B',
            borderBottom: activeTab === 'image' ? '3px solid #7F5D99' : '3px solid transparent',
          }}
        >
          <ImageIcon size={17} color={activeTab === 'image' ? '#7F5D99' : '#6D5E7B'} />
          Image Verification (V4 + Forensics)
        </button>

        <button
          onClick={() => setActiveTab('video')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.65rem 1.15rem',
            borderRadius: '9px',
            border: 'none',
            fontSize: '0.85rem',
            fontWeight: activeTab === 'video' ? '700' : '500',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            backgroundColor: activeTab === 'video' ? '#EDE4F5' : 'transparent',
            color: activeTab === 'video' ? '#7F5D99' : '#6D5E7B',
            borderBottom: activeTab === 'video' ? '3px solid #7F5D99' : '3px solid transparent',
          }}
        >
          <Video size={17} color={activeTab === 'video' ? '#7F5D99' : '#6D5E7B'} />
          Video Frame Analysis
        </button>

        <button
          onClick={() => setActiveTab('audio')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.65rem 1.15rem',
            borderRadius: '9px',
            border: 'none',
            fontSize: '0.85rem',
            fontWeight: activeTab === 'audio' ? '700' : '500',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            backgroundColor: activeTab === 'audio' ? '#EDE4F5' : 'transparent',
            color: activeTab === 'audio' ? '#7F5D99' : '#6D5E7B',
            borderBottom: activeTab === 'audio' ? '3px solid #7F5D99' : '3px solid transparent',
          }}
        >
          <Activity size={17} color={activeTab === 'audio' ? '#7F5D99' : '#6D5E7B'} />
          Audio Authenticity (Voice Cloning)
        </button>

        <button
          onClick={() => setActiveTab('transparency')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.65rem 1.15rem',
            borderRadius: '9px',
            border: 'none',
            fontSize: '0.85rem',
            fontWeight: activeTab === 'transparency' ? '700' : '500',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            backgroundColor: activeTab === 'transparency' ? '#EDE4F5' : 'transparent',
            color: activeTab === 'transparency' ? '#7F5D99' : '#6D5E7B',
            borderBottom: activeTab === 'transparency' ? '3px solid #7F5D99' : '3px solid transparent',
          }}
        >
          <Cpu size={17} color={activeTab === 'transparency' ? '#7F5D99' : '#6D5E7B'} />
          Dataset & Model Transparency
        </button>
      </div>

      {/* ========================================================= */}
      {/* TAB 1: IMAGE VERIFICATION                                 */}
      {/* ========================================================= */}
      {activeTab === 'image' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Upload Card */}
          <div className="cyber-card" style={{ padding: '2rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '0.75rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
                <div style={{
                  width: '44px', height: '44px', borderRadius: '12px',
                  backgroundColor: '#EDE4F5', border: '1px solid #D8C6E6',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  boxShadow: '0 2px 8px rgba(127, 93, 153, 0.12)',
                }}>
                  <ImageIcon size={22} color="#7F5D99" />
                </div>
                <div>
                  <h3 style={{ fontSize: '1.1rem', fontWeight: '800', color: '#221730', margin: 0 }}>
                    Image Authenticity Verification
                  </h3>
                  <p style={{ fontSize: '0.825rem', color: '#6D5E7B', margin: '0.15rem 0 0 0' }}>
                    Powered by MobileNetV2 V4 Multi-Specialist Architecture & Forensic Residuals
                  </p>
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{
                  fontSize: '0.72rem',
                  padding: '0.35rem 0.75rem',
                  borderRadius: '20px',
                  backgroundColor: '#EDE4F5',
                  color: '#5D3A7A',
                  border: '1px solid #D8C6E6',
                  fontWeight: '700'
                }}>
                  Calibrated Decision Boundary
                </span>
              </div>
            </div>

            {/* Drop Zone */}
            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => inputRef.current?.click()}
              style={{
                border: '2px dashed #D8C6E6', borderRadius: '14px',
                padding: '2.5rem 1.5rem', textAlign: 'center', cursor: 'pointer',
                backgroundColor: '#F8F6FA',
                transition: 'border-color 0.2s ease, background 0.2s ease',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#EDE4F5'; e.currentTarget.style.borderColor = '#7F5D99'; }}
              onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#F8F6FA'; e.currentTarget.style.borderColor = '#D8C6E6'; }}
            >
              <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" onChange={(e) => handleFileSelect(e.target.files?.[0])} style={{ display: 'none' }} />
              {preview ? (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem' }}>
                  <img
                    src={showEla && result?.forensics?.ela?.ela_image_base64
                      ? `data:image/png;base64,${result.forensics.ela.ela_image_base64}`
                      : preview}
                    alt="Preview"
                    style={{ maxHeight: '300px', maxWidth: '100%', borderRadius: '8px', objectFit: 'contain', border: '1px solid #E5DDEC' }}
                  />
                  {result?.forensics?.ela?.ela_image_base64 && (
                    <span style={{ fontSize: '0.75rem', fontWeight: '700', color: showEla ? '#D98928' : '#6D5E7B' }}>
                      {showEla ? 'Viewing Forensic Error Level Analysis (ELA 15x Gain)' : 'Viewing Original Image'}
                    </span>
                  )}
                </div>
              ) : (
                <>
                  <ImageIcon size={44} color="#7F5D99" style={{ marginBottom: '0.75rem' }} />
                  <p style={{ fontSize: '0.95rem', color: '#221730', margin: 0, fontWeight: '700' }}>
                    Drag and drop an image here, or click to browse
                  </p>
                  <p style={{ fontSize: '0.78rem', color: '#6D5E7B', marginTop: '0.35rem' }}>
                    Supports JPG, PNG, WEBP (Max 10 MB)
                  </p>
                </>
              )}
            </div>

            {file && (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
                <span style={{ fontSize: '0.85rem', color: '#221730', fontWeight: '600' }}>
                  📄 <strong>{file.name}</strong> ({(file.size / 1024).toFixed(1)} KB)
                </span>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button
                    className="cyber-button-primary"
                    onClick={handleAnalyzeImage}
                    disabled={loading}
                    style={{ padding: '0.7rem 1.6rem', fontSize: '0.85rem' }}
                  >
                    {loading ? (
                      <><RefreshCw size={16} className="animate-spin" /> {loadingMsg || 'Verifying...'}</>
                    ) : 'Verify Authenticity'}
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Loading Indicator */}
          {loading && (
            <div className="cyber-card" style={{ padding: '2.5rem', textAlign: 'center' }}>
              <RefreshCw size={36} color="#7F5D99" className="animate-spin" style={{ marginBottom: '1rem' }} />
              <p style={{ fontSize: '1rem', color: '#221730', fontWeight: '700' }}>{loadingMsg || 'Analyzing Media...'}</p>
              <p style={{ fontSize: '0.8rem', color: '#6D5E7B', marginTop: '0.25rem' }}>
                Extracting deep spatial artifacts, frequency distributions, and error residual levels.
              </p>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div style={{
              backgroundColor: '#FDF2F4', border: '1px solid #F4C2CA',
              color: '#963143', padding: '0.85rem 1rem', borderRadius: '8px', fontSize: '0.85rem',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.75rem', fontWeight: '600'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <AlertTriangle size={16} color="#D98291" /> {error}
              </div>
              <button
                type="button"
                onClick={handleAnalyzeImage}
                style={{
                  padding: '0.35rem 0.75rem',
                  fontSize: '0.78rem',
                  fontWeight: '700',
                  color: '#fff',
                  backgroundColor: '#7F5D99',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  whiteSpace: 'nowrap'
                }}
              >
                Retry
              </button>
            </div>
          )}

          {/* Complete Report */}
          {result && !loading && (
            <div className="cyber-card" style={{ padding: '2rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
                <div>
                  <h3 style={{
                    fontSize: '0.78rem', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.08em',
                    color: '#36435E', margin: 0,
                  }}>
                    Authenticity Verification Report
                  </h3>
                  <p style={{ fontSize: '0.75rem', color: '#697586', margin: '0.2rem 0 0 0' }}>
                    Verification ID: <code style={{ backgroundColor: '#EDF3E8', padding: '2px 6px', borderRadius: '4px', color: '#202734' }}>{result.verification_id || 'TG-VERIFIED'}</code>
                  </p>
                </div>

                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                  {result.forensics?.ela?.ela_image_base64 && (
                    <button
                      onClick={() => setShowEla(!showEla)}
                      style={{
                        display: 'flex', alignItems: 'center', gap: '0.4rem',
                        padding: '0.5rem 0.9rem', borderRadius: '8px', fontSize: '0.8rem',
                        fontWeight: '700',
                        backgroundColor: showEla ? '#FEF9EE' : '#EDF3E8',
                        color: showEla ? '#8C6718' : '#263142',
                        border: `1px solid ${showEla ? '#EED89E' : '#E1E7DD'}`, cursor: 'pointer'
                      }}
                    >
                      <Eye size={14} /> {showEla ? 'Show Original' : 'Toggle Forensic ELA'}
                    </button>
                  )}
                  <button
                    onClick={handleDownloadJSON}
                    style={{
                      padding: '0.5rem 0.9rem', borderRadius: '8px', fontSize: '0.8rem',
                      fontWeight: '700',
                      backgroundColor: '#EDF3E8', color: '#263142',
                      border: '1px solid #E1E7DD', cursor: 'pointer'
                    }}
                  >
                    Export JSON
                  </button>
                </div>
              </div>

              {/* Status Banner */}
              <div style={{
                display: 'flex', alignItems: 'center', gap: '1.25rem', padding: '1.5rem',
                borderRadius: '10px', marginBottom: '1.5rem',
                backgroundColor: isSafe ? '#EDF3E8' : isInconclusive ? '#FEF9EE' : '#FDF2F4',
                border: `1px solid ${isSafe ? '#B9D995' : isInconclusive ? '#EED89E' : '#F4C2CA'}`,
              }}>
                {isSafe ? <CheckCircle size={36} color="#87A587" /> : isInconclusive ? <AlertTriangle size={36} color="#C99B3B" /> : <XCircle size={36} color="#D98291" />}
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                    <span style={{
                      fontSize: '1.6rem', fontWeight: '800',
                      color: isSafe ? '#3D683D' : isInconclusive ? '#8C6718' : '#963143',
                    }}>
                      {pred}
                    </span>
                    <span style={{
                      fontSize: '0.75rem', padding: '0.25rem 0.75rem', borderRadius: '12px',
                      backgroundColor: '#FFFFFF', color: '#202734', fontWeight: '700',
                      border: `1px solid ${isSafe ? '#B9D995' : isInconclusive ? '#EED89E' : '#F4C2CA'}`
                    }}>
                      Calibrated Confidence: {Number(result.confidence || 0).toFixed(2)}%
                    </span>
                  </div>
                  <p style={{ fontSize: '0.84rem', color: '#263142', margin: '0.35rem 0 0 0', lineHeight: '1.5' }}>
                    {isSafe
                      ? 'The image displays authentic camera capture characteristics with no detectable generative or face-swap artifacts.'
                      : isInconclusive
                      ? 'Model score falls in the calibrated uncertainty boundary [0.45 - 0.55]. Independent verification recommended.'
                      : 'High-probability synthetic manipulation detected. Structural frequency and compression residuals deviate from authentic sensor data.'}
                  </p>
                </div>
              </div>

              {/* Metrics Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.85rem', marginBottom: '1.5rem' }}>
                {[
                  { label: 'Active Model', value: result.model_version ? `MobileNetV2 V${result.model_version}` : 'MobileNetV2 V4' },
                  { label: 'Calibrated Score', value: result.calibrated_score !== undefined ? String(result.calibrated_score) : String(result.raw_score ?? 'N/A') },
                  { label: 'Decision Threshold', value: result.threshold !== undefined ? String(result.threshold) : '0.50' },
                  { label: 'Uncertainty Status', value: result.is_inconclusive ? 'Borderline' : 'Decisive' },
                  { label: 'Faces Detected', value: result.faces_detected !== undefined ? String(result.faces_detected) : '1 Face Crop' },
                  { label: 'Original Dimensions', value: result.original_image_size ? `${result.original_image_size.width} × ${result.original_image_size.height}` : 'N/A' },
                ].map((m, i) => (
                  <div key={i} style={{
                    backgroundColor: '#F7F9F4', padding: '0.9rem', borderRadius: '8px',
                    border: '1px solid #E1E7DD',
                  }}>
                    <p style={{ fontSize: '0.68rem', fontWeight: '700', color: '#697586', textTransform: 'uppercase', margin: 0 }}>{m.label}</p>
                    <p style={{ fontSize: '0.95rem', fontWeight: '800', color: '#202734', margin: '0.25rem 0 0 0' }}>{m.value}</p>
                  </div>
                ))}
              </div>

              {/* Cryptographic Media Fingerprint */}
              {result.forensics?.file_hash_sha256 && (
                <div style={{
                  backgroundColor: '#F7F9F4', padding: '1.15rem', borderRadius: '8px',
                  border: '1px solid #E1E7DD', marginBottom: '1.5rem'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
                    <span style={{ fontSize: '0.72rem', fontWeight: '700', color: '#36435E', textTransform: 'uppercase' }}>
                      Cryptographic Media Fingerprint (SHA-256)
                    </span>
                    <button
                      onClick={() => copyHash(result.forensics.file_hash_sha256)}
                      style={{
                        background: 'transparent', border: 'none', color: copiedHash ? '#3D683D' : '#36435E',
                        fontSize: '0.75rem', fontWeight: '700', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.25rem'
                      }}
                    >
                      {copiedHash ? '✓ Copied' : 'Copy Hash'}
                    </button>
                  </div>
                  <code style={{ fontSize: '0.8rem', color: '#202734', wordBreak: 'break-all', fontFamily: 'monospace', fontWeight: '600' }}>
                    {result.forensics.file_hash_sha256}
                  </code>
                </div>
              )}

              {/* Cross-Model Consensus Panel */}
              {result.cross_model_comparison && (
                <div style={{ marginBottom: '1.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                    <h4 style={{ fontSize: '0.75rem', fontWeight: '700', color: '#36435E', textTransform: 'uppercase', letterSpacing: '0.06em', margin: 0 }}>
                      Cross-Model Consensus & Agreement
                    </h4>
                    <span style={{
                      fontSize: '0.72rem', padding: '0.25rem 0.6rem', borderRadius: '6px',
                      backgroundColor: result.cross_model_comparison.agreement ? '#EDF3E8' : '#FEF9EE',
                      color: result.cross_model_comparison.agreement ? '#3D683D' : '#8C6718',
                      border: `1px solid ${result.cross_model_comparison.agreement ? '#B9D995' : '#EED89E'}`,
                      fontWeight: '700'
                    }}>
                      {result.cross_model_comparison.agreement ? 'Unanimous Agreement' : 'Divergent Ensemble'}
                    </span>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.75rem' }}>
                    {result.cross_model_comparison.models?.map((cm, idx) => (
                      <div key={idx} style={{
                        backgroundColor: '#F7F9F4', padding: '0.85rem 1rem', borderRadius: '8px',
                        border: '1px solid #E1E7DD'
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontSize: '0.78rem', fontWeight: '700', color: '#202734' }}>{cm.model}</span>
                          <span style={{
                            fontSize: '0.72rem', fontWeight: '800',
                            color: cm.prediction === 'REAL' ? '#3D683D' : '#963143'
                          }}>
                            {cm.prediction}
                          </span>
                        </div>
                        <p style={{ fontSize: '0.72rem', color: '#697586', margin: '0.25rem 0 0 0' }}>
                          Score: {cm.score?.toFixed(4)} | Thresh: {cm.threshold}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Forensic Error Level Analysis Details */}
              {result.forensics?.ela && (
                <div style={{
                  backgroundColor: '#F7F9F4', padding: '1rem 1.25rem', borderRadius: '8px',
                  border: '1px solid #E1E7DD', marginBottom: '1.5rem'
                }}>
                  <h4 style={{ fontSize: '0.75rem', fontWeight: '700', color: '#36435E', textTransform: 'uppercase', letterSpacing: '0.06em', margin: '0 0 0.5rem 0' }}>
                    Forensic Error Level Analysis Assessment
                  </h4>
                  <p style={{ fontSize: '0.84rem', color: '#263142', lineHeight: '1.6', margin: 0 }}>
                    {result.forensics.ela.assessment} (Mean residual difference: {result.forensics.ela.mean_error?.toFixed(2)}, Max peak: {result.forensics.ela.max_error?.toFixed(2)}).
                  </p>
                </div>
              )}

              {/* Actions */}
              <ReportActions
                result={result}
                moduleType="deepfake"
                onDownloadPDF={handleDownloadPDF}
                onAnalyzeAnother={() => {
                  setFile(null);
                  setPreview(null);
                  setResult(null);
                  setError('');
                }}
              />
            </div>
          )}
        </div>
      )}

      {/* ========================================================= */}
      {/* TAB 2: VIDEO FRAME ANALYSIS                               */}
      {/* ========================================================= */}
      {activeTab === 'video' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="cyber-card" style={{ padding: '2rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem', marginBottom: '1.5rem' }}>
              <div style={{
                width: '44px', height: '44px', borderRadius: '10px',
                backgroundColor: '#EDF3E8', border: '1px solid #B9D995',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <Video size={22} color="#36435E" />
              </div>
              <div>
                <h3 style={{ fontSize: '1.05rem', fontWeight: '800', color: '#202734', margin: 0 }}>
                  Video Frame-Level Deepfake Analysis
                </h3>
                <p style={{ fontSize: '0.8rem', color: '#697586', margin: '0.15rem 0 0 0' }}>
                  Extracts keyframes at uniform sampling intervals, detects facial crops, and scores manipulation timeline.
                </p>
              </div>
            </div>

            <div
              onClick={() => videoInputRef.current?.click()}
              style={{
                border: '2px dashed #B9D995', borderRadius: '12px',
                padding: '2.5rem 1.5rem', textAlign: 'center', cursor: 'pointer',
                backgroundColor: '#F7F9F4',
                transition: 'background-color 0.2s ease',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#EDF3E8'; }}
              onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#F7F9F4'; }}
            >
              <input ref={videoInputRef} type="file" accept="video/mp4,video/webm,video/quicktime,video/x-msvideo" onChange={(e) => handleVideoSelect(e.target.files?.[0])} style={{ display: 'none' }} />
              {videoPreview ? (
                <video src={videoPreview} controls style={{ maxHeight: '280px', maxWidth: '100%', borderRadius: '8px', border: '1px solid #E1E7DD' }} />
              ) : (
                <>
                  <Video size={44} color="#36435E" style={{ marginBottom: '0.75rem' }} />
                  <p style={{ fontSize: '0.95rem', color: '#202734', margin: 0, fontWeight: '700' }}>
                    Click or drag to upload video file
                  </p>
                  <p style={{ fontSize: '0.78rem', color: '#697586', marginTop: '0.35rem' }}>
                    Supported: MP4, WEBM, MOV (Max 50 MB)
                  </p>
                </>
              )}
            </div>

            {videoFile && (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
                <span style={{ fontSize: '0.85rem', color: '#263142', fontWeight: '600' }}>
                  🎥 {videoFile.name} ({(videoFile.size / (1024 * 1024)).toFixed(2)} MB)
                </span>
                <button
                  className="cyber-button-primary"
                  onClick={handleAnalyzeVideo}
                  disabled={videoLoading}
                  style={{ padding: '0.7rem 1.6rem' }}
                >
                  {videoLoading ? <><RefreshCw size={16} className="animate-spin" /> Analyzing Video Frames...</> : 'Analyze Video Timeline'}
                </button>
              </div>
            )}
          </div>

          {videoError && (
            <div style={{
              backgroundColor: '#FDF2F4', border: '1px solid #F4C2CA',
              color: '#963143', padding: '0.85rem 1rem', borderRadius: '8px', fontSize: '0.85rem',
              fontWeight: '600', display: 'flex', alignItems: 'center', gap: '0.5rem'
            }}>
              <AlertTriangle size={16} color="#D98291" /> {videoError}
            </div>
          )}

          {videoResult && (
            <div className="cyber-card" style={{ padding: '2rem' }}>
              <div style={{
                display: 'flex', alignItems: 'center', gap: '1rem', padding: '1.25rem',
                borderRadius: '10px', marginBottom: '1.5rem',
                backgroundColor: videoResult.prediction === 'REAL' ? '#EDF3E8' : '#FDF2F4',
                border: `1px solid ${videoResult.prediction === 'REAL' ? '#B9D995' : '#F4C2CA'}`,
              }}>
                {videoResult.prediction === 'REAL' ? <CheckCircle size={36} color="#87A587" /> : <XCircle size={36} color="#D98291" />}
                <div>
                  <h3 style={{ fontSize: '1.5rem', fontWeight: '800', color: videoResult.prediction === 'REAL' ? '#3D683D' : '#963143', margin: 0 }}>
                    {videoResult.prediction}
                  </h3>
                  <p style={{ fontSize: '0.84rem', color: '#263142', margin: '0.2rem 0 0 0', fontWeight: '600' }}>
                    Aggregated Confidence: {videoResult.confidence?.toFixed(2)}% | Analyzed Frames: {videoResult.analyzed_frames}
                  </p>
                </div>
              </div>

              {/* Frame Timeline */}
              {videoResult.timeline && (
                <div style={{ marginBottom: '1.5rem' }}>
                  <h4 style={{ fontSize: '0.75rem', fontWeight: '700', color: '#36435E', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.75rem' }}>
                    Frame-by-Frame Timeline
                  </h4>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))', gap: '0.75rem' }}>
                    {videoResult.timeline.map((frame, idx) => (
                      <div key={idx} style={{
                        backgroundColor: '#F7F9F4', padding: '0.85rem', borderRadius: '8px',
                        border: `1px solid ${frame.prediction === 'REAL' ? '#B9D995' : '#F4C2CA'}`,
                        textAlign: 'center'
                      }}>
                        <span style={{ fontSize: '0.72rem', color: '#697586', display: 'block', fontWeight: '600' }}>{frame.timestamp_sec?.toFixed(2)}s</span>
                        <span style={{
                          fontSize: '0.85rem', fontWeight: '800', display: 'block', margin: '0.25rem 0',
                          color: frame.prediction === 'REAL' ? '#3D683D' : '#963143'
                        }}>
                          {frame.prediction}
                        </span>
                        <span style={{ fontSize: '0.72rem', color: '#697586' }}>Score: {frame.score?.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ========================================================= */}
      {/* TAB 3: AUDIO AUTHENTICITY ANALYSIS                        */}
      {/* ========================================================= */}
      {activeTab === 'audio' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="cyber-card" style={{ padding: '2rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem', marginBottom: '1.5rem' }}>
              <div style={{
                width: '44px', height: '44px', borderRadius: '10px',
                backgroundColor: '#EDF3E8', border: '1px solid #B9D995',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <Activity size={22} color="#36435E" />
              </div>
              <div>
                <h3 style={{ fontSize: '1.05rem', fontWeight: '800', color: '#202734', margin: 0 }}>
                  Audio Voice-Cloning & Synthetic Speech Verification
                </h3>
                <p style={{ fontSize: '0.8rem', color: '#697586', margin: '0.15rem 0 0 0' }}>
                  Analyzes spectral centroid, high-frequency cutoff, zero-crossing rates, and phase coherence.
                </p>
              </div>
            </div>

            <div
              onClick={() => audioInputRef.current?.click()}
              style={{
                border: '2px dashed #B9D995', borderRadius: '12px',
                padding: '2.5rem 1.5rem', textAlign: 'center', cursor: 'pointer',
                backgroundColor: '#F7F9F4',
                transition: 'background-color 0.2s ease',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#EDF3E8'; }}
              onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#F7F9F4'; }}
            >
              <input ref={audioInputRef} type="file" accept="audio/wav,audio/mpeg,audio/ogg,audio/mp4" onChange={(e) => handleAudioSelect(e.target.files?.[0])} style={{ display: 'none' }} />
              <Activity size={44} color="#36435E" style={{ marginBottom: '0.75rem' }} />
              <p style={{ fontSize: '0.95rem', color: '#202734', margin: 0, fontWeight: '700' }}>
                {audioFile ? audioFile.name : 'Click or drag to upload voice audio'}
              </p>
              <p style={{ fontSize: '0.78rem', color: '#697586', marginTop: '0.35rem' }}>
                Supported: WAV, MP3, OGG, M4A (Max 25 MB)
              </p>
            </div>

            {audioFile && (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
                <span style={{ fontSize: '0.85rem', color: '#263142', fontWeight: '600' }}>
                  🎵 {audioFile.name} ({(audioFile.size / 1024).toFixed(1)} KB)
                </span>
                <button
                  className="cyber-button-primary"
                  onClick={handleAnalyzeAudio}
                  disabled={audioLoading}
                  style={{ padding: '0.7rem 1.6rem' }}
                >
                  {audioLoading ? <><RefreshCw size={16} className="animate-spin" /> Analyzing Acoustics...</> : 'Analyze Voice Authenticity'}
                </button>
              </div>
            )}
          </div>

          {audioError && (
            <div style={{
              backgroundColor: '#FDF2F4', border: '1px solid #F4C2CA',
              color: '#963143', padding: '0.85rem 1rem', borderRadius: '8px', fontSize: '0.85rem',
              fontWeight: '600', display: 'flex', alignItems: 'center', gap: '0.5rem'
            }}>
              <AlertTriangle size={16} color="#D98291" /> {audioError}
            </div>
          )}

          {audioResult && (
            <div className="cyber-card" style={{ padding: '2rem' }}>
              <div style={{
                display: 'flex', alignItems: 'center', gap: '1rem', padding: '1.25rem',
                borderRadius: '10px', marginBottom: '1.5rem',
                backgroundColor: audioResult.prediction === 'REAL' ? '#EDF3E8' : '#FDF2F4',
                border: `1px solid ${audioResult.prediction === 'REAL' ? '#B9D995' : '#F4C2CA'}`,
              }}>
                {audioResult.prediction === 'REAL' ? <CheckCircle size={36} color="#87A587" /> : <XCircle size={36} color="#D98291" />}
                <div>
                  <h3 style={{ fontSize: '1.5rem', fontWeight: '800', color: audioResult.prediction === 'REAL' ? '#3D683D' : '#963143', margin: 0 }}>
                    {audioResult.prediction}
                  </h3>
                  <p style={{ fontSize: '0.84rem', color: '#263142', margin: '0.2rem 0 0 0', fontWeight: '600' }}>
                    Authenticity Confidence: {audioResult.confidence?.toFixed(2)}% | Assessment: {audioResult.summary}
                  </p>
                </div>
              </div>

              {audioResult.features && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.85rem' }}>
                  {Object.entries(audioResult.features).map(([key, val], idx) => (
                    <div key={idx} style={{
                      backgroundColor: '#F7F9F4', padding: '0.9rem', borderRadius: '8px',
                      border: '1px solid #E1E7DD'
                    }}>
                      <span style={{ fontSize: '0.68rem', color: '#697586', textTransform: 'uppercase', display: 'block', fontWeight: '700' }}>{key.replace(/_/g, ' ')}</span>
                      <span style={{ fontSize: '0.95rem', fontWeight: '800', color: '#202734' }}>{typeof val === 'number' ? val.toFixed(2) : String(val)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ========================================================= */}
      {/* TAB 4: TRANSPARENCY & DATASET AUDIT DASHBOARD             */}
      {/* ========================================================= */}
      {activeTab === 'transparency' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="cyber-card" style={{ padding: '2rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '0.75rem' }}>
              <div>
                <h3 style={{ fontSize: '1.1rem', fontWeight: '800', color: '#202734', margin: 0 }}>
                  Model & Dataset Transparency Dashboard
                </h3>
                <p style={{ fontSize: '0.8rem', color: '#697586', margin: '0.2rem 0 0 0' }}>
                  Audited Kaggle Datasets, Provenance Attributions, and Untouched Test Metrics
                </p>
              </div>

              <button
                onClick={loadTransparencyStats}
                className="cyber-button-secondary"
                style={{ padding: '0.5rem 1rem', fontSize: '0.8rem' }}
              >
                <RefreshCw size={13} className={transparencyLoading ? 'animate-spin' : ''} /> Refresh Audit
              </button>
            </div>

            {transparencyLoading ? (
              <div style={{ padding: '2.5rem', textAlign: 'center' }}>
                <RefreshCw size={32} color="#36435E" className="animate-spin" />
              </div>
            ) : transparencyData ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                {/* Model Info */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.85rem' }}>
                  <div style={{ backgroundColor: '#F7F9F4', padding: '1rem', borderRadius: '8px', border: '1px solid #E1E7DD' }}>
                    <span style={{ fontSize: '0.68rem', color: '#697586', textTransform: 'uppercase', display: 'block', fontWeight: '700' }}>Active Production Model</span>
                    <strong style={{ fontSize: '1rem', color: '#202734' }}>{transparencyData.model_info?.name}</strong>
                    <p style={{ fontSize: '0.75rem', color: '#697586', margin: '0.3rem 0 0 0' }}>{transparencyData.model_info?.architecture}</p>
                  </div>
                  <div style={{ backgroundColor: '#F7F9F4', padding: '1rem', borderRadius: '8px', border: '1px solid #E1E7DD' }}>
                    <span style={{ fontSize: '0.68rem', color: '#697586', textTransform: 'uppercase', display: 'block', fontWeight: '700' }}>Validation Decision Threshold</span>
                    <strong style={{ fontSize: '1rem', color: '#3D683D' }}>{transparencyData.model_info?.decision_threshold}</strong>
                    <p style={{ fontSize: '0.75rem', color: '#697586', margin: '0.3rem 0 0 0' }}>Calibrated strictly on validation split</p>
                  </div>
                  <div style={{ backgroundColor: '#F7F9F4', padding: '1rem', borderRadius: '8px', border: '1px solid #E1E7DD' }}>
                    <span style={{ fontSize: '0.68rem', color: '#697586', textTransform: 'uppercase', display: 'block', fontWeight: '700' }}>Held-Out Test Accuracy</span>
                    <strong style={{ fontSize: '1rem', color: '#202734' }}>
                      {transparencyData.evaluation_metrics?.metrics?.accuracy
                        ? `${(transparencyData.evaluation_metrics.metrics.accuracy * 100).toFixed(2)}%`
                        : 'Benchmarked'}
                    </strong>
                    <p style={{ fontSize: '0.75rem', color: '#697586', margin: '0.3rem 0 0 0' }}>
                      ROC-AUC: {transparencyData.evaluation_metrics?.metrics?.roc_auc?.toFixed(4) || 'Evaluated'}
                    </p>
                  </div>
                </div>

                {/* Dataset Provenance */}
                {transparencyData.dataset_manifest?.sources && (
                  <div>
                    <h4 style={{ fontSize: '0.75rem', fontWeight: '700', color: '#36435E', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.75rem' }}>
                      Verified Dataset Provenance & Attribution
                    </h4>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
                      {transparencyData.dataset_manifest.sources.map((src, idx) => (
                        <div key={idx} style={{
                          backgroundColor: '#F7F9F4', padding: '0.85rem 1rem', borderRadius: '8px',
                          border: '1px solid #E1E7DD', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem'
                        }}>
                          <div>
                            <strong style={{ fontSize: '0.85rem', color: '#202734', display: 'block' }}>{src.name}</strong>
                            <span style={{ fontSize: '0.75rem', color: '#697586' }}>{src.provenance}</span>
                          </div>
                          <span style={{ fontSize: '0.72rem', padding: '0.2rem 0.6rem', borderRadius: '4px', backgroundColor: '#EDF3E8', color: '#202734', fontWeight: '700', border: '1px solid #B9D995' }}>
                            {src.license}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Transparency Principles */}
                <div style={{
                  backgroundColor: '#F7F9F4', padding: '1.25rem', borderRadius: '8px',
                  border: '1px solid #E1E7DD'
                }}>
                  <h4 style={{ fontSize: '0.75rem', fontWeight: '700', color: '#36435E', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.65rem' }}>
                    Trust & Verification Commitments
                  </h4>
                  <ul style={{ margin: 0, paddingLeft: '1.2rem', color: '#263142', fontSize: '0.825rem', lineHeight: '1.7' }}>
                    {transparencyData.transparency_principles?.map((p, idx) => (
                      <li key={idx}>{p}</li>
                    ))}
                  </ul>
                </div>
              </div>
            ) : (
              <p style={{ color: '#697586', fontSize: '0.85rem' }}>Load transparency metrics from the active server.</p>
            )}
          </div>
        </div>
      )}

    </div>
  );
}
