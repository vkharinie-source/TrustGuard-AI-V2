import React, { useRef, useState } from 'react';
import {
    AlertTriangle,
    CheckCircle,
    Download,
    Film,
    Pause,
    Play,
    RefreshCw,
    Shield,
    Square,
    Upload,
    Volume2,
    XCircle,
} from 'lucide-react';
import { api } from '../services/api';
import { generatePDFReport } from '../utils/pdfReport';
import { tts } from '../utils/voice';
import { transcribeAudioSamples } from '../utils/offlineSpeech';
import { captionSampledFrame } from '../utils/videoCaption';

const VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv', '.webm'];
const FRAME_POSITIONS = [0.1, 0.3, 0.5, 0.7, 0.9];

const createBrowserSampleVideo = () => new Promise((resolve, reject) => {
    if (!HTMLCanvasElement.prototype.captureStream || !window.MediaRecorder) {
        reject(new Error('This browser cannot generate a sample video. Choose a video file instead.'));
        return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 360;
    const context = canvas.getContext('2d');
    const stream = canvas.captureStream(12);
    const supportedMime = ['video/mp4;codecs=avc1.42E01E', 'video/mp4', 'video/webm;codecs=vp8', 'video/webm']
        .find(type => MediaRecorder.isTypeSupported(type));
    if (!supportedMime) {
        stream.getTracks().forEach(track => track.stop());
        reject(new Error('This browser does not support the sample video format. Choose an MP4 or WEBM file instead.'));
        return;
    }

    const recorder = new MediaRecorder(stream, { mimeType: supportedMime });
    const chunks = [];
    let startedAt = 0;
    let timeoutId;
    const cleanup = () => {
        window.clearTimeout(timeoutId);
        stream.getTracks().forEach(track => track.stop());
    };
    recorder.ondataavailable = event => {
        if (event.data.size > 0) chunks.push(event.data);
    };
    recorder.onerror = () => {
        cleanup();
        reject(new Error('The browser could not create its sample video. Choose a video file instead.'));
    };
    recorder.onstop = () => {
        cleanup();
        const blob = new Blob(chunks, { type: recorder.mimeType || supportedMime });
        if (blob.size === 0) reject(new Error('The browser created an empty sample. Choose a video file instead.'));
        else resolve(blob);
    };

    const drawFrame = now => {
        const elapsed = (now - startedAt) / 1000;
        context.fillStyle = '#101b2a';
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.fillStyle = '#203448';
        context.fillRect(0, 0, canvas.width, 58);
        context.fillStyle = '#e5edf5';
        context.font = 'bold 24px sans-serif';
        context.fillText('TRUSTGUARD AI - SAMPLE CLIP', 24, 38);
        context.fillStyle = '#83ded5';
        context.beginPath();
        context.arc(80 + (elapsed * 145) % 480, 190, 42, 0, Math.PI * 2);
        context.fill();
        context.fillStyle = '#e6b763';
        const boxX = 500 - (elapsed * 105) % 430;
        context.fillRect(boxX, 250, 60, 60);
        context.fillStyle = '#a9b8c7';
        context.font = '16px sans-serif';
        context.fillText('Synthetic motion for upload testing', 24, 338);
        if (recorder.state === 'recording') requestAnimationFrame(drawFrame);
    };

    recorder.start();
    startedAt = performance.now();
    requestAnimationFrame(drawFrame);
    timeoutId = window.setTimeout(() => {
        if (recorder.state === 'recording') recorder.stop();
    }, 3500);
});

const formatTime = (seconds) => {
    if (!Number.isFinite(seconds)) return 'Unavailable';
    const minutes = Math.floor(seconds / 60);
    const remainder = Math.floor(seconds % 60);
    return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`;
};

const waitForVideoMetadata = (video) => new Promise((resolve, reject) => {
    if (video.readyState >= 1 && video.videoWidth > 0) {
        resolve();
        return;
    }
    const cleanup = () => {
        video.removeEventListener('loadedmetadata', onLoaded);
        video.removeEventListener('error', onError);
    };
    const onLoaded = () => { cleanup(); resolve(); };
    const onError = () => { cleanup(); reject(new Error('This video format could not be decoded by your browser.')); };
    video.addEventListener('loadedmetadata', onLoaded, { once: true });
    video.addEventListener('error', onError, { once: true });
    video.load();
});

const seekTo = (video, time) => new Promise((resolve, reject) => {
    if (Math.abs(video.currentTime - time) < 0.01) {
        resolve();
        return;
    }
    const timeout = window.setTimeout(() => {
        cleanup();
        reject(new Error('The browser could not read a video frame at the requested time.'));
    }, 10000);
    const cleanup = () => {
        window.clearTimeout(timeout);
        video.removeEventListener('seeked', onSeeked);
        video.removeEventListener('error', onError);
    };
    const onSeeked = () => { cleanup(); resolve(); };
    const onError = () => { cleanup(); reject(new Error('The browser could not decode this video frame.')); };
    video.addEventListener('seeked', onSeeked, { once: true });
    video.addEventListener('error', onError, { once: true });
    video.currentTime = time;
});

const canvasToBlob = (canvas) => new Promise((resolve, reject) => {
    canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('Could not extract a frame from this video.')), 'image/jpeg', 0.9);
});

const classifySampledFrames = (frames) => {
    const labels = frames.map(frame => String(frame.prediction || '').toUpperCase());
    if (labels.some(label => !['REAL', 'FAKE'].includes(label))) return 'UNVERIFIED';
    const distinctLabels = new Set(labels);
    return distinctLabels.size === 1 ? labels[0] : 'SUSPICIOUS';
};

export default function VideoAnalysis() {
    const [file, setFile] = useState(null);
    const [videoUrl, setVideoUrl] = useState('');
    const [metadata, setMetadata] = useState(null);
    const [frameResults, setFrameResults] = useState([]);
    const [report, setReport] = useState(null);
    const [contentSummary, setContentSummary] = useState('');
    const [captionStatus, setCaptionStatus] = useState('');
    const [captionError, setCaptionError] = useState('');
    const [captionLoading, setCaptionLoading] = useState(false);
    const [loading, setLoading] = useState(false);
    const [loadingMessage, setLoadingMessage] = useState('');
    const [error, setError] = useState('');
    const [dragOver, setDragOver] = useState(false);
    const [sampleLoading, setSampleLoading] = useState(false);
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState('');
    const [voiceListening, setVoiceListening] = useState(false);
    const [voiceTranscribing, setVoiceTranscribing] = useState(false);
    const [voiceStatus, setVoiceStatus] = useState('');
    const [speaking, setSpeaking] = useState(false);
    const [paused, setPaused] = useState(false);
    const inputRef = useRef(null);
    const videoRef = useRef(null);
    const questionCaptureRef = useRef(null);
    const questionRecordingStartedAtRef = useRef(0);

    const releaseFrameUrls = (frames) => {
        frames.forEach(frame => URL.revokeObjectURL(frame.previewUrl));
    };

    const handleFileSelect = (selectedFile) => {
        if (!selectedFile) return;
        const extension = `.${selectedFile.name.split('.').pop().toLowerCase()}`;
        if (!VIDEO_EXTENSIONS.includes(extension)) {
            setError('Choose an MP4, MOV, AVI, MKV, or WEBM video. Playback depends on browser codec support.');
            return;
        }
        if (videoUrl) URL.revokeObjectURL(videoUrl);
        releaseFrameUrls(frameResults);
        setVideoUrl(URL.createObjectURL(selectedFile));
        setFile(selectedFile);
        setMetadata(null);
        setFrameResults([]);
        setReport(null);
        setContentSummary('');
        setCaptionStatus('');
        setCaptionError('');
        setQuestion('');
        setAnswer('');
        setError('');
    };

    const handleUseSample = async (event) => {
        event.stopPropagation();
        setError('');
        setSampleLoading(true);
        try {
            const sampleBlob = await createBrowserSampleVideo();
            const extension = sampleBlob.type.includes('webm') ? 'webm' : 'mp4';
            const sampleFile = new File(
                [sampleBlob],
                `trustguard-synthetic-upload-test.${extension}`,
                { type: sampleBlob.type },
            );
            handleFileSelect(sampleFile);
        } catch (sampleError) {
            setError(sampleError.message || 'The sample clip could not be loaded.');
        } finally {
            setSampleLoading(false);
        }
    };

    const handleMetadata = (event) => {
        const video = event.currentTarget;
        if (!video.videoWidth || !video.videoHeight || !Number.isFinite(video.duration)) {
            setError('The browser could not read this video metadata. Try a browser-supported MP4 or WEBM file.');
            return;
        }
        setMetadata({
            duration: video.duration,
            width: video.videoWidth,
            height: video.videoHeight,
            frameRate: null,
        });
        setError('');
    };

    const handleAnalyze = async () => {
        if (!file || !metadata || !videoUrl) {
            setError('Choose a browser-readable video before starting analysis.');
            return;
        }

        setLoading(true);
        setLoadingMessage('Reading video metadata...');
        setError('');
        setReport(null);
        releaseFrameUrls(frameResults);
        setFrameResults([]);
        setContentSummary('');
        setCaptionStatus('');
        setCaptionError('');
        setAnswer('');

        const sampler = document.createElement('video');
        sampler.preload = 'auto';
        sampler.muted = true;
        sampler.src = videoUrl;

        try {
            await waitForVideoMetadata(sampler);
            const sampleTimes = FRAME_POSITIONS.map(position => Math.min(
                Math.max(metadata.duration * position, 0.05),
                Math.max(metadata.duration - 0.05, 0),
            ));
            const analyzedFrames = [];
            const baseName = file.name.replace(/\.[^.]+$/, '').replace(/[^a-zA-Z0-9_-]/g, '_');

            for (let index = 0; index < sampleTimes.length; index += 1) {
                const timestamp = sampleTimes[index];
                setLoadingMessage(`Extracting frame ${index + 1} of ${sampleTimes.length}...`);
                await seekTo(sampler, timestamp);

                const scale = Math.min(1, 1280 / sampler.videoWidth, 720 / sampler.videoHeight);
                const canvas = document.createElement('canvas');
                canvas.width = Math.max(1, Math.round(sampler.videoWidth * scale));
                canvas.height = Math.max(1, Math.round(sampler.videoHeight * scale));
                const context = canvas.getContext('2d');
                if (!context) throw new Error('Canvas frame extraction is unavailable in this browser.');
                context.drawImage(sampler, 0, 0, canvas.width, canvas.height);
                const frameBlob = await canvasToBlob(canvas);
                const frameFile = new File(
                    [frameBlob],
                    `${baseName}_frame_${String(index + 1).padStart(2, '0')}.jpg`,
                    { type: 'image/jpeg' },
                );

                setLoadingMessage(`Analyzing sampled frame ${index + 1} of ${sampleTimes.length}...`);
                const prediction = await api.predictImage(frameFile);
                const frame = {
                    timestamp,
                    previewUrl: URL.createObjectURL(frameBlob),
                    prediction: String(prediction.prediction || '').toUpperCase(),
                    confidence: prediction.confidence,
                    rawScore: prediction.raw_score,
                    facesDetected: prediction.faces_detected,
                };
                analyzedFrames.push(frame);
                setFrameResults([...analyzedFrames]);
            }

            const prediction = classifySampledFrames(analyzedFrames);
            const frameConfidence = analyzedFrames
                .map(frame => Number(frame.confidence))
                .filter(Number.isFinite);
            const meanFrameConfidence = frameConfidence.length
                ? frameConfidence.reduce((total, value) => total + value, 0) / frameConfidence.length
                : undefined;
            const counts = analyzedFrames.reduce((totals, frame) => {
                totals[frame.prediction] = (totals[frame.prediction] || 0) + 1;
                return totals;
            }, {});
            const analysisSummary = `The video was sampled at ${analyzedFrames.length} points across ${formatTime(metadata.duration)}. The image detector returned ${counts.REAL || 0} REAL and ${counts.FAKE || 0} FAKE frame results. This checks selected still frames only; it does not verify the whole video, motion, or audio.`;
            setLoadingMessage('Preparing video report...');
            setReport({
                prediction,
                confidence: meanFrameConfidence,
                analyzedFrames,
                analysisSummary,
                counts,
            });

            setCaptionLoading(true);
            setCaptionError('');
            setCaptionStatus('Loading the visual caption model. First use may download about 240 MB...');
            const captionIndexes = [...new Set([0, Math.floor((analyzedFrames.length - 1) / 2), analyzedFrames.length - 1])];
            const captions = [];
            try {
                for (let index = 0; index < captionIndexes.length; index += 1) {
                    setCaptionStatus(`Describing sampled frame ${index + 1} of ${captionIndexes.length}...`);
                    const frame = analyzedFrames[captionIndexes[index]];
                    const caption = await captionSampledFrame(
                        frame.previewUrl,
                        percent => setCaptionStatus(`Downloading visual caption model: ${percent}%`),
                    );
                    const words = new Set(caption.toLowerCase().split(/[^a-z0-9]+/).filter(word => word.length > 3));
                    const isNearDuplicate = captions.some(existing => {
                        const existingWords = new Set(existing.toLowerCase().split(/[^a-z0-9]+/).filter(word => word.length > 3));
                        const overlap = [...words].filter(word => existingWords.has(word)).length;
                        return overlap / Math.max(1, Math.min(words.size, existingWords.size)) > 0.6;
                    });
                    if (caption && !isNearDuplicate && captions.length < 2) {
                        captions.push(caption);
                    }
                }
                if (!captions.length) throw new Error('The caption model did not return a description.');
                setContentSummary(`${captions.map(caption => caption.replace(/[.!?]+$/, '')).join('. Later, ')}. This AI description is based on three sampled frames, may misidentify details, and can miss events between samples.`);
                setCaptionStatus('Content summary ready. Check it against the full video.');
            } catch (error) {
                setCaptionError(error.message || 'The visual caption model could not describe this video.');
                setCaptionStatus('');
            } finally {
                setCaptionLoading(false);
            }
        } catch (analysisError) {
            setError(analysisError.message || 'Video analysis failed.');
        } finally {
            sampler.removeAttribute('src');
            sampler.load();
            setLoading(false);
            setLoadingMessage('');
        }
    };

    const handleReset = () => {
        if (videoUrl) URL.revokeObjectURL(videoUrl);
        releaseFrameUrls(frameResults);
        setFile(null);
        setVideoUrl('');
        setMetadata(null);
        setFrameResults([]);
        setReport(null);
        setContentSummary('');
        setCaptionStatus('');
        setCaptionError('');
        setQuestion('');
        setAnswer('');
        setError('');
    };

    const handleDownloadPDF = () => {
        if (!report || !file) return;
        generatePDFReport({
            title: 'Video Authenticity Report',
            moduleType: 'Video Analysis',
            result: {
                prediction: report.prediction,
                confidence: report.confidence,
            },
            inputInfo: {
                Filename: file.name,
                'File Size': `${(file.size / (1024 * 1024)).toFixed(2)} MB`,
                Duration: formatTime(metadata.duration),
                Resolution: `${metadata.width} x ${metadata.height}`,
                'Audio Analysis': 'Not performed',
            },
            findings: report.analyzedFrames.map(frame =>
                `${formatTime(frame.timestamp)}: ${frame.prediction} (${Number(frame.confidence).toFixed(2)}% frame confidence)`,
            ),
            details: [
                contentSummary ? `What appears in the video: ${contentSummary}` : null,
                report.analysisSummary,
                'Only sampled video frames were analyzed. No motion, temporal consistency, or audio analysis was performed.',
            ].filter(Boolean),
            recommendations: [
                'Treat this as a sampled-frame screening result, not proof of video authenticity.',
                'Verify the original video and its source before relying on it.',
            ],
            technicalInfo: {
                Model: 'MobileNetV2 V3 image detector applied to sampled frames',
                'Frames Analyzed': report.analyzedFrames.length,
                'Frame Rate': 'Not exposed by browser metadata',
                'Audio Track': 'Not analyzed',
            },
            extraSections: [{
                title: 'Frame Analysis',
                content: report.analyzedFrames.map(frame => `${formatTime(frame.timestamp)}: ${frame.prediction}, ${Number(frame.confidence).toFixed(2)}% confidence`).join('\n'),
            }],
        });
    };

    const handleAsk = (prompt = question) => {
        if (!report || !prompt.trim()) return;
        const normalizedQuestion = prompt.toLowerCase();
        const fakeFrames = report.analyzedFrames.filter(frame => frame.prediction === 'FAKE');
        let response;
        if (normalizedQuestion.includes('audio') || normalizedQuestion.includes('sound') || normalizedQuestion.includes('transcript')) {
            response = 'Audio and transcript analysis were not performed. This report only contains still-frame image detector results.';
        } else if (normalizedQuestion.includes('which frame') || normalizedQuestion.includes('suspicious')) {
            response = fakeFrames.length
                ? `The image detector classified these sampled frames as FAKE: ${fakeFrames.map(frame => formatTime(frame.timestamp)).join(', ')}. This is frame-level evidence only, not a whole-video conclusion.`
                : 'No sampled frame was classified as FAKE by the image detector. This does not establish that the full video is authentic.';
        } else if (normalizedQuestion.includes('summar') || normalizedQuestion.includes('about') || normalizedQuestion.includes('explain')) {
            response = contentSummary || report.analysisSummary;
        } else {
            response = `${contentSummary ? `${contentSummary} ` : ''}The sampled-frame result is ${report.prediction}. ${report.analyzedFrames.length} frames were checked with MobileNetV2 V3. Audio and motion were not analyzed.`;
        }
        setQuestion(prompt);
        setAnswer(response);
    };

    const startVoiceQuestion = async () => {
        const AudioContextType = window.AudioContext || window.webkitAudioContext;
        const OfflineAudioContextType = window.OfflineAudioContext || window.webkitOfflineAudioContext;
        if (!navigator.mediaDevices?.getUserMedia || !AudioContextType || !OfflineAudioContextType) {
            setVoiceStatus('Microphone dictation is unavailable. Type your question instead.');
            return;
        }
        setError('');
        setVoiceStatus('Requesting microphone access...');
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const audioContext = new AudioContextType();
            await audioContext.resume();
            const source = audioContext.createMediaStreamSource(stream);
            const processor = audioContext.createScriptProcessor(4096, 1, 1);
            const silentOutput = audioContext.createGain();
            const chunks = [];
            silentOutput.gain.value = 0;
            processor.onaudioprocess = event => chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
            source.connect(processor);
            processor.connect(silentOutput);
            silentOutput.connect(audioContext.destination);
            questionCaptureRef.current = { stream, audioContext, source, processor, silentOutput, chunks };
            questionRecordingStartedAtRef.current = Date.now();
            setVoiceListening(true);
            setVoiceStatus('Recording your question. Speak for at least one second, then select Stop.');
        } catch (microphoneError) {
            setVoiceStatus(microphoneError.name === 'NotAllowedError'
                ? 'Microphone access is blocked. Allow access or type your question.'
                : 'Could not start microphone recording. Type your question instead.');
        }
    };

    const stopVoiceQuestion = async () => {
        const capture = questionCaptureRef.current;
        if (!capture) return;
        questionCaptureRef.current = null;
        capture.processor.onaudioprocess = null;
        capture.source.disconnect();
        capture.processor.disconnect();
        capture.silentOutput.disconnect();
        capture.stream.getTracks().forEach(track => track.stop());
        setVoiceListening(false);

        if (Date.now() - questionRecordingStartedAtRef.current < 800) {
            await capture.audioContext.close();
            setVoiceStatus('Speak for at least one second, then stop recording.');
            return;
        }

        const sampleCount = capture.chunks.reduce((total, chunk) => total + chunk.length, 0);
        if (!sampleCount) {
            await capture.audioContext.close();
            setVoiceStatus('No microphone audio was captured. Type your question instead.');
            return;
        }

        const samples = new Float32Array(sampleCount);
        let offset = 0;
        capture.chunks.forEach(chunk => { samples.set(chunk, offset); offset += chunk.length; });
        setVoiceTranscribing(true);
        setVoiceStatus('Transcribing your question on this device...');
        try {
            const transcript = await transcribeAudioSamples(
                samples,
                capture.audioContext.sampleRate,
                percent => setVoiceStatus(`Downloading offline speech model: ${percent}%`),
            );
            const spokenQuestion = String(transcript.text || '').trim();
            if (!spokenQuestion) throw new Error('No speech was detected. Try again or type your question.');
            setQuestion(spokenQuestion);
            setVoiceStatus('You said: Review or edit the question, then select Ask.');
        } catch (transcriptionError) {
            setVoiceStatus(transcriptionError.message || 'Transcription failed. Type your question instead.');
        } finally {
            await capture.audioContext.close();
            setVoiceTranscribing(false);
        }
    };

    const speakAnswer = () => {
        if (!tts.isSupported() || !report) return;
        const speech = answer || `${contentSummary ? `${contentSummary} ` : ''}${report.analysisSummary}`;
        setSpeaking(true);
        setPaused(false);
        tts.speak(speech, () => { setSpeaking(false); setPaused(false); });
    };

    const handlePauseSpeech = () => {
        if (tts.isPaused()) {
            tts.resume();
            setPaused(false);
        } else {
            tts.pause();
            setPaused(true);
        }
    };

    const handleStopSpeech = () => {
        tts.stop();
        setSpeaking(false);
        setPaused(false);
    };

    const resultColor = report?.prediction === 'REAL'
        ? '#34D399'
        : report?.prediction === 'FAKE' ? '#F87171'
            : report?.prediction === 'SUSPICIOUS' ? '#FBBF24' : '#AAB7C5';

    return (
        <div className="video-analysis-page">
            <section className="cyber-card video-input-card">
                <div className="video-page-title">
                    <span className="video-page-icon"><Film size={21} /></span>
                    <div>
                        <h3>Analyze Video</h3>
                        <p>Analyze sampled frames for possible visual manipulation. Audio and motion are not analyzed.</p>
                    </div>
                </div>

                {!file ? (
                    <div
                        className={`video-dropzone${dragOver ? ' is-dragging' : ''}`}
                        onClick={() => { setError(''); inputRef.current?.click(); }}
                        onDragOver={event => { event.preventDefault(); setDragOver(true); setError(''); }}
                        onDragLeave={() => setDragOver(false)}
                        onDrop={event => { event.preventDefault(); setDragOver(false); handleFileSelect(event.dataTransfer.files?.[0]); }}
                    >
                        <input
                            ref={inputRef}
                            type="file"
                            accept=".mp4,.mov,.avi,.mkv,.webm,video/*"
                            onChange={event => handleFileSelect(event.target.files?.[0])}
                            hidden
                        />
                        <span className="video-drop-icon"><Upload size={22} /></span>
                        <strong>Drop a video to begin</strong>
                        <span>or choose a file: MP4, MOV, AVI, MKV, WEBM</span>
                        <button type="button" className="cyber-button-secondary" onClick={event => { event.stopPropagation(); inputRef.current?.click(); }}>
                            <Upload size={16} /> Choose video
                        </button>
                        <button type="button" className="cyber-button-secondary" onClick={handleUseSample} disabled={sampleLoading}>
                            {sampleLoading ? <><RefreshCw size={15} className="animate-spin" /> Creating sample...</> : <><Play size={15} /> Try sample clip</>}
                        </button>
                        <small className="video-sample-note">Creates a short synthetic clip in your browser to test upload and frame extraction. It is not labeled real or fake.</small>
                    </div>
                ) : (
                    <div className="video-preview-layout">
                        <div className="video-preview-frame">
                            <video
                                ref={videoRef}
                                src={videoUrl}
                                controls
                                playsInline
                                preload="metadata"
                                onLoadedMetadata={handleMetadata}
                                onError={() => setError('This browser cannot play this video codec. Try another MP4 or WEBM encoding.')}
                            />
                        </div>
                        <div className="video-file-details">
                            <div className="video-file-heading"><Film size={18} /><strong>{file.name}</strong></div>
                            <div className="video-metadata-grid">
                                <div><span>File size</span><strong>{(file.size / (1024 * 1024)).toFixed(2)} MB</strong></div>
                                <div><span>Duration</span><strong>{metadata ? formatTime(metadata.duration) : 'Reading...'}</strong></div>
                                <div><span>Resolution</span><strong>{metadata ? `${metadata.width} x ${metadata.height}` : 'Reading...'}</strong></div>
                                <div><span>Frame rate</span><strong>Not provided by browser</strong></div>
                                <div><span>Audio analysis</span><strong>Not performed</strong></div>
                                <div><span>Frame sampling</span><strong>5 representative points</strong></div>
                            </div>
                            <div className="video-actions">
                                <button type="button" className="cyber-button-primary" onClick={handleAnalyze} disabled={loading || !metadata}>
                                    {loading ? <><RefreshCw size={16} className="animate-spin" /> {loadingMessage || 'Analyzing video...'}</> : 'Analyze video'}
                                </button>
                                <button type="button" className="cyber-button-secondary" onClick={handleReset} disabled={loading}>Choose another</button>
                            </div>
                        </div>
                    </div>
                )}

                {error && (
                    <div className="video-error" role="alert" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><AlertTriangle size={16} />{error}</span>
                        <button
                            type="button"
                            aria-label="Dismiss error"
                            onClick={() => setError('')}
                            style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'inherit', padding: '0 2px', fontSize: '16px', lineHeight: 1, opacity: 0.7 }}
                        >✕</button>
                    </div>
                )}
            </section>

            {loading && (
                <section className="video-progress" role="status">
                    <RefreshCw size={19} className="animate-spin" />
                    <div><strong>{loadingMessage}</strong><span>Analyzing frames with the existing image detector. No full-video classification is performed.</span></div>
                </section>
            )}

            {frameResults.length > 0 && (
                <section className="cyber-card video-frames-card">
                    <div className="video-section-heading"><div><span>VISUAL EVIDENCE</span><h3>Sampled frame analysis</h3></div><span>{frameResults.length} / {FRAME_POSITIONS.length}</span></div>
                    <div className="video-frame-grid">
                        {frameResults.map((frame, index) => (
                            <article className="video-frame-item" key={`${frame.timestamp}-${index}`}>
                                <img src={frame.previewUrl} alt={`Video frame sampled at ${formatTime(frame.timestamp)}`} />
                                <div className="video-frame-meta"><span>{formatTime(frame.timestamp)}</span><strong style={{ color: frame.prediction === 'REAL' ? '#34D399' : '#F87171' }}>{frame.prediction}</strong></div>
                                <p>{Number(frame.confidence).toFixed(2)}% frame confidence</p>
                                {frame.facesDetected !== undefined && <p>{frame.facesDetected} faces detected</p>}
                            </article>
                        ))}
                    </div>
                </section>
            )}

            {report && (
                <>
                    <section className="cyber-card video-report-card">
                        <div className="video-section-heading"><div><span>VIDEO AUTHENTICITY REPORT</span><h3>Sampled visual assessment</h3></div><Shield size={19} color="#83DED5" /></div>
                        <div className="video-result-banner">
                            {report.prediction === 'REAL' ? <CheckCircle size={26} color={resultColor} /> : report.prediction === 'FAKE' ? <XCircle size={26} color={resultColor} /> : <AlertTriangle size={26} color={resultColor} />}
                            <div><strong style={{ color: resultColor }}>{report.prediction}</strong><span>Sampled-frame result, not a whole-video verdict</span></div>
                            <div className="video-result-metric"><strong>{report.confidence !== undefined ? `${report.confidence.toFixed(2)}%` : 'N/A'}</strong><span>Mean frame confidence</span></div>
                        </div>
                        <div className="video-report-metrics">
                            <div><span>Duration</span><strong>{formatTime(metadata.duration)}</strong></div>
                            <div><span>Frames analyzed</span><strong>{report.analyzedFrames.length}</strong></div>
                            <div><span>Audio</span><strong>Not analyzed</strong></div>
                            <div><span>Risk score</span><strong>Not provided</strong></div>
                        </div>
                        <div className="video-report-section video-content-summary">
                            <h4>AI visual summary</h4>
                            {contentSummary ? (
                                <p>{contentSummary}</p>
                            ) : captionLoading ? (
                                <p role="status">{captionStatus}</p>
                            ) : (
                                <p>{captionError || 'A visual content summary is unavailable. Frame analysis is shown below.'}</p>
                            )}
                            <small>Generated from three sampled frames. The caption model can misidentify details and miss events between samples.</small>
                        </div>
                        <div className="video-report-section"><h4>Sampled-frame analysis</h4><p>{report.analysisSummary}</p></div>
                        <div className="video-report-section"><h4>Key findings</h4><ul>{report.analyzedFrames.map((frame, index) => <li key={index}>{formatTime(frame.timestamp)}: {frame.prediction}, {Number(frame.confidence).toFixed(2)}% frame confidence.</li>)}</ul></div>
                        <div className="video-report-section"><h4>Audio and temporal analysis</h4><p>No audio, transcript, synchronization, or motion analysis was performed. This report describes only the sampled still frames.</p></div>
                        <div className="video-report-section"><h4>Recommendation</h4><p>Review the source video and its provenance independently. A few sampled frames cannot establish that the complete video is authentic or manipulated.</p></div>
                        <div className="video-report-actions">
                            <button type="button" className="cyber-button-primary" onClick={handleDownloadPDF}><Download size={16} /> Download PDF</button>
                            {!speaking ? (
                                <button type="button" className="cyber-button-secondary" onClick={speakAnswer}><Volume2 size={16} /> Read result aloud</button>
                            ) : (
                                <>
                                    <button type="button" className="cyber-button-secondary" onClick={handlePauseSpeech}>{paused ? <Play size={15} /> : <Pause size={15} />}{paused ? 'Resume' : 'Pause'}</button>
                                    <button type="button" className="cyber-button-secondary" onClick={handleStopSpeech}><Square size={14} /> Stop</button>
                                </>
                            )}
                            <button type="button" className="cyber-button-secondary" onClick={handleReset}>Analyze another</button>
                        </div>
                    </section>

                    <section className="cyber-card video-qa-card">
                        <div className="video-section-heading"><div><span>REPORT CONTEXT</span><h3>Ask about this video</h3></div></div>
                        <p className="video-qa-note">Answers use the frame results above. No generative question-answering service is configured.</p>
                        <div className="video-suggestions">
                            {['What is this video about?', 'Which frames were classified as fake?', 'Explain the result in simple words.', 'Was audio analyzed?'].map(prompt => (
                                <button type="button" key={prompt} onClick={() => handleAsk(prompt)}>{prompt}</button>
                            ))}
                        </div>
                        <form className="video-question-form" onSubmit={event => { event.preventDefault(); handleAsk(); }}>
                            <input value={question} onChange={event => setQuestion(event.target.value)} placeholder="Ask something about this video..." className="cyber-input" />
                            <button type="submit" className="cyber-button-primary" disabled={!question.trim() || voiceListening || voiceTranscribing}>Ask</button>
                            <button type="button" className="cyber-button-secondary" onClick={voiceListening ? stopVoiceQuestion : startVoiceQuestion} disabled={voiceTranscribing}>
                                {voiceListening ? <Square size={14} /> : <Volume2 size={15} />}
                                {voiceListening ? 'Stop' : voiceTranscribing ? 'Transcribing...' : 'Ask by voice'}
                            </button>
                        </form>
                        {voiceStatus && <p className="video-qa-note" role="status">{voiceStatus}</p>}
                        {answer && <div className="video-answer"><strong>Report-based answer</strong><p>{answer}</p></div>}
                    </section>
                </>
            )}
        </div>
    );
}
