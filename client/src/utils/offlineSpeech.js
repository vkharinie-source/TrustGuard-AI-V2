let whisperTranscriberPromise;

const getWhisperTranscriber = (onProgress) => {
    if (!whisperTranscriberPromise) {
        whisperTranscriberPromise = import('@huggingface/transformers')
            .then(({ pipeline }) => pipeline(
                'automatic-speech-recognition',
                'onnx-community/whisper-tiny.en',
                {
                    dtype: 'q8',
                    progress_callback: (progress) => {
                        if (progress.status === 'progress' && Number.isFinite(progress.progress)) {
                            onProgress?.(Math.round(progress.progress));
                        }
                    },
                },
            ))
            .catch((error) => {
                whisperTranscriberPromise = undefined;
                throw error;
            });
    }
    return whisperTranscriberPromise;
};

const resampleAudio = async (samples, sourceRate, targetRate) => {
    if (sourceRate === targetRate) return samples;
    const OfflineAudioContextType = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    if (!OfflineAudioContextType) throw new Error('Audio resampling is not supported in this browser.');
    const frameCount = Math.ceil(samples.length * targetRate / sourceRate);
    const context = new OfflineAudioContextType(1, frameCount, targetRate);
    const buffer = context.createBuffer(1, samples.length, sourceRate);
    buffer.copyToChannel(samples, 0);
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    source.start();
    return (await context.startRendering()).getChannelData(0);
};

export const transcribeAudioSamples = async (samples, sampleRate, onProgress) => {
    const transcriber = await getWhisperTranscriber(onProgress);
    const modelSamples = await resampleAudio(samples, sampleRate, 16000);
    return transcriber(modelSamples, { chunk_length_s: 30, stride_length_s: 5 });
};
