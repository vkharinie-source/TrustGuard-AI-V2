let captionerPromise;

const getCaptioner = (onProgress) => {
    if (!captionerPromise) {
        captionerPromise = import('@huggingface/transformers')
            .then(({ pipeline }) => pipeline(
                'image-to-text',
                'Xenova/vit-gpt2-image-captioning',
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
                captionerPromise = undefined;
                throw error;
            });
    }
    return captionerPromise;
};

export const captionSampledFrame = async (imageUrl, onProgress) => {
    const captioner = await getCaptioner(onProgress);
    const output = await captioner(imageUrl, { max_new_tokens: 32, num_beams: 2 });
    return String(output?.[0]?.generated_text || '').trim();
};
