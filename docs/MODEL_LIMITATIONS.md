# TrustGuard AI — Model Limitations and Forensic Boundaries

Media authenticity verification is an adversarial, rapidly-evolving computer vision and acoustic engineering problem. To preserve trust, transparency, and research integrity, TrustGuard AI explicitly documents the boundaries, assumptions, and known failure modes of each detection pipeline.

---

## 1. Image Model Limitations (MobileNetV2 V4)

1. **Resolution & Compression Degradation**:
   - The model expects 224×224×3 RGB input. Extreme downsampling of high-resolution images or aggressive re-compression (e.g., multiple rounds of WhatsApp/Telegram lossy JPEG compression) removes high-frequency artifact residues, which can degrade detection precision.
2. **Out-of-Distribution Generative Models**:
   - While trained on StyleGAN, StyleGAN2, and FaceForensics++ face-swaps, newer latent diffusion architectures (e.g., FLUX, Midjourney v6, Stable Diffusion 3.5) introduce different noise scheduling and attention patterns. An `INCONCLUSIVE` classification is explicitly emitted when scores fall within the uncertainty margin `[0.45, 0.55]`.
3. **Non-Facial Imagery**:
   - The model is specialized for human portrait, headshot, and facial media authenticity. While it can detect artifacts on general images, its highest validation guarantees apply to human faces.
4. **Adversarial Perturbations**:
   - Deliberate adversarial attacks (FGSM, PGD, subtle noise injection) can shift prediction probabilities without altering human visual perception.

---

## 2. Explainable AI & Error Level Analysis (ELA) Limitations

1. **Attention Map vs. Ground Truth Tampering**:
   - Gradient-based model attention heatmaps (Grad-CAM) indicate the receptive regions that most strongly influenced the classification head. **They must never be presented as definitive forensic proof of localized splicing or physical tampering.**
2. **Compression Inconsistency**:
   - Error Level Analysis operates by computing difference residuals against an 85% JPEG re-compression baseline. Uniformly compressed images or raw lossless PNG files will exhibit minimal ELA variance regardless of authenticity.

---

## 3. Video Frame-Level Analysis Limitations

1. **Temporal Artifacts**:
   - The current video verification pipeline samples individual keyframes, computes face crops, and aggregates frame-level scores through dynamic interval weighting.
   - It does **not** evaluate inter-frame optical flow, temporal eye-blinking cadence, or heartbeat photoplethysmography (rPPG). Results reflect per-frame spatial authenticity, not temporal continuity.
2. **Variable Frame Rate & Lighting**:
   - Low-light videos with motion blur produce lower facial landmark detection confidence.

---

## 4. Audio Verification Limitations

1. **Acoustic Signal Processing**:
   - Audio authenticity evaluates spectral centroid distribution, high-frequency energy cutoffs, zero-crossing rates, and dynamic range to flag synthetic vocoders.
   - It is not a speaker biometric verification system and cannot confirm the true legal identity of the speaker.
2. **Audio Compression & Codec Artifacts**:
   - Low-bitrate telephony codecs (e.g., AMR-NB at 8 kHz or GSM) naturally truncate high frequencies above 3.4 kHz, which can mimic synthetic speech cutoffs.

---

## 5. Provenance & Cryptographic Metadata Boundaries

1. **Metadata Stripping**:
   - Social media platforms routinely strip EXIF metadata, camera serial numbers, and C2PA content credentials during upload. **The absence of metadata does not indicate manipulation.**
2. **SHA-256 Immutability**:
   - SHA-256 hashes confirm file integrity against tampering *after* generation; they cannot verify the historical origin or truthfulness of the media itself.
