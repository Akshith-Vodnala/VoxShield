import numpy as np
import soundfile as sf
from scipy import signal
import io

class AudioDeepfakeDetector:
    """
    Forensic Audio Signal Analysis Engine.
    Detects AI-synthesized speech (ElevenLabs, Tacotron, VITS vocoder artifacts)
    vs authentic human vocal tract acoustics.
    """
    def __init__(self):
        self.sample_rate = 22050

    def load_audio(self, audio_source):
        """
        Loads audio from file path or bytes buffer.
        Normalizes audio to mono float32.
        """
        if isinstance(audio_source, (str, bytes, io.BytesIO)):
            data, sr = sf.read(audio_source)
        else:
            data, sr = sf.read(io.BytesIO(audio_source.read()))

        # Convert stereo to mono if needed
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # Normalize amplitude to [-1.0, 1.0]
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val

        # Resample to standard analysis rate if different
        if sr != self.sample_rate:
            num_samples = int(len(data) * self.sample_rate / sr)
            data = signal.resample(data, num_samples)
            sr = self.sample_rate

        return data, sr

    def compute_spectrogram(self, data, sr, n_fft=1024, hop_length=256):
        """
        Computes Short-Time Fourier Transform (STFT) power spectrogram.
        """
        frequencies, times, Zxx = signal.stft(
            data, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length
        )
        spec_power = np.abs(Zxx) ** 2
        spec_db = 10 * np.log10(np.maximum(spec_power, 1e-10))
        return frequencies, times, spec_db

    def extract_acoustic_features(self, data, sr):
        """
        Extracts biometric vocal tract indicators and vocoder artifacts.
        """
        # 1. Zero Crossing Rate (ZCR)
        zero_crossings = np.nonzero(np.diff(data > 0))[0]
        zcr = len(zero_crossings) / float(len(data))

        # 2. Spectral Centroid & Bandwidth
        frequencies, times, Zxx = signal.stft(data, fs=sr, nperseg=1024, noverlap=512)
        magnitude = np.abs(Zxx)
        
        freq_grid = frequencies[:, np.newaxis]
        centroid = np.sum(freq_grid * magnitude, axis=0) / np.maximum(np.sum(magnitude, axis=0), 1e-10)
        mean_centroid = float(np.mean(centroid))

        # 3. High-Frequency Vocoder Energy Ratio (Above 7500 Hz)
        # Deepfakes often drop off sharply above 7.5kHz due to vocoder compression
        high_freq_idx = np.where(frequencies >= 7500)[0]
        if len(high_freq_idx) > 0:
            high_freq_energy = np.mean(magnitude[high_freq_idx, :])
            total_energy = np.mean(magnitude)
            hf_ratio = float(high_freq_energy / np.maximum(total_energy, 1e-10))
        else:
            hf_ratio = 0.05

        # 4. Spectral Flatness (Wiener Entropy)
        # AI speech vocoders exhibit unnaturally flat spectral distributions in unvoiced regions
        geo_mean = np.exp(np.mean(np.log(np.maximum(magnitude, 1e-10)), axis=0))
        arith_mean = np.mean(magnitude, axis=0)
        spectral_flatness = float(np.mean(geo_mean / np.maximum(arith_mean, 1e-10)))

        # 5. Pitch Perturbation / Micro-Jitter
        # Real human vocal cords have natural non-linear micro-tremors (jitter).
        # Neural vocoders generate unnaturally smooth glottal pulses.
        autocorr = signal.correlate(data[:min(len(data), sr * 2)], data[:min(len(data), sr * 2)], mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        peak_diffs = np.diff(autocorr[:200])
        jitter_estimate = float(np.std(peak_diffs) / np.maximum(np.mean(np.abs(peak_diffs)), 1e-10))

        return {
            "zcr": zcr,
            "spectral_centroid": mean_centroid,
            "high_freq_ratio": hf_ratio,
            "spectral_flatness": spectral_flatness,
            "jitter_estimate": jitter_estimate
        }

    def analyze(self, audio_source):
        """
        Performs full deepfake speech analysis and forensic classification.
        """
        data, sr = self.load_audio(audio_source)
        features = self.extract_acoustic_features(data, sr)
        freqs, times, spec_db = self.compute_spectrogram(data, sr)

        # Deepfake Probability Scoring Heuristic
        # High AI probability occurs when:
        # - High frequency cutoff is steep (low hf_ratio < 0.03)
        # - Spectral flatness is unnaturally uniform (> 0.25)
        # - Pitch micro-tremor jitter is robotic (< 0.60)
        ai_evidence = 0.0
        
        if features["high_freq_ratio"] < 0.04:
            ai_evidence += 0.35
        elif features["high_freq_ratio"] < 0.08:
            ai_evidence += 0.15

        if features["spectral_flatness"] > 0.18:
            ai_evidence += 0.30
        
        if features["jitter_estimate"] < 0.85:
            ai_evidence += 0.25

        if features["zcr"] < 0.03 or features["zcr"] > 0.18:
            ai_evidence += 0.10

        deepfake_prob = float(np.clip(ai_evidence, 0.05, 0.98))
        is_deepfake = deepfake_prob >= 0.50

        confidence = deepfake_prob if is_deepfake else (1.0 - deepfake_prob)

        return {
            "is_deepfake": is_deepfake,
            "prediction": "AI-SYNTHESIZED (DEEPFAKE)" if is_deepfake else "AUTHENTIC HUMAN VOICE",
            "deepfake_probability": deepfake_prob,
            "confidence": confidence,
            "features": features,
            "waveform": data,
            "sample_rate": sr,
            "spectrogram": {
                "frequencies": freqs,
                "times": times,
                "db": spec_db
            }
        }
