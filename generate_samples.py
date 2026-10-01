import numpy as np
import soundfile as sf
import os

def generate_samples(output_dir="samples"):
    """
    Synthesizes two controlled acoustic test signals:
    1. authentic_human_voice.wav (Natural human formants, vibrato, micro-jitter, wideband breath noise)
    2. ai_cloned_deepfake.wav (Quantized harmonic vocoder, steep cutoff above 7.5kHz, zero vocal jitter)
    """
    os.makedirs(output_dir, exist_ok=True)
    sr = 22050
    duration = 3.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 1. Authentic Human Voice Simulation
    # Natural fundamental pitch (~130Hz) with natural vibrato and micro-jitter
    pitch_contour = 130.0 + 3.5 * np.sin(2 * np.pi * 4.5 * t) + np.random.normal(0, 0.4, len(t))
    phase = 2 * np.pi * np.cumsum(pitch_contour) / sr
    
    # Human vowel formants (F1, F2, F3) + natural biological breath noise
    f1 = 0.5 * np.sin(phase * 4.0)  # ~520Hz
    f2 = 0.3 * np.sin(phase * 11.5) # ~1500Hz
    f3 = 0.2 * np.sin(phase * 19.0) # ~2500Hz
    breath_noise = np.random.normal(0, 0.04, len(t))
    
    human_signal = (np.sin(phase) + f1 + f2 + f3 + breath_noise)
    envelope = np.sin(np.pi * t / duration) ** 0.5
    human_signal = human_signal * envelope
    human_signal = human_signal / np.max(np.abs(human_signal))

    human_path = os.path.join(output_dir, "authentic_human_voice.wav")
    sf.write(human_path, human_signal.astype(np.float32), sr)

    # 2. AI Cloned Deepfake Voice Simulation
    # Neural vocoders generate ultra-rigid harmonics with abrupt 7.5kHz cutoff
    rigid_pitch = 140.0
    synth_phase = 2 * np.pi * rigid_pitch * t
    deepfake_signal = np.zeros_like(t)
    
    # Sum strictly quantized harmonics up to 7500Hz (vocoder band limit)
    for harmonic in range(1, 50):
        h_freq = rigid_pitch * harmonic
        if h_freq < 7500:
            deepfake_signal += (1.0 / (harmonic ** 0.85)) * np.sin(harmonic * synth_phase)
            
    # Add vocoder phase discontinuity / quantization artifact
    quant_artifact = np.round(deepfake_signal * 32.0) / 32.0
    deepfake_signal = 0.8 * deepfake_signal + 0.2 * quant_artifact
    deepfake_signal = deepfake_signal * envelope
    deepfake_signal = deepfake_signal / np.max(np.abs(deepfake_signal))

    deepfake_path = os.path.join(output_dir, "ai_cloned_deepfake.wav")
    sf.write(deepfake_path, deepfake_signal.astype(np.float32), sr)

    return human_path, deepfake_path

if __name__ == "__main__":
    h_path, d_path = generate_samples()
    print(f"Generated sample files:\n- {h_path}\n- {d_path}")
