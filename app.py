import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
from audio_analyzer import AudioDeepfakeDetector
from generate_samples import generate_samples

st.set_page_config(
    page_title="VoxShield - AI Voice Deepfake Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme styling
st.markdown("""
<style>
    .main { background-color: #0b0f19; color: #f8fafc; }
    .stMetric { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 12px; }
    .verdict-box-human { background: rgba(16, 185, 129, 0.1); border: 2px solid #10b981; border-radius: 12px; padding: 18px; text-align: center; }
    .verdict-box-fake { background: rgba(239, 68, 68, 0.1); border: 2px solid #ef4444; border-radius: 12px; padding: 18px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("🎙️ VoxShield — Forensic AI Voice Deepfake & Audio Clone Detector")
st.caption("Acoustic Biometric Verification, Spectral Vocoder Artifact Analysis & Neural Speech Detection")

detector = AudioDeepfakeDetector()

# Sidebar: Controls & Sample Loading
st.sidebar.header("📁 Audio Source")
source_option = st.sidebar.radio(
    "Choose Audio Input:",
    ["Upload Audio File", "Pre-loaded Evaluation Samples"]
)

audio_file = None

if source_option == "Upload Audio File":
    uploaded = st.sidebar.file_uploader("Upload Audio (WAV, MP3, OGG, FLAC):", type=["wav", "mp3", "ogg", "flac"])
    if uploaded:
        audio_file = uploaded
else:
    # Ensure samples exist
    if not os.path.exists("samples/authentic_human_voice.wav"):
        generate_samples("samples")
        
    sample_choice = st.sidebar.selectbox(
        "Select Benchmark Sample:",
        ["Authentic Human Voice Sample", "AI Cloned Deepfake Voice Sample"]
    )
    if sample_choice == "Authentic Human Voice Sample":
        audio_file = "samples/authentic_human_voice.wav"
    else:
        audio_file = "samples/ai_cloned_deepfake.wav"

if audio_file is not None:
    # Analyze Audio
    with st.spinner("🔍 Running acoustic feature extraction and spectral analysis..."):
        results = detector.analyze(audio_file)

    # Top Section: Audio Player & Verdict
    col1, col2 = st.columns([5, 7])

    with col1:
        st.subheader("🎧 Audio Stream Inspection")
        if isinstance(audio_file, str):
            st.audio(audio_file)
        else:
            audio_file.seek(0)
            st.audio(audio_file)
            
        is_fake = results["is_deepfake"]
        prob_pct = results["deepfake_probability"] * 100
        conf_pct = results["confidence"] * 100

        st.markdown("<br>", unsafe_allow_html=True)
        if is_fake:
            st.markdown(f"""
            <div class="verdict-box-fake">
                <h2 style="color: #ef4444; margin-bottom: 4px;">🚨 AI-SYNTHESIZED (DEEPFAKE)</h2>
                <p style="color: #fca5a5; margin-bottom: 0;"><b>Deepfake Probability: {prob_pct:.1f}%</b> (Confidence: {conf_pct:.1f}%)</p>
                <small style="color: #94a3b8;">Vocoder high-frequency quantization & unnatural pitch smoothness detected.</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-box-human">
                <h2 style="color: #10b981; margin-bottom: 4px;">✅ AUTHENTIC HUMAN VOICE</h2>
                <p style="color: #6ee7b7; margin-bottom: 0;"><b>Human Authenticity Score: {100 - prob_pct:.1f}%</b> (Confidence: {conf_pct:.1f}%)</p>
                <small style="color: #94a3b8;">Natural glottal jitter, organic wideband harmonics, and biological breathing dynamics confirmed.</small>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.subheader("📊 Acoustic Biometric Telemetry")
        m1, m2, m3 = st.columns(3)
        m1.metric("Spectral Flatness", f"{results['features']['spectral_flatness']:.3f}", 
                  help="Unnaturally high flatness indicates synthetic neural vocoder artifacts.")
        m2.metric("High-Freq Energy Ratio", f"{results['features']['high_freq_ratio']:.3f}", 
                  help="Values < 0.04 indicate sharp post-filter vocoder cutoffs common in speech synthesis.")
        m3.metric("Micro-Jitter Index", f"{results['features']['jitter_estimate']:.3f}", 
                  help="Biological human vocal cords exhibit natural pitch micro-tremors (higher jitter).")

        m4, m5, m6 = st.columns(3)
        m4.metric("Zero Crossing Rate", f"{results['features']['zcr']:.4f}")
        m5.metric("Spectral Centroid", f"{results['features']['spectral_centroid']:.0f} Hz")
        m6.metric("Sampling Rate", f"{results['sample_rate']} Hz")

    st.markdown("---")

    # Visualizations: Waveform & Spectrogram Heatmap
    tab_spec, tab_wave, tab_forensics = st.tabs(["🔥 Mel-Frequency Spectrogram Heatmap", "🌊 Raw Audio Waveform", "📑 Forensic Inspection Audit"])

    with tab_spec:
        spec = results["spectrogram"]
        # Downsample for snappy Plotly rendering
        step_t = max(1, len(spec["times"]) // 200)
        step_f = max(1, len(spec["frequencies"]) // 100)
        
        fig_spec = go.Figure(data=go.Heatmap(
            z=spec["db"][::step_f, ::step_t],
            x=spec["times"][::step_t],
            y=spec["frequencies"][::step_f],
            colorscale="Viridis",
            colorbar=dict(title="Energy (dB)")
        ))
        fig_spec.update_layout(
            title="Short-Time Fourier Transform (STFT) Spectrogram Energy Distribution",
            xaxis_title="Time (seconds)",
            yaxis_title="Frequency (Hz)",
            template="plotly_dark",
            height=380,
            margin=dict(l=30, r=30, t=40, b=30)
        )
        st.plotly_chart(fig_spec, use_container_width=True)

    with tab_wave:
        waveform = results["waveform"]
        step_w = max(1, len(waveform) // 2000)
        time_axis = np.linspace(0, len(waveform) / results["sample_rate"], len(waveform))
        
        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(
            x=time_axis[::step_w],
            y=waveform[::step_w],
            mode="lines",
            line=dict(color="#38bdf8", width=1.5)
        ))
        fig_wave.update_layout(
            title="Normalized Audio Amplitude Envelope",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            template="plotly_dark",
            height=280,
            margin=dict(l=30, r=30, t=40, b=30)
        )
        st.plotly_chart(fig_wave, use_container_width=True)

    with tab_forensics:
        st.write("#### Forensic Parameter Verification Checklist")
        report_data = [
            {"Biometric Indicator": "High-Frequency Cutoff (>7.5kHz)", "Observed Value": f"{results['features']['high_freq_ratio']:.4f}", "Benchmark Threshold": "≥ 0.05 (Human)", "Classification": "PASS (Natural)" if results['features']['high_freq_ratio'] >= 0.05 else "FAIL (Synthesized Artifact)"},
            {"Biometric Indicator": "Spectral Flatness (Wiener Entropy)", "Observed Value": f"{results['features']['spectral_flatness']:.4f}", "Benchmark Threshold": "< 0.18 (Organic)", "Classification": "PASS (Natural)" if results['features']['spectral_flatness'] < 0.18 else "FAIL (Neural Vocoder Bias)"},
            {"Biometric Indicator": "Glottal Micro-Jitter Tremor", "Observed Value": f"{results['features']['jitter_estimate']:.4f}", "Benchmark Threshold": "≥ 0.85 (Human)", "Classification": "PASS (Natural)" if results['features']['jitter_estimate'] >= 0.85 else "FAIL (Synthetic Pitch Lock)"},
            {"Biometric Indicator": "Mean Spectral Centroid", "Observed Value": f"{results['features']['spectral_centroid']:.1f} Hz", "Benchmark Threshold": "800 - 3200 Hz", "Classification": "NOMINAL"}
        ]
        st.table(pd.DataFrame(report_data))
        
else:
    st.info("👈 Upload an audio clip or select a pre-loaded sample in the sidebar to begin forensic voice analysis.")
