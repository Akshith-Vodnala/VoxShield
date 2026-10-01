# 🎙️ VoxShield — Forensic AI Voice Deepfake & Audio Clone Detection System

VoxShield is an automated acoustic forensics platform designed to detect AI-synthesized speech, voice clones (e.g., ElevenLabs, Tacotron, VITS, HiFi-GAN), and audio spoofing attacks in digital communications.

---

## 📌 Problem Statement & Motivation
With recent breakthroughs in diffusion models and neural vocoders, voice cloning has become a significant vector for social engineering, financial fraud, and CEO impersonation scams. Traditional anti-spoofing systems struggle to differentiate synthetic waveforms from real voices. **VoxShield** analyzes microscopic glottal dynamics, high-frequency energy attenuation, and spectral Wiener entropy to identify synthetic audio with high precision.

---

## 🚀 Key Architectural Features

1. **Acoustic Biometric Feature Extraction:**
   * **Glottal Micro-Jitter Analysis:** Measures cycle-to-cycle pitch perturbations naturally caused by human vocal fold tissues. Synthetic speech generators lack biological tremors and exhibit robotic pitch trajectories.
   * **High-Frequency Attenuation (>7.5 kHz):** Identifies steep energy cutoffs typical of 16kHz/22kHz neural vocoder sampling pipelines.
   * **Spectral Flatness (Wiener Entropy):** Quantifies tonal vs. noise-like energy distribution to catch vocoder quantization artifacts:
     $$SF = \frac{\exp\left(\frac{1}{N}\sum_{k=1}^N \ln |X(k)|\right)}{\frac{1}{N}\sum_{k=1}^N |X(k)|}$$

2. **Spectrogram Forensic Analysis:**
   * Computes Short-Time Fourier Transform (STFT) power distributions.
   * Renders dynamic interactive 2D frequency-time heatmaps using **Plotly**.

3. **Interactive Forensics Dashboard:**
   * Built with **Streamlit** in a high-tech dark theme.
   * Multi-format ingestion: Supports WAV, MP3, OGG, and FLAC files.
   * Comprehensive **Forensic Inspection Audit Checklist** breaking down decision factors for security analysts.

---

## 🛠️ Technology Stack
* **Programming Language:** Python 3.9+
* **Audio Signal Processing:** SciPy Signal, SoundFile
* **Mathematical Analytics:** NumPy
* **Interactive Dashboard:** Streamlit
* **Telemetry Visualizations:** Plotly Graph Objects
* **Tabular Reporting:** Pandas

---

## 📂 Project Structure
```
VoxShield/
├── audio_analyzer.py    # Core Acoustic Signal Processing & Deepfake Engine
├── generate_samples.py  # Generates Controlled Benchmark Samples (Human vs AI)
├── app.py               # Streamlit Forensics Hub
├── requirements.txt     # Python Dependencies
├── .gitignore           # Git ignore rules
└── README.md            # Project Documentation
```

---

## ⚙️ Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/<username>/VoxShield.git
cd VoxShield
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Benchmark Audio Samples
```bash
python generate_samples.py
```

### 4. Launch the Forensics Dashboard
```bash
streamlit run app.py
```

### 5. Running a Quick Demonstration:
* In the left sidebar, choose **"Pre-loaded Evaluation Samples"**.
* Switch between:
  * **Authentic Human Voice Sample:** The system displays a green **"AUTHENTIC HUMAN VOICE"** banner, showing natural pitch micro-jitter and broad spectral energy.
  * **AI Cloned Deepfake Voice Sample:** The system triggers a red **"AI-SYNTHESIZED (DEEPFAKE)"** alert, highlighting the steep vocoder frequency cutoff and synthetic flatness!
