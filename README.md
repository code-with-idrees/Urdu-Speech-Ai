# Urdu Speech AI 🚀

![Urdu Speech AI Logo](file:///C:/Users/HP/.gemini/antigravity/brain/cfe5fbcc-864d-4799-9521-a559d3dc5e87/artifacts/urdu_speech_ai_logo.png)

**Urdu Speech AI** is a comprehensive, end‑to‑end pipeline for extracting, cleaning, transcribing, and emotion‑annotating Urdu poetry performances from YouTube.  It is built for both **Kaggle** notebooks (with strict storage limits) and local Windows environments.

---

## ✨ Features

- **YouTube audio extraction** via `yt‑dlp`
- **Vocal isolation** using `Demucs`
- **Silence‑aware segmentation** (VAD) with optional overlap
- **High‑quality Whisper transcription** (large‑v3, forced Urdu)
- **AI‑driven emotion annotation** powered by **Google Gemini**
- **Four‑dimensional benchmark evaluation** (clarity, fluency, accent authenticity, language distance)
- **Dataset builder** that produces HuggingFace‑compatible JSON/CSV splits
- **Kaggle‑friendly**: intermediate WAVs are deleted and only tiny JSON files are zipped to stay well under the 30 GB quota

---

## 🛠️ Quickstart (Kaggle)

1. **Create a Kaggle dataset** with your raw audio (≈15 GB) and attach it as an input.
2. In a notebook cell:
   ```python
   !git clone https://github.com/code-with-idrees/Urdu-Speech-Ai.git
   %cd Urdu-Speech-Ai
   !pip install -r requirements.txt
   ```
3. Run the pipeline:
   ```bash
   python src/download_audio.py --url "<YOUR_YOUTUBE_URL>"
   python src/segment_audio.py   # creates /kaggle/working/segments
   python src/preprocess_audio.py
   python kaggle_pipeline/run_pipeline.py   # produces urdu_transcripts_output.zip
   ```
4. Download the zip from the notebook output.

---

## 🖥️ Local Windows Setup

```powershell
# Clone the repo
git clone https://github.com/code-with-idrees/Urdu-Speech-Ai.git
cd Urdu-Speech-Ai

# Install dependencies (use user flag to avoid admin rights)
python -m pip install --user -r requirements.txt

# Create a .env file (copy from .env.example) and provide your Gemini API key
copy .env.example .env
# edit .env → set GEMINI_API_KEY=...

# Run the full pipeline on your local audio collection
python src/download_audio.py --url "<URL>"
python src/segment_audio.py --no-vad   # optional VAD
python src/preprocess_audio.py
python kaggle_pipeline/run_pipeline.py
```

---

## 📂 Repository Layout

```
Urdu-Speech-Ai/
├─ config/                # YAML settings
├─ data/                  # raw, segments, processed, annotations
├─ kaggle_pipeline/       # Kaggle‑compatible entry point
├─ src/                   # core scripts (download, segment, preprocess, transcribe…)
├─ README.md              # **You are here**
└─ .github/               # CI, issue templates, contribution guide
```

---

## 🤝 Contributing

We follow the classic **fork‑branch‑pull‑request** workflow.

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feat/awesome‑feature
   ```
3. Make your changes and ensure the pipeline still runs end‑to‑end.
4. Commit with a clear message (e.g., `feat: add support for multi‑speaker diarization`).
5. Open a Pull Request – describe the change, link any related issues, and add screenshots if applicable.

All contributions are welcome!  For major architectural changes, please open an issue first.

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 📣 Acknowledgements

- **OpenAI Whisper** – for state‑of‑the‑art speech‑to‑text.
- **Google Gemini** – for emotion classification.
- **Demucs** – vocal isolation.
- **yt‑dlp** – reliable YouTube downloads.

---

# 🚀 Get Started Now!

```bash
# Clone & install
git clone https://github.com/code-with-idrees/Urdu-Speech-Ai.git && cd Urdu-Speech-Ai
pip install -r requirements.txt

# Run!
python src/download_audio.py --url "https://youtu.be/…"
python kaggle_pipeline/run_pipeline.py
```

Happy coding! 🎉
