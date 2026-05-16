# Speech AI for Urdu Poetry (Shayari) 🎙️📖

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![AI](https://img.shields.io/badge/AI-Speech--to--Text-orange)
![Dataset](https://img.shields.io/badge/Dataset-Urdu%20Poetry-brightgreen)

## 📌 Project Purpose
This project aims to develop the **first Speech AI system** trained exclusively on professional Urdu poetry (Shayari) performances. The model will learn from actual Urdu poets (*Shayars*) and their emotionally expressive recitations to understand and capture the nuances of acting and emotion in Urdu poetry.

### Key Focus Areas:
- **Tone and Delivery:** Capturing precise tonal variations that convey emotion.
- **Acting Quality:** Understanding what makes a poetry performance compelling and authentic.
- **Emotional Range:** Capturing soulfulness, romance, nostalgia, and other emotions deeply rooted in Urdu poetry.

---

## 🚀 Why This Project Matters
1. **First of Its Kind:** Currently, there is no acting dataset created from Urdu poetry performances.
2. **Essential for Understanding Acting:** Crucial for training AI models that can understand emotional nuances and acting techniques specific to Urdu poetry.
3. **Project Innovation:** By creating this dataset, we can train a model that intimately understands emotional expression in this unique cultural art form.

---

## 🧠 Model Training Approach
- **Model Type:** Single GPU voice-to-voice training model (utilizing AICO or similar frameworks).
- **Training Process:** The dataset will be fed to the model, utilizing specific prompts to guide its emotional understanding.
- **Data Collection Sources:**
  - YouTube channels featuring professional Urdu poetry performances.
  - Video recordings from established Urdu poets and Shayars.
  - Performances demonstrating highly specific and intentional emotional expressions.

*Key Insight:* In Urdu poetry, performers express emotions very specifically and deliberately. This intentional emotional delivery is the core element we aim to capture.

---

## 📊 Dataset Creation Process

### Step 1: Audio Collection and Annotation
- **Extract Audio:** Automated Python scripts to extract audio from performance videos.
- **Annotate Dataset:** Human annotators to label emotional content.
- **AI-Assisted Analysis:** Using **Gemini AI** to assist in understanding and categorizing emotions.
- **Quality Control:** Using platforms like Qwen for validation.

### Step 2: Emotion Classification
Targeting a minimum of 5 and maximum of 10 emotion types, including:
- Nostalgia (yearning for the past)
- Belonging (connection and identity)
- Joy and celebration
- Abstract conceptual emotions
- Travel and journey-related emotions

### Step 3: Constructive Pipeline Development
- **Prompt Engineering:** Utilizing Gemini or Qwen for accurate identification of emotional audio snippets.
- **Automated Extraction:** Python scripts for processing segments.
- **Dataset Specifications:**
  - **Volume:** 100-200 hours of curated performances.
  - **Segment Length:** 1-minute fragments per sample.
  - **Size:** 1,000-2,000 emotionally labeled audio segments.

---

## 📈 Benchmarking and Evaluation Strategy
The model will be trained on platforms like LAION, aiming for ~900 annotations for comprehensive coverage. Human judges will evaluate the model based on a **Four-Dimensional (4D) Benchmark Criteria**:

1. **Clarity:** Intelligibility of speech output.
2. **Fluency:** Smoothness and naturalness of speech flow/rhythm.
3. **Accent Authenticity:** Accuracy of the Urdu accent and pronunciation.
4. **Language Distance:** Measuring distance from Hindi to maintain strict Urdu linguistic authenticity.

---

## 📝 Summary
This project is a pioneering effort to create an authentic dataset (100-200 hours, 1,000-2,000 snippets) from professional Urdu poets' performances. Through AI-assisted categorization and human validation, we will build a resource capable of teaching AI the deep emotional nuances of Urdu poetry.
