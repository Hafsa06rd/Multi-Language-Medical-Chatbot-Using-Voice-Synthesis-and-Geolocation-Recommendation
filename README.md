# Enhanced Multilingual AI Medical Chatbot with Voice Support and Geolocation Recommendation

> An AI-powered medical symptom analysis assistant built with **Streamlit**, **Google Gemini API**, and multilingual voice support. Designed to provide professional-grade medical insights in multiple languages, with an emphasis on accessibility and user experience.

---

##  Project Overview

This project is a conversational medical chatbot that allows users to describe their symptoms — via text or voice — and receive structured medical analysis, urgency assessments, specialist referrals, and actionable recommendations. It combines a local medical knowledge base with optional AI enhancement via the **Google Gemini API** (`gemini-1.5-flash`), and supports **7 languages** with audio response synthesis.

The system is built around two core modules:
- **`ChatBot.py`** — Core NLP engine for symptom extraction, diagnosis logic, multilingual translation, and audio generation.
- **`app.py`** — Streamlit-based frontend that provides an interactive chat interface, dataset explorer, voice input, and emergency guidance.

---

##  Key Features

###  AI-Powered Symptom Analysis
- Dual-mode diagnosis: **Google Gemini AI** (when configured) for advanced analysis, with automatic fallback to a **local knowledge base** using zero-shot classification (`facebook/bart-large-mnli`).
- Extracts symptoms from free-text input using enhanced pattern matching with severity inference (mild/moderate/severe/emergency).

###  Multilingual Support
Supports input and output in **7 languages**:
`English` · `French` · `Spanish` · `German` · `Italian` · `Portuguese` · `Arabic`

- Auto-detects the language of user input using `langdetect`.
- Translates user messages to English for analysis, then translates the response back via `deep-translator`.
- Chat history translation when the user switches languages mid-session.

###  Voice Input & Audio Output
- Microphone-based voice input using `PyAudio` and `SpeechRecognition` with real-time recording progress.
- Text-to-speech audio responses generated with `gTTS`, played back directly in the browser.

###  Medical Dataset Integration
- Loads a local `Symptom_dataset.csv` to enrich diagnoses with real patient statistics (average age, gender distribution, case counts).
- Dataset statistics are displayed on demand in the UI.

###  Specialist Referral System
- Recommends the appropriate medical specialist based on the diagnosed condition.
- Supports fallback from an external `specialist-recommendations.csv` to a built-in mapping of 50+ conditions.

###  Urgency & Severity Assessment
- Each diagnosis is tagged with an urgency level: `Routine` / `Urgent` / `Emergency`.
- Emergency situations trigger prominent warnings with international emergency numbers (MR, EU, UK, US).

---

##  Architecture & How It Works

```
User Input (Text / Voice)
        │
        ▼
Language Detection (langdetect)
        │
        ▼
Translation to English (deep-translator) ──► if non-English
        │
        ▼
Symptom Extraction (pattern matching + Gemini AI)
        │
        ▼
Diagnosis Engine
   ├── Gemini AI Analysis (primary, if configured)
   └── Local Knowledge Base (fallback via BART zero-shot)
        │
        ▼
Specialist Recommendation + Urgency Scoring
        │
        ▼
Response Generation
        │
        ▼
Translation Back to User Language + Audio Synthesis (gTTS)
        │
        ▼
Streamlit Chat Interface
```

---

##  Technologies Used

| Category | Technology |
|---|---|
| Frontend / UI | Streamlit |
| AI / LLM | Google Gemini API (`gemini-1.5-flash`) |
| NLP Model | HuggingFace Transformers (`facebook/bart-large-mnli`) |
| Translation | `deep-translator` (GoogleTranslator) |
| Language Detection | `langdetect` |
| Speech Recognition | `SpeechRecognition`, `PyAudio` |
| Text-to-Speech | `gTTS` (Google Text-to-Speech), `pyttsx3` |
| Data Processing | `pandas`, `numpy` |
| Audio I/O | `soundfile`, `wave` |

---

##  Getting Started

### Prerequisites

- Python 3.9+
- A Google Gemini API key (free tier available at [https://ai.google.dev/](https://ai.google.dev/))
- A working microphone (for voice input)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/medical-chatbot.git
cd medical-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your Gemini API key
# Open ChatBot.py and replace the placeholder:
# self.api_key = "YOUR_GEMINI_API_KEY_HERE"

# 4. Add required data files
# Place Symptom_dataset.csv and specialist-recommendations.csv
# in the project root directory.

# 5. Run the application
streamlit run app.py
```

### Required Data Files

| File | Description |
|---|---|
| `Symptom_dataset.csv` | Medical dataset with columns: `Disease`, `Age`, `Gender`, `Fever`, `Cough`, `Fatigue`, `Difficulty Breathing`, `Blood Pressure`, `Cholesterol Level` |
| `specialist-recommendations.csv` | CSV with columns: `Disease`, `Symptom`, `Specialist Recommended` |

> **Note:** If `specialist-recommendations.csv` is not found, the system falls back to a built-in mapping of 50+ conditions.

---

##  Usage Tips

- **Be descriptive**: Include severity (*"severe"*, *"throbbing"*), duration (*"for 3 days"*), and context (*"when breathing deeply"*) for better accuracy.
- **Example input**: `"I have a severe throbbing headache for 2 days, with nausea and sensitivity to light."`
- **Switch languages** at any time using the language selector — the entire conversation history is retranslated automatically.
- **Voice mode**: Click "Voice Input", speak clearly for 5 seconds when prompted, and the app handles the rest.

---

##  Medical Disclaimer

> **This application is for educational and informational purposes only.** It is not intended as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any medical concerns. In case of emergency, call your local emergency services immediately.

---

##  Future Work

| Area | Planned Improvement |
|---|---|
| **API Key Management** | Replace hardcoded API key with environment variable loading (`.env` / `secrets.toml` for Streamlit Cloud) |
| **Extended Dataset** | Integrate larger, more diverse medical datasets (e.g., MIMIC, ICD-10 coded data) |
| **Better NLP** | Fine-tune a dedicated symptom NER model instead of relying on zero-shot classification |
| **Conversation Memory** | Add multi-turn conversation awareness so the chatbot can refine its diagnosis as the user provides more information |
| **User Profiles** | Allow users to save medical history (age, chronic conditions, allergies) for more personalized analysis |
| **Symptom Visualization** | Add body map UI for users to point to affected areas |
| **Cloud Deployment** | Package for Streamlit Cloud / Docker deployment with proper secrets management |
| **Evaluation Metrics** | Add benchmarking against known symptom-to-diagnosis datasets to measure accuracy |
| **Authentication** | Optional user login for session persistence across visits |

---

##  License

This project is released for educational and research purposes.
---
