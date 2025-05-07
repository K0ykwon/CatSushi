# 🐱🍣 Cat Sushi  
**AI-Powered Medication Assistant for Visually Impaired Parents**

Cat Sushi is an **AI-based interactive assistant** designed to help visually impaired parents safely determine whether their child can take a given medication.  
By receiving an image of the medicine and a spoken question, the system analyzes both the medication content and the child’s condition to provide guidance on **safety, dosage, and precautions** — via **text and voice**.

[👉 Try the Demo](https://cat-sushi.streamlit.app/)

---

## 🧠 Technologies Used

| Component         | Technology                                 |
|-------------------|---------------------------------------------|
| Language Model     | **OpenAI GPT-4.0** (question refinement & guidance generation) |
| Speech-to-Text     | **Whisper** (high-quality voice-to-text conversion) |
| Text-to-Speech     | pyttsx3 or equivalent (audio feedback generation) |
| OCR                | **UpstageAI OCR API** (extracting info from medicine image) |
| External Info Search | Crawling Drug.com, WHO, etc.             |
| Web Framework      | Streamlit                                  |
| Session Management | Streamlit SessionState                     |

---

## 🎯 Project Goals

- Enable visually impaired parents to **safely administer medication** to their children based on symptoms.
- Use AI to **clarify and restructure vague or complex user questions**.
- Combine child health information and medicine data for **automated medication guidance**.

---

## 📲 Core Features

### 🗣️ 1. Voice Input → STT → Question Refinement
- Converts speech to text using **Whisper** for high accuracy.
- Refines and restructures questions using **GPT-4.0** for clarity.
- Automatically extracts child information: **age, weight, symptoms, allergies**.

### 📷 2. Medicine Image Recognition
- Users upload a photo of the medicine.
- **UpstageAI OCR API** is used to extract key info: **active ingredients, dosage instructions, warnings**.

### 🧾 3. AI-Based Medication Guidance
- Combines child data and drug information to determine **suitability and dosage**.
- Optionally consults **external databases** (e.g. Drug.com, WHO) to enrich guidance.
- Returns answers in both **text and voice (TTS)** formats for accessibility.

---

## 🗂️ System Architecture

```plaintext
[User Input]
  ├── Voice (question, child details)
  └── Medicine image

    ↓

[Processing Pipeline]
  ├── Whisper STT: Converts voice to text
  ├── GPT-4: Refines question and structures input
  ├── Upstage OCR: Extracts info from image
  ├── External drug info crawling
  └── Session management (Streamlit)

    ↓

[System Output]
  ├── Medication guide tailored to child’s condition
  ├── Safety and dosage recommendation
  └── Audio response via TTS

```

## ⚙️ Optimization Highlights

- 🔈 Enhanced speech recognition via Whisper STT
- 🧠 Clearer and safer queries through GPT-4.0 question refinement
- 📷 High OCR accuracy using UpstageAI for reliable medicine info extraction
- 🧭 Context-aware answers using previous conversation history
- 🔁 Persistent session with Streamlit SessionState (no data loss on refresh)
- 🗣️ Instant audio feedback for a more accessible user experience
