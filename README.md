# HealthVoice AI 🩺

**Multilingual clinical Q&A over voice and SMS — no smartphone required.**

HealthVoice AI delivers RAG-grounded medical education to patients across Africa via toll-free calls and SMS. Answers are sourced from indexed clinical literature and scored for hallucination risk before being returned to the user.

---

## The Problem

Over 40% of patients in low-income and rural communities across Africa lack reliable smartphone access, cutting them off from digital health tools. Complex medical jargon creates confusion and poor health outcomes — especially for patients managing chronic conditions like HIV, malaria, diabetes, and hypertension.

## The Solution

HealthVoice AI meets patients where they are:
- **Voice calls** — works on any phone, including basic feature phones
- **SMS** — works without data or internet
- No app installation required
- Responds in the user's language automatically

---

## Features

- Voice and SMS interface via Twilio
- RAG-grounded answers indexed from PubMed and clinical guidelines
- Two-layer hallucination detection (NLI entailment + BERT NSP)
- Emergency detection in any language — routes to local emergency services
- Multilingual support: Swahili, Hausa, Yoruba, Amharic, Zulu, French, Arabic, and more
- Country inference from phone number for localized context
- Fail-closed classifier — defaults to refusal if topic classification fails

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Voice / SMS | Twilio |
| LLM | OpenAI GPT-3.5-turbo |
| Retrieval | FAISS + sentence-transformers |
| Hallucination scoring | cross-encoder NLI + BERT NSP |
| Data source | PubMed via NCBI E-utilities API |

---

## Architecture

```
SMS / Voice (Twilio)
        │
   [Classifier] — language detection, emergency check, topic gate
        │
   [RAG Retriever] — FAISS vector search over clinical literature
        │
   [LLM Generator] — GPT-3.5-turbo, grounded in retrieved chunks
        │
   [Hallucination Scorer] — NLI + BERT NSP per-sentence scoring
        │
   Response → SMS reply or Twilio TTS
```

---

## Project Structure

```
healthvoiceai/
├── rag/
│   ├── __init__.py
│   └── pipeline.py          # Embedding, FAISS retrieval, generation
├── hallucination/
│   ├── __init__.py
│   └── scorer.py            # NLI + NSP hallucination scoring
├── utils/
│   └── pubMedIngest.py      # PubMed abstract fetcher and indexer
├── templates/
│   └── index.html
├── app.py                   # Flask app — voice, SMS, and API routes
├── .env
└── requirements.txt
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/yourusername/healthvoice-ai.git
cd healthvoice-ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
```

### 3. Ingest clinical data

```bash
python utils/pubMedIngest.py --query "malaria treatment Africa" --max 100
python utils/pubMedIngest.py --query "HIV antiretroviral therapy" --max 100
python utils/pubMedIngest.py --query "maternal health Africa" --max 100
```

### 4. Run locally

```bash
python app.py
```

### 5. Expose to Twilio (for voice/SMS testing)

```bash
brew install ngrok/ngrok/ngrok
ngrok http 8000
```

Set the ngrok URL as your Twilio webhook:
- Voice: `https://your-ngrok-url/voice`
- SMS: `https://your-ngrok-url/sms`

---

## API

### `POST /api/ask`
```json
{ "question": "What are the symptoms of malaria?" }
```

### `POST /api/ingest`
```json
{ "texts": [{ "text": "...", "source": "PubMed:12345" }] }
```

### `GET /test`
Returns server health status.

---

## Known Limitations

- Bantu language disambiguation is imperfect — the classifier occasionally confuses closely related languages (e.g. Ekegusii and Kinyarwanda). Improving this requires either a dedicated African language classifier or user-initiated language preference setting.
- Voice latency: RAG retrieval + hallucination scoring adds ~2–4 seconds. Twilio's response timeout is 15 seconds — sufficient for most queries but worth monitoring.
- The FAISS index is in-memory and file-backed. For production, swap for a hosted vector DB (Pinecone, Weaviate).

---

## Roadmap

- [ ] BiomedBERT embeddings for better clinical retrieval
- [ ] Dedicated African language classifier (Afro-XLMR)
- [ ] Conversation history / multi-turn voice sessions
- [ ] WhatsApp Business API integration
- [ ] Hosted vector DB for scalable ingestion
- [ ] Eval on MedQA-USMLE with hallucination precision/recall metrics