# Dental Clinic AI Chatbot

A RAG-powered customer support chatbot for a dental clinic, built with Flask, LangChain, FAISS, and the Groq LLM API. It answers patient questions about services, pricing, staff, and clinic information using a local knowledge base.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT (Browser)                        │
│                          index.html                             │
│          [Chat UI · Quick Replies · Typing Indicator]           │
└────────────────────────┬────────────────────────────────────────┘
                         │  HTTP  GET /
                         │  HTTP  POST /chat  { "message": "..." }
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FLASK SERVER  (app.py)                      │
│                       localhost:5000                            │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                   RAG PIPELINE                          │  │
│   │                                                         │  │
│   │  1. DOCUMENT LOADING                                    │  │
│   │     dental_knowledge.txt  ──►  TextLoader               │  │
│   │                                                         │  │
│   │  2. CHUNKING                                            │  │
│   │     RecursiveCharacterTextSplitter                      │  │
│   │     (chunk_size=500, overlap=50)                        │  │
│   │                                                         │  │
│   │  3. EMBEDDINGS                                          │  │
│   │     HuggingFace all-MiniLM-L6-v2  ──►  384-dim vectors │  │
│   │                                                         │  │
│   │  4. VECTOR STORE                                        │  │
│   │     FAISS (in-memory index, rebuilt on startup)         │  │
│   │                                                         │  │
│   │  5. RETRIEVAL                                           │  │
│   │     User Query  ──►  Top-4 similar chunks               │  │
│   │                                                         │  │
│   │  6. GENERATION                                          │  │
│   │     Context + Question  ──►  Groq API                   │  │
│   │                               (Llama 3.1 8B Instant)    │  │
│   │                           ──►  Response string          │  │
│   └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼  REST API
┌─────────────────────────────────────────────────────────────────┐
│                      EXTERNAL SERVICES                          │
│                                                                 │
│   ┌──────────────────────┐    ┌──────────────────────────────┐ │
│   │    Groq Cloud API    │    │  HuggingFace Model Hub       │ │
│   │  Llama 3.1 8B Instant│    │  all-MiniLM-L6-v2            │ │
│   │  (LLM inference)     │    │  (downloaded on first run)   │ │
│   └──────────────────────┘    └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Request Flow

```
User types question
       │
       ▼
POST /chat  { "message": "What is the price of a root canal?" }
       │
       ▼
FAISS vector search  ──►  retrieves top 4 relevant text chunks
       │
       ▼
LangChain prompt template  ──►  [system prompt] + [context] + [question]
       │
       ▼
Groq API  (Llama 3.1 8B, temperature=0.2)
       │
       ▼
JSON response  { "response": "Root canal treatment costs NRS 3,500–8,000..." }
       │
       ▼
Chat UI renders answer
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Flask |
| LLM provider | Groq API — Llama 3.1 8B Instant |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector store | FAISS (in-memory, CPU) |
| Orchestration | LangChain |
| Frontend | HTML5 + Vanilla JavaScript |
| Config | python-dotenv |

---

## Project Structure

```
dental-chatbot/
├── app.py                 # Flask app, RAG pipeline, API routes
├── index.html             # Chat UI (served by Flask)
├── dental_knowledge.txt   # Clinic knowledge base (services, pricing, FAQs)
├── requirements.txt       # Python dependencies
├── .env                   # API keys (not committed)
├── .env.example           # Environment variable template
└── .gitignore
```

---

## Prerequisites

- Python 3.8+
- A [Groq API key](https://console.groq.com) (free tier available)
- Internet access on first run (to download the HuggingFace embedding model)

---

## Setup & Running the Server

### 1. Clone / navigate to the project

```bash
cd dental-chatbot
```

### 2. Create and activate a virtual environment

```bash
# Create venv
python -m venv venv

# Activate — Linux / macOS
source venv/bin/activate

# Activate — Windows (PowerShell)
venv\Scripts\Activate.ps1

# Activate — Windows (CMD / Git Bash)
source venv/Scripts/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> The first install will download PyTorch and the sentence-transformers model (~90 MB).

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and add your key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Start the server

```bash
python app.py
```

Expected output:

```
* Running on http://127.0.0.1:5000
* Debug mode: on
```

### 6. Open the chat

Navigate to [http://localhost:5000](http://localhost:5000) in your browser.

---

## API Reference

### `GET /`

Serves the chat UI (`index.html`).

---

### `POST /chat`

Send a user message and receive an AI-generated response.

**Request body:**

```json
{
  "message": "What services do you offer?"
}
```

**Response:**

```json
{
  "response": "We offer scaling & polishing, fillings, root canal treatment, crowns, extractions, orthodontics, dental implants, pediatric dentistry, and cosmetic dentistry..."
}
```

**Error response (empty message):**

```json
{
  "error": "No message provided"
}
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | API key from [console.groq.com](https://console.groq.com) |

---

## Knowledge Base

The chatbot's knowledge is sourced entirely from `dental_knowledge.txt`. It covers:

- Clinic name, location, phone, and email
- Medical staff and their specialisations
- Services and pricing (in NRS)
- Appointment booking process
- FAQs and patient success stories

To update clinic information, edit `dental_knowledge.txt` and restart the server (the FAISS index is rebuilt on startup).

---

## Clinic Information

| Detail | Info |
|---|---|
| Name | Neupane Dental Clinic |
| Location | Tilottama-03, Divertole, Rupandehi, Nepal |
| Phone | +977 9866578969 / +977 9857064310 |
| Email | neupanedental@gmail.com |

---

## Notes

- The FAISS vector index is **in-memory only** — it is rebuilt every time the server starts.
- There is **no conversation history** stored; each request is independent.
- The LLM temperature is set to **0.2** for consistent, factual answers.
- If the knowledge base does not contain an answer, the bot directs the user to contact the clinic directly.
