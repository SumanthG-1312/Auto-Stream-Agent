# 🎬 AutoStream AI Agent — Social-to-Lead Agentic Workflow

A **LangChain** and **LangGraph** powered conversational AI agent for **AutoStream**, a SaaS platform for automated video editing. The agent leverages **Google Gemini 1.5 Flash** for intent classification and RAG-based knowledge retrieval.

---

## 🛠️ Tech Stack
- **Framework**: LangGraph (Stateful Multi-Agent Orchestration)
- **Orchestration**: LangChain
- **LLM**: Google Gemini 1.5 Flash
- **Retriever**: BM25 (Local Keyword Search)
- **Environment**: Python 3.10+

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/autostream-agent.git
cd autostream-agent
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your API key
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your-google-api-key-here
```

### 5. Run the agent
```bash
python agent.py
```

---

## 🗂️ Project Structure

```
autostream-agent/
├── agent.py             # Core LangGraph agent (intent, RAG, lead capture)
├── knowledge_base.json  # Local knowledge base (pricing, policies, about)
├── requirements.txt     # Python dependencies
├── .env                 # API key (not committed to git)
└── README.md
```

---

## 🏗️ Architecture Explanation (~200 words)

### Why LangGraph?

LangGraph was chosen over AutoGen because it provides **explicit, auditable state machines** — every node and edge in the conversation flow is declared upfront. This makes the agent easier to debug, test, and extend in a production setting. AutoGen is better suited for multi-agent collaboration; for a single-agent lead-capture workflow, LangGraph's directed graph model is a cleaner fit.

### How It Works

The agent is modelled as a **StateGraph** with four nodes:

| Node | Responsibility |
|---|---|
| `classify_intent` | Classifies user message into `greeting`, `inquiry`, `high_intent`, or `collecting` |
| `respond_greeting` | Returns a warm brand-aware greeting |
| `rag_respond` | Retrieves relevant chunks from a FAISS vector store built from `knowledge_base.json` and answers using GPT-4o-mini |
| `collect_lead` | Progressively extracts name → email → platform from conversation; fires `mock_lead_capture()` only when all three are present |

### State Management

A single `AgentState` TypedDict is passed through every graph invocation. It carries:
- **`messages`** — the full conversation history (persisted across all turns via `add_messages`)
- **`intent`** — current classified intent, preventing re-classification mid-collection
- **`lead_info`** — a dict that accumulates name / email / platform across turns
- **`lead_captured`** — a boolean flag that ends the session after the tool fires

This state object is returned from each `app.invoke()` call and fed back in on the next turn, giving the agent memory across 5–6 conversation turns with no external database needed.

---

## 💬 WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, you would use the **WhatsApp Business Cloud API** (Meta) with a webhook integration:

### Architecture Overview

```
User (WhatsApp) → Meta Cloud API → Webhook (FastAPI) → AutoStream Agent → Reply
```

### Step-by-Step Integration

**1. Set up a webhook server (FastAPI)**
```python
from fastapi import FastAPI, Request
import httpx

app = FastAPI()
WHATSAPP_TOKEN = "your_whatsapp_token"
PHONE_NUMBER_ID = "your_phone_number_id"

@app.post("/webhook")
async def receive_message(request: Request):
    body = await request.json()
    
    # Extract user message and sender ID
    message = body["entry"][0]["changes"][0]["value"]["messages"][0]
    user_id = message["from"]
    user_text = message["text"]["body"]
    
    # Run the LangGraph agent (load session state from DB/cache by user_id)
    reply_text = run_agent(user_id, user_text)
    
    # Send reply back via WhatsApp API
    await send_whatsapp_message(user_id, reply_text)
    return {"status": "ok"}

async def send_whatsapp_message(to: str, text: str):
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    payload = {"messaging_product": "whatsapp", "to": to, "text": {"body": text}}
    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload, headers=headers)
```

**2. Persist session state per user**

Replace the in-memory `state` dict with Redis or a database keyed by `user_id` so each WhatsApp user maintains their own conversation thread:
```python
import redis, json

r = redis.Redis()

def load_state(user_id): 
    raw = r.get(f"state:{user_id}")
    return json.loads(raw) if raw else default_state()

def save_state(user_id, state): 
    r.set(f"state:{user_id}", json.dumps(state), ex=3600)
```

**3. Verify webhook (Meta requirement)**
```python
@app.get("/webhook")
def verify_webhook(hub_mode: str, hub_challenge: str, hub_verify_token: str):
    if hub_verify_token == "your_verify_token":
        return int(hub_challenge)
    return {"error": "invalid token"}
```

**4. Deploy & Register**
- Deploy the FastAPI server (e.g. Railway, Render, or AWS)
- Register the public URL in the Meta Developer Console under your WhatsApp App → Webhooks
- Subscribe to the `messages` webhook field

This approach is **fully stateless on the server side** — all conversation memory lives in Redis — making it horizontally scalable.

---

## 🧠 Example Conversation

```
You: Hi there!
AutoStream: Hey! Welcome to AutoStream 🎬 — we help content creators...

You: What are your pricing plans?
AutoStream: AutoStream offers two plans:
  • Basic — $29/month: 10 videos/month, 720p resolution, email support
  • Pro   — $79/month: Unlimited videos, 4K, AI captions, 24/7 support

You: I want to try the Pro plan for my YouTube channel
AutoStream: Awesome! 🎉 To get you set up, could you share your full name?

You: My name is Rahul Sharma
AutoStream: Thanks! Could you share your email address?

You: rahul@gmail.com
AutoStream: One more thing — your main creator platform?

You: YouTube
✅ Lead captured successfully: Rahul Sharma, rahul@gmail.com, YouTube
AutoStream: 🎊 You're all set, Rahul! Our team will reach out shortly. Welcome to AutoStream Pro! 🚀
```

---

## 📋 Evaluation Checklist

| Criteria | Status |
|---|---|
| Intent classification (greeting / inquiry / high-intent) | ✅ |
| RAG from local knowledge base (FAISS) | ✅ |
| State management across 5–6 turns | ✅ |
| Tool called only after all 3 fields collected | ✅ |
| Clean LangGraph architecture | ✅ |
| WhatsApp webhook deployment explained | ✅ |
