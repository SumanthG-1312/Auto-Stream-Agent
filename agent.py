"""
AutoStream Conversational AI Agent
===================================
A LangGraph-powered agent for AutoStream SaaS that handles:
  - Intent classification (greeting / inquiry / high-intent)
  - RAG-powered product Q&A from a local knowledge base
  - Progressive lead collection & mock lead capture tool
"""

import json
import os
import re
import sys
from typing import Annotated, TypedDict

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Using gemini-1.5-flash which typically has higher free-tier limits than experimental versions
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.1,  # Lower temperature for more consistent "fine-tuned" like behavior
)

def llm_invoke_with_retry(messages, max_retries=3):
    """Wrapper to handle 429 errors with simple backoff."""
    import time
    from google.api_core import exceptions
    
    for i in range(max_retries):
        try:
            return llm.invoke(messages)
        except exceptions.ResourceExhausted as e:
            if i == max_retries - 1:
                raise e
            wait_time = (i + 1) * 2
            print(f"⚠️ Quota hit. Retrying in {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            raise e


# ── Knowledge Base & RAG Setup ────────────────────────────────────────────────

def load_knowledge_base() -> dict:
    """Load the local JSON knowledge base."""
    kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
    with open(kb_path, "r") as f:
        return json.load(f)


def build_retriever():
    """
    Build a BM25 retriever from the knowledge base (fully local, no embeddings needed).
    BM25 is a proven keyword-based ranking algorithm — great for small, structured KBs.
    Each top-level section of the JSON is stored as a separate document.
    """
    kb = load_knowledge_base()
    docs = []
    for section, content in kb.items():
        text = f"[Section: {section}]\n{json.dumps(content, indent=2)}"
        docs.append(Document(page_content=text, metadata={"section": section}))
    return BM25Retriever.from_documents(docs, k=3)


# Build retriever once at startup
retriever = build_retriever()


# ── Mock Lead Capture Tool ─────────────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Simulates a CRM / backend lead capture API call.
    In production this would POST to a real endpoint.
    """
    print("\n" + "=" * 55)
    print(f"  ✅  Lead captured successfully!")
    print(f"      Name     : {name}")
    print(f"      Email    : {email}")
    print(f"      Platform : {platform}")
    print("=" * 55 + "\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"


# ── Agent State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # full conversation history
    intent: str                                # greeting | inquiry | high_intent | collecting | done
    lead_info: dict                            # progressively collected: name, email, platform
    lead_captured: bool                        # True once mock_lead_capture is called


# ── Node: Classify Intent ─────────────────────────────────────────────────────

def classify_intent(state: AgentState) -> AgentState:
    """
    Classify the user's latest message.
    Skips LLM classification if we're already mid lead-collection.
    """
    current_intent = state.get("intent", "")
    already_collecting = current_intent in ("high_intent", "collecting")

    if already_collecting and not state.get("lead_captured", False):
        # Stay in collection flow; don't re-classify
        return {**state, "intent": "collecting"}

    last_msg = state["messages"][-1].content.lower()

    # --- Step 1: Enhanced Regex Classification (Saves Quota) ---
    # Greeting detection
    if re.search(r"\b(hi|hello|hey|greetings|morning|evening|howdy|yo|sup)\b", last_msg):
        return {**state, "intent": "greeting"}
    
    # Exit/Polite detection
    if re.search(r"\b(bye|goodbye|thanks|thank you|thx|awesome|great|cool)\b", last_msg):
        return {**state, "intent": "greeting"}

    # High Intent / Signup detection
    high_intent_keywords = [
        "buy", "sign up", "purchase", "subscribe", "get started", 
        "start today", "join", "upgrade", "pro plan", "want pro",
        "demo", "try", "trial", "account"
    ]
    if any(kw in last_msg for kw in high_intent_keywords):
        # But if they ask "how much" or "what is", it might be an inquiry
        if not any(q in last_msg for q in ["how much", "what is", "pricing", "cost", "tell me about"]):
            return {**state, "intent": "high_intent"}

    # Inquiry detection
    inquiry_keywords = ["price", "cost", "how much", "features", "what does", "refund", "support", "help", "work"]
    if any(kw in last_msg for kw in inquiry_keywords):
        return {**state, "intent": "inquiry"}

    # --- Step 2: Optimized LLM Classification (Few-Shot for "Fine-Tuned" Quality) ---
    classify_prompt = (
        "You are an expert intent classifier for AutoStream. "
        "Classify the user's message into one of these labels:\n"
        "- greeting: Casual hello, hi, or small talk.\n"
        "- inquiry: Questions about product, pricing, features, or policies.\n"
        "- high_intent: Clear signal they want to buy, sign up, or start now.\n\n"
        "Examples:\n"
        "User: 'Tell me about the pro plan' -> inquiry\n"
        "User: 'I want to sign up now' -> high_intent\n"
        "User: 'Hey there' -> greeting\n"
        "User: 'How much does it cost?' -> inquiry\n\n"
        f"User Message: '{last_msg}'\n"
        "Label:"
    )

    response = llm_invoke_with_retry([HumanMessage(content=classify_prompt)])
    raw = response.content.strip().lower()

    if "high" in raw:
        intent = "high_intent"
    elif "greeting" in raw:
        intent = "greeting"
    else:
        intent = "inquiry"

    return {**state, "intent": intent}


# ── Node: Respond to Greeting ─────────────────────────────────────────────────

def respond_greeting(state: AgentState) -> AgentState:
    system = (
        "You are an enthusiastic and helpful AI assistant for AutoStream — "
        "an AI-powered SaaS platform for automated video editing. "
        "Greet the user warmly, briefly mention what AutoStream does, "
        "and invite them to ask about plans or features. Keep it short and punchy."
    )
    response = llm_invoke_with_retry([SystemMessage(content=system), *state["messages"]])
    return {**state, "messages": [AIMessage(content=response.content)]}


# ── Node: RAG-Powered Product Response ────────────────────────────────────────

def rag_respond(state: AgentState) -> AgentState:
    """Retrieve relevant KB chunks and answer the user's question."""
    last_msg = state["messages"][-1].content
    docs = retriever.invoke(last_msg)
    context = "\n\n".join(d.page_content for d in docs)

    system = (
        "You are a knowledgeable and friendly AI assistant for AutoStream, "
        "an AI-powered video editing SaaS for content creators.\n\n"
        "Answer the user's question using ONLY the context provided below. "
        "Be clear, helpful, and concise. Use bullet points for features/pricing if possible. "
        "If the context doesn't cover the question, say so honestly and offer to connect them with support.\n\n"
        f"--- Knowledge Base Context ---\n{context}\n-----------------------------"
    )
    response = llm_invoke_with_retry([SystemMessage(content=system), *state["messages"]])
    return {**state, "messages": [AIMessage(content=response.content)]}


# ── Node: Collect Lead Info ────────────────────────────────────────────────────

def collect_lead(state: AgentState) -> AgentState:
    """
    Progressively extract lead details (name, email, platform) from
    the conversation. Calls mock_lead_capture only when all three are present.
    """
    lead_info = dict(state.get("lead_info") or {})
    last_msg = state["messages"][-1].content

    # --- Step 1: Extract any newly provided info from the latest message ---
    # --- Step 1: Optimized Extraction (Few-Shot for accuracy) ---
    extract_prompt = (
        "Extract lead information from the user message. "
        "Return a JSON object with 'name', 'email', and 'platform'. "
        "Only include fields found in the message. If none, return {}.\n\n"
        "Examples:\n"
        "Message: 'I am John Doe, email me at john@example.com'\n"
        "Output: {\"name\": \"John Doe\", \"email\": \"john@example.com\"}\n\n"
        "Message: 'I use YouTube mainly'\n"
        "Output: {\"platform\": \"YouTube\"}\n\n"
        f"User Message: \"{last_msg}\"\n"
        "Output (JSON only):"
    )

    extract_response = llm_invoke_with_retry([HumanMessage(content=extract_prompt)])
    raw = extract_response.content.strip()

    # Strip accidental markdown code fences
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        extracted = json.loads(raw)
        for key in ("name", "email", "platform"):
            if key in extracted and extracted[key]:
                lead_info[key] = extracted[key]
    except json.JSONDecodeError:
        pass  # If extraction fails, just continue with what we have

    # --- Step 2: Identify missing fields ---
    missing_labels = {
        "name": "your full name",
        "email": "your email address",
        "platform": "your main creator platform (e.g. YouTube, Instagram, TikTok)",
    }
    missing = [label for key, label in missing_labels.items() if key not in lead_info]

    # --- Step 3: Either ask for next missing field OR fire the tool ---
    if missing:
        if not lead_info:
            # First ask — enthusiastic opener
            reply = (
                "Awesome! 🎉 I'd love to get you set up on the Pro plan. "
                f"To get started, could you please share {missing[0]}?"
            )
        else:
            reply = f"Thanks! One more thing — could you share {missing[0]}?"

        return {
            **state,
            "lead_info": lead_info,
            "messages": [AIMessage(content=reply)],
            "intent": "collecting",
        }

    else:
        # All three collected — fire the lead capture tool
        mock_lead_capture(
            name=lead_info["name"],
            email=lead_info["email"],
            platform=lead_info["platform"],
        )
        reply = (
            f"🎊 You're all set, {lead_info['name']}! "
            f"Our team will reach out to you at {lead_info['email']} shortly. "
            f"Welcome to AutoStream Pro — can't wait to see your {lead_info['platform']} content soar! 🚀"
        )
        return {
            **state,
            "lead_info": lead_info,
            "messages": [AIMessage(content=reply)],
            "lead_captured": True,
            "intent": "done",
        }


# ── Routing Logic ─────────────────────────────────────────────────────────────

def route_after_classify(state: AgentState) -> str:
    """Decide which node to call after intent classification."""
    intent = state.get("intent", "inquiry")
    if intent == "greeting":
        return "respond_greeting"
    elif intent in ("high_intent", "collecting"):
        return "collect_lead"
    elif intent == "done":
        return END
    else:
        return "rag_respond"


# ── Build the LangGraph ───────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("respond_greeting", respond_greeting)
    graph.add_node("rag_respond", rag_respond)
    graph.add_node("collect_lead", collect_lead)

    # Entry point
    graph.set_entry_point("classify_intent")

    # Conditional routing after classification
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "respond_greeting": "respond_greeting",
            "rag_respond": "rag_respond",
            "collect_lead": "collect_lead",
            END: END,
        },
    )

    # All response nodes lead back to END (next turn starts fresh invoke)
    graph.add_edge("respond_greeting", END)
    graph.add_edge("rag_respond", END)
    graph.add_edge("collect_lead", END)

    return graph.compile()


# ── Main Chat Loop ────────────────────────────────────────────────────────────

def main():
    print("\n" + "🎬 " * 10)
    print("   Welcome to AutoStream AI Assistant")
    print("   Automated Video Editing for Content Creators")
    print("🎬 " * 10)
    print("\nAsk about our plans, features, or get started today!")
    print("Type 'quit' to exit.\n")

    app = build_graph()

    # Initial state — persisted across all turns in this session
    state: AgentState = {
        "messages": [],
        "intent": "",
        "lead_info": {},
        "lead_captured": False,
    }

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! 👋")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q", "bye"):
            print("\nAutoStream: Thanks for chatting! Visit autostream.io to learn more. 👋\n")
            break

        # Append user message and invoke the graph
        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]
        state = app.invoke(state)

        # Print the latest AI reply
        ai_msgs = [m for m in state["messages"] if isinstance(m, AIMessage)]
        if ai_msgs:
            print(f"\nAutoStream: {ai_msgs[-1].content}\n")

        # End session after successful lead capture
        if state.get("lead_captured"):
            print("─" * 55)
            print("✅  Lead successfully captured. Session complete!")
            print("─" * 55 + "\n")
            break


if __name__ == "__main__":
    main()
