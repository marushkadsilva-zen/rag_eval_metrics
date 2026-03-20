import os
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-not-used")
from dotenv import load_dotenv
load_dotenv()

from trulens.core import TruSession, Feedback
from trulens.apps.basic import TruBasicApp
from trulens.providers.huggingface import Huggingface
from rag_pipeline import build_rag_pipeline, run_rag
import numpy as np

TEST_DATA = [
    {"question": "When was Albert Einstein born?",
     "ground_truth": "Albert Einstein was born on March 14, 1879."},
    {"question": "What is Einstein's famous equation?",
     "ground_truth": "Einstein's famous equation is E = mc2."},
    {"question": "What prize did Einstein win?",
     "ground_truth": "Einstein received the Nobel Prize in Physics in 1921."},
    {"question": "Where did Einstein work in the US?",
     "ground_truth": "Einstein worked at the Institute for Advanced Study in Princeton."},
]

# ── Step 1: Build RAG pipeline ────────────────────────────────────
print("Building pipeline...")
llm, retriever = build_rag_pipeline("documents/sample.txt")

# ── Step 2: Initialize TruLens ────────────────────────────────────
print("Initializing TruLens...")
session = TruSession()
session.reset_database()

# ── Step 3: Use HuggingFace provider (correct methods) ───────────
print("Loading HuggingFace provider...")
provider = Huggingface()

# List available methods on provider
print("\nAvailable feedback methods:")
methods = [m for m in dir(provider) if not m.startswith("_")]
for m in methods:
    print(f"  {m}")

# ── Step 4: Use correct available method ──────────────────────────
# Use sentiment / language_match / positive_sentiment that HF supports
f_relevance = (
    Feedback(
        provider.language_match,
        name="Language Match"
    )
    .on_input_output()
)

# ── Step 5: Custom LLM-based feedback using Groq ─────────────────
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

def groq_answer_relevance(question: str, answer: str) -> float:
    """Custom feedback: is the answer relevant to the question?"""
    prompt = f"""Score how relevant this answer is to the question.
Return ONLY a decimal number between 0.0 and 1.0. Nothing else.

Question: {question}
Answer: {answer}

Score (0.0 to 1.0):"""
    try:
        response = groq_llm.invoke([HumanMessage(content=prompt)])
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except:
        return 0.5

def groq_faithfulness(question: str, answer: str) -> float:
    """Custom feedback: is the answer grounded in facts?"""
    prompt = f"""Score how faithful and factually accurate this answer is.
Return ONLY a decimal number between 0.0 and 1.0. Nothing else.

Question: {question}
Answer: {answer}

Score (0.0 to 1.0):"""
    try:
        response = groq_llm.invoke([HumanMessage(content=prompt)])
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except:
        return 0.5

def groq_coherence(question: str, answer: str) -> float:
    """Custom feedback: is the answer coherent and well-formed?"""
    prompt = f"""Score how coherent and well-structured this answer is.
Return ONLY a decimal number between 0.0 and 1.0. Nothing else.

Answer: {answer}

Score (0.0 to 1.0):"""
    try:
        response = groq_llm.invoke([HumanMessage(content=prompt)])
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except:
        return 0.5

# ── Step 6: Define Feedback objects with Groq functions ───────────
f_answer_relevance = (
    Feedback(groq_answer_relevance, name="Answer Relevance")
    .on_input_output()
)

f_faithfulness = (
    Feedback(groq_faithfulness, name="Faithfulness")
    .on_input_output()
)

f_coherence = (
    Feedback(groq_coherence, name="Coherence")
    .on_input_output()
)

# ── Step 7: Define RAG app ────────────────────────────────────────
def rag_app(question: str) -> str:
    answer, contexts = run_rag(llm, retriever, question)
    return answer

# ── Step 8: Wrap with TruLens ─────────────────────────────────────
tru_rag = TruBasicApp(
    rag_app,
    app_name="Einstein-RAG",
    app_version="v1",
    feedbacks=[f_answer_relevance, f_faithfulness, f_coherence],
)

# ── Step 9: Run all questions ─────────────────────────────────────
print("\nRunning TruLens evaluation...\n")
for item in TEST_DATA:
    q = item["question"]
    print(f"  Q: {q}")
    with tru_rag as recording:
        response = tru_rag.app(q)
    print(f"  A: {response}\n")

# ── Step 10: Print results ────────────────────────────────────────
print("\n" + "="*55)
print("  TRULENS RESULTS")
print("="*55)

try:
    records, feedback_cols = session.get_records_and_feedback(
        app_name="Einstein-RAG"
    )
    if records is not None and len(records) > 0:
        print(f"\nTotal records: {len(records)}")
        print("\nPer-question scores:")
        display_cols = ["input", "output"] + (feedback_cols if feedback_cols else [])
        available = [c for c in display_cols if c in records.columns]
        print(records[available].to_string())
        if feedback_cols:
            print("\nAverage scores:")
            for col in feedback_cols:
                avg = records[col].mean()
                bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
                status = "✅" if avg >= 0.7 else "⚠️ " if avg >= 0.5 else "❌"
                print(f"  {status} {col:<25}: {avg:.3f}  [{bar}]")
except Exception as e:
    print(f"Could not fetch records: {e}")

try:
    print("\nLeaderboard:")
    print(session.get_leaderboard())
except Exception as e:
    print(f"Leaderboard error: {e}")

print("\n" + "="*55)
print("Launching dashboard at http://localhost:8501")
print("Press Ctrl+C to stop")
print("="*55)
session.run_dashboard()