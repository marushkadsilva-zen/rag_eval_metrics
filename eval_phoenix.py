import os
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-not-used")
from dotenv import load_dotenv
load_dotenv()

# LiteLLM reads GROQ_API_KEY from environment automatically
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

import phoenix as px
from phoenix.evals import (
    HallucinationEvaluator,
    QAEvaluator,
    RelevanceEvaluator,
    run_evals,
)
from phoenix.evals.models import LiteLLMModel
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register
import pandas as pd
from rag_pipeline import build_rag_pipeline, run_rag

TEST_DATA = [
    {"question": "When was Albert Einstein born?",
     "ground_truth": "Albert Einstein was born on March 14, 1879."},
    {"question": "What is Einstein's famous equation?",
     "ground_truth": "Einstein's famous equation is E = mc2."},
    {"question": "What prize did Einstein win and for what?",
     "ground_truth": "Einstein received the Nobel Prize in Physics in 1921 for the photoelectric effect."},
    {"question": "Where did Einstein work in the United States?",
     "ground_truth": "Einstein worked at the Institute for Advanced Study in Princeton, New Jersey."},
    {"question": "When did Einstein die?",
     "ground_truth": "Einstein died on April 18, 1955, in Princeton, New Jersey."},
]

# ── Step 1: Launch Phoenix dashboard ─────────────────────────────
print("Launching Arize Phoenix...")
session = px.launch_app()
print(f"Phoenix dashboard: {session.url}")

# ── Step 2: Set up OpenTelemetry tracing ─────────────────────────
tracer_provider = register(
    project_name="rag-eval-einstein",
    endpoint="http://localhost:6006/v1/traces",
)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# ── Step 3: Build RAG pipeline ────────────────────────────────────
print("\nBuilding RAG pipeline (all calls will be traced)...")
llm, retriever = build_rag_pipeline("documents/sample.txt")

# ── Step 4: Run RAG — Phoenix auto-traces every call ─────────────
print("\nRunning RAG (Phoenix is tracing everything)...\n")
questions, answers, contexts_list, ground_truths = [], [], [], []

for item in TEST_DATA:
    q  = item["question"]
    gt = item["ground_truth"]
    answer, contexts = run_rag(llm, retriever, q)
    questions.append(q)
    answers.append(answer)
    contexts_list.append(contexts)
    ground_truths.append(gt)
    print(f"  Q: {q}")
    print(f"  A: {answer}\n")

# ── Step 5: Build eval dataframe ─────────────────────────────────
eval_df = pd.DataFrame({
    "input":     questions,
    "output":    answers,
    "reference": ground_truths,
    "context":   ["\n\n".join(c) for c in contexts_list],
})

# ── Step 6: LiteLLM model — reads GROQ_API_KEY from env ──────────
print("Setting up LiteLLM model with Groq...")
model = LiteLLMModel(
    model="groq/llama-3.3-70b-versatile",
)

hallucination_eval = HallucinationEvaluator(model)
qa_eval            = QAEvaluator(model)
relevance_eval     = RelevanceEvaluator(model)

# ── Step 7: Run evaluations ───────────────────────────────────────
print("\nRunning Phoenix evaluators...\n")
results = run_evals(
    dataframe=eval_df,
    evaluators=[hallucination_eval, qa_eval, relevance_eval],
    provide_explanation=True,
)

# ── Step 8: Print results ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  ARIZE PHOENIX RESULTS")
print("=" * 60)

evaluator_names = ["HallucinationEvaluator", "QAEvaluator", "RelevanceEvaluator"]

for result_df, name in zip(results, evaluator_names):
    print(f"\n  {name}:")
    print("  " + "-" * 50)

    if "label" in result_df.columns:
        label_counts = result_df["label"].value_counts().to_dict()
        total = len(result_df)
        for label, count in label_counts.items():
            pct = count / total * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"    {label:<20}: {count}/{total}  ({pct:.0f}%)  [{bar}]")

    if "score" in result_df.columns:
        avg = result_df["score"].dropna().mean()
        if avg is not None and str(avg) != "nan":
            bar    = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
            status = "✅" if avg >= 0.7 else "⚠️ " if avg >= 0.5 else "❌"
            print(f"    {status} Avg score : {avg:.3f}  [{bar}]")

    if "explanation" in result_df.columns:
        print(f"\n    Per-question explanations:")
        for i, (q, exp) in enumerate(zip(questions, result_df["explanation"]), 1):
            if exp and str(exp) != "nan":
                print(f"    Q{i}: {q[:50]}")
                print(f"        {str(exp)[:100]}")

print("\n" + "=" * 60)
print(f"  Full traces at: {session.url}")
print("  → Click 'Traces' to see every RAG call traced")
print("  → Click 'Evals'  to see scores per span")
print("=" * 60)
print("\nPhoenix evaluation complete!")
print("Dashboard is live — open your browser to:", session.url)
input("\nPress Enter to exit and close dashboard...")