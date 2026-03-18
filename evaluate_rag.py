# import os
# from dotenv import load_dotenv
# from datasets import Dataset
# from ragas import evaluate
# from ragas.metrics import (
#     faithfulness,
#     answer_relevancy,
#     context_precision,
#     context_recall,
#     answer_correctness,
# )
# from ragas.llms import LangchainLLMWrapper
# from ragas.embeddings import LangchainEmbeddingsWrapper
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
# from rag_pipeline import build_rag_pipeline, run_rag

# load_dotenv()

# # ── Test data ─────────────────────────────────────────────────────────────────
# TEST_DATA = [
#     {
#         "question": "When was Albert Einstein born?",
#         "ground_truth": "Albert Einstein was born on March 14, 1879."
#     },
#     {
#         "question": "What prize did Einstein win and for what?",
#         "ground_truth": "Einstein received the Nobel Prize in Physics in 1921 for the photoelectric effect."
#     },
#     {
#         "question": "Where did Einstein work after moving to the United States?",
#         "ground_truth": "Einstein worked at the Institute for Advanced Study in Princeton, New Jersey."
#     },
#     {
#         "question": "What is Einstein's famous equation?",
#         "ground_truth": "Einstein's famous equation is E = mc², the mass–energy equivalence formula."
#     },
# ]

# # ── Build RAG pipeline ────────────────────────────────────────────────────────
# print("🔧 Building RAG pipeline...")
# llm, retriever = build_rag_pipeline("documents/sample.txt")

# # ── Run RAG ───────────────────────────────────────────────────────────────────
# print("🤖 Running RAG on test questions...\n")
# questions, answers, contexts_list, ground_truths = [], [], [], []

# for item in TEST_DATA:
#     q, gt = item["question"], item["ground_truth"]
#     answer, contexts = run_rag(llm, retriever, q)
#     questions.append(q)
#     answers.append(answer)
#     contexts_list.append(contexts)
#     ground_truths.append(gt)
#     print(f"Q: {q}")
#     print(f"A: {answer}")
#     print(f"Chunks retrieved: {len(contexts)}")
#     print("-" * 60)

# # ── Build dataset ─────────────────────────────────────────────────────────────
# eval_dataset = Dataset.from_dict({
#     "question":     questions,
#     "answer":       answers,
#     "contexts":     contexts_list,
#     "ground_truth": ground_truths,
# })

# # ── Configure Groq + HuggingFace for RAGAs ────────────────────────────────────
# print("\n⚙️  Configuring RAGAs...")

# ragas_llm = LangchainLLMWrapper(ChatGroq(
#     model="llama-3.3-70b-versatile",
#     temperature=0,
#     api_key=os.getenv("GROQ_API_KEY")
# ))

# ragas_embeddings = LangchainEmbeddingsWrapper(
#     HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# )

# # Assign LLM and embeddings to each metric
# faithfulness.llm           = ragas_llm
# answer_relevancy.llm       = ragas_llm
# answer_relevancy.embeddings = ragas_embeddings
# context_precision.llm      = ragas_llm
# context_recall.llm         = ragas_llm
# answer_correctness.llm     = ragas_llm

# # ── Evaluate ──────────────────────────────────────────────────────────────────
# print("\n📊 Running RAGAs evaluation...\n")

# results = evaluate(
#     dataset=eval_dataset,
#     metrics=[
#         faithfulness,
#         answer_relevancy,
#         context_precision,
#         context_recall,
#         answer_correctness,
#     ]
# )

# # ── Print results ─────────────────────────────────────────────────────────────
# print("\n" + "=" * 65)
# print("       RAG EVALUATION RESULTS")
# print("=" * 65)

# df = results.to_pandas()

# metrics_info = {
#     "faithfulness":       ("Faithfulness",       "No hallucination?        "),
#     "answer_relevancy":   ("Answer Relevancy",   "On-topic answer?         "),
#     "context_precision":  ("Context Precision",  "Chunks relevant?         "),
#     "context_recall":     ("Context Recall",     "Context covers truth?    "),
#     "answer_correctness": ("Answer Correctness", "Factually correct?       "),
# }

# for col, (name, desc) in metrics_info.items():
#     if col in df.columns:
#         score  = df[col].mean()
#         bar    = "█" * int(score * 20) + "░" * (20 - int(score * 20))
#         status = "✅" if score >= 0.7 else "⚠️ " if score >= 0.5 else "❌"
#         print(f"{status} {name:<22} {desc} Score: {score:.3f}  [{bar}]")

# print("=" * 65)
# print("\n📁 Per-question breakdown:")
# cols = ["question", "faithfulness", "answer_relevancy",
#         "context_precision", "context_recall", "answer_correctness"]
# available = [c for c in cols if c in df.columns]
# print(df[available].to_string())
# print("\n✅ Evaluation complete!")


import os
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-not-used")  # stops ragas internal fallback

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from rag_pipeline import build_rag_pipeline, run_rag

# ── Test data ─────────────────────────────────────────────────────────────────
TEST_DATA = [
    {
        "question": "When was Albert Einstein born?",
        "ground_truth": "Albert Einstein was born on March 14, 1879."
    },
    {
        "question": "What prize did Einstein win and for what?",
        "ground_truth": "Einstein received the Nobel Prize in Physics in 1921 for the photoelectric effect."
    },
    {
        "question": "Where did Einstein work after moving to the United States?",
        "ground_truth": "Einstein worked at the Institute for Advanced Study in Princeton, New Jersey."
    },
    {
        "question": "What is Einstein's famous equation?",
        "ground_truth": "Einstein's famous equation is E = mc², the mass–energy equivalence formula."
    },
]

# ── Build RAG pipeline ────────────────────────────────────────────────────────
print("🔧 Building RAG pipeline...")
llm, retriever = build_rag_pipeline("documents/sample.txt")

# ── Run RAG ───────────────────────────────────────────────────────────────────
print("🤖 Running RAG on test questions...\n")
questions, answers, contexts_list, ground_truths = [], [], [], []

for item in TEST_DATA:
    q, gt = item["question"], item["ground_truth"]
    answer, contexts = run_rag(llm, retriever, q)
    questions.append(q)
    answers.append(answer)
    contexts_list.append(contexts)
    ground_truths.append(gt)
    print(f"Q: {q}")
    print(f"A: {answer}")
    print(f"Chunks retrieved: {len(contexts)}")
    print("-" * 60)

# ── Build dataset ─────────────────────────────────────────────────────────────
eval_dataset = Dataset.from_dict({
    "question":     questions,
    "answer":       answers,
    "contexts":     contexts_list,
    "ground_truth": ground_truths,
})

# ── Configure Groq + HuggingFace for RAGAs ───────────────────────────────────
print("\n⚙️  Configuring RAGAs...")

ragas_llm = LangchainLLMWrapper(ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
))

hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

# ── Assign to every metric explicitly ────────────────────────────────────────
faithfulness.llm            = ragas_llm
answer_relevancy.llm        = ragas_llm
answer_relevancy.embeddings = ragas_embeddings
context_precision.llm       = ragas_llm
context_recall.llm          = ragas_llm
answer_correctness.llm      = ragas_llm

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n📊 Running RAGAs evaluation...\n")

results = evaluate(
    dataset=eval_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ],
    llm=ragas_llm,                # ← pass globally too
    embeddings=ragas_embeddings,  # ← this stops the OpenAI fallback
)

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("       RAG EVALUATION RESULTS")
print("=" * 65)

df = results.to_pandas()

metrics_info = {
    "faithfulness":       ("Faithfulness",       "No hallucination?        "),
    "answer_relevancy":   ("Answer Relevancy",   "On-topic answer?         "),
    "context_precision":  ("Context Precision",  "Chunks relevant?         "),
    "context_recall":     ("Context Recall",     "Context covers truth?    "),
    "answer_correctness": ("Answer Correctness", "Factually correct?       "),
}

for col, (name, desc) in metrics_info.items():
    if col in df.columns:
        score  = df[col].mean()
        bar    = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        status = "✅" if score >= 0.7 else "⚠️ " if score >= 0.5 else "❌"
        print(f"{status} {name:<22} {desc} Score: {score:.3f}  [{bar}]")

print("=" * 65)
print("\n📁 Per-question breakdown:")
cols = ["question", "faithfulness", "answer_relevancy",
        "context_precision", "context_recall", "answer_correctness"]
available = [c for c in cols if c in df.columns]
print(df[available].to_string())
print("\n✅ Evaluation complete!")