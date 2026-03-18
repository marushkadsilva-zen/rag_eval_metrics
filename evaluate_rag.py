import os
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-not-used")

from dotenv import load_dotenv
load_dotenv()

import re
import math
import nltk
import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
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

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Big test dataset (20 questions) ──────────────────────────────────────────
TEST_DATA = [
    {"question": "When was Albert Einstein born?",
     "ground_truth": "Albert Einstein was born on March 14, 1879."},
    {"question": "Where was Einstein born?",
     "ground_truth": "Einstein was born in Ulm, in the Kingdom of Württemberg in the German Empire."},
    {"question": "What is Einstein's famous equation?",
     "ground_truth": "Einstein's famous equation is E = mc², the mass–energy equivalence formula."},
    {"question": "What prize did Einstein win and for what?",
     "ground_truth": "Einstein received the Nobel Prize in Physics in 1921 for the photoelectric effect."},
    {"question": "When did Einstein move to the United States?",
     "ground_truth": "Einstein moved to the United States in 1933 when Hitler came to power."},
    {"question": "Where did Einstein work in the United States?",
     "ground_truth": "Einstein worked at the Institute for Advanced Study in Princeton, New Jersey."},
    {"question": "When did Einstein die?",
     "ground_truth": "Einstein died on April 18, 1955, in Princeton, New Jersey, at the age of 76."},
    {"question": "How many scientific papers did Einstein publish?",
     "ground_truth": "Einstein published more than 300 scientific papers."},
    {"question": "What is the photoelectric effect?",
     "ground_truth": "The photoelectric effect explains how light can eject electrons from a metal surface."},
    {"question": "What is the Annus Mirabilis?",
     "ground_truth": "Annus Mirabilis refers to 1905, when Einstein published four groundbreaking papers."},
    {"question": "Who was Einstein's first wife?",
     "ground_truth": "Einstein's first wife was Mileva Marić, whom he married in 1903."},
    {"question": "Was Einstein offered any political position?",
     "ground_truth": "Einstein was offered the presidency of Israel in 1952, which he declined."},
    {"question": "What did Einstein predict about light near massive objects?",
     "ground_truth": "Einstein's general theory of relativity predicted the bending of light around massive objects."},
    {"question": "What are gravitational waves?",
     "ground_truth": "Gravitational waves were predicted by Einstein's general relativity and first detected in 2015 by LIGO."},
    {"question": "What instrument did Einstein play?",
     "ground_truth": "Einstein played the violin and loved the works of Mozart and Bach."},
    {"question": "What is an Einstein ring?",
     "ground_truth": "An Einstein ring is an optical phenomenon caused by gravitational lensing of light from a distant source."},
    {"question": "What is spacetime?",
     "ground_truth": "Einstein's special theory of relativity introduced the concept that space and time are interwoven into spacetime."},
    {"question": "Who did Einstein collaborate with?",
     "ground_truth": "Einstein collaborated with physicists including Niels Bohr, Max Planck, and Werner Heisenberg."},
    {"question": "Why did Einstein sign a letter to President Roosevelt?",
     "ground_truth": "Einstein signed the letter urging nuclear research, fearing Nazi Germany would develop nuclear weapons first."},
    {"question": "What happened to Einstein's brain after death?",
     "ground_truth": "Einstein's brain was preserved after death and has been the subject of several studies."},
]

# ── Build RAG pipeline ────────────────────────────────────────────────────────
print("🔧 Building RAG pipeline...")
llm, retriever = build_rag_pipeline("documents/sample.txt")

# ── Run RAG on all questions ──────────────────────────────────────────────────
print("🤖 Running RAG on 20 questions...\n")
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
    print("-" * 60)

# ── Build RAGAs dataset ───────────────────────────────────────────────────────
eval_dataset = Dataset.from_dict({
    "question":     questions,
    "answer":       answers,
    "contexts":     contexts_list,
    "ground_truth": ground_truths,
})

# ── Configure Groq + HuggingFace ─────────────────────────────────────────────
print("\n⚙️  Configuring RAGAs with Groq + HuggingFace...")

ragas_llm = LangchainLLMWrapper(ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
))

hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

faithfulness.llm            = ragas_llm
answer_relevancy.llm        = ragas_llm
answer_relevancy.embeddings = ragas_embeddings
context_precision.llm       = ragas_llm
context_recall.llm          = ragas_llm
answer_correctness.llm      = ragas_llm

# ── RAGAs Evaluation ──────────────────────────────────────────────────────────
print("\n📊 Running RAGAs evaluation...\n")
results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision,
             context_recall, answer_correctness],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)
df = results.to_pandas()

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM METRICS
# ══════════════════════════════════════════════════════════════════════════════

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def tokenize(text: str) -> list:
    return normalize(text).split()

# 1. Exact Match ───────────────────────────────────────────────────────────────
def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize(prediction) == normalize(ground_truth))

# 2. F1 Score ──────────────────────────────────────────────────────────────────
def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = tokenize(prediction)
    gt_tokens   = tokenize(ground_truth)
    common      = Counter(pred_tokens) & Counter(gt_tokens)
    num_common  = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall    = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

# 3. BLEU Score ────────────────────────────────────────────────────────────────
def bleu_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = tokenize(prediction)
    gt_tokens   = tokenize(ground_truth)
    smoothie    = SmoothingFunction().method4
    return sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoothie)

# 4. ROUGE Score ───────────────────────────────────────────────────────────────
def rouge_scores(prediction: str, ground_truth: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }

# 5. Exact Match in Context (for Hit Rate & MRR) ───────────────────────────────
def answer_in_context(ground_truth: str, contexts: list) -> list:
    """Returns list of bools — True if ground truth keywords found in that chunk."""
    gt_tokens = set(tokenize(ground_truth))
    hits = []
    for ctx in contexts:
        ctx_tokens = set(tokenize(ctx))
        overlap = len(gt_tokens & ctx_tokens) / max(len(gt_tokens), 1)
        hits.append(overlap >= 0.3)   # 30% keyword overlap = hit
    return hits

# 6. Hit Rate ──────────────────────────────────────────────────────────────────
def hit_rate(ground_truth: str, contexts: list) -> float:
    hits = answer_in_context(ground_truth, contexts)
    return float(any(hits))

# 7. Mean Reciprocal Rank (MRR) ───────────────────────────────────────────────
def reciprocal_rank(ground_truth: str, contexts: list) -> float:
    hits = answer_in_context(ground_truth, contexts)
    for i, hit in enumerate(hits):
        if hit:
            return 1.0 / (i + 1)
    return 0.0

# 8. Groundedness ─────────────────────────────────────────────────────────────
def groundedness(answer: str, contexts: list) -> float:
    """What fraction of answer tokens appear in the retrieved context."""
    combined_ctx  = " ".join(contexts)
    answer_tokens = set(tokenize(answer))
    ctx_tokens    = set(tokenize(combined_ctx))
    if not answer_tokens:
        return 0.0
    overlap = len(answer_tokens & ctx_tokens)
    return overlap / len(answer_tokens)

# ── Compute all custom metrics ────────────────────────────────────────────────
print("🔢 Computing custom metrics (F1, BLEU, ROUGE, MRR, Hit Rate, Exact Match, Groundedness)...\n")

em_scores, f1_scores, bleu_scores       = [], [], []
rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
hit_scores, mrr_scores, ground_scores   = [], [], []

for ans, gt, ctxs in zip(answers, ground_truths, contexts_list):
    em_scores.append(exact_match(ans, gt))
    f1_scores.append(f1_score(ans, gt))
    bleu_scores.append(bleu_score(ans, gt))
    rouge = rouge_scores(ans, gt)
    rouge1_scores.append(rouge["rouge1"])
    rouge2_scores.append(rouge["rouge2"])
    rougeL_scores.append(rouge["rougeL"])
    hit_scores.append(hit_rate(gt, ctxs))
    mrr_scores.append(reciprocal_rank(gt, ctxs))
    ground_scores.append(groundedness(ans, ctxs))

# ── Add custom metrics to dataframe ──────────────────────────────────────────
df["exact_match"]   = em_scores
df["f1_score"]      = f1_scores
df["bleu_score"]    = bleu_scores
df["rouge1"]        = rouge1_scores
df["rouge2"]        = rouge2_scores
df["rougeL"]        = rougeL_scores
df["hit_rate"]      = hit_scores
df["mrr"]           = mrr_scores
df["groundedness"]  = ground_scores

# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("            FULL RAG EVALUATION REPORT — 12 METRICS")
print("=" * 70)

all_metrics = {
    # RAGAs metrics
    "faithfulness":       ("Faithfulness",       "RAGAs  ", "No hallucination?          "),
    "answer_relevancy":   ("Answer Relevancy",   "RAGAs  ", "Answer on-topic?           "),
    "context_precision":  ("Context Precision",  "RAGAs  ", "Retrieved chunks relevant? "),
    "context_recall":     ("Context Recall",     "RAGAs  ", "Context covers truth?      "),
    "answer_correctness": ("Answer Correctness", "RAGAs  ", "Factually correct?         "),
    # Custom metrics
    "faithfulness":       ("Faithfulness",       "RAGAs  ", "No hallucination?          "),
    "exact_match":        ("Exact Match",        "Custom ", "Exact string match?        "),
    "f1_score":           ("F1 Score",           "Custom ", "Precision + recall balance?"),
    "bleu_score":         ("BLEU Score",         "Custom ", "N-gram overlap with truth? "),
    "rouge1":             ("ROUGE-1",            "Custom ", "Unigram overlap?           "),
    "rouge2":             ("ROUGE-2",            "Custom ", "Bigram overlap?            "),
    "rougeL":             ("ROUGE-L",            "Custom ", "Longest common subsequence?"),
    "hit_rate":           ("Hit Rate",           "Custom ", "Truth found in context?    "),
    "mrr":                ("MRR",                "Custom ", "Rank of first correct ctx? "),
    "groundedness":       ("Groundedness",       "Custom ", "Answer grounded in context?"),
}

print(f"\n{'Metric':<22} {'Type':<9} {'Description':<32} {'Score':>6}  {'Bar'}")
print("-" * 70)

for col, (name, mtype, desc) in all_metrics.items():
    if col in df.columns:
        score  = df[col].mean()
        bar    = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        status = "✅" if score >= 0.7 else "⚠️ " if score >= 0.5 else "❌"
        print(f"{status} {name:<22} {mtype:<9} {desc:<32} {score:.3f}  [{bar}]")

print("=" * 70)

# ── Per-question breakdown ────────────────────────────────────────────────────
print("\n📁 Per-question scores:")
report_cols = ["question", "faithfulness", "f1_score", "bleu_score",
               "rouge1", "rougeL", "hit_rate", "mrr",
               "groundedness", "exact_match"]
available = [c for c in report_cols if c in df.columns]
print(df[available].to_string())

# ── Save to CSV ───────────────────────────────────────────────────────────────
df.to_csv("rag_eval_results.csv", index=False)
print("\n💾 Full results saved to rag_eval_results.csv")
print("✅ Evaluation complete!")