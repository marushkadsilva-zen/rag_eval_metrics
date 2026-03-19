import os
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-not-used")

from dotenv import load_dotenv
load_dotenv()

import re
import warnings
warnings.filterwarnings("ignore")

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rouge_lib
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

# ══════════════════════════════════════════════════════════════════
# ANSWER QUALITY HELPERS
# ══════════════════════════════════════════════════════════════════

def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower().strip())

def tokenize(text: str) -> list:
    return normalize(text).split()

def score_bar(score: float) -> str:
    filled = int(score * 20)
    return "█" * filled + "░" * (20 - filled)

def status_icon(score: float) -> str:
    if score >= 0.7: return "✅"
    if score >= 0.5: return "⚠️ "
    return "❌"

def print_section(title: str):
    print("\n" + "═" * 65)
    print(f"  {title}")
    print("═" * 65)

def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize(prediction) == normalize(ground_truth))

def token_precision(prediction: str, ground_truth: str) -> float:
    """Of all tokens in the answer, what fraction appear in ground truth?"""
    pred_tokens = tokenize(prediction)
    gt_tokens   = tokenize(ground_truth)
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    return sum(common.values()) / len(pred_tokens)

def token_recall(prediction: str, ground_truth: str) -> float:
    """Of all tokens in ground truth, what fraction appear in the answer?"""
    pred_tokens = tokenize(prediction)
    gt_tokens   = tokenize(ground_truth)
    if not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    return sum(common.values()) / len(gt_tokens)

def f1_score(prediction: str, ground_truth: str) -> float:
    prec = token_precision(prediction, ground_truth)
    rec  = token_recall(prediction, ground_truth)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def bleu_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = tokenize(prediction)
    gt_tokens   = tokenize(ground_truth)
    smoothie    = SmoothingFunction().method4
    return sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoothie)

def rouge_scores(prediction: str, ground_truth: str) -> dict:
    scorer = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = scorer.score(ground_truth, prediction)
    return {
        "rouge1": s["rouge1"].fmeasure,
        "rouge2": s["rouge2"].fmeasure,
        "rougeL": s["rougeL"].fmeasure,
    }

def groundedness(answer: str, contexts: list) -> float:
    """Fraction of answer tokens that appear in retrieved context."""
    combined      = " ".join(contexts)
    answer_tokens = set(tokenize(answer))
    ctx_tokens    = set(tokenize(combined))
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & ctx_tokens) / len(answer_tokens)

# ══════════════════════════════════════════════════════════════════
# RETRIEVAL PRECISION & RECALL  ← correct IR definitions
# ══════════════════════════════════════════════════════════════════

RELEVANCE_THRESHOLD = 0.3  # chunk needs to cover ≥30% of GT tokens to count as relevant

def is_relevant_chunk(chunk: str, ground_truth: str) -> bool:
    """A chunk is relevant if its token recall vs ground truth >= threshold."""
    chunk_tokens = set(tokenize(chunk))
    gt_tokens    = set(tokenize(ground_truth))
    if not gt_tokens:
        return False
    overlap = len(chunk_tokens & gt_tokens)
    return (overlap / len(gt_tokens)) >= RELEVANCE_THRESHOLD

def retrieval_precision(contexts: list, ground_truth: str) -> float:
    """
    Retrieval Precision = relevant chunks retrieved / total chunks retrieved
    → Of ALL chunks the system returned, how many were actually useful?
    """
    if not contexts:
        return 0.0
    relevant = sum(1 for ctx in contexts if is_relevant_chunk(ctx, ground_truth))
    return relevant / len(contexts)

def retrieval_recall(contexts: list, ground_truth: str) -> float:
    """
    Retrieval Recall = relevant chunks retrieved / total relevant chunks needed
    → Did the system retrieve enough useful chunks to answer the question?
    We assume 1 relevant chunk = sufficient to answer (standard RAG assumption).
    """
    if not contexts:
        return 0.0
    relevant = sum(1 for ctx in contexts if is_relevant_chunk(ctx, ground_truth))
    return min(relevant / max(1, len(contexts)), 1.0) if relevant > 0 else 0.0

def retrieval_f1(contexts: list, ground_truth: str) -> float:
    prec = retrieval_precision(contexts, ground_truth)
    rec  = retrieval_recall(contexts, ground_truth)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def chunk_coverage(chunk: str, ground_truth: str) -> float:
    """
    Per-chunk token recall:
    How much of the ground truth does THIS single chunk cover?
    """
    chunk_tokens = set(tokenize(chunk))
    gt_tokens    = set(tokenize(ground_truth))
    if not gt_tokens:
        return 0.0
    return len(chunk_tokens & gt_tokens) / len(gt_tokens)

def chunk_relevance(chunk: str, ground_truth: str) -> float:
    """
    Per-chunk token precision:
    How focused is THIS chunk on the ground truth?
    """
    chunk_tokens = set(tokenize(chunk))
    gt_tokens    = set(tokenize(ground_truth))
    if not chunk_tokens:
        return 0.0
    return len(chunk_tokens & gt_tokens) / len(chunk_tokens)

def hit_rate(contexts: list, ground_truth: str) -> float:
    """1.0 if at least one relevant chunk was retrieved, else 0.0"""
    return 1.0 if any(is_relevant_chunk(ctx, ground_truth) for ctx in contexts) else 0.0

def reciprocal_rank(contexts: list, ground_truth: str) -> float:
    """1 / rank of the first relevant chunk."""
    for i, ctx in enumerate(contexts):
        if is_relevant_chunk(ctx, ground_truth):
            return 1.0 / (i + 1)
    return 0.0

# ══════════════════════════════════════════════════════════════════
# SETUP — runs once at startup
# ══════════════════════════════════════════════════════════════════

print("\n" + "═" * 65)
print("       RAG INTERACTIVE EVALUATOR")
print("═" * 65)

print("\n🔧 Building RAG pipeline...")
llm, retriever = build_rag_pipeline("documents/sample.txt")

print("⚙️  Setting up RAGAs evaluator...")

ragas_llm = LangchainLLMWrapper(ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
))

hf_embeddings    = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

faithfulness.llm            = ragas_llm
answer_relevancy.llm        = ragas_llm
answer_relevancy.embeddings = ragas_embeddings
context_precision.llm       = ragas_llm
context_recall.llm          = ragas_llm
answer_correctness.llm      = ragas_llm

print("✅ Ready! Type your question below.\n")

# ══════════════════════════════════════════════════════════════════
# MAIN INTERACTIVE LOOP
# ══════════════════════════════════════════════════════════════════

while True:
    print("\n" + "─" * 65)
    question = input("❓ Enter your question (or 'quit' to exit): ").strip()

    if question.lower() in ("quit", "exit", "q"):
        print("\n👋 Goodbye!\n")
        break

    if not question:
        print("⚠️  Please enter a question.")
        continue

    ground_truth = input("📝 Enter the ground truth answer (for evaluation): ").strip()

    if not ground_truth:
        print("⚠️  Ground truth is needed to compute metrics. Please try again.")
        continue

    # ── Step 1: Run RAG ───────────────────────────────────────────
    print("\n🤖 Running RAG pipeline...")
    answer, contexts = run_rag(llm, retriever, question)

    # ── Step 2: Question & Answer ─────────────────────────────────
    print_section("QUESTION & ANSWER")
    print(f"  Question : {question}")
    print(f"  Answer   : {answer}")
    print(f"  Truth    : {ground_truth}")

    # ── Step 3: Retrieved Chunks + Retrieval Metrics ──────────────
    ret_prec       = retrieval_precision(contexts, ground_truth)
    ret_rec        = retrieval_recall(contexts, ground_truth)
    ret_f1_val     = retrieval_f1(contexts, ground_truth)
    relevant_count = sum(1 for ctx in contexts if is_relevant_chunk(ctx, ground_truth))
    hr             = hit_rate(contexts, ground_truth)
    mrr            = reciprocal_rank(contexts, ground_truth)

    print_section(f"RETRIEVED CHUNKS ({len(contexts)} total)")

    # ── Retrieval-level summary (across ALL chunks) ───────────────
    print(f"\n  🎯 RETRIEVAL METRICS (across all {len(contexts)} chunks):")
    print(f"     Relevant chunks found : {relevant_count} / {len(contexts)}")
    print(f"\n  {status_icon(ret_prec)} Retrieval Precision : {ret_prec:.3f}  [{score_bar(ret_prec)}]")
    print(f"       → of {len(contexts)} chunks returned, {relevant_count} were relevant")
    print(f"       → formula: relevant retrieved ({relevant_count}) / total retrieved ({len(contexts)})")
    print(f"\n  {status_icon(ret_rec)} Retrieval Recall    : {ret_rec:.3f}  [{score_bar(ret_rec)}]")
    print(f"       → did the system retrieve enough relevant chunks to answer?")
    print(f"       → formula: relevant retrieved ({relevant_count}) / total needed")
    print(f"\n  {status_icon(ret_f1_val)} Retrieval F1        : {ret_f1_val:.3f}  [{score_bar(ret_f1_val)}]")
    print(f"       → harmonic mean of retrieval precision and recall")
    print(f"\n  {status_icon(hr)} Hit Rate            : {hr:.3f}  [{score_bar(hr)}]")
    print(f"       → was at least 1 relevant chunk found? {'YES ✅' if hr else 'NO ❌'}")
    print(f"\n  {status_icon(mrr)} MRR                 : {mrr:.3f}  [{score_bar(mrr)}]")
    first_hit = int(1/mrr) if mrr > 0 else "N/A"
    print(f"       → first relevant chunk was at rank {first_hit}")
    print("\n  " + "─" * 61)

    # ── Per-chunk breakdown ───────────────────────────────────────
    print(f"\n  📄 PER-CHUNK BREAKDOWN:")
    for i, chunk in enumerate(contexts, 1):
        coverage  = chunk_coverage(chunk, ground_truth)
        relevance = chunk_relevance(chunk, ground_truth)
        relevant  = is_relevant_chunk(chunk, ground_truth)
        preview   = chunk.strip().replace("\n", " ")[:110]

        print(f"\n  Chunk {i}: {'✅ RELEVANT' if relevant else '❌ NOT RELEVANT'}")
        print(f"  \"{preview}...\"")
        print(f"\n     Coverage  (token recall — how much of GT this chunk covers)")
        print(f"     → {coverage:.3f}  [{score_bar(coverage)}]  ({coverage*100:.1f}% of ground truth tokens found here)")
        print(f"     Relevance (token precision — how focused this chunk is on GT)")
        print(f"     → {relevance:.3f}  [{score_bar(relevance)}]  ({relevance*100:.1f}% of chunk tokens appear in ground truth)")

    # ── Step 4: Answer Quality Metrics ───────────────────────────
    print_section("ANSWER QUALITY METRICS (answer vs ground truth)")

    em    = exact_match(answer, ground_truth)
    prec  = token_precision(answer, ground_truth)
    rec   = token_recall(answer, ground_truth)
    f1    = f1_score(answer, ground_truth)
    bleu  = bleu_score(answer, ground_truth)
    rouge = rouge_scores(answer, ground_truth)
    grnd  = groundedness(answer, contexts)

    print(f"\n  {status_icon(em)}  Exact Match   : {em:.3f}  [{score_bar(em)}]")
    print(f"       (1.0 = answer exactly matches ground truth)")

    print(f"\n  {status_icon(prec)} Precision      : {prec:.3f}  [{score_bar(prec)}]")
    print(f"       (of words in answer, how many are in ground truth?)")

    print(f"\n  {status_icon(rec)}  Recall         : {rec:.3f}  [{score_bar(rec)}]")
    print(f"       (of words in ground truth, how many appear in answer?)")

    print(f"\n  {status_icon(f1)}  F1 Score       : {f1:.3f}  [{score_bar(f1)}]")
    print(f"       (harmonic mean of answer precision and recall)")

    print(f"\n  {status_icon(bleu)} BLEU Score     : {bleu:.3f}  [{score_bar(bleu)}]")
    print(f"       (n-gram overlap between answer and ground truth)")

    print(f"\n  {status_icon(rouge['rouge1'])} ROUGE-1        : {rouge['rouge1']:.3f}  [{score_bar(rouge['rouge1'])}]")
    print(f"       (unigram overlap)")

    print(f"\n  {status_icon(rouge['rouge2'])} ROUGE-2        : {rouge['rouge2']:.3f}  [{score_bar(rouge['rouge2'])}]")
    print(f"       (bigram overlap)")

    print(f"\n  {status_icon(rouge['rougeL'])} ROUGE-L        : {rouge['rougeL']:.3f}  [{score_bar(rouge['rougeL'])}]")
    print(f"       (longest common subsequence)")

    print(f"\n  {status_icon(grnd)} Groundedness   : {grnd:.3f}  [{score_bar(grnd)}]")
    print(f"       (how much of the answer is supported by retrieved context?)")

    # ── Step 5: RAGAs Metrics ─────────────────────────────────────
    print_section("RAGAs METRICS (LLM-based deep evaluation)")
    print("  ⏳ Running RAGAs evaluation (this takes ~20-30 seconds)...")

    eval_dataset = Dataset.from_dict({
        "question":     [question],
        "answer":       [answer],
        "contexts":     [contexts],
        "ground_truth": [ground_truth],
    })

    try:
        ragas_results = evaluate(
            dataset=eval_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision,
                     context_recall, answer_correctness],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        ragas_df = ragas_results.to_pandas()

        ragas_metric_info = {
            "faithfulness":       ("Faithfulness",       "answer not hallucinated?"),
            "answer_relevancy":   ("Answer Relevancy",   "answer addresses the question?"),
            "context_precision":  ("Context Precision",  "retrieved chunks are relevant?"),
            "context_recall":     ("Context Recall",     "context covers ground truth?"),
            "answer_correctness": ("Answer Correctness", "answer factually matches truth?"),
        }

        print()
        for col, (name, desc) in ragas_metric_info.items():
            if col in ragas_df.columns:
                score = ragas_df[col].iloc[0]
                if score is not None and str(score) != "nan":
                    score = float(score)
                    print(f"  {status_icon(score)} {name:<22} : {score:.3f}  [{score_bar(score)}]")
                    print(f"       ({desc})")
                    print()

    except Exception as e:
        print(f"  ⚠️  RAGAs evaluation failed: {e}")

    # ── Step 6: Quick Summary ─────────────────────────────────────
    print_section("QUICK SUMMARY")
    print(f"  Question  : {question}")
    print(f"  Answer    : {answer}")
    print(f"\n  Retrieval → Precision={ret_prec:.2f}  Recall={ret_rec:.2f}  "
          f"F1={ret_f1_val:.2f}  Hit={hr:.2f}  MRR={mrr:.2f}")
    print(f"  Answer    → F1={f1:.2f}  BLEU={bleu:.2f}  "
          f"ROUGE-L={rouge['rougeL']:.2f}  Grounded={grnd:.2f}")
    print("\n  ✅ Evaluation complete for this question!")
    print("  💡 Ask another question or type 'quit' to exit.")