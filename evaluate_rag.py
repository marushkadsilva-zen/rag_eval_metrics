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
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower().strip())

def tokenize(text: str) -> list:
    return normalize(text).split()

def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize(prediction) == normalize(ground_truth))

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

def precision_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = tokenize(prediction)
    gt_tokens   = tokenize(ground_truth)
    common      = Counter(pred_tokens) & Counter(gt_tokens)
    num_common  = sum(common.values())
    return num_common / len(pred_tokens) if pred_tokens else 0.0

def recall_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = tokenize(prediction)
    gt_tokens   = tokenize(ground_truth)
    common      = Counter(pred_tokens) & Counter(gt_tokens)
    num_common  = sum(common.values())
    return num_common / len(gt_tokens) if gt_tokens else 0.0

def bleu_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = tokenize(prediction)
    gt_tokens   = tokenize(ground_truth)
    smoothie    = SmoothingFunction().method4
    return sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoothie)

def rouge_scores(prediction: str, ground_truth: str) -> dict:
    scorer = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }

def hit_rate(ground_truth: str, contexts: list) -> float:
    gt_tokens = set(tokenize(ground_truth))
    for ctx in contexts:
        ctx_tokens = set(tokenize(ctx))
        overlap = len(gt_tokens & ctx_tokens) / max(len(gt_tokens), 1)
        if overlap >= 0.3:
            return 1.0
    return 0.0

def reciprocal_rank(ground_truth: str, contexts: list) -> float:
    gt_tokens = set(tokenize(ground_truth))
    for i, ctx in enumerate(contexts):
        ctx_tokens = set(tokenize(ctx))
        overlap = len(gt_tokens & ctx_tokens) / max(len(gt_tokens), 1)
        if overlap >= 0.3:
            return 1.0 / (i + 1)
    return 0.0

def groundedness(answer: str, contexts: list) -> float:
    combined_ctx  = " ".join(contexts)
    answer_tokens = set(tokenize(answer))
    ctx_tokens    = set(tokenize(combined_ctx))
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & ctx_tokens) / len(answer_tokens)

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

    # ── Step 2: Show Question + Answer ───────────────────────────
    print_section("QUESTION & ANSWER")
    print(f"  Question : {question}")
    print(f"  Answer   : {answer}")
    print(f"  Truth    : {ground_truth}")

    # ── Step 3: Show Retrieved Chunks ────────────────────────────
    print_section(f"RETRIEVED CHUNKS ({len(contexts)} total)")
    for i, chunk in enumerate(contexts, 1):
        print(f"\n  📄 Chunk {i}:")
        print(f"  {chunk.strip()}")

        # Per-chunk stats vs ground truth
        gt_tokens  = set(tokenize(ground_truth))
        ctx_tokens = set(tokenize(chunk))
        overlap    = len(gt_tokens & ctx_tokens)
        coverage   = overlap / max(len(gt_tokens), 1)
        relevance  = overlap / max(len(ctx_tokens), 1)

        print(f"\n  📊 Chunk {i} Metrics (vs ground truth):")
        print(f"     Token overlap count : {overlap}")
        print(f"     Coverage (recall)   : {coverage:.3f}  [{score_bar(coverage)}]")
        print(f"     Relevance (prec)    : {relevance:.3f}  [{score_bar(relevance)}]")
        print(f"     Hit (≥30% overlap)  : {'✅ YES' if coverage >= 0.3 else '❌ NO'}")

    # ── Step 4: Answer-level Custom Metrics ──────────────────────
    print_section("ANSWER QUALITY METRICS (answer vs ground truth)")

    em   = exact_match(answer, ground_truth)
    f1   = f1_score(answer, ground_truth)
    prec = precision_score(answer, ground_truth)
    rec  = recall_score(answer, ground_truth)
    bleu = bleu_score(answer, ground_truth)
    rouge = rouge_scores(answer, ground_truth)
    grnd = groundedness(answer, contexts)
    hr   = hit_rate(ground_truth, contexts)
    mrr  = reciprocal_rank(ground_truth, contexts)

    print(f"\n  {status_icon(em)}   Exact Match   : {em:.3f}  [{score_bar(em)}]")
    print(       f"       (1.0 = answer exactly matches ground truth)")

    print(f"\n  {status_icon(prec)} Precision      : {prec:.3f}  [{score_bar(prec)}]")
    print(       f"       (of words in answer, how many are in ground truth?)")

    print(f"\n  {status_icon(rec)}  Recall         : {rec:.3f}  [{score_bar(rec)}]")
    print(       f"       (of words in ground truth, how many appear in answer?)")

    print(f"\n  {status_icon(f1)}  F1 Score       : {f1:.3f}  [{score_bar(f1)}]")
    print(       f"       (harmonic mean of precision and recall)")

    print(f"\n  {status_icon(bleu)} BLEU Score     : {bleu:.3f}  [{score_bar(bleu)}]")
    print(       f"       (n-gram overlap between answer and ground truth)")

    print(f"\n  {status_icon(rouge['rouge1'])} ROUGE-1        : {rouge['rouge1']:.3f}  [{score_bar(rouge['rouge1'])}]")
    print(       f"       (unigram overlap)")

    print(f"\n  {status_icon(rouge['rouge2'])} ROUGE-2        : {rouge['rouge2']:.3f}  [{score_bar(rouge['rouge2'])}]")
    print(       f"       (bigram overlap)")

    print(f"\n  {status_icon(rouge['rougeL'])} ROUGE-L        : {rouge['rougeL']:.3f}  [{score_bar(rouge['rougeL'])}]")
    print(       f"       (longest common subsequence)")

    # ── Step 5: Retrieval Metrics ─────────────────────────────────
    print_section("RETRIEVAL METRICS (context vs ground truth)")

    print(f"\n  {status_icon(hr)}  Hit Rate       : {hr:.3f}  [{score_bar(hr)}]")
    print(       f"       (was the answer found in any retrieved chunk?)")

    print(f"\n  {status_icon(mrr)} MRR            : {mrr:.3f}  [{score_bar(mrr)}]")
    print(       f"       (1/rank of first chunk containing the answer)")

    print(f"\n  {status_icon(grnd)} Groundedness   : {grnd:.3f}  [{score_bar(grnd)}]")
    print(       f"       (how much of the answer is supported by context?)")

    # ── Step 6: RAGAs Metrics ─────────────────────────────────────
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
            "faithfulness":       ("Faithfulness",       "answer not hallucinated?          "),
            "answer_relevancy":   ("Answer Relevancy",   "answer addresses the question?    "),
            "context_precision":  ("Context Precision",  "retrieved chunks are relevant?    "),
            "context_recall":     ("Context Recall",     "context covers ground truth?      "),
            "answer_correctness": ("Answer Correctness", "answer factually matches truth?   "),
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

    # ── Step 7: Summary ───────────────────────────────────────────
    print_section("QUICK SUMMARY")
    print(f"  Question    : {question}")
    print(f"  Answer      : {answer}")
    print(f"  F1={f1:.2f}  BLEU={bleu:.2f}  ROUGE-L={rouge['rougeL']:.2f}  "
          f"Hit={hr:.2f}  MRR={mrr:.2f}  Grounded={grnd:.2f}")

    print("\n  ✅ Evaluation complete for this question!")
    print("  💡 Ask another question or type 'quit' to exit.")