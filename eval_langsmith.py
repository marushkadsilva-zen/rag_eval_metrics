"""
LangSmith — tracing + dataset-based evaluation.
Get free API key at: smith.langchain.com
"""
import os
from dotenv import load_dotenv
load_dotenv()

# ── LangSmith needs these env vars ────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"]  = "true"
os.environ["LANGCHAIN_PROJECT"]     = "rag-eval-project"
os.environ["LANGCHAIN_API_KEY"]     = os.getenv("LANGCHAIN_API_KEY", "")

from langsmith import Client
from langsmith.evaluation import evaluate as ls_evaluate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from rag_pipeline import build_rag_pipeline, run_rag

TEST_DATA = [
    {"question": "When was Albert Einstein born?",
     "ground_truth": "Albert Einstein was born on March 14, 1879."},
    {"question": "What is Einstein's famous equation?",
     "ground_truth": "Einstein's famous equation is E = mc²."},
    {"question": "What prize did Einstein win?",
     "ground_truth": "Einstein received the Nobel Prize in Physics in 1921."},
]

print("🔧 Building pipeline...")
llm, retriever = build_rag_pipeline("documents/sample.txt")

# ── Step 1: Create a dataset in LangSmith ─────────────────────────
client = Client()
dataset_name = "Einstein-RAG-Eval"

# Delete old dataset if exists
try:
    client.delete_dataset(dataset_name=dataset_name)
except:
    pass

dataset = client.create_dataset(dataset_name=dataset_name)
client.create_examples(
    inputs=[{"question": d["question"]} for d in TEST_DATA],
    outputs=[{"ground_truth": d["ground_truth"]} for d in TEST_DATA],
    dataset_id=dataset.id,
)
print(f"✅ Dataset '{dataset_name}' created in LangSmith")

# ── Step 2: Define the RAG function to evaluate ───────────────────
def rag_function(inputs: dict) -> dict:
    question = inputs["question"]
    answer, contexts = run_rag(llm, retriever, question)
    return {"answer": answer, "contexts": contexts}

# ── Step 3: Define evaluators ─────────────────────────────────────
judge_llm = ChatGroq(
    model="llama-3.3-70b-versatile", temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

def correctness_evaluator(run, example):
    """LLM-as-judge: is the answer correct vs ground truth?"""
    answer = run.outputs.get("answer", "")
    ground_truth = example.outputs.get("ground_truth", "")
    question = example.inputs.get("question", "")

    prompt = f"""Score the answer from 0.0 to 1.0 based on correctness.
Question: {question}
Ground Truth: {ground_truth}
Answer: {answer}
Return ONLY a number between 0.0 and 1.0. Nothing else."""

    from langchain_core.messages import HumanMessage
    response = judge_llm.invoke([HumanMessage(content=prompt)])
    try:
        score = float(response.content.strip())
        score = max(0.0, min(1.0, score))
    except:
        score = 0.0

    return {"key": "correctness", "score": score}

def relevance_evaluator(run, example):
    """Is the answer relevant to the question?"""
    answer = run.outputs.get("answer", "")
    question = example.inputs.get("question", "")

    prompt = f"""Score relevance from 0.0 to 1.0.
Question: {question}
Answer: {answer}
Return ONLY a number between 0.0 and 1.0."""

    from langchain_core.messages import HumanMessage
    response = judge_llm.invoke([HumanMessage(content=prompt)])
    try:
        score = float(response.content.strip())
        score = max(0.0, min(1.0, score))
    except:
        score = 0.5
    return {"key": "relevance", "score": score}

def faithfulness_evaluator(run, example):
    """Is the answer grounded in the retrieved context?"""
    answer   = run.outputs.get("answer", "")
    contexts = run.outputs.get("contexts", [])
    context  = "\n".join(contexts)

    prompt = f"""Score faithfulness from 0.0 to 1.0.
Is this answer fully supported by the context? (1.0 = fully supported, 0.0 = hallucinated)
Context: {context}
Answer: {answer}
Return ONLY a number between 0.0 and 1.0."""

    from langchain_core.messages import HumanMessage
    response = judge_llm.invoke([HumanMessage(content=prompt)])
    try:
        score = float(response.content.strip())
        score = max(0.0, min(1.0, score))
    except:
        score = 0.5
    return {"key": "faithfulness", "score": score}

# ── Step 4: Run evaluation ─────────────────────────────────────────
print("\n📊 Running LangSmith evaluation...")
results = ls_evaluate(
    rag_function,
    data=dataset_name,
    evaluators=[correctness_evaluator, relevance_evaluator, faithfulness_evaluator],
    experiment_prefix="rag-eval",
)

print("\n════ LangSmith Results ════")
print(f"  View full results at: https://smith.langchain.com")
print(f"  Project: rag-eval-project")
print(f"  Dataset: {dataset_name}")

scores = {"correctness": [], "relevance": [], "faithfulness": []}
for result in results:
    for eval_result in result.get("evaluation_results", {}).get("results", []):
        key = eval_result.key
        if key in scores and eval_result.score is not None:
            scores[key].append(eval_result.score)

for metric, vals in scores.items():
    if vals:
        print(f"  {metric:<20}: {sum(vals)/len(vals):.3f}")

print("\n✅ Full traces visible at smith.langchain.com")