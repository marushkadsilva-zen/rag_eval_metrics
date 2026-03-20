import os
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-not-used")
from dotenv import load_dotenv
load_dotenv()

import re
import json
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from rag_pipeline import build_rag_pipeline, run_rag


class GroqDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.client = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )

    def load_model(self):
        return self.client

    def _clean_json(self, text: str) -> str:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group()
        def fix_escapes(m):
            char = m.group(1)
            valid = set('bfnrtu"\\/0123456789')
            if char.lower() in valid:
                return m.group(0)
            return '\\\\' + char
        text = re.sub(r'\\([a-zA-Z])', fix_escapes, text)
        text = re.sub(r',\s*([}\]])', r'\1', text)
        return text

    def generate(self, prompt: str) -> str:
        json_prompt = (
            prompt
            + "\n\nIMPORTANT: Respond with valid JSON only. "
            + "No extra text. No markdown. No code blocks."
        )
        response = self.client.invoke([HumanMessage(content=json_prompt)])
        raw = response.content.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        cleaned = self._clean_json(raw)
        try:
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError:
            return raw

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "groq-llama-3.3-70b"


TEST_DATA = [
    {
        "question": "When was Albert Einstein born?",
        "ground_truth": "Albert Einstein was born on March 14, 1879."
    },
    {
        "question": "What is Einstein's famous equation?",
        "ground_truth": "Einstein's famous equation is E = mc2, the mass-energy equivalence formula."
    },
    {
        "question": "What prize did Einstein win and for what?",
        "ground_truth": "Einstein received the Nobel Prize in Physics in 1921 for the photoelectric effect."
    },
    {
        "question": "Where did Einstein work in the United States?",
        "ground_truth": "Einstein worked at the Institute for Advanced Study in Princeton, New Jersey."
    },
]

print("Building pipeline...")
llm, retriever = build_rag_pipeline("documents/sample.txt")
groq_judge = GroqDeepEvalLLM()

print("\nGenerating answers...")
test_cases = []
for item in TEST_DATA:
    q  = item["question"]
    gt = item["ground_truth"]
    answer, contexts = run_rag(llm, retriever, q)
    print(f"  Q: {q}")
    print(f"  A: {answer}\n")
    test_cases.append(LLMTestCase(
        input=q,
        actual_output=answer,
        expected_output=gt,
        retrieval_context=contexts,
        context=contexts,
    ))

metrics = [
    AnswerRelevancyMetric(
        threshold=0.5,
        model=groq_judge,
        include_reason=True,
        async_mode=False,
    ),
    FaithfulnessMetric(
        threshold=0.5,
        model=groq_judge,
        include_reason=True,
        async_mode=False,
    ),
    ContextualRecallMetric(
        threshold=0.5,
        model=groq_judge,
        include_reason=True,
        async_mode=False,
    ),
]

print("\nRunning DeepEval evaluation...\n")
results = evaluate(test_cases, metrics)

print("\n" + "=" * 60)
print("  DEEPEVAL RESULTS")
print("=" * 60)

all_scores = {}

for test_result in results.test_results:
    print(f"\n  Q: {test_result.input}")
    print(f"  A: {test_result.actual_output}")
    print()
    for metric_data in test_result.metrics_data:
        score  = metric_data.score if metric_data.score is not None else 0.0
        icon   = "✅" if metric_data.success else "❌"
        status = "PASS" if metric_data.success else "FAIL"
        bar    = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {icon} {metric_data.name:<28}: {score:.3f}  [{bar}]  {status}")
        if metric_data.reason:
            print(f"     Reason: {metric_data.reason[:100]}")
        if metric_data.name not in all_scores:
            all_scores[metric_data.name] = []
        if metric_data.score is not None:
            all_scores[metric_data.name].append(metric_data.score)

print("\n" + "=" * 60)
print("  AGGREGATE SUMMARY")
print("=" * 60)
for metric_name, scores in all_scores.items():
    avg    = sum(scores) / len(scores) if scores else 0.0
    bar    = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
    status = "✅" if avg >= 0.7 else "⚠️ " if avg >= 0.5 else "❌"
    print(f"  {status} {metric_name:<28}: {avg:.3f}  [{bar}]")

print("\n✅ DeepEval complete!")
print("💡 Tip: deepeval test run eval_deepeval.py for pytest output")