"""
main.py — Run the Gas Turbine Maintenance Q&A system.

Usage:
    python main.py              # Answer the 5 preset questions
    python main.py --interactive  # Enter your own questions
"""

import argparse
import sys
from typing import Any, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

# We use a small open-source LLM that runs locally (no API key needed).
# google/flan-t5-base is ~250 MB and works well for Q&A tasks.
MODEL_ID = "google/flan-t5-base"

QUESTIONS = [
    "What are the three main types of inspection?",
    "What does a maintenance factor of two mean?",
    "What is the purchase price of this gas turbine?",
    "What is the melting point of titanium?",
    "What is the recommended combustion inspection interval in hours?",
]


class LocalFlanT5LLM(LLM):
    """Custom LangChain LLM wrapper for Flan-T5 compatible with transformers v5+."""
    model: Any = None
    tokenizer: Any = None
    max_new_tokens: int = 512

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @property
    def _llm_type(self) -> str:
        return "flan_t5_local"


def load_llm():
    """Load local Hugging Face model (flan-t5-base)."""
    print(f"🤖 Loading LLM: {MODEL_ID} …")
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    llm = LocalFlanT5LLM(model=model, tokenizer=tokenizer, max_new_tokens=512)
    print("   → LLM ready\n")
    return llm


def print_separator():
    print("\n" + "=" * 70 + "\n")


def run_preset_questions(chain, retriever):
    from rag import answer_question

    print_separator()
    print("📋  GAS TURBINE MAINTENANCE ASSISTANT — Preset Questions")
    print_separator()

    results = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"Q{i}: {q}")
        result = answer_question(chain, retriever, q)
        print(f"Answer: {result['answer']}")
        print(f"Sources: Pages {result['pages']}")
        print("-" * 70)
        results.append(result)

    return results


def run_interactive(chain, retriever):
    from rag import answer_question

    print_separator()
    print("💬  Interactive Q&A Mode  (type 'quit' to exit)")
    print_separator()

    while True:
        try:
            q = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        result = answer_question(chain, retriever, q)
        print(f"\nAnswer: {result['answer']}")
        print(f"Sources: Pages {result['pages']}\n")


def main():
    parser = argparse.ArgumentParser(description="GE Gas Turbine RAG Q&A")
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run in interactive Q&A mode"
    )
    parser.add_argument(
        "--ingest", action="store_true",
        help="(Re-)ingest the PDF before answering"
    )
    args = parser.parse_args()

    # Optionally re-ingest
    if args.ingest:
        from ingest import ingest
        ingest()

    # Import after possible ingest
    from rag import build_rag_chain

    llm = load_llm()
    chain, retriever = build_rag_chain(llm)

    if args.interactive:
        run_interactive(chain, retriever)
    else:
        run_preset_questions(chain, retriever)


if __name__ == "__main__":
    main()
