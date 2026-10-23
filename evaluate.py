"""
evaluate.py — Run the 5 official questions and dump results to results.json
"""

import json
from pathlib import Path

QUESTIONS = [
    "What are the three main types of inspection?",
    "What does a maintenance factor of two mean?",
    "What is the purchase price of this gas turbine?",
    "What is the melting point of titanium?",
    "What is the recommended combustion inspection interval in hours?",
]

def main():
    from main import load_llm
    from rag import build_rag_chain, answer_question

    llm = load_llm()
    chain, retriever = build_rag_chain(llm)

    results = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n[Q{i}] {q}")
        r = answer_question(chain, retriever, q)
        print(f"→ {r['answer']}")
        print(f"   Pages: {r['pages']}")
        results.append({
            "q_num": i,
            "question": q,
            "answer": r["answer"],
            "source_pages": r["pages"],
        })

    out_path = Path("results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n✅ Results written to {out_path}")


if __name__ == "__main__":
    main()
