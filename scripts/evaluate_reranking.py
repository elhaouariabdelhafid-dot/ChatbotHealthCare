import csv
import json
import time
import urllib.request
from pathlib import Path
from datetime import datetime

from sentence_transformers import CrossEncoder
from healthbot.settings import settings
from healthbot.retrieval import Retriever

DATASET = Path("data/evaluation/reranking_test.csv")
OUTPUT = Path("artifacts") / (
    "reranking_evaluation_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
    + ".csv"
)

cfg = dict(settings())
cfg["retriever"] = "dense"

print("Chargement E5...", flush=True)
retriever = Retriever(cfg)

print("Chargement mMiniLM...", flush=True)
reranker = CrossEncoder(
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    device="cpu",
    max_length=384
)


def rewrite(question):
    prompt = (
        "Rewrite this medical question into one short English "
        "search question. Preserve exactly the disease, "
        "the medical intent and all important details. "
        "Do not invent information. Return only the question.\n\n"
        f"Question: {question}"
    )

    payload = {
        "model": "qwen2.5:3b",
        "prompt": prompt,
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "temperature": 0,
            "num_ctx": 1024,
            "num_predict": 100,
        }
    }

    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)

    rewritten = result.get("response", "").strip().strip('"')
    if not rewritten:
        raise ValueError("Reformulation vide")
    return rewritten


def rerank(question, hits):
    pairs = [
        (question, h.get("title", "") + ". " + h.get("text", ""))
        for h in hits
    ]

    scores = reranker.predict(
        pairs, batch_size=4, show_progress_bar=False
    )

    return [
        item[0] for item in sorted(
            zip(hits, scores),
            key=lambda x: float(x[1]),
            reverse=True
        )
    ]


def get_rank(hits, expected):
    for i, hit in enumerate(hits, 1):
        if hit.get("title", "").strip().casefold() == expected.strip().casefold():
            return i
    return 0


with DATASET.open(encoding="utf-8", newline="") as f:
    cases = list(csv.DictReader(f))

results = []

for case in cases:
    case_id = case["id"]
    question = case["question"]
    expected = case["expected_title"]

    print("\n" + "=" * 70, flush=True)
    print(case_id, question, flush=True)

    start = time.perf_counter()
    hits = [
        h for h in retriever.search(question, k=10)
        if h.get("accepted", False)
    ]
    retrieval_time = time.perf_counter() - start

    rank_e5 = get_rank(hits, expected)

    rank_original = 0
    rank_rewritten = 0
    rerank_time = 0
    rewrite_time = 0
    rewritten = ""
    error = ""

    if hits:
        start = time.perf_counter()
        original_hits = rerank(question, hits)
        rerank_time = time.perf_counter() - start
        rank_original = get_rank(original_hits, expected)

        start = time.perf_counter()
        try:
            rewritten = rewrite(question)
            rewrite_time = time.perf_counter() - start

            start = time.perf_counter()
            rewritten_hits = rerank(rewritten, hits)
            rewritten_rerank_time = time.perf_counter() - start
            rank_rewritten = get_rank(rewritten_hits, expected)
        except Exception as exc:
            error = str(exc)
            rewrite_time = time.perf_counter() - start
            rewritten_rerank_time = 0
    else:
        rewritten_rerank_time = 0
        error = "Aucun candidat accepté"

    result = {
        "id": case_id,
        "question": question,
        "expected_title": expected,
        "rewritten_question": rewritten,
        "rank_e5": rank_e5,
        "rank_mminilm": rank_original,
        "rank_rewriting_mminilm": rank_rewritten,
        "retrieval_seconds": round(retrieval_time, 3),
        "rerank_seconds": round(rerank_time, 3),
        "rewrite_seconds": round(rewrite_time, 3),
        "rerank_rewritten_seconds": round(rewritten_rerank_time, 3),
        "error": error,
    }

    results.append(result)

    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader()
        writer.writerows(results)

    print("Reformulation :", rewritten or error)
    print(
        f"Rang E5={rank_e5} | "
        f"mMiniLM={rank_original} | "
        f"Rewriting+mMiniLM={rank_rewritten}"
    )
    print(
        f"Temps rewriting={rewrite_time:.2f}s | "
        f"reranking={rewritten_rerank_time:.2f}s",
        flush=True
    )


def summarize(key):
    ranks = [int(r[key]) for r in results]
    n = len(ranks)

    hit1 = sum(r == 1 for r in ranks) / n
    hit5 = sum(1 <= r <= 5 for r in ranks) / n
    mrr5 = sum(1 / r for r in ranks if 1 <= r <= 5) / n

    return hit1, hit5, mrr5


print("\n" + "=" * 70)
print("RÉSUMÉ FINAL")

for name, key in [
    ("E5 seul", "rank_e5"),
    ("E5 + mMiniLM", "rank_mminilm"),
    ("E5 + Rewriting + mMiniLM", "rank_rewriting_mminilm"),
]:
    hit1, hit5, mrr5 = summarize(key)
    print(
        f"{name}: Hit@1={hit1:.1%} | "
        f"Hit@5={hit5:.1%} | MRR@5={mrr5:.3f}"
    )

print("\nRapport :", OUTPUT)
print("Attention : vérifier manuellement les reformulations.")
