import csv
import json
import sys
import time
import urllib.request
from pathlib import Path
from datetime import datetime

SOURCE = Path("data/evaluation/rewrite_audit.csv")
OUTPUT = Path("artifacts") / (
    "rewrite_v2_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
    + ".csv"
)

PROMPT = """
You are a medical search-query translator.

Your ONLY task is to translate the user's question
into English for document retrieval.

STRICT RULES:
1. Preserve the exact meaning of the original question.
2. Preserve the medical condition, disease and body part.
3. Preserve the question intent: definition, identification,
   symptoms, causes, prevention or other intent.
4. Never replace a broad medical condition with a more
   specific condition.
5. Never add symptoms, treatments, causes, risk factors,
   diagnoses or other details not present in the question.
6. Preserve negations, uncertainty and important qualifiers.
7. If the question asks WHICH disease or syndrome, keep
   that identification question. Do not replace it with
   a question asking which cancer or organ is involved.
8. If the question is already in English, return it
   exactly unchanged.
9. If you cannot translate faithfully, return the
   original question unchanged.
10. Return ONLY one question. No explanation.

Original question:
{question}
"""

def rewrite(question):
    payload = {
        "model": "qwen2.5:3b",
        "prompt": PROMPT.format(question=question),
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "temperature": 0,
            "num_ctx": 1024,
            "num_predict": 130
        }
    }

    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(
        request, timeout=180
    ) as response:
        result = json.load(response)

    rewritten = result.get("response", "").strip()
    rewritten = rewritten.strip('"').strip("'")

    if not rewritten:
        raise ValueError("Reformulation vide")

    return rewritten


with SOURCE.open(encoding="utf-8", newline="") as f:
    cases = list(csv.DictReader(f))

# Par défaut : tester uniquement les trois erreurs.
# Avec --all : tester les vingt questions.
if "--all" not in sys.argv:
    cases = [
        row for row in cases
        if row["id"] in {"Q10", "Q15", "Q18"}
    ]

results = []

for row in cases:
    question = row["question_originale"]

    print("\n" + "=" * 70, flush=True)
    print("QUESTION :", row["id"], flush=True)
    print("Originale :", question, flush=True)
    print(
        "Ancienne :", row["question_reformulee"],
        flush=True
    )

    start = time.perf_counter()
    error = ""

    try:
        rewritten = rewrite(question)
    except Exception as exc:
        rewritten = ""
        error = str(exc)

    duration = round(time.perf_counter() - start, 2)

    print("Nouvelle :", rewritten or "ERREUR")
    print("Temps :", duration, "s", flush=True)

    results.append({
        "id": row["id"],
        "original": question,
        "ancienne_reformulation": row["question_reformulee"],
        "nouvelle_reformulation": rewritten,
        "evaluation_manuelle": "",
        "commentaire": "",
        "temps_secondes": duration,
        "erreur": error
    })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

print("\nTEST TERMINÉ")
print("Rapport :", OUTPUT)
print("Vérification humaine encore nécessaire.")
