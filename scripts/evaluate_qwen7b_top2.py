import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from healthbot.settings import settings
from healthbot.engine import HealthBot
import healthbot.engine as engine

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

dataset = ROOT / "data/evaluation/evaluation_30_proposee.json"
data = json.loads(dataset.read_text(encoding="utf-8"))

cases = [
    c for c in data["cases"]
    if c.get("enabled")
    and c["category"] != "medical_absence_unverified"
]

if len(cases) != 25 or len({c["id"] for c in cases}) != 25:
    raise SystemExit("Le test exige les 25 cas actifs, sans doublons.")

cfg = dict(settings())
cfg.update(
    retriever="dense",
    generation="ollama",
    ollama_model="qwen2.5:7b",
    compact_prompt=False,
    reranking_enabled=True,
    reranking_candidates=5,
    top_k=2,
    timeout_seconds=600,
)

if sha256(cfg["corpus"]) != data["corpus_sha256"]:
    raise SystemExit("Corpus différent : vérifier le jeu avant de tester.")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output = ROOT / "artifacts" / f"qwen7b_top2_25_{stamp}.json"
output.parent.mkdir(parents=True, exist_ok=True)

report = {
    "experiment": "Qwen 7B, ancien prompt, E5 + reranking, top_k=2",
    "dataset_sha256": sha256(dataset),
    "corpus_sha256": sha256(cfg["corpus"]),
    "configuration": cfg,
    "code_sha256": {
        name: sha256(ROOT / name)
        for name in [
            "healthbot/engine.py",
            "healthbot/generation.py",
            "healthbot/retrieval.py",
            "healthbot/reranking.py",
        ]
    },
    "limitations": [
        "Jeu de développement déjà utilisé, pas un test indépendant.",
        "Statut conforme ne signifie pas réponse fidèle ou correcte.",
        "Latence influencée par le chargement et les caches.",
    ],
    "completed": False,
    "details": [],
}

def save():
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(output)

save()
print("Rapport :", output, flush=True)
print("Chargement unique du chatbot...", flush=True)
start = time.perf_counter()
bot = HealthBot(cfg)
report["initialization_seconds"] = round(time.perf_counter() - start, 3)
save()

real_generate = engine.generate
captured = {}

def capture_generate(question, hits, generation_cfg):
    captured["passages"] = json.loads(json.dumps(hits, default=str))
    return real_generate(question, hits, generation_cfg)

for index, case in enumerate(cases, 1):
    print(f"\n[{index}/25] {case['id']} : {case['question']}", flush=True)
    captured.clear()
    start = time.perf_counter()

    try:
        with patch.object(engine, "generate", capture_generate):
            result = bot.ask(case["question"])
        error = None
    except Exception as exc:
        result = {
            "status": "error",
            "mode": "error",
            "answer": "",
        }
        error = f"{type(exc).__name__}: {exc}"

    seconds = round(time.perf_counter() - start, 3)
    row = {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expected_statuses": case["expected_statuses"],
        "status_matches_expectation":
            result.get("status") in case["expected_statuses"],
        "wall_seconds": seconds,
        "result": result,
        "generation_passages": captured.get("passages", []),
        "reference_text": case.get("reference_text", ""),
        "evidence_quotes": case.get("evidence_quotes", []),
        "error": error,
        "manual_review": {
            "faithful_to_passages": None,
            "answers_question": None,
            "citations_support_claims": None,
            "language_quality": None,
            "comment": "",
        },
    }
    report["details"].append(row)
    save()

    print("Mode :", result.get("mode"), "| Temps :", seconds, "s", flush=True)
    print("Réponse :", result.get("answer"), flush=True)
    print("Avertissement :", result.get("warning") or error or "Aucun",
          flush=True)

summary = {}
for category in sorted({c["category"] for c in cases}):
    rows = [r for r in report["details"] if r["category"] == category]
    times = [r["wall_seconds"] for r in rows]
    summary[category] = {
        "questions": len(rows),
        "statuses_conforming": sum(r["status_matches_expectation"] for r in rows),
        "modes": dict(Counter(r["result"].get("mode") for r in rows)),
        "mean_seconds": round(statistics.mean(times), 2),
        "median_seconds": round(statistics.median(times), 2),
        "max_seconds": round(max(times), 2),
    }

report["summary"] = summary
report["completed"] = True
save()

print("\n=== BILAN : STATUTS ET LATENCE, QUALITÉ À RELIRE ===")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\nRapport :", output)
