import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from healthbot.settings import settings
from healthbot.retrieval import Retriever

# Documents de référence connus dans notre corpus.
# D'autres documents pourraient aussi être pertinents :
# leurs jugements seront à compléter lors d'une vraie évaluation.
cases = [
    (
        "What is cardiac arrest?",
        "medquad_081d51ce858725c65cc6",
    ),
    (
        "Qu’est-ce qu’un arrêt cardiaque ?",
        "medquad_081d51ce858725c65cc6",
    ),
    (
        "Que signifie un arrêt du cœur ?",
        "medquad_081d51ce858725c65cc6",
    ),
    (
        "Qu’est-ce que le diabète ?",
        "diabete_definition",
    ),
    (
        "Quels sont les symptômes de l’asthme ?",
        "asthme_definition",
    ),
    (
        "Quels facteurs favorisent l’hypertension ?",
        "hypertension_risques",
    ),
    (
        "Quel est le prix d’un billet de train ?",
        None,
    ),
    (
        "Comment installer un jeu vidéo ?",
        None,
    ),
]

cfg = settings()

print("Chargement de TF-IDF...", flush=True)
lexical = Retriever({**cfg, "retriever": "lexical"})

print("Chargement d’E5 et du cache existant...", flush=True)
semantic = Retriever({**cfg, "retriever": "dense"})

available = {c["document_id"] for c in lexical.chunks}
missing = {
    expected for _, expected in cases
    if expected is not None and expected not in available
}
if missing:
    raise SystemExit(f"Documents de référence absents : {sorted(missing)}")

details = []
summary = []

for mode in ("lexical", "dense", "hybrid"):
    retriever = lexical if mode == "lexical" else semantic

    # Modification en mémoire seulement : config.json reste inchangé.
    retriever.cfg["retriever"] = mode

    hits_at_1 = []
    recalls_at_5 = []
    reciprocal_ranks = []
    rejected_outside = []

    print(f"\n=== {mode.upper()} ===", flush=True)

    for question, expected in cases:
        hits = retriever.search(question, k=5)

        rank = next(
            (
                position
                for position, hit in enumerate(hits, 1)
                if hit["document_id"] == expected
            ),
            None,
        ) if expected else None

        # Le moteur transmet normalement au maximum top_k passages.
        accepted = [
            hit for hit in hits[:cfg["top_k"]]
            if hit["accepted"]
        ]

        if expected:
            hits_at_1.append(int(rank == 1))
            recalls_at_5.append(int(rank is not None))
            reciprocal_ranks.append(1 / rank if rank else 0)
        else:
            rejected_outside.append(int(not accepted))

        details.append({
            "method": mode,
            "question": question,
            "expected_document": expected or "",
            "rank_at_5": rank or "",
            "top_document": hits[0]["document_id"] if hits else "",
            "accepted_passages": len(accepted),
            "expected_document_accepted": (
                any(h["document_id"] == expected for h in accepted)
                if expected else ""
            ),
        })

        print(
            f"{question}\n"
            f"  Rang attendu : {rank if expected else 'hors domaine'}"
            f" | Passages acceptés : {len(accepted)}",
            flush=True,
        )

    mean = lambda values: sum(values) / len(values)

    result = {
        "method": mode,
        "hit_at_1": mean(hits_at_1),
        "recall_at_5": mean(recalls_at_5),
        "mrr_at_5": mean(reciprocal_ranks),
        "out_of_domain_rejection_rate": mean(rejected_outside),
    }
    summary.append(result)

out = ROOT / "artifacts"
out.mkdir(exist_ok=True)

with (out / "retrieval_comparison.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as f:
    writer = csv.DictWriter(f, fieldnames=list(details[0]))
    writer.writeheader()
    writer.writerows(details)

report = {
    "scope": "Test de contrôle sur 6 questions couvertes et 2 hors domaine.",
    "limitations": (
        "Questions connues pendant le développement ; "
        "un document de référence par question ; "
        "pertinence des autres documents non annotée."
    ),
    "corpus_hash": semantic.hash,
    "results": summary,
}

(out / "retrieval_comparison.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n=== BILAN ===")
for result in summary:
    print(
        f"{result['method']:8s}"
        f" | Hit@1={result['hit_at_1']:.3f}"
        f" | Recall@5={result['recall_at_5']:.3f}"
        f" | MRR@5={result['mrr_at_5']:.3f}"
        f" | Rejet hors domaine="
        f"{result['out_of_domain_rejection_rate']:.0%}"
    )

print("\nRapports enregistrés dans artifacts/retrieval_comparison.*")
