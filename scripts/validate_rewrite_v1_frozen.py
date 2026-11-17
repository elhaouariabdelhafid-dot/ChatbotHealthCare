import csv
import re
from pathlib import Path

source = Path("artifacts/rewrite_v2_20261116_170843.csv")

def validate(original, rewritten):
    original = original.strip().lower()
    rewritten = rewritten.strip().lower()

    if not rewritten:
        return False, "Reformulation vide"

    # Vérifier une question demandant une maladie précise
    if re.match(r"^quelle maladie\b", original):
        if not re.match(r"^(which|what) disease\b", rewritten):
            return False, "Type de réponse changé : maladie"

    # Vérifier les questions demandant un syndrome
    if re.match(r"^quel syndrome\b", original):
        if not re.match(r"^(which|what) syndrome\b", rewritten):
            return False, "Type de réponse changé : syndrome"

    return True, "Aucune anomalie détectée par les règles"

with source.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    original = row["original"]
    rewritten = row["nouvelle_reformulation"]

    valid, reason = validate(original, rewritten)
    effective_query = rewritten if valid else original

    print(f"\n{row['id']} : {'PASS' if valid else 'FALLBACK'}")
    print("Raison :", reason)

    if not valid:
        print("Question utilisée :", effective_query)

print("\nVALIDATION TERMINÉE")
print("Attention : validation partielle, pas une garantie médicale.")
