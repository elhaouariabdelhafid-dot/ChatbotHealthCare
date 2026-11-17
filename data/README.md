# Données incluses

- `demo_corpus.jsonl` : six synthèses originales en français, réparties sur trois sources institutionnelles. `approved: true` signifie admis dans la démonstration, **pas approuvé cliniquement**. Les sources officielles sont liées dans chaque entrée.
- `intents.csv` : 48 questions pédagogiques rédigées pour montrer l’entraînement. Elles ne proviennent pas de patients.
- `evaluation.jsonl` : 16 questions de régression. Ce jeu est connu pendant le développement et ne constitue pas un test indépendant.

## Convention d’annotation NER à étendre

Types : MALADIE, SYMPTOME, MEDICAMENT, EXAMEN. Conserver des offsets [start, end) dans le texte original. Distinguer négation, hypothèse et antécédent familial dans les futures annotations. Une mention n’implique jamais la présence d’une maladie.

## Provenance

Synthèses et exemples construits le 15 novembre 2026. Aucun dossier de patient réel n’est inclus. Les corpus externes restent dans `data/external/`, ignoré par Git. Ne jamais marquer un corpus entier comme approuvé sans contrôle de qualité et de conditions d’usage.
