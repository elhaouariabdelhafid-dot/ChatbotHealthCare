# Validation de la livraison — 15 novembre 2026

## Vérifications exécutées

- 26 tests unitaires et API : réussis (`artifacts/tests.txt`).
- Compilation des modules Python et contrôle syntaxique JavaScript : réussis.
- API testée avec FastAPI TestClient : question générale, erreurs de saisie, sources, routes de sécurité et ressources statiques.
- Import MedQuAD vérifié sur XML miniature : réponse vide exclue, provenance conservée et revue requise.
- Import MedDialog vérifié sur fixture : masquage de contact et statut de revue.
- Ollama simulé : réponse valide, abstention, JSON invalide, référence invalide et indisponibilité.
- Recherche dense/hybride testée avec encodeur factice pour vérifier le câblage ; aucune qualité sémantique n’en est déduite.

## Mesures réellement obtenues

Configuration : recherche lexicale et réponse extractive ; 16 questions, dont 8 avec documents pertinents annotés.

| Mesure | Résultat |
|---|---:|
| Recall@5 avant filtrage | 1.000 |
| MRR@10 avant filtrage | 1.000 |
| Statuts conformes aux attentes | 100.0% |
| Latence p50, moteur chaud, machine de réalisation | 0.88 ms |
| Latence p95, moteur chaud, machine de réalisation | 1.46 ms |
| Macro-F1 intentions sur 12 exemples de validation | 0.262 |

Ces scores portent sur six petits documents et un jeu connu pendant le développement. Le Recall@5 est particulièrement facile avec seulement six documents. La latence ne comprend ni démarrage, ni téléchargement, ni génération LLM ; elle ne prédit pas les performances sur le PC de l’utilisateur.

Le classifieur est entraîné sur 36 exemples et évalué sur 12 pour ce rapport. Sa macro-F1 faible montre que le jeu est insuffisant. Le modèle de l’application utilise les 48 exemples au lancement, sans prétendre améliorer la qualité mesurée. L’intention n’est pas utilisée pour les décisions de sécurité ni pour bloquer la recherche.

## Limites de vérification

Les poids E5 et Qwen n’ont pas été téléchargés, et aucun serveur Ollama réel n’a été utilisé pendant la réalisation. La disponibilité de tous les fichiers MedDialog et l’import complet de MedQuAD n’ont pas été testés. Les tests réseau Ollama sont simulés.

Une vérification visuelle automatisée a été tentée, mais le navigateur local n’a pas pu démarrer à cause d’une restriction système de socket. Il n’y a donc pas de validation visuelle dans un navigateur réel. Les fichiers HTML/CSS/JS sont fournis ; les ressources HTTP et la syntaxe JS ont été vérifiées.

Aucune validation clinique, multilingue ou en production n’a été réalisée. Une revue indépendante du contenu, des réponses et des échecs reste nécessaire avant tout usage hors démonstration.

## Environnement

Python 3.12.14, scikit-learn 1.8.0, NumPy 2.3.5, FastAPI 0.141.1, HTTPX 0.28.1. Un avertissement de dépréciation Starlette/HTTPX est affiché par le client de test ; il ne provoque pas d’échec des tests.
