# Intégration Groq — prototype mono-processus

Configuration séparée config.groq.json : E5 dense, reranking 5 candidats,
2 passages, ancien prompt + précisions de fidélité V3, GPT-OSS 120B.
La qualité V3 n'est pas encore validée par des appels réels. Le JSON structuré
et les citations syntaxiquement valides ne garantissent pas la fidélité.

Clé : GROQ_API_KEY dans l'environnement du backend seulement. Aucun appel
cloud n'est fait à l'installation ou par les tests unitaires.

## États
- answered + mode groq : réponse générée contrôlée syntaxiquement.
- abstain : contexte jugé insuffisant ; ne pas le contourner automatiquement.
- degraded + mode extractive : génération impossible ou résultat rejeté.
  Les sources restent consultables ; ne pas présenter les extraits comme
  une synthèse française ou comme une validation clinique.
- generation_trace.status précise missing_key, rate_limited, cooldown,
  local_rate_limit, busy, timeout, network_error, provider_unavailable,
  access_denied, billing_required, request_rejected, invalid_response,
  incomplete_response, invalid_citations ou content_rejected.
- retry_after_seconds est un délai conseillé, pas une garantie de disponibilité.

## Capacité et coût
Un seul processus serveur (--workers 1), une seule instance backend.
Admission atomique à l'échelle du processus : 2 appels concurrents maximum,
30 requêtes/minute et 8000 tokens/minute avec marge de 10%, configurables.
Réservation estimée à partir de la taille UTF-8 du prompt + plafond de sortie ;
estimation conservatrice, pas un tokenizer exact. Réconciliation sur total_tokens
après succès. Les échecs gardent leur réservation durant la fenêtre de 60 s.
Un prompt très long peut être refusé localement : vérifier reserved_tokens_estimate.
Les autres applications de la même organisation ne sont pas connues du limiteur.
Le 429 distant demeure donc possible. Respect de Retry-After, sinon pause de 60 s.
503 : pause commune de 30 s. Pas de répétition automatique ni de long sleep.
Le modèle local n'est jamais lancé implicitement en secours.

Pas de file d'attente : si aucune capacité n'est disponible, retour documentaire
immédiat. Pas encore de cache partagé, budget journalier en euros, authentification,
ni limite par utilisateur. Les limites financières sont à configurer chez le
fournisseur. Le délai HTTP de 20 s est un timeout réseau, pas un SLA global.
Après redémarrage, les compteurs locaux repartent à zéro ; ceux du fournisseur non.
Pour plusieurs processus/serveurs, remplacer cette admission par un service commun
ou un stockage atomique partagé avant de multiplier les workers.

## Tester
python -m unittest discover -s tests -v
python -m healthbot.cli --config config.groq.json ask "Qu’est-ce que le diabète ?"
Ne pas lancer plusieurs CLI pour tester le partage du quota : chaque CLI est un
processus différent. Le test de concurrence unitaire vérifie l'admission commune.

## Activation API pour le test manuel puis Streamlit
L'API actuelle lit config.json. Pour l'activer, sauvegarder config.json puis
copier config.groq.json vers config.json. L'installeur ne fait pas cette activation.
uvicorn healthbot.api:app --host 127.0.0.1 --port 8000 --workers 1
Ne pas utiliser --reload pour les mesures, ni plusieurs backends avec ce limiteur.
Streamlit sera un client de /api/chat, sans clé cloud et sans modèle chargé.
La gestion du dossier personnel (M04) reste celle d'origine et doit encore être
améliorée avant présentation finale. Aucun nouveau support arabe/darija ajouté.

## Rollback
L'installeur crée artifacts/backup_groq_DATE avec les copies originales et un
manifest.json. Restaurer ces copies pour annuler les modifications ; les nouveaux
fichiers sont listés dans le manifeste. Conserver les rapports d'évaluation.
