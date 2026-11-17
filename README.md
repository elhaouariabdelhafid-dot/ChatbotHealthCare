# ChatBotHealthCare

Assistant documentaire de santé • projet académique NLP + RAG • version 1.0

**Application locale fonctionnelle en français**, livrée avec son interface web, API FastAPI, recherche TF-IDF, classification d’intentions, extraction d’entités par lexique, règles d’orientation, connecteur Ollama, recherche E5 optionnelle, imports de données et tests.

Le démarrage par défaut ne télécharge aucun modèle : il affiche des passages sourcés. La génération RAG avec un LLM s’active après installation d’Ollama. Le corpus fourni est volontairement petit : **6 synthèses pédagogiques françaises sur le diabète, l’hypertension et l’asthme**. Les grands corpus et les poids de modèles ne sont pas dans l’archive.

## 1. Lancer en cinq minutes

Décompresser l’archive, puis ouvrir un terminal **dans le dossier ChatBotHealthCare**. Utiliser Python **3.11 ou 3.12**. Les commandes ci-dessous supposent Python 3.12 installé.

### Linux / WSL

```bash
cd ChatBotHealthCare
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m healthbot.cli build
python -m uvicorn healthbot.api:app --host 127.0.0.1 --port 8000
```

### Windows PowerShell

L’activation n’est pas nécessaire avec ces commandes ; elles fonctionnent aussi quand PowerShell bloque Activate.ps1.

```powershell
cd ChatBotHealthCare
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m healthbot.cli build
.\.venv\Scripts\python.exe -m uvicorn healthbot.api:app --host 127.0.0.1 --port 8000
```

Ouvrir **http://127.0.0.1:8000**. La documentation API est sur **http://127.0.0.1:8000/docs**. Garder le terminal ouvert. Arrêter avec Ctrl+C. Sur Windows, remplacer `python` dans les commandes suivantes par `.\.venv\Scripts\python.exe` si l’environnement n’est pas activé.

Exemples :

- « Qu’est-ce que le diabète ? »
- « Quels facteurs favorisent l’hypertension ? »
- « Quels sont les symptômes de l’asthme ? »
- « Quels examens pour la maladie de Lyme ? » → abstention sur le corpus initial.

L’installation des dépendances exige Internet. Après installation, le mode lexical/extractif fonctionne hors ligne. Les liens vers les sources nécessitent Internet pour leur consultation.

## 2. Ce qui est réellement réalisé

| Composant | Implémentation | État |
|---|---|---|
| Interface | HTML/CSS/JavaScript responsive, chat, sources, analyse NLP | Servie par FastAPI ; endpoints et ressources testés |
| Recherche | TF-IDF mots et bigrammes, cosinus, filtrage des mots génériques | Testée sur le corpus inclus |
| Recherche sémantique | E5-small CPU, vecteurs normalisés, cache NumPy contrôlé par empreinte | Code fourni ; poids non téléchargés ici |
| Recherche hybride | Fusion réciproque des rangs (RRF) | Code fourni ; intégration E5 à tester localement |
| Classification | TF-IDF + régression logistique, entraînement au lancement | 48 exemples pédagogiques ; qualité insuffisante pour une utilisation réelle |
| NER | Lexique de quatre types, offsets et indice de négation | Testée ; pas un modèle biomédical NER entraîné |
| Génération | Ollama `/api/chat`, prompt documenté, contrôle des citations | Connecteur et erreurs testés par simulation ; LLM réel non exécuté ici |
| Réponse de secours | Passages exacts du corpus si Ollama échoue | Testée |
| Orientation | Règles urgence, demande individuelle, langue non prise en charge | Cas de régression testés ; couverture limitée |
| Données | Import MedQuAD XML ; adaptateur MedDialog JSONL | Testés sur petites fixtures, pas sur la totalité des corpus |
| Évaluation | Recherche, statuts, latence et classification | Rapports réels dans `artifacts/` |

**Ce n’est pas un système clinique validé.** Les citations sont contrôlées syntaxiquement, mais leur présence ne garantit pas que chaque affirmation générée soit médicalement correcte. Les règles d’urgence peuvent manquer des cas ou produire des fausses alertes. Le français est le périmètre évalué ; arabe et darija ne sont pas implémentés comme langues de réponse validées.

## 3. Activer la génération RAG avec Ollama

Installer Ollama depuis sa documentation officielle : https://docs.ollama.com/quickstart
Puis, dans un terminal :

```bash
ollama pull qwen2.5:1.5b
ollama serve
```

Si le service tourne déjà, ne pas démarrer un deuxième `ollama serve`. Dans `config.json`, changer seulement :

```json
"generation": "ollama"
```

Redémarrer le serveur Python. L’application contacte `http://127.0.0.1:11434`. Sans réponse valide, elle revient automatiquement aux extraits avec une indication visible. Le modèle reçoit la question courante et les passages récupérés ; **l’historique affiché n’est pas utilisé comme mémoire conversationnelle**.

Le modèle 1,5B est un candidat léger, pas un modèle médical validé. Le choix d’un plus grand modèle doit être conditionné à la RAM disponible et à une évaluation humaine des réponses.

## 4. Activer E5 et la recherche hybride

Installation CPU, sans téléchargement volontaire des roues CUDA :

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-semantic.txt
```

Dans `config.json`, mettre `"retriever": "hybrid"` (ou `"dense"`). Puis :

```bash
python -m healthbot.cli build --fresh
python -m healthbot.cli evaluate --output artifacts/evaluation_hybrid.json
```

Le premier lancement télécharge `intfloat/multilingual-e5-small`. Le modèle et ses fichiers doivent rester en cache pour un usage hors ligne. Les préfixes `query:` et `passage:` sont appliqués automatiquement. L’encodeur tronque à 512 tokens ; réduire `chunk_words` pour les textes dont la tokenisation est longue. La recherche dense utilise NumPy : adaptée à un petit corpus, sans dépendance FAISS supplémentaire.

Les seuils 0,18 lexical et 0,80 dense sont des **valeurs initiales heuristiques**. Ils ne sont pas calibrés médicalement. Ajuster sur un jeu de validation indépendant, puis figer les paramètres avant l’évaluation finale. Les scores RRF ne sont pas comparés directement aux seuils cosinus.

## 5. Importer MedQuAD

Récupérer le dépôt officiel après examen de ses conditions :

```bash
git clone https://github.com/abachaa/MedQuAD.git data/external/MedQuAD
python -m healthbot.cli import-medquad data/external/MedQuAD data/external/medquad_pending.jsonl
```

L’importateur supprime les réponses vides et les doublons, conserve le type de question et la provenance. Il n’essaie pas de reconstruire les réponses supprimées pour des raisons de droits. Les entrées ont `approved: false`.

**Procédure de revue :** sélectionner les thèmes, contrôler les droits et l’actualité du contenu, vérifier le lien original, examiner les informations sensibles, puis créer `data/corpus_reviewed.jsonl` à partir des seules entrées retenues et passer leur champ `approved` à `true`. Le chargeur exige des URLs HTTPS : pour une ancienne URL HTTP, vérifier son URL HTTPS ou sa nouvelle destination avant de la remplacer. Ne pas modifier toutes les URLs en masse sans vérification.

Mettre `"corpus": "data/corpus_reviewed.jsonl"` dans la configuration, reconstruire l’index et redémarrer. MedQuAD est anglophone : le mode lexical ne réalise pas une recherche interlangue. Pour des questions françaises sur ce corpus, évaluer E5 et/ou ajouter des traductions relues en conservant la provenance. Le modèle génératif est configuré pour répondre en français.

Schéma d’une ligne JSONL :

```json
{"id":"unique_id","title":"Titre","text":"Contenu documentaire relu","source_url":"https://source.example/document","language":"fr","license_status":"Conditions vérifiées et attribution","approved":true}
```

Ne pas utiliser l’URL d’exemple comme vraie source.

## 6. Utiliser MedDialog

MedDialog n’est pas ajouté automatiquement au corpus de preuves. Il sert à étudier les formulations conversationnelles et à préparer des annotations. Selon la distribution que tu obtiens légalement, convertir ses enregistrements vers ce format JSONL :

```json
{"id":"conversation_1","patient":"Question anonymisée","doctor":"Réponse à revoir","source_url":"https://source.example/conversation","language":"en"}
```

Puis :

```bash
python -m healthbot.cli import-meddialog data/external/meddialog_input.jsonl data/external/meddialog_review.jsonl
```

L’adaptateur masque certains emails et numéros. **Ce n’est pas une anonymisation complète** : noms, lieux, dates et descriptions identifiantes nécessitent une revue humaine. Le résultat n’est pas compatible avec l’index documentaire par conception. Le format brut des différentes versions MedDialog n’est pas pris en charge universellement ; l’étape de normalisation ci-dessus est nécessaire.

## 7. Classification et NER

`data/intents.csv` contient `text,label`. Six classes : définition, symptômes, prévention, causes, examens, hors domaine. Remplacer/compléter ce fichier par des questions annotées et relues. L’application entraîne le petit classifieur à son lancement ; aucun pickle externe n’est chargé.

```bash
python -m healthbot.cli train-intent
```

Cette commande entraîne et évalue une partition 75/25 stratifiée et écrit le rapport. La validation fournie est trop petite pour sélectionner un modèle définitif. Les règles de sécurité sont indépendantes du classifieur. La NER reconnaît seulement les entrées explicites de `LEXICON` dans `healthbot/nlp.py` ; une entité reconnue n’est pas une pathologie diagnostiquée. L’attribut `negation_candidate` est une heuristique de proximité, pas une compréhension complète de la phrase.

## 8. Vérifier et reproduire

```bash
python -m unittest discover -s tests -v
python -m healthbot.cli build
python -m healthbot.cli evaluate
python -m healthbot.cli train-intent
python -m healthbot.cli ask "Que signifie le diabète ?"
```

Rapports : `artifacts/evaluation.json`, `artifacts/intent_evaluation.json` et `docs/VALIDATION.md`. Le jeu fourni est un **jeu de régression pédagogique**, rédigé avec le corpus. Ne pas annoncer ses scores comme une évaluation médicale indépendante.

L’API accepte `POST /api/chat` avec `{"question":"..."}`. La réponse contient `answer`, `status`, `mode`, `sources`, `intent`, `entities`, `trace_id`, `corpus_hash` et `latency_ms`. Statuts : `answered`, `abstain`, `clarify`, `personal`, `safety`, `unsupported`.

## 9. Organisation

| Chemin | Rôle |
|---|---|
| `healthbot/corpus.py` | Validation, découpage, empreinte |
| `healthbot/nlp.py` | Classifieur, lexique, règles |
| `healthbot/retrieval.py` | Recherche lexicale, dense et hybride |
| `healthbot/generation.py` | Ollama, citations et retour aux extraits |
| `healthbot/engine.py` | Orchestration et statuts |
| `healthbot/api.py` | API et interface locale |
| `healthbot/cli.py` | Import, indexation, entraînement et évaluation |
| `web/` | Interface sans CDN ni service distant |
| `data/` | Corpus, annotations et questions de régression |
| `tests/` | Tests unitaires et API |
| `docs/` | Architecture, validation et dossier de conception |

## 10. Dépannage

- **Module introuvable** : lancer depuis la racine du projet avec le Python de `.venv`.
- **Port occupé** : utiliser `--port 8001`, puis ouvrir le même port dans le navigateur.
- **Aucune réponse Ollama** : vérifier le service et `ollama list` ; l’extraction reste disponible.
- **Erreur E5 / Internet** : revenir à `"retriever": "lexical"` et redémarrer.
- **Aucun document approuvé** : vérifier le chemin du corpus, son format et la revue des entrées.
- **Modification non prise en compte** : redémarrer le serveur ; le moteur est mis en cache en mémoire.
- **Réponse faible** : inspecter les passages, améliorer le corpus, tester les seuils sur validation. Ne pas augmenter simplement la confiance affichée.

Le serveur est destiné à un usage local et ne fournit pas d’authentification. Les messages ne sont pas écrits sur disque par l’application. Garder l’écoute sur `127.0.0.1` ; une mise en service publique demanderait un travail distinct sur authentification, confidentialité, disponibilité et validation clinique.

## 11. Git

```bash
git init
git add .
git commit -m "feat: add local healthcare NLP and RAG prototype"
# Ajouter ton propre dépôt distant, puis :
git branch -M main
git remote add origin URL_DE_TON_DEPOT
git push -u origin main
```

Les données externes et l’environnement virtuel sont ignorés. Avant chaque commit suivant : `git status`, `git diff`, tests, `git add` des fichiers voulus, `git commit`, `git push`. Ne pas versionner des conversations de patients.

## Sources techniques et documentaires

- MedQuAD : https://github.com/abachaa/MedQuAD
- MedDialog : https://aclanthology.org/2020.emnlp-main.743/
- E5-small : https://huggingface.co/intfloat/multilingual-e5-small
- Qwen : https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- API Ollama : https://docs.ollama.com/api/chat
- Fiches OMS : https://www.who.int/news-room/fact-sheets/detail/diabetes ; https://www.who.int/news-room/fact-sheets/detail/hypertension ; https://www.who.int/news-room/fact-sheets/detail/asthma

Les synthèses incluses ont été rédigées pour cette démonstration, avec ces sources consultées le 15 novembre 2026. Elles ne sont ni les textes officiels, ni une traduction officielle, ni un corpus validé par un médecin.
