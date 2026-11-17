# Démarrer ChatBotHealthCare

1. Lire `README.md`, section 1 : choisir Windows ou Linux/WSL.
2. Créer l’environnement virtuel et installer `requirements.txt`.
3. Lancer `python -m healthbot.cli build`.
4. Lancer `python -m uvicorn healthbot.api:app --host 127.0.0.1 --port 8000`.
5. Ouvrir http://127.0.0.1:8000.

Le démarrage par défaut utilise 6 fiches et affiche des extraits sourcés. Pour la génération avec un LLM, suivre la section Ollama du README. Pour les données plus volumineuses, suivre les imports MedQuAD/MedDialog.

Le code est fonctionnel pour une démonstration académique. Lire `docs/VALIDATION.md` pour les tests exécutés et les limites réelles.
