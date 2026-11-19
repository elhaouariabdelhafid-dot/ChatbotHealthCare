"""Interface locale : python -m streamlit run streamlit_app.py.

Le backend FastAPI garde les modèles et le contrôle de débit en mémoire.
HEALTHBOT_API_URL vaut http://127.0.0.1:8000 par défaut.
Aucune clé Groq n'est lue ni conservée par cette interface.
"""
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import streamlit as st

API_URL = os.environ.get("HEALTHBOT_API_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_TURNS = 20


def call_backend(path, body=None, timeout=90):
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        API_URL + path,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        if not isinstance(result, dict):
            raise ValueError("Réponse du backend invalide.")
        return result
    except HTTPError as exc:
        # Ne pas afficher le corps brut : il peut contenir des détails internes.
        if exc.code == 429:
            raise RuntimeError("Service occupé. Réessaie un peu plus tard.") from None
        if exc.code == 422:
            raise RuntimeError("Question refusée : vérifie sa longueur et son contenu.") from None
        raise RuntimeError(f"Le backend a renvoyé une erreur HTTP {exc.code}.") from None
    except (TimeoutError, URLError):
        raise RuntimeError(
            "Backend inaccessible ou délai dépassé. Vérifie le terminal FastAPI. "
            "La demande peut encore être en cours ; évite de la relancer immédiatement."
        ) from None
    except (ValueError, UnicodeError):
        raise RuntimeError("Le backend a renvoyé une réponse illisible.") from None


def display_result(result, elapsed=None):
    mode = result.get("mode", "inconnu")
    if result.get("status") == "degraded" or mode == "extractive":
        st.warning("Réponse de secours : les passages originaux sont affichés, sans synthèse générée.")
    elif result.get("status") == "abstain":
        st.info("Les informations disponibles ne permettent pas de répondre.")
    st.markdown(str(result.get("answer", "Aucune réponse reçue.")))
    if result.get("warning"):
        st.warning(str(result["warning"]))
    trace = result.get("generation_trace") or {}
    retry = trace.get("retry_after_seconds")
    if isinstance(retry, (int, float)) and retry > 0:
        st.caption(f"Délai indiqué par le backend : environ {retry:.0f} secondes. Aucun nouvel appel automatique.")
    if elapsed is not None:
        st.caption(f"Temps de réponse de l’interface : {elapsed:.2f} s · Mode : {mode}")

    sources = result.get("sources") or []
    if sources:
        with st.expander(f"Consulter les sources ({len(sources)})"):
            st.caption("Passages sélectionnés pour la réponse. Les références entre crochets indiquent ceux cités dans le texte.")
            for source in sources:
                st.write(f"[{source.get('citation', '?')}] {source.get('title', 'Document')}")
                url = source.get("url", "")
                if isinstance(url, str) and urlparse(url).scheme in {"http", "https"}:
                    st.link_button("Ouvrir la source", url)
                st.text(str(source.get("text", "")))
                st.caption(f"Langue : {source.get('language', 'inconnue')} · Date de collecte : {source.get('retrieved_at', 'inconnue')}")
                st.divider()

    with st.expander("Détails du test"):
        # Liste explicite : aucun en-tête HTTP ni secret de configuration.
        st.json({
            "trace_id": result.get("trace_id"),
            "status": result.get("status"),
            "mode": mode,
            "latency_ms": result.get("latency_ms"),
            "intent": result.get("intent"),
            "entities": result.get("entities", []),
            "retrieval": result.get("retrieval"),
            "generation_trace": trace,
        })


st.set_page_config(page_title="ChatBotHealthCare", page_icon="💬", layout="centered")
st.title("ChatBotHealthCare")
st.caption("Des réponses documentées à vos questions générales sur la santé.")
st.info("Prototype académique : information générale, sans diagnostic ni prescription. Ne saisis pas de données médicales personnelles.")

if "turns" not in st.session_state:
    st.session_state.turns = []

with st.sidebar:
    st.header("Votre session")
    st.caption("Chaque question est traitée indépendamment. L’historique affiché n’est pas transmis au modèle.")
    st.caption("Les questions et les passages sélectionnés peuvent être envoyés au fournisseur de génération configuré.")
    if st.button("Vider la conversation", use_container_width=True):
        st.session_state.turns = []
        st.rerun()
    if st.button("Vérifier le backend", use_container_width=True):
        try:
            with st.spinner("Connexion et chargement initial des modèles…"):
                info = call_backend("/api/info", timeout=180)
            st.success("Backend prêt")
            st.json(info)
        except RuntimeError as exc:
            st.error(str(exc))
    st.caption("Le premier chargement peut être plus long. Le backend doit rester ouvert pendant les tests.")

for turn in st.session_state.turns:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        if "error" in turn:
            st.error(turn["error"])
        else:
            display_result(turn["result"], turn["elapsed"])

question = st.chat_input("Pose une question générale sur la santé…", max_chars=1200)
if question and question.strip():
    question = question.strip()
    with st.chat_message("user"):
        st.markdown(question)
    turn = {"question": question}
    with st.chat_message("assistant"):
        start = time.perf_counter()
        try:
            with st.spinner("Recherche des passages et préparation de la réponse…"):
                result = call_backend("/api/chat", {"question": question})
            turn.update(result=result, elapsed=time.perf_counter() - start)
            display_result(result, turn["elapsed"])
        except RuntimeError as exc:
            turn["error"] = str(exc)
            st.error(str(exc))
    st.session_state.turns.append(turn)
    st.session_state.turns = st.session_state.turns[-MAX_TURNS:]
