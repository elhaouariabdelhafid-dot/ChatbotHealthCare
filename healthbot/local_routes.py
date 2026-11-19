"""Routes déterministes limitées ; aucune donnée personnelle déduite du corpus."""
import re
import unicodedata

def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))

def local_reply(question):
    q = normalize(question)
    greeting = re.sub(r"[^a-z ]", "", q).strip()
    if greeting in {"bnj", "bjr", "bonjour", "bonsoir", "salut", "hello", "hi", "coucou", "merci", "merci beaucoup"}:
        return {"status": "answered", "mode": "local", "answer": "Bonjour ! Je peux expliquer des termes de santé à partir des documents disponibles. Quelle est ta question ?"}
    personal = bool(re.search(r"\b(ma|mon|mes|my|moi|ai[- ]je|did i|was i)\b", q))
    record = bool(re.search(r"\b(dossier|compte rendu|rendez.vous|record|report|appointment)\b", q))
    measure = bool(re.search(r"\b(glycemie|tension|hemoglobine|groupe sanguin|irm|vaccin|allergie|blood|glucose|hemoglobin|vaccine|vaccination|allergy|scan)\b", q))
    past = bool(re.search(r"\b(dernier|derniere|derniers|dernieres|last|latest|recent|recentes|recently|mesuree|mesure|recu|recue|received|indique|inscrit|inscrite|etait|was|resultat|result|valeur|value)\b", q))
    if personal and (record or (measure and past)):
        return {"status": "abstain", "mode": "local", "answer": "Je n’ai pas accès à ton dossier médical ni à tes résultats personnels. Je ne peux donc pas retrouver cette information. Consulte ton compte rendu ou le professionnel qui te suit. Tu peux me demander une explication générale d’un terme médical."}
    return None
