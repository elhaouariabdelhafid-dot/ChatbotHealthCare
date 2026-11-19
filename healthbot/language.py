"""FR/EN routing without an extra API call; ambiguous short input defaults to FR."""
import re
import unicodedata

FR = set("bonjour bonsoir salut bnj bjr merci quoi quel quelle quels quelles que est ce c pourquoi comment peut peux sont les des une un le la du de mon ma mes dans selon signifie symptomes causes derniere dernier etait et ou".split())
EN = set("hello hi thanks thank what which how why when where does do is are can could would should the a an of for my in according mean means symptoms causes last latest was and or".split())

def question_language(question):
    text = ''.join(c for c in unicodedata.normalize('NFKD', question.casefold()) if not unicodedata.combining(c))
    words = re.findall(r"[a-z]+", text)
    # Shared medical terms alone cannot determine a language.
    fr = sum(w in FR - EN for w in words)
    en = sum(w in EN - FR for w in words)
    return 'en' if en > fr else 'fr'

TRANSLATIONS = {
    "Bonjour ! Je peux expliquer des termes de santé à partir des documents disponibles. Quelle est ta question ?": "Hello! I can explain health terms using the available documents. What would you like to know?",
    "Je n’ai pas accès à ton dossier médical ni à tes résultats personnels. Je ne peux donc pas retrouver cette information. Consulte ton compte rendu ou le professionnel qui te suit. Tu peux me demander une explication générale d’un terme médical.": "I do not have access to your medical records or personal test results, so I cannot retrieve this information. Please check your report or contact your healthcare professional. You can ask me for a general explanation of a medical term.",
    "Les passages disponibles ne suffisent pas pour répondre à cette question.": "The available passages do not provide enough information to answer this question.",
    "Je ne trouve pas de passage suffisamment pertinent dans le corpus chargé. Précise la question ou consulte une source adaptée.": "I could not find a sufficiently relevant passage in the loaded documents. Please clarify your question or consult an appropriate source.",
    "Peux-tu préciser le sujet de santé ? Par exemple : définition du diabète ou prévention de l’hypertension.": "Could you specify the health topic? For example: the definition of diabetes or hypertension prevention.",
    "Je ne peux pas déterminer un diagnostic ni recommander une dose ou un traitement individuel. Adresse cette demande à un professionnel de santé. Tu peux poser une question générale sur une maladie ou un terme médical.": "I cannot diagnose you or recommend an individual dose or treatment. Please contact a healthcare professional. You can ask a general question about a condition or medical term.",
    "Ce message peut décrire une urgence. Contacte immédiatement les secours locaux ou rends-toi aux urgences. Ne retarde pas la recherche d'aide pour poursuivre cette conversation. Cet outil ne peut pas évaluer ta situation.": "This message may describe an emergency. Contact local emergency services immediately or go to an emergency department. Do not delay getting help to continue this conversation. This tool cannot assess your situation.",
}

def localize(text, language):
    return TRANSLATIONS.get(text, text) if language == 'en' else text
