import csv
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

STOP = set('le la les un une des du de d l au aux et ou a en pour par dans sur est sont ce ces cette cet que qui quoi comment quels quelles quel quelle qu il elle je j me moi tu vous nous mon ma mes son sa ses ne pas plus tres avec sans pourquoi peut on faire general generale information informations expliquer explique definition symptomes prevention causes maladie maladies medical medicale medicaux terme termes examen examens role principe'.split())
def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFD',text.lower()) if not unicodedata.combining(c)).replace('’',"'")
def keywords(text):
    return {w for w in re.findall(r'[a-z]{2,}',normalize(text)) if w not in STOP}

# Règles de démonstration, pas un triage médical validé.
EMERGENCY = [r"(?:je |j'|il |elle |mon |ma ).{0,40}(?:ne respire plus|n'arrive plus a respirer|peux plus respirer|douleur thoracique|douleur dans la poitrine)",
             r'\b(?:inconscient|inconsciente|perte de connaissance|hemorragie|overdose)\b',
             r"\b(?:i cannot breathe|i can't breathe|chest pain now|unconscious)\b",
             r'(?:ما ?قادرش نتنفس|لا استطيع التنفس|لا أستطيع التنفس)']
PERSONAL = [r'\b(?:posologie|dosage|dose|doses|comprime|comprimes|prescris|prescription)\b',
            r'\b(?:dois.je|puis.je|je dois|je peux).{0,35}(?:prendre|arreter|augmenter|diminuer)\b',
            r'\b(?:quel|quels|quelle).{0,25}(?:medicament|traitement).{0,25}(?:moi|prendre|mon|ma)\b',
            r"\b(?:ai.je|suis.je|j'ai|je souffre|mon enfant|ma mere|mon pere|mes symptomes)\b",
            r'\b(?:should i take|what dose|do i have|i have symptoms)\b']
def safety_route(text):
    t=normalize(text)
    if any(re.search(p,t) for p in EMERGENCY):
        return 'safety'
    if any(re.search(p,t) for p in PERSONAL):
        return 'personal'
    return None

LEXICON = {'MALADIE':['diabète','hypertension','asthme'], 'SYMPTOME':['toux','fatigue','soif','douleur thoracique','fièvre'], 'MEDICAMENT':['insuline','metformine','paracétamol'], 'EXAMEN':['glycémie','prise de sang','pression artérielle']}
def entities(text):
    # Recherche Unicode insensible à la casse sur le texte original : offsets conservés.
    found=[]
    for label,terms in LEXICON.items():
        for term in terms:
            for m in re.finditer(r'(?<!\w)'+re.escape(term)+r'(?!\w)',text,re.I):
                left=normalize(text[max(0,m.start()-35):m.start()])
                found.append({'text':m.group(),'label':label,'start':m.start(),'end':m.end(),
                    'negation_candidate':bool(re.search(r"\b(?:pas de|sans|aucun|aucune)\s*$",left))})
    return sorted(found,key=lambda x:x['start'])

class IntentModel:
    def __init__(self,path):
        with open(path,encoding='utf-8',newline='') as f:
            rows=list(csv.DictReader(f))
        self.pipeline=make_pipeline(TfidfVectorizer(strip_accents='unicode',ngram_range=(1,2)),LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42))
        self.pipeline.fit([r['text'] for r in rows],[r['label'] for r in rows])
        self.training_count=len(rows)
    def predict(self,text):
        probabilities=self.pipeline.predict_proba([text])[0]
        i=int(probabilities.argmax())
        return {'label':str(self.pipeline.classes_[i]),'score':round(float(probabilities[i]),3),'method':'TF-IDF + régression logistique (jeu pédagogique)'}
