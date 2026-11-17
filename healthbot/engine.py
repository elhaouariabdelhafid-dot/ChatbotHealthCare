import re
import time
import uuid
from .settings import settings
from .nlp import safety_route,entities,IntentModel,keywords
from .retrieval import Retriever
from .generation import generate

class HealthBot:
    def __init__(self,cfg=None):
        self.cfg=cfg or settings()
        self.retriever=Retriever(self.cfg)
        self.intent=IntentModel(self.cfg['intents'])

    def ask(self,question):
        if not isinstance(question,str) or not question.strip():
            raise ValueError('Écris une question non vide.')
        question=question.strip()
        if len(question)>self.cfg['max_question_chars']:
            raise ValueError('Question trop longue.')
        start=time.perf_counter()
        result={'trace_id':uuid.uuid4().hex,'answer':'','status':'abstain','mode':'none','sources':[],
                'intent':self.intent.predict(question),'entities':entities(question),'corpus_hash':self.retriever.hash,
                'notice':'Prototype académique : information générale, sans diagnostic ni prescription. Les règles ne détectent pas toutes les situations à risque.'}
        route=safety_route(question)
        if route=='safety':
            result.update(status='safety',answer="Ce message peut décrire une urgence. Contacte immédiatement les secours locaux ou rends-toi aux urgences. Ne retarde pas la recherche d'aide pour poursuivre cette conversation. Cet outil ne peut pas évaluer ta situation.")
        elif route=='personal':
            result.update(status='personal',answer="Je ne peux pas déterminer un diagnostic ni recommander une dose ou un traitement individuel. Adresse cette demande à un professionnel de santé. Tu peux poser une question générale sur une maladie ou un terme médical.")
        elif re.search(r'[\u0600-\u06ff]',question):
            result.update(status='unsupported',answer="Cette version est évaluée en français uniquement. Merci de reformuler en français ; l'arabe et la darija ne sont pas encore validés.")
        elif len(keywords(question))==0:
            result.update(status='clarify',answer='Peux-tu préciser le sujet de santé ? Par exemple : définition du diabète ou prévention de l’hypertension.')
        else:
            hits=[h for h in self.retriever.search(question) if h['accepted']]
            if not hits:
                result['answer']='Je ne trouve pas de passage suffisamment pertinent dans le corpus chargé. Précise la question ou consulte une source adaptée.'
            else:
                answer,mode,warning=generate(question,hits,self.cfg)
                result.update(answer=answer,mode=mode,status='abstain' if mode=='abstain' else 'answered')
                if warning:result['warning']=warning
                if mode!='abstain':
                    result['sources']=[{'citation':i,'id':h['id'],'document_id':h['document_id'],'title':h['title'],'url':h['source_url'],'text':h['text'],'language':h['language'],'content_kind':h.get('content_kind','document'),'retrieved_at':h.get('retrieved_at','inconnue')} for i,h in enumerate(hits,1)]
        result['latency_ms']=round((time.perf_counter()-start)*1000,2)
        return result
