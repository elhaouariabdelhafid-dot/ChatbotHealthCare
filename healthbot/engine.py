import re
import time
import uuid
from .settings import settings
from .nlp import safety_route,entities,IntentModel,keywords
from .retrieval import Retriever
from .generation import generate
from . import groq_provider
from .reranking import PassageReranker, configure_reranking

class HealthBot:
    def __init__(self,cfg=None):
        self.cfg=groq_provider.configure_groq(configure_reranking(dict(cfg) if cfg is not None else settings()))
        self.retriever=Retriever(self.cfg)
        self.intent=IntentModel(self.cfg['intents'])
        self.reranker=PassageReranker(self.cfg['reranking_model']) if self.cfg['reranking_enabled'] else None

    def _select_passages(self, question):
        started=time.perf_counter()
        if self.reranker is None:
            candidates=self.retriever.search(question)
            hits=[h for h in candidates if h['accepted']]
            return hits, {'reranking_status':'disabled','candidate_count':len(candidates),
                          'selected_count':len(hits),'rerank_ms':0.0}
        candidates=self.retriever.search(question,k=self.cfg['reranking_candidates'])
        ordered=candidates
        status='skipped_no_accepted'
        warning=None
        rerank_ms=0.0
        if any(h['accepted'] for h in candidates):
            rerank_start=time.perf_counter()
            try:
                ordered=self.reranker.rerank(question,candidates)
                status='applied'
            except (ImportError,OSError,RuntimeError,ValueError,TypeError) as exc:
                # Repli explicite sur les rangs E5, sans élargir le seuil.
                ordered=candidates
                status='fallback_e5'
                warning='Reranking indisponible ('+type(exc).__name__+'); classement E5 conservé.'
            rerank_ms=round((time.perf_counter()-rerank_start)*1000,2)
        # Même ordre que le protocole évalué : rerank -> top_k -> accepted.
        hits=[h for h in ordered[:self.cfg['top_k']] if h['accepted']]
        trace={'reranking_status':status,'candidate_count':len(candidates),
               'selected_count':len(hits),'rerank_ms':rerank_ms,
               'retrieval_ms':round((time.perf_counter()-started)*1000,2),
               'candidate_order':[{'id':h['id'],'document_id':h['document_id'],
                    'accepted':h['accepted'],'dense_score':h.get('dense_score'),
                    'reranker_score':h.get('reranker_score')} for h in ordered]}
        if warning:trace['warning']=warning
        return hits,trace

    def ask(self,question):
        if not isinstance(question,str) or not question.strip():
            raise ValueError('Écris une question non vide.')
        question=question.strip()
        if len(question)>self.cfg['max_question_chars']:
            raise ValueError('Question trop longue.')
        start=time.perf_counter()
        groq_provider.TRACE.set(None)
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
            hits,retrieval_trace=self._select_passages(question)
            result['retrieval']=retrieval_trace
            if not hits:
                result['answer']='Je ne trouve pas de passage suffisamment pertinent dans le corpus chargé. Précise la question ou consulte une source adaptée.'
            else:
                answer,mode,warning=generate(question,hits,self.cfg)
                result.update(answer=answer,mode=mode,status='abstain' if mode=='abstain' else 'answered')
                if self.cfg['generation']=='groq':
                    result['generation_trace']=groq_provider.snapshot()
                    if mode=='extractive': result['status']='degraded'
                if warning:result['warning']=warning
                if mode!='abstain':
                    result['sources']=[{'citation':i,'id':h['id'],'document_id':h['document_id'],'title':h['title'],'url':h['source_url'],'text':h['text'],'language':h['language'],'content_kind':h.get('content_kind','document'),'retrieved_at':h.get('retrieved_at','inconnue')} for i,h in enumerate(hits,1)]
        result['latency_ms']=round((time.perf_counter()-start)*1000,2)
        return result

