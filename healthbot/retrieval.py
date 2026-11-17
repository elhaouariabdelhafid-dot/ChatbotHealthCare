import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .corpus import load_documents,chunk_documents,fingerprint
from .nlp import keywords
from .settings import ROOT

class Retriever:
    def __init__(self,cfg):
        self.cfg=cfg
        self.chunks=chunk_documents(load_documents(cfg['corpus']),cfg['chunk_words'],cfg['overlap_words'])
        self.hash=fingerprint(self.chunks)
        self.vectorizer=TfidfVectorizer(strip_accents='unicode',ngram_range=(1,2),sublinear_tf=True)
        texts=[d['title']+' '+d['text'] for d in self.chunks]
        self.matrix=self.vectorizer.fit_transform(texts)
        self.model=None
        if cfg['retriever'] in ('dense','hybrid'):
            from sentence_transformers import SentenceTransformer
            self.model=SentenceTransformer(cfg['embedding_model'],device='cpu')
            self.model.max_seq_length=512
            mp=ROOT/'artifacts/manifest.json';vp=ROOT/'artifacts/embeddings.npy'
            manifest=json.loads(mp.read_text()) if mp.exists() else {}
            if vp.exists() and manifest.get('corpus_hash')==self.hash and manifest.get('embedding_model')==cfg['embedding_model']:
                self.vectors=np.load(vp,allow_pickle=False)
                if self.vectors.shape!=(len(self.chunks),self.model.get_sentence_embedding_dimension()):
                    raise ValueError('Cache embeddings invalide. Relancer build avec --fresh.')
            else:
                self.vectors=self.model.encode(['passage: '+t for t in texts],normalize_embeddings=True,batch_size=8,show_progress_bar=False)

    def save(self):
        out=ROOT/'artifacts';out.mkdir(exist_ok=True)
        (out/'chunks.json').write_text(json.dumps(self.chunks,ensure_ascii=False,indent=2),encoding='utf-8')
        if self.model is not None:
            np.save(out/'embeddings.npy',self.vectors)
        manifest={'corpus_hash':self.hash,'documents':len({c['document_id'] for c in self.chunks}),'chunks':len(self.chunks),'retriever':self.cfg['retriever'],'embedding_model':self.cfg['embedding_model'] if self.model is not None else None}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        return manifest

    def search(self,question,k=None):
        n=len(self.chunks); k=min(k or self.cfg['top_k'],n)
        lex=(self.matrix @ self.vectorizer.transform([question]).T).toarray().ravel()
        dense=np.zeros(n)
        if self.model is not None:
            v=self.model.encode(['query: '+question],normalize_embeddings=True,show_progress_bar=False)[0]
            dense=self.vectors @ v
        mode=self.cfg['retriever']
        if mode=='lexical':score=lex
        elif mode=='dense':score=dense
        else:
            score=np.zeros(n)
            for values in (lex,dense):
                order=np.argsort(-values,kind='stable')
                for rank,index in enumerate(order):score[index]+=1/(60+rank+1)
        hits=[]; qwords=keywords(question)
        for i in np.argsort(-score,kind='stable')[:k]:
            c=self.chunks[int(i)]
            overlap=bool(qwords & keywords(c['title']+' '+c['text']))
            lexical_ok=bool(lex[i]>=self.cfg['lexical_threshold'] and overlap)
            dense_ok=bool(self.model is not None and dense[i]>=self.cfg['dense_threshold'])
            accepted=lexical_ok if mode=='lexical' else dense_ok if mode=='dense' else lexical_ok or dense_ok
            hits.append({**c,'score':float(score[i]),'lexical_score':float(lex[i]),'dense_score':float(dense[i]),'accepted':accepted})
        return hits
