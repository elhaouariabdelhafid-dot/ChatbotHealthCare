"""Commands: python -m healthbot.cli --help"""
import argparse
import csv
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from .settings import ROOT,settings
from .engine import HealthBot
from .retrieval import Retriever


def dump(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def import_medquad(source,output):
    """Local checkout only; empty answers removed; review is required before indexing."""
    files=sorted(Path(source).rglob('*.xml'))
    if not files:raise ValueError('Aucun fichier XML dans ce dossier.')
    counts={'files':len(files),'imported':0,'empty_answers':0,'invalid':0,'duplicates':0}
    seen=set()
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    with open(output,'w',encoding='utf-8') as f:
        for file in files:
            if file.stat().st_size>10_000_000:
                counts['invalid']+=1;continue
            raw=file.read_bytes()
            if b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
                counts['invalid']+=1;continue
            try:root=ET.fromstring(raw)
            except ET.ParseError:counts['invalid']+=1;continue
            url=root.attrib.get('url','')
            if not url.startswith(('http://','https://')):
                counts['invalid']+=1;continue
            # Preserve original HTTP links via HTTPS only if the source already provides it.
            # Legacy HTTP sources remain in staging, and must be verified before approval.
            focus=root.findtext('Focus') or file.stem
            for pair in root.findall('.//QAPair'):
                q=pair.find('Question');a=pair.find('Answer')
                question=' '.join(q.itertext()).strip() if q is not None else ''
                answer=' '.join(a.itertext()).strip() if a is not None else ''
                if not question or not answer:
                    counts['empty_answers']+=1;continue
                import hashlib
                digest=hashlib.sha256((question+'\n'+answer).encode()).hexdigest()
                if digest in seen:counts['duplicates']+=1;continue
                seen.add(digest)
                record={'id':'medquad_'+digest[:20],'title':question,'text':answer,'source_url':url,'language':'en','license_status':'MedQuAD CC BY 4.0 ; vérifier les droits de la source et attribuer les auteurs','content_kind':'medquad','approved':False,'question_type':q.attrib.get('qtype','unknown'),'focus':focus,'source_file':str(file.relative_to(source))}
                f.write(json.dumps(record,ensure_ascii=False)+'\n');counts['imported']+=1
    return counts


def import_meddialog(source,output):
    """Input JSONL: {id, patient, doctor, source_url}. Conversation research only."""
    n=0; Path(output).parent.mkdir(parents=True,exist_ok=True)
    def scrub(text):
        text=re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}','[EMAIL]',text)
        text=re.sub(r'(?<!\w)\+?\d[\d ()-]{7,}\d','[NUMERO]',text)
        return text
    with open(source,encoding='utf-8') as src,open(output,'w',encoding='utf-8') as out:
        for line in src:
            if not line.strip():continue
            d=json.loads(line)
            if not all(isinstance(d.get(k),str) and d[k].strip() for k in ('patient','doctor','source_url')):
                raise ValueError('MedDialog : patient, doctor, source_url requis pour chaque ligne.')
            record={'id':str(d.get('id',n+1)),'patient':scrub(d['patient']),'doctor':scrub(d['doctor']),'source_url':d['source_url'],'language':d.get('language','en'),'review_status':'pending_manual_privacy_and_quality_review','use':'conversation_research_only'}
            out.write(json.dumps(record,ensure_ascii=False)+'\n');n+=1
    return {'imported':n,'warning':'Masquage partiel seulement. Ne pas indexer ce fichier comme preuve médicale. Revue humaine des noms, lieux, dates et identifiants requise.'}


def evaluate(cfg,output):
    bot=HealthBot(cfg); rows=[];recalls=[];reciprocal=[];durations=[]
    for line in (ROOT/'data/evaluation.jsonl').read_text(encoding='utf-8').splitlines():
        case=json.loads(line);start=time.perf_counter();r=bot.ask(case['question']);durations.append((time.perf_counter()-start)*1000)
        relevant=set(case['relevant_documents']);hits=bot.retriever.search(case['question'],k=10)
        ids=[x['document_id'] for x in hits]
        if relevant:
            recalls.append(len(relevant & set(ids[:5]))/len(relevant))
            reciprocal.append(next((1/(i+1) for i,x in enumerate(ids) if x in relevant),0))
        rows.append({'id':case['id'],'question':case['question'],'expected_status':case['expected_status'],'actual_status':r['status'],'status_correct':r['status']==case['expected_status'],'returned_documents':[s['document_id'] for s in r['sources']]})
    import numpy as np
    report={'scope':'Petit jeu pédagogique rédigé avec le corpus. Test de régression, pas évaluation médicale indépendante. Les métriques de recherche sont calculées avant filtrage de pertinence.','corpus_hash':bot.retriever.hash,'config':{k:v for k,v in cfg.items() if k not in ('corpus','intents')},'questions':len(rows),'retrieval_questions':len(recalls),'recall_at_5':float(np.mean(recalls)) if recalls else None,'mrr_at_10':float(np.mean(reciprocal)) if reciprocal else None,'status_accuracy':sum(r['status_correct'] for r in rows)/len(rows),'latency_p50_ms':float(np.percentile(durations,50)),'latency_p95_ms':float(np.percentile(durations,95)),'details':rows}
    dump(output,report);return report


def evaluate_intent(cfg,output):
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import classification_report,confusion_matrix
    with open(cfg['intents'],encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    x=[r['text'] for r in rows];y=[r['label'] for r in rows]
    a,b,c,d=train_test_split(x,y,test_size=.25,stratify=y,random_state=42)
    model=make_pipeline(TfidfVectorizer(strip_accents='unicode',ngram_range=(1,2)),LogisticRegression(class_weight='balanced',max_iter=1000,random_state=42))
    model.fit(a,c);pred=model.predict(b);labels=sorted(set(y))
    report={'scope':'Validation pédagogique : 48 exemples synthétiques, séparation stratifiée 75/25. Insuffisant pour conclure sur des données réelles. Le modèle de l’application est réentraîné sur tout le CSV au lancement.','train_count':len(a),'validation_count':len(b),'metrics':classification_report(d,pred,zero_division=0,output_dict=True),'labels':labels,'confusion_matrix':confusion_matrix(d,pred,labels=labels).tolist(),'predictions':[{'text':q,'expected':e,'predicted':str(p)} for q,e,p in zip(b,d,pred)]}
    dump(output,report);return report


def main():
    p=argparse.ArgumentParser(description='ChatBotHealthCare — outils locaux')
    p.add_argument('--config',help='Fichier JSON de configuration')
    sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('--fresh',action='store_true')
    a=sub.add_parser('ask');a.add_argument('question')
    e=sub.add_parser('evaluate');e.add_argument('--output',default=str(ROOT/'artifacts/evaluation.json'))
    t=sub.add_parser('train-intent');t.add_argument('--output',default=str(ROOT/'artifacts/intent_evaluation.json'))
    for name in ('import-medquad','import-meddialog'):
        imp=sub.add_parser(name);imp.add_argument('source');imp.add_argument('output')
    args=p.parse_args();cfg=settings(args.config)
    if args.command=='build':
        if args.fresh:(ROOT/'artifacts/embeddings.npy').unlink(missing_ok=True)
        result=Retriever(cfg).save()
    elif args.command=='ask':result=HealthBot(cfg).ask(args.question)
    elif args.command=='evaluate':result=evaluate(cfg,args.output)
    elif args.command=='train-intent':result=evaluate_intent(cfg,args.output)
    elif args.command=='import-medquad':result=import_medquad(args.source,args.output)
    else:result=import_meddialog(args.source,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
