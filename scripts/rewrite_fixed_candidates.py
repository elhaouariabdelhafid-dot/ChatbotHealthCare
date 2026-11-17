#!/usr/bin/env python3
"""Traduction puis reranking sur les candidats fixes du nouveau jeu.
translate : Qwen 3B uniquement, checkpoint reprenable.
score : mMiniLM uniquement ; réutilise les résultats E5 et mMiniLM originaux.
Pas de nouvelle recherche E5. Les règles ne valident pas la fidélité sémantique.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASELINE = Path('artifacts/reranking_fixed5_20261117_190113_030265.json')
PROMPT_SOURCE = Path('scripts/test_rewrite_v3_frozen.py')
CACHE = Path('artifacts/rewrite_new25.json')
MODEL = 'qwen2.5:3b'
OPTIONS = dict(temperature=0, num_ctx=1024, num_predict=130)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


def get_prompt(path):
    # Lire la constante sans importer le script, qui lancerait ses anciens tests.
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    values=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROMPT' for t in node.targets):
            values.append(ast.literal_eval(node.value))
    if len(values)!=1 or not isinstance(values[0],str) or '{question}' not in values[0]:
        raise ValueError('Une constante PROMPT contenant {question} est requise.')
    return values[0]


def validate(original, rewritten):
    a,b=original.strip().lower(),rewritten.strip().lower()
    if not b: return False,'Reformulation vide'
    if re.match(r'^quelle maladie\b',a) and not re.match(r'^(which|what) disease\b',b):
        return False,'Type de réponse changé : maladie'
    if re.match(r'^quel syndrome\b',a) and not re.match(r'^(which|what) syndrome\b',b):
        return False,'Type de réponse changé : syndrome'
    return True,'Aucune anomalie détectée par les règles partielles'


def load_baseline(path):
    data=read(path)
    if data.get('completed')!=data.get('planned') or not data.get('details'):
        raise ValueError('Rapport de référence incomplet.')
    if len({r['id'] for r in data['details']})!=len(data['details']):
        raise ValueError('Identifiants dupliqués.')
    for r in data['details']:
        if {h['id'] for h in r['before']}!={h['id'] for h in r['after']}:
            raise ValueError('Candidats incohérents : '+r['id'])
    return data


def translate(args):
    baseline=load_baseline(args.baseline)
    prompt=get_prompt(args.prompt_source)
    cfg=read(Path('config.json'))
    endpoint=cfg['ollama_url'].rstrip('/')+'/api/generate'
    identity=dict(baseline_sha256=sha(args.baseline), prompt=prompt, model=MODEL, options=OPTIONS, endpoint=endpoint)
    if args.cache.exists():
        data=read(args.cache)
        if any(data.get(k)!=v for k,v in identity.items()):
            raise ValueError('Checkpoint incompatible : conserver le fichier et choisir un autre --cache.')
    else:
        data=dict(identity, baseline=str(args.baseline), prompt_source=str(args.prompt_source),
                  prompt_source_sha256=sha(args.prompt_source), planned=len(baseline['details']),
                  rules='Deux règles historiques maladie/syndrome + contrôle de transport et troncature.',
                  human_review_required=True, details=[])
    by_id={r['id']:r for r in data['details']}
    for case in baseline['details']:
        if case['id'] in by_id:
            if by_id[case['id']]['original']!=case['question']:
                raise ValueError('Question modifiée dans le checkpoint.')
            continue
        original=case['question']
        payload=dict(model=MODEL, prompt=prompt.format(question=original), stream=False,keep_alive='5m',options=OPTIONS)
        start=time.perf_counter();raw='';error='';done_reason=None;response_model=None
        try:
            req=Request(endpoint,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
            with urlopen(req,timeout=180) as response: result=json.load(response)
            if result.get('error'): raise ValueError(str(result['error']))
            raw=result.get('response','')
            response_model=result.get('model')
            done_reason=result.get('done_reason')
            if not isinstance(raw,str): raise ValueError('Texte de réponse invalide')
            if done_reason=='length': raise ValueError('Sortie tronquée : retour à la question originale')
            if result.get('done') is False: raise ValueError('Réponse non terminée')
        except HTTPError as exc:
            error=f'HTTP {exc.code}: '+exc.read(5000).decode('utf-8',errors='replace')
        except (OSError,ValueError,KeyError,TypeError) as exc:
            error=f'{type(exc).__name__}: {exc}'
        rewritten=raw.strip().strip('"').strip("'") if isinstance(raw,str) else ''
        valid,reason=validate(original,rewritten)
        if error: valid,reason=False,error
        record=dict(id=case['id'],category=case['category'],original=original,raw=raw,rewritten=rewritten,
                    rule_pass=valid,rule_reason=reason,effective_query=rewritten if valid else original,
                    error=error,done_reason=done_reason,response_model=response_model,
                    rewrite_seconds=time.perf_counter()-start,human_review=None,human_comment='')
        data['details'].append(record);by_id[case['id']]=record
        data['completed']=len(data['details']);save(args.cache,data)
        print('\n'+case['id']+' | '+original,flush=True)
        print('Proposée :',rewritten or '(vide)',flush=True)
        print('Règles :', 'PASS PARTIEL' if valid else 'FALLBACK', '|',reason,flush=True)
    print('\n=== TRADUCTIONS TERMINÉES ===')
    print('Cas enregistrés :',len(data['details']))
    print('Erreurs techniques :',sum(bool(r['error']) for r in data['details']))
    print('Retours à l’original :',sum(not r['rule_pass'] for r in data['details']))
    print('Rapport :',args.cache)
    print('PASS PARTIEL ne signifie pas traduction fidèle. Relire sujet, intention, négations et détails ajoutés.')


def score(args):
    baseline=load_baseline(args.baseline);data=read(args.cache)
    if data['baseline_sha256']!=sha(args.baseline):raise ValueError('Rapport de référence modifié.')
    lookup={r['id']:r for r in data['details']}
    if len(lookup)!=len(data['details']) or set(lookup)!={r['id'] for r in baseline['details']}:
        raise ValueError('Traductions incomplètes ou dupliquées.')
    module_path=Path(__file__).with_name('rerank_fixed_candidates.py')
    spec=importlib.util.spec_from_file_location('fixed_metrics',module_path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    if baseline['reranker']!=dict(model=m.MODEL,max_length=384,device='cpu',batch_size=4):
        raise ValueError('Paramètres du reranker différents de la référence.')
    model=None;rows=[]
    output=Path('artifacts')/('rewrite_fixed5_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
    report=dict(baseline=str(args.baseline),baseline_sha256=sha(args.baseline),rewrites_sha256=sha(args.cache),
                protocol='Question reformulée pour le reranker uniquement ; pas de nouvelle recherche E5.',
                provisional=True,reranker=baseline['reranker'],configuration=baseline['configuration'],details=rows)
    for base in baseline['details']:
        t=lookup[base['id']]
        if t['original']!=base['question']:raise ValueError('Question incohérente : '+base['id'])
        # Évaluer la politique automatique gelée, sans utiliser les jugements humains pour améliorer les rangs.
        valid,_=validate(t['original'],t['rewritten'])
        expected=t['rewritten'] if valid and not t['error'] else t['original']
        if t['effective_query']!=expected:raise ValueError('Requête effective altérée : '+base['id'])
        if expected==base['question']:
            hits=base['after'];seconds=0.0;reused=True
        else:
            if model is None:
                from sentence_transformers import CrossEncoder
                print('Chargement mMiniLM, sans E5 ni appel à Ollama...',flush=True)
                model=CrossEncoder(m.MODEL,device='cpu',max_length=384)
            start=time.perf_counter();hits=m.rerank(model,expected,base['before']);seconds=time.perf_counter()-start;reused=False
        metrics=m.measure(hits,base['relevant_documents'],baseline['configuration']['top_k'])
        row=dict(id=base['id'],question=base['question'],category=base['category'],
                 e5=base['e5'],reranked=base['reranked'],rewritten=metrics,
                 proposed=t['rewritten'],effective_query=expected,rule_pass=t['rule_pass'],
                 human_review=t.get('human_review'),human_comment=t.get('human_comment',''),
                 rewrite_seconds=t['rewrite_seconds'],rerank_seconds=seconds,reused_original_reranking=reused,
                 after_rewrite=hits)
        rows.append(row);report['completed']=len(rows);save(output,report)
        print(f"{base['id']} | E5={base['e5']['rank']} | mMiniLM={base['reranked']['rank']} | rewriting+mMiniLM={metrics['rank']}",flush=True)
    report['metrics']={v:m.summary(rows,v) for v in ['e5','reranked','rewritten']}
    report['technical_errors']=sum(bool(r['error']) for r in data['details'])
    report['automatic_fallbacks']=sum(not r['rule_pass'] for r in data['details'])
    report['human_reviewed_count']=sum(r.get('human_review') in ['faithful','unfaithful'] for r in data['details'])
    save(output,report)
    print('\n=== COMPARAISON REWRITING ===')
    print(json.dumps(report['metrics'],ensure_ascii=False,indent=2))
    print('Erreurs techniques :',report['technical_errors'])
    print('Retours automatiques à l’original :',report['automatic_fallbacks'])
    print('Reformulations relues :',report['human_reviewed_count'],'/',len(rows))
    print('Rapport :',output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['translate','score'])
    p.add_argument('--baseline',type=Path,default=BASELINE)
    p.add_argument('--prompt-source',type=Path,default=PROMPT_SOURCE)
    p.add_argument('--cache',type=Path,default=CACHE)
    args=p.parse_args()
    try:
        (translate if args.command=='translate' else score)(args)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise SystemExit('Test interrompu : '+str(exc))

if __name__=='__main__':main()
