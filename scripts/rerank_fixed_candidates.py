#!/usr/bin/env python3
"""Compare E5 et mMiniLM sur les mêmes cinq candidats enregistrés.
Aucun appel à E5 ou Ollama ; config.json inchangé. Scores documentaires provisoires.
"""
import argparse
import hashlib
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path

MODEL = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(hits, relevant, top_k):
    relevant = set(relevant)
    rank = next((i for i, h in enumerate(hits[:5], 1) if h['document_id'] in relevant), None)
    kept = [h for h in hits[:top_k] if h['accepted']]
    return dict(rank=rank,
                recall_at_5=len(relevant & {h['document_id'] for h in hits[:5]})/len(relevant) if relevant else None,
                reference_retained=bool(relevant & {h['document_id'] for h in kept}),
                accepted_count=len(kept))


def rerank(model, question, hits):
    if not hits:
        return []
    pairs = [(question, h.get('title', '')+'. '+h['text']) for h in hits]
    scores = [float(s) for s in model.predict(pairs, batch_size=4, show_progress_bar=False)]
    if len(scores) != len(hits) or not all(math.isfinite(s) for s in scores):
        raise ValueError('Scores du reranker invalides.')
    # Tri stable : en cas d'égalité, conserver le rang E5.
    scored = [dict(h, reranker_score=s) for h,s in zip(hits,scores)]
    return sorted(scored, key=lambda h: h['reranker_score'], reverse=True)


def summary(rows, variant):
    pos = [r[variant] for r in rows if r['category']=='covered']
    mean = lambda xs: statistics.mean(xs) if xs else None
    out = dict(positive_questions=len(pos),
               hit_at_1=mean([int(r['rank']==1) for r in pos]),
               recall_at_5=mean([r['recall_at_5'] for r in pos]),
               mrr_at_5=mean([1/r['rank'] if r['rank'] else 0 for r in pos]),
               reference_retained_rate=mean([int(r['reference_retained']) for r in pos]))
    for cat in ['missing_personal_information','out_of_domain']:
        group = [r[variant] for r in rows if r['category']==cat]
        out[cat] = dict(questions=len(group), no_accepted_passage_rate=mean([int(r['accepted_count']==0) for r in group]))
    return out


def validate_inputs(report, dataset):
    if report['corpus_sha256'] != dataset['corpus_sha256']:
        raise ValueError('Empreintes de corpus différentes.')
    cases = {c['id']:c for c in dataset['cases'] if c.get('enabled', True) and c['category']!='medical_absence_unverified'}
    rows = report['details']
    if len(rows)!=len(cases) or {r['id'] for r in rows}!=set(cases):
        raise ValueError('Le rapport doit couvrir exactement les cas actifs du jeu.')
    top_k = report['configuration']['top_k']
    if type(top_k) is not int or not 1 <= top_k <= 5:
        raise ValueError('top_k doit être compris entre 1 et 5 pour ce cache.')
    for r in rows:
        c = cases[r['id']]
        if r['question']!=c['question'] or r['category']!=c['category']:
            raise ValueError('Question ou catégorie différente : '+r['id'])
        if len(r['top5'])>5 or len({h['id'] for h in r['top5']})!=len(r['top5']):
            raise ValueError('Cache de candidats invalide : '+r['id'])
        measured = measure(r['top5'], c['relevant_documents'], top_k)
        for key in ['rank','recall_at_5','reference_retained','accepted_count']:
            if measured[key]!=r[key]:
                raise ValueError('Annotations ou baseline différentes : '+r['id']+' / '+key)
    return cases, top_k


def write(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


def evaluate(args):
    report, dataset = read(args.report), read(args.dataset)
    cases, top_k = validate_inputs(report, dataset)
    from sentence_transformers import CrossEncoder
    print('Chargement mMiniLM sur CPU ; E5 et Ollama ne sont pas chargés.', flush=True)
    model = CrossEncoder(MODEL, device='cpu', max_length=384)
    outdir = Path('artifacts'); outdir.mkdir(exist_ok=True)
    output = outdir / ('reranking_fixed5_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
    result = dict(protocol='Reranking des mêmes top-5 bruts E5 ; puis top_k et filtre accepted inchangé.',
                  provisional=True, baseline_report=str(args.report), baseline_sha256=sha(args.report),
                  dataset=str(args.dataset), dataset_sha256=sha(args.dataset),
                  corpus_sha256=report['corpus_sha256'], configuration=report['configuration'],
                  reranker=dict(model=MODEL, max_length=384, device='cpu', batch_size=4),
                  limitations=['Annotations documentaires non exhaustives ; vérifier le texte de N05.',
                               'Recall@5 ne peut pas changer, car les cinq candidats sont identiques.',
                               'Aucun seuil de rejet appris pour mMiniLM ; accepted conserve la décision E5.',
                               'Aucune génération ni mesure de fidélité médicale ; cinq cas U exclus.',
                               'Latence = reranking uniquement, hors chargement ; premier appel potentiellement plus lent.'],
                  planned=len(report['details']), completed=0, details=[])
    for r in report['details']:
        before = r['top5']
        start = time.perf_counter()
        after = rerank(model, r['question'], before)
        elapsed = time.perf_counter()-start
        relevant = cases[r['id']]['relevant_documents']
        a, b = measure(before,relevant,top_k), measure(after,relevant,top_k)
        row = dict(id=r['id'], question=r['question'], category=r['category'],
                   relevant_documents=relevant, e5=a, reranked=b, rerank_seconds=elapsed,
                   before=before, after=after)
        result['details'].append(row)
        result['completed']+=1
        write(output,result)
        print(f"{r['id']} | rang E5={a['rank']} -> mMiniLM={b['rank']} | {elapsed:.2f}s", flush=True)
    result['metrics']={v:summary(result['details'],v) for v in ['e5','reranked']}
    changes={'improved':[], 'worsened':[], 'unchanged':[]}
    for r in result['details']:
        if r['category']!='covered': continue
        a,b=r['e5']['rank'] or 6,r['reranked']['rank'] or 6
        changes['improved' if b<a else 'worsened' if b>a else 'unchanged'].append(r['id'])
    result['rank_changes']=changes
    result['rerank_mean_seconds']=statistics.mean(r['rerank_seconds'] for r in result['details'])
    write(output,result)
    review=output.with_suffix('.md')
    lines=['# Audit des passages : N05 et changements de classement', '',
           'Lire le texte : un document différent de la référence peut aussi répondre.',
           'Toute révision des références doit être documentée et appliquée aux deux variantes.', '']
    for r in result['details']:
        if r['category']!='covered': continue
        if r['id']!='N05' and r['e5']==r['reranked']: continue
        lines += [f"## {r['id']} — {r['question']}", 'Références attendues : '+', '.join(r['relevant_documents'])]
        for name in ['before','after']:
            lines += ['','### '+name]
            for rank,h in enumerate(r[name],1):
                lines += [f"{rank}. {h['title']} — {h['document_id']} — accepté={h['accepted']}", h['text'], '']
    review.write_text('\n'.join(lines),encoding='utf-8')
    print('\n=== COMPARAISON SUR CANDIDATS FIXES ===')
    print(json.dumps(result['metrics'],ensure_ascii=False,indent=2))
    print('Rangs améliorés :',changes['improved'])
    print('Rangs dégradés :',changes['worsened'])
    print('Temps moyen reranking :',round(result['rerank_mean_seconds'],3),'s')
    print('Rapport :',output)
    print('Passages à relire :',review)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=Path('data/evaluation/evaluation_30_proposee.json'))
    args=p.parse_args()
    try: evaluate(args)
    except (OSError,ValueError,KeyError,TypeError) as e:
        raise SystemExit('Évaluation interrompue : '+str(e))

if __name__=='__main__': main()
