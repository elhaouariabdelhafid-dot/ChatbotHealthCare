#!/usr/bin/env python3
"""Validation ChatBotHealthCare. Python 3.10+, aucune dépendance supplémentaire.
Depuis la racine du projet :
  python scripts/validation_healthcare.py prepare --translate
  python scripts/validation_healthcare.py run
  python scripts/validation_healthcare.py run --answers
  python scripts/validation_healthcare.py self-test
Ne change pas config.json. Ne modifie pas le corpus. Refuse d'écraser le jeu existant.
"""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

TYPES = ('information', 'symptoms', 'causes', 'prevention')
TEMPLATES = {
    'information': 'Comment définir la maladie ou le sujet suivant : {topic} ?',
    'symptoms': 'Quels signes ou symptômes sont associés à {topic} ?',
    'causes': 'Quels facteurs peuvent expliquer {topic} ?',
    'prevention': 'Quelles mesures de prévention sont décrites pour {topic} ?',
}
# Informations individuelles absentes d'un corpus documentaire public.
# Ce lot ne remplace pas un test de connaissances médicales hors corpus.
MISSING = [
    'Quelle était la valeur exacte de ma glycémie lors de ma dernière prise de sang ?',
    'Quelle tension artérielle a été mesurée pendant ma dernière consultation ?',
    'Quel résultat figure sur mon dernier test respiratoire ?',
    'Quelle est la date de mon prochain rendez-vous médical ?',
    'Quel groupe sanguin est indiqué dans mon dossier médical ?',
]
OUTSIDE = [
    'Quel est le prix d’un billet de train entre Fès et Casablanca ?',
    'Comment installer un jeu vidéo sur mon ordinateur ?',
    'Comment calculer la surface d’un triangle ?',
    'Quelle commande Git permet de créer une nouvelle branche ?',
    'Comment préparer une présentation PowerPoint ?',
]


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def root_and_config():
    # Le dossier courant prime pour éviter d'utiliser un autre projet par erreur.
    root = Path.cwd()
    if not (root / 'healthbot').is_dir() or not (root / 'config.json').is_file():
        raise ValueError('Exécuter depuis ~/ChatBotHealthCare, pas depuis scripts/.')
    sys.path.insert(0, str(root))
    from healthbot.settings import settings
    return root, settings()


def corpus_hash(cfg):
    return hashlib.sha256(Path(cfg['corpus']).read_bytes()).hexdigest()


def choose_documents(rows):
    groups = {label: [] for label in TYPES}
    previous_topics = {'cardiac arrest', 'diabetes', 'hypertension', 'asthma'}
    for row in rows:
        label = row.get('question_type')
        topic = str(row.get('focus', '')).strip()
        if (row.get('approved') is True and row.get('content_kind') == 'medquad'
                and label in groups and topic and topic.casefold() not in previous_topics
                and 40 <= len(row.get('text', '').split()) <= 700):
            groups[label].append(row)
    rng = random.Random(20261116)
    for label in TYPES:
        groups[label].sort(key=lambda r: r['id'])
        rng.shuffle(groups[label])
    selected, topics = [], set()
    while len(selected) < 15:
        changed = False
        for label in TYPES:
            while groups[label]:
                row = groups[label].pop()
                topic = row['focus'].strip().casefold()
                if topic not in topics:
                    selected.append(row)
                    topics.add(topic)
                    changed = True
                    break
            if len(selected) == 15:
                break
        if not changed:
            break
    if len(selected) != 15:
        raise ValueError(f'Seulement {len(selected)} sujets distincts disponibles ; jeu non créé.')
    return selected


def translate_question(row, cfg):
    # Pas de génération de vérité terrain par le modèle évalué : traduction uniquement.
    payload = {
        'model': cfg['ollama_model'], 'stream': False,
        'format': {'type': 'object', 'properties': {'question_fr': {'type': 'string'}},
                   'required': ['question_fr'], 'additionalProperties': False},
        'messages': [
            {'role': 'system', 'content': 'Traduis la question fournie en français naturel. Conserve strictement le sujet et le sens. Ne réponds pas à la question. Ne crée aucun détail. Conserve les noms propres et sigles si leur traduction est incertaine. Retourne uniquement le JSON demandé.'},
            {'role': 'user', 'content': json.dumps({'question': row['title']}, ensure_ascii=False)},
        ],
        'options': {'temperature': 0, 'num_predict': 180, 'num_ctx': 2048},
    }
    req = Request(cfg['ollama_url'].rstrip('/') + '/api/chat',
                  data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=cfg.get('timeout_seconds', 180)) as response:
        result = json.loads(response.read(1_000_000))
    if result.get('done_reason') == 'length':
        raise ValueError('Traduction tronquée')
    question = json.loads(result['message']['content'])['question_fr']
    if not isinstance(question, str) or not 8 <= len(question.strip()) <= 500:
        raise ValueError('Traduction vide ou trop longue')
    return question.strip()


def prepare(root, cfg, path, translate):
    if path.exists():
        raise ValueError(f'{path} existe déjà : conserver ce jeu pour des comparaisons stables.')
    rows = [json.loads(line) for line in Path(cfg['corpus']).read_text(encoding='utf-8').splitlines() if line.strip()]
    selected = choose_documents(rows)
    cache_path = root / 'artifacts/validation_translation_cache.json'
    cache = load_json(cache_path) if cache_path.exists() else {}
    cases = []
    for i, row in enumerate(selected, 1):
        question = TEMPLATES[row['question_type']].format(topic=row['focus'])
        method = 'french_template_original_topic'
        if translate:
            key = hashlib.sha256((cfg['ollama_model'] + '\n' + row['title']).encode()).hexdigest()
            if key not in cache:
                print(f'Traduction {i}/15 : {row["focus"]}', flush=True)
                cache[key] = translate_question(row, cfg)
                write_json(cache_path, cache)
            question = cache[key]
            method = 'ollama_translation_to_review'
        cases.append({'id': f'P{i:02}', 'category': 'covered', 'question': question,
                      'original_question': row['title'], 'topic': row['focus'],
                      'question_type': row['question_type'],
                      'relevant_documents': [row['id']], 'expected_statuses': ['answered'],
                      'reference_text': row['text'], 'source_url': row['source_url'],
                      'reviewed': False, 'question_method': method})
    for prefix, category, questions in [('M', 'missing_personal_information', MISSING), ('O', 'out_of_domain', OUTSIDE)]:
        for i, q in enumerate(questions, 1):
            cases.append({'id': f'{prefix}{i:02}', 'category': category, 'question': q,
                          'relevant_documents': [],
                          'expected_statuses': ['abstain', 'clarify', 'personal'] if prefix == 'M' else ['abstain'],
                          'reviewed': False})
    data = {'version': 1, 'corpus_sha256': corpus_hash(cfg), 'seed': 20261116,
            'scope': 'Validation de développement, annotations proposées à relire. Pas un test final indépendant.',
            'limitations': ['Les documents positifs sont choisis dans le corpus ; les autres documents pertinents ne sont pas annotés exhaustivement.',
                           'Les traductions automatiques peuvent être erronées. Les relire avec les questions originales.',
                           'Les 5 questions médicales négatives visent des informations personnelles absentes, pas des maladies absentes du corpus.',
                           'Si Qwen traduit les questions puis répond, ces résultats ne sont pas indépendants de ce modèle.'],
            'cases': cases}
    write_json(path, data)
    review = root / 'artifacts/validation_questions_a_relire.md'
    lines = ['# Jeu proposé : 25 questions', '', 'Corriger les questions et reviewed dans data/validation_v1.json après relecture.',
             'Ne pas marquer reviewed=true sans contrôler question, document et réponse attendue.', '']
    for case in cases:
        lines += [f'## {case["id"]} — {case["category"]}', case['question'], '']
        if case['category'] == 'covered':
            lines += ['Original : ' + case['original_question'], 'Document : ' + case['relevant_documents'][0],
                      'Source : ' + case['source_url'], '', case['reference_text'], '']
    review.parent.mkdir(parents=True, exist_ok=True)
    review.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Créé : {path}\nRelecture : {review}\n15 sujets distincts + 5 informations personnelles absentes + 5 hors domaine.')


def metrics(records):
    positive = [r for r in records if r['category'] == 'covered']
    def avg(values):
        return sum(values) / len(values) if values else None
    result = {'positive_questions': len(positive),
              'hit_at_1': avg([int(r['rank'] == 1) for r in positive]),
              'recall_at_5': avg([r['recall_at_5'] for r in positive]),
              'mrr_at_5': avg([1/r['rank'] if r['rank'] else 0 for r in positive]),
              'reference_retained_rate': avg([int(r['reference_retained']) for r in positive])}
    for category in ('missing_personal_information', 'out_of_domain'):
        group = [r for r in records if r['category'] == category]
        result[category] = {'questions': len(group), 'no_accepted_passage_rate': avg([int(r['accepted_count'] == 0) for r in group])}
    return result


def run(root, cfg, path, answers, reviewed_only):
    from healthbot.retrieval import Retriever
    from healthbot.engine import HealthBot
    data = load_json(path)
    if data['corpus_sha256'] != corpus_hash(cfg):
        raise ValueError('Le corpus a changé depuis la création du jeu. Comparaison interrompue.')
    cases = [c for c in data['cases'] if not reviewed_only or c.get('reviewed') is True]
    if not cases:
        raise ValueError('Aucune question relue. Relire le jeu ou retirer --reviewed-only pour un diagnostic provisoire.')
    if not all(c.get('reviewed') is True for c in cases):
        print('ATTENTION : scores provisoires, questions ou annotations non relues.', flush=True)
    print('Chargement de la recherche dense et du cache E5...', flush=True)
    dense_cfg = {**cfg, 'retriever': 'dense'}
    bot = HealthBot(dense_cfg) if answers else None
    retriever = bot.retriever if bot else Retriever(dense_cfg)
    artifact = root / 'artifacts'
    artifact.mkdir(exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    report_path = artifact / f'validation_{"answers" if answers else "retrieval"}_{stamp}.json'
    records = []
    for i, case in enumerate(cases, 1):
        print(f'[{i}/{len(cases)}] {case["id"]} : {case["question"]}', flush=True)
        hits = retriever.search(case['question'], k=5)
        relevant = set(case['relevant_documents'])
        ranks = [j for j, hit in enumerate(hits, 1) if hit['document_id'] in relevant]
        accepted = [h for h in hits[:cfg['top_k']] if h['accepted']]
        row = {'id': case['id'], 'question': case['question'], 'category': case['category'],
               'reviewed': case.get('reviewed', False), 'rank': min(ranks) if ranks else None,
               'recall_at_5': len(relevant & {h['document_id'] for h in hits}) / len(relevant) if relevant else None,
               'reference_retained': bool(relevant & {h['document_id'] for h in accepted}),
               'accepted_count': len(accepted), 'top5': [
                   {k: h[k] for k in ('id', 'document_id', 'title', 'text', 'dense_score', 'accepted')} for h in hits]}
        if answers:
            start = time.perf_counter()
            result = bot.ask(case['question'])
            row.update(result=result, status_matches_expectation=result['status'] in case['expected_statuses'],
                       elapsed_ms=round((time.perf_counter()-start)*1000, 2),
                       human_review={'correctness': None, 'citation_support': None, 'relevance': None, 'notes': ''})
        records.append(row)
        report = {'scope': data['scope'], 'provisional': not all(c.get('reviewed') is True for c in cases),
                  'corpus_sha256': data['corpus_sha256'], 'index_hash': retriever.hash,
                  'configuration': dense_cfg, 'completed': len(records), 'planned': len(cases),
                  'retrieval_metrics': metrics(records), 'details': records}
        if answers:
            report['status_match_rate'] = sum(r['status_matches_expectation'] for r in records)/len(records)
            report['answer_quality'] = 'Non mesurée automatiquement : relecture humaine nécessaire.'
        write_json(report_path, report)  # checkpoint après chaque question
    print('\n=== BILAN VALIDATION DENSE ===')
    print(json.dumps(report['retrieval_metrics'], ensure_ascii=False, indent=2))
    if answers:
        print('Statuts attendus :', f'{report["status_match_rate"]:.1%}', '(pas une mesure de justesse médicale)')
    print('Rapport :', report_path)
    if answers:
        csv_path = report_path.with_suffix('.csv')
        with csv_path.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'category', 'question', 'answer', 'status', 'mode', 'warning', 'sources_json', 'correctness_0_2', 'citation_support_0_2', 'relevance_0_2', 'notes'])
            for r in records:
                result = r['result']
                writer.writerow([r['id'], r['category'], r['question'], result['answer'], result['status'], result['mode'], result.get('warning', ''), json.dumps(result['sources'], ensure_ascii=False), '', '', '', ''])
        print('Grille de relecture des réponses :', csv_path)


class Checks(unittest.TestCase):
    def test_selection_reproducible_distinct(self):
        rows = [{'id': str(i), 'focus': f'Topic {i}', 'question_type': TYPES[i % 4], 'text': 'word ' * 60,
                 'approved': True, 'content_kind': 'medquad'} for i in range(40)]
        a = choose_documents(rows)
        self.assertEqual(a, choose_documents(rows))
        self.assertEqual(len({r['focus'] for r in a}), 15)
    def test_insufficient_topics(self):
        with self.assertRaises(ValueError): choose_documents([])
    def test_metrics(self):
        rows = [{'category': 'covered', 'rank': 2, 'recall_at_5': 1, 'reference_retained': False},
                {'category': 'covered', 'rank': None, 'recall_at_5': 0, 'reference_retained': False},
                {'category': 'out_of_domain', 'accepted_count': 0}]
        m = metrics(rows)
        self.assertEqual(m['hit_at_1'], 0)
        self.assertEqual(m['recall_at_5'], .5)
        self.assertEqual(m['mrr_at_5'], .25)
        self.assertEqual(m['out_of_domain']['no_accepted_passage_rate'], 1)
    def test_missing_group_not_zero(self):
        self.assertIsNone(metrics([])['hit_at_1'])
    def test_write_read_unicode(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'x.json'
            write_json(p, {'question': 'définition ?'})
            self.assertEqual(load_json(p)['question'], 'définition ?')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--translate', action='store_true')
    p = sub.add_parser('run'); p.add_argument('--answers', action='store_true'); p.add_argument('--reviewed-only', action='store_true')
    sub.add_parser('self-test')
    args = parser.parse_args()
    if args.command == 'self-test':
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
        sys.exit(0 if result.wasSuccessful() else 1)
    root, cfg = root_and_config()
    path = root / 'data/validation_v1.json'
    if args.command == 'prepare': prepare(root, cfg, path, args.translate)
    else: run(root, cfg, path, args.answers, args.reviewed_only)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'ERREUR : {exc}', file=sys.stderr)
        sys.exit(1)
