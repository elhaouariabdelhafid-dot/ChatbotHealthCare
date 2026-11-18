import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from healthbot.engine import HealthBot
from healthbot.reranking import PassageReranker, configure_reranking
from healthbot.settings import settings


def hit(name, accepted=True):
    return dict(id=name+':0', document_id=name, title=name, text='Passage '+name,
                source_url='https://example.org/'+name, language='fr', accepted=accepted,
                score=.9, dense_score=.9, lexical_score=.1)


def bot(enabled=True, top_k=1):
    cfg=dict(settings(), reranking_enabled=enabled, retriever='dense', top_k=top_k)
    with patch('healthbot.engine.Retriever') as retriever, patch('healthbot.engine.IntentModel') as intent:
        retriever.return_value.hash='test'
        intent.return_value.predict.return_value={'label':'definition','score':1}
        return HealthBot(cfg)


class RerankingTests(unittest.TestCase):
    def test_lazy_load_and_preserve_scores_and_ties(self):
        model=Mock(); model.predict.return_value=[.5,.5,2.0]
        factory=Mock(return_value=model)
        fake=types.SimpleNamespace(CrossEncoder=factory)
        hits=[hit('a'),hit('b'),hit('c',False)]
        r=PassageReranker();factory.assert_not_called()
        with patch.dict('sys.modules', {'sentence_transformers':fake}):
            ordered=r.rerank('question',hits);r.rerank('question',hits)
        factory.assert_called_once()
        self.assertEqual([h['document_id'] for h in ordered],['c','a','b'])
        self.assertFalse(ordered[0]['accepted']);self.assertEqual(ordered[0]['dense_score'],.9)
        self.assertNotIn('reranker_score',hits[0])
        self.assertEqual(model.predict.call_args.kwargs['batch_size'],4)

    def test_invalid_scores_rejected(self):
        for scores in [[float('nan')],[float('inf')],[]]:
            r=PassageReranker();r._model=Mock();r._model.predict.return_value=scores
            with self.assertRaises(ValueError):r.rerank('q',[hit('a')])

    def test_disabled_uses_original_search(self):
        b=bot(False);b.retriever.search.return_value=[hit('a'),hit('b',False)]
        selected,trace=b._select_passages('q')
        b.retriever.search.assert_called_once_with('q')
        self.assertEqual([h['document_id'] for h in selected],['a'])
        self.assertEqual(trace['reranking_status'],'disabled')

    def test_rerank_then_topk_then_filter(self):
        b=bot();hits=[hit('a'),hit('b',False),hit('c')]
        b.retriever.search.return_value=hits;b.reranker=Mock()
        b.reranker.rerank.return_value=[hits[1],hits[2],hits[0]]
        selected,trace=b._select_passages('q')
        b.retriever.search.assert_called_once_with('q',k=5)
        self.assertEqual(selected,[])
        self.assertEqual(trace['reranking_status'],'applied')

    def test_no_accepted_skips_model(self):
        b=bot();b.retriever.search.return_value=[hit('a',False)];b.reranker=Mock()
        selected,trace=b._select_passages('q')
        b.reranker.rerank.assert_not_called();self.assertEqual(selected,[])
        self.assertEqual(trace['reranking_status'],'skipped_no_accepted')

    def test_failure_falls_back_to_e5_explicitly(self):
        b=bot();b.retriever.search.return_value=[hit('a'),hit('b')]
        b.reranker=Mock();b.reranker.rerank.side_effect=RuntimeError('offline')
        selected,trace=b._select_passages('q')
        self.assertEqual(selected[0]['document_id'],'a')
        self.assertEqual(trace['reranking_status'],'fallback_e5');self.assertIn('warning',trace)

    def test_sources_follow_reranked_order(self):
        b=bot();a,c=hit('a'),hit('c')
        b.retriever.search.return_value=[a,c];b.reranker=Mock();b.reranker.rerank.return_value=[c,a]
        with patch('healthbot.engine.safety_route',return_value=None), patch('healthbot.engine.generate',return_value=('Réponse [1]','ollama',None)) as gen:
            result=b.ask('Définition diabète')
        self.assertEqual(gen.call_args.args[1],[c])
        self.assertEqual(result['sources'][0]['document_id'],'c')
        self.assertEqual(result['sources'][0]['citation'],1)
        self.assertEqual(result['retrieval']['reranking_status'],'applied')

    def test_routes_bypass_retrieval_and_reranker(self):
        for route in ['safety','personal']:
            b=bot();b.reranker=Mock()
            with patch('healthbot.engine.safety_route',return_value=route):result=b.ask('Question santé')
            self.assertEqual(result['status'],route)
            b.retriever.search.assert_not_called();b.reranker.rerank.assert_not_called()

    def test_generator_abstention_preserved(self):
        b=bot();h=hit('a');b.retriever.search.return_value=[h]
        b.reranker=Mock();b.reranker.rerank.return_value=[h]
        with patch('healthbot.engine.safety_route',return_value=None), patch('healthbot.engine.generate',return_value=('Insuffisant','abstain',None)):
            result=b.ask('Question diabète')
        self.assertEqual(result['status'],'abstain');self.assertEqual(result['sources'],[])

    def test_invalid_configuration(self):
        cfg=dict(retriever='dense',top_k=3,reranking_enabled=True)
        for update in [dict(reranking_enabled='true'),dict(reranking_candidates=2),dict(reranking_candidates=True),dict(retriever='lexical'),dict(reranking_model='')]:
            with self.assertRaises(ValueError):configure_reranking(dict(cfg,**update))
        self.assertFalse(configure_reranking(dict(retriever='lexical',top_k=3))['reranking_enabled'])

    def test_settings_reads_new_keys(self):
        cfg=dict(settings(),reranking_enabled=True,reranking_candidates=5)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'cfg.json';p.write_text(json.dumps(cfg));loaded=settings(p)
        self.assertTrue(loaded['reranking_enabled']);self.assertEqual(loaded['reranking_candidates'],5)

if __name__=='__main__':unittest.main()
