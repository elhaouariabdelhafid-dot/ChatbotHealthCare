import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from healthbot.settings import settings
from healthbot.engine import HealthBot
from healthbot.nlp import safety_route,entities
from healthbot.corpus import chunk_documents,load_documents
from healthbot.generation import valid_citations,generate
from healthbot.cli import import_medquad,import_meddialog
from healthbot.api import app

class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = {
            **settings(),
            'generation': 'extractive',
            'retriever': 'lexical',
        }
        cls.bot = HealthBot(cls.cfg)

        # Les tests API utilisent également ce moteur sans appel réel à Ollama.
        api_patch = patch('healthbot.api.get_bot', return_value=cls.bot)
        api_patch.start()
        cls.addClassCleanup(api_patch.stop)

        cls.client = TestClient(app)
    def test_definition_returns_source(self):
        r=self.bot.ask('Que signifie le diabète ?')
        self.assertEqual(r['status'],'answered')
        self.assertEqual(r['sources'][0]['document_id'],'diabete_definition')
    def test_unknown_medical_topic_abstains(self):
        self.assertEqual(self.bot.ask('Quels examens pour la maladie de Lyme ?')['status'],'abstain')
    def test_off_topic(self):
        self.assertEqual(self.bot.ask('Programme une boucle Python')['status'],'abstain')
    def test_empty(self):
        with self.assertRaises(ValueError):self.bot.ask('   ')
    def test_too_long(self):
        with self.assertRaises(ValueError):self.bot.ask('x'*1201)
    def test_safety_bypasses_retrieval(self):
        with patch.object(self.bot.retriever,'search',side_effect=AssertionError('called')):
            self.assertEqual(self.bot.ask('Je ne respire plus')['status'],'safety')
    def test_urgent_variants(self):
        for q in ["J’ai une douleur dans la poitrine",'Je n’arrive plus à respirer','Elle est inconsciente','ما قادرش نتنفس']:
            self.assertEqual(safety_route(q),'safety',q)
    def test_personal_dose(self):
        self.assertEqual(self.bot.ask('Combien de comprimés prendre ?')['status'],'personal')
    def test_arabic_explicit_limit(self):
        self.assertEqual(self.bot.ask('ما هو السكري')['status'],'unsupported')
    def test_clarification(self):
        self.assertEqual(self.bot.ask('Comment faire ?')['status'],'clarify')
    def test_entity_offsets_and_negation(self):
        q='Pas de toux mais une fatigue.'
        e=entities(q)
        for x in e:self.assertEqual(q[x['start']:x['end']],x['text'])
        self.assertTrue(e[0]['negation_candidate']);self.assertFalse(e[1]['negation_candidate'])
    def test_chunks_cover_words(self):
        text=' '.join('mot'+str(i) for i in range(87))
        chunks=chunk_documents([{'id':'x','text':text}],size=20,overlap=4)
        self.assertEqual(set(text.split()),{w for c in chunks for w in c['text'].split()})
    def test_invalid_chunk_parameters(self):
        with self.assertRaises(ValueError):chunk_documents([],size=20,overlap=20)
    def test_citation_validation(self):
        self.assertTrue(valid_citations('Texte [1].',1))
        for text in ['Texte [7].','Texte sans référence','Texte [1] https://evil.example']:
            self.assertFalse(valid_citations(text,1))
    def test_ollama_offline_fallback(self):
        cfg={**self.cfg,'generation':'ollama'}
        hits=[{'text':'Passage de test.'}]
        with patch('healthbot.generation.urlopen',side_effect=OSError('offline')):
            text,mode,warning=generate('question',hits,cfg)
        self.assertEqual(mode,'extractive');self.assertIn('Passage de test',text);self.assertTrue(warning)
    def test_ollama_bad_citation_fallback(self):
        from unittest.mock import MagicMock
        response=MagicMock();response.__enter__.return_value.read.return_value=b'{"message":{"content":"Texte [99]"}}'
        with patch('healthbot.generation.urlopen',return_value=response):
            self.assertEqual(generate('q',[{'text':'preuve'}],{**self.cfg,'generation':'ollama'})[1],'extractive')
    def test_ollama_valid_response(self):
        from unittest.mock import MagicMock
        response=MagicMock();response.__enter__.return_value.read.return_value=b'{"message":{"content":"Information [1]."}}'
        with patch('healthbot.generation.urlopen',return_value=response):
            text,mode,warning=generate('q',[{'text':'preuve'}],{**self.cfg,'generation':'ollama'})
        self.assertEqual(mode,'ollama');self.assertIsNone(warning)
    def test_ollama_abstention(self):
        from unittest.mock import MagicMock
        response=MagicMock();response.__enter__.return_value.read.return_value=b'{"message":{"content":"INSUFFICIENT"}}'
        with patch('healthbot.generation.urlopen',return_value=response):
            self.assertEqual(generate('q',[{'text':'preuve'}],{**self.cfg,'generation':'ollama'})[1],'abstain')
    def test_ollama_malformed_response(self):
        from unittest.mock import MagicMock
        response=MagicMock();response.__enter__.return_value.read.return_value=b'not json'
        with patch('healthbot.generation.urlopen',return_value=response):
            self.assertEqual(generate('q',[{'text':'preuve'}],{**self.cfg,'generation':'ollama'})[1],'extractive')
    def test_dense_and_hybrid_with_fake_encoder(self):
        import types,numpy as np
        from healthbot.retrieval import Retriever
        class FakeEncoder:
            def __init__(self,*args,**kwargs):pass
            def get_sentence_embedding_dimension(self):return 3
            def encode(self,texts,**kwargs):
                vectors=np.array([[float('diab' in t.lower()),float('hypertension' in t.lower()),float('asthme' in t.lower())] for t in texts])
                return vectors / np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1)
        fake=types.SimpleNamespace(SentenceTransformer=FakeEncoder)
        with patch.dict('sys.modules',{'sentence_transformers':fake}):
            for mode in ['dense','hybrid']:
                retriever=Retriever({**self.cfg,'retriever':mode})
                hits=retriever.search('Definition diabete')
                self.assertTrue(hits[0]['accepted'])
                self.assertIn('diabete',hits[0]['document_id'])
    def test_medquad_import_filters_and_keeps_pending(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'a.xml').write_text('<Document url="https://example.org/doc"><Focus>Topic</Focus><QAPairs><QAPair><Question qtype="information">Question?</Question><Answer>Texte</Answer></QAPair><QAPair><Question>Vide?</Question><Answer/></QAPair></QAPairs></Document>')
            out=p/'out.jsonl';r=import_medquad(p,out)
            self.assertEqual(r['imported'],1);self.assertEqual(r['empty_answers'],1)
            self.assertFalse(json.loads(out.read_text())['approved'])
            with self.assertRaises(ValueError):load_documents(out)
    def test_meddialog_redacts_contact(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);src=p/'input.jsonl';out=p/'out.jsonl'
            src.write_text(json.dumps({'patient':'Email a@b.com','doctor':'Texte','source_url':'https://example.org'}))
            import_meddialog(src,out)
            self.assertNotIn('a@b.com',out.read_text());self.assertIn('pending_manual',out.read_text())
    def test_api(self):
        self.assertEqual(self.client.get('/health').status_code,200)
        r=self.client.post('/api/chat',json={'question':'Que signifie le diabète ?'})
        self.assertEqual(r.status_code,200);self.assertEqual(r.json()['status'],'answered')
    def test_api_invalid_input(self):
        for q in ['', 'x'*1201,'   ']:
            self.assertEqual(self.client.post('/api/chat',json={'question':q}).status_code,422)
    def test_static_ui(self):
        self.assertIn('ChatBotHealthCare',self.client.get('/').text)
        self.assertEqual(self.client.get('/static/app.js').status_code,200)
    def test_configuration(self):
        self.assertEqual(self.bot.cfg['generation'],'extractive')
        self.assertEqual(len(self.bot.retriever.chunks),6)
if __name__=='__main__':unittest.main()
