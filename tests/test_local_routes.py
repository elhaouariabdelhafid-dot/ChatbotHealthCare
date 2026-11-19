import unittest
from unittest.mock import patch
from healthbot.local_routes import local_reply
from healthbot.engine import HealthBot
from healthbot.settings import settings

class LocalRoutesTests(unittest.TestCase):
    def test_simple_greetings_only(self):
        self.assertEqual(local_reply("bnj")['mode'], 'local')
        self.assertIsNone(local_reply("Bonjour, qu’est-ce que le diabète ?"))
    def test_personal_results(self):
        for q in ["Quelle était ma dernière glycémie ?", "Quelle allergie est inscrite dans mon dossier hospitalier ?", "Quel vaccin ai-je reçu lors de ma dernière consultation ?", "What was the exact hemoglobin value in my most recent blood test?"]:
            with self.subTest(q=q): self.assertEqual(local_reply(q)['status'], 'abstain')
    def test_general_questions_preserved(self):
        for q in ["Qu’est-ce que la glycémie ?", "Comment mesure-t-on la tension ?", "What is blood sugar?", "Bonjour, ma glycémie est élevée, que signifie ce terme ?"]:
            with self.subTest(q=q): self.assertIsNone(local_reply(q))
    def test_routes_skip_search_and_generation(self):
        cfg=dict(settings(),retriever='lexical',reranking_enabled=False)
        with patch('healthbot.engine.Retriever') as retriever, patch('healthbot.engine.IntentModel') as intent:
            retriever.return_value.hash='test'
            intent.return_value.predict.return_value={}
            bot=HealthBot(cfg)
        with patch('healthbot.engine.generate') as gen, patch('healthbot.engine.safety_route', return_value=None):
            for q in ['bnj','Quelle était ma dernière glycémie ?']:
                result=bot.ask(q)
                self.assertEqual(result['mode'],'local')
                self.assertEqual(result['sources'],[])
            retriever.return_value.search.assert_not_called()
            gen.assert_not_called()
    def test_emergency_keeps_priority(self):
        cfg=dict(settings(),retriever='lexical',reranking_enabled=False)
        with patch('healthbot.engine.Retriever'), patch('healthbot.engine.IntentModel'):
            bot=HealthBot(cfg)
        with patch('healthbot.engine.safety_route',return_value='safety'):
            self.assertEqual(bot.ask('Quelle était ma dernière glycémie ?')['status'],'safety')
