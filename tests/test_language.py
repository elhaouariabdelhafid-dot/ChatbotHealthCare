import json
import unittest
from unittest.mock import patch
from healthbot.language import question_language, localize
from healthbot.generation import generate
from healthbot.settings import settings
from healthbot.engine import HealthBot

class LanguageTests(unittest.TestCase):
    def test_fr_en(self):
        for q, expected in [("What does the acronym TARP stand for in TARP syndrome?",'en'),("What is diabetes?",'en'),("What was my last blood glucose value?",'en'),("Hi",'en'),("Qu’est-ce que le diabète ?",'fr'),("c est quoi le diabète ?",'fr'),("Quels sont les symptômes ?",'fr'),("bnj",'fr'),("TARP",'fr')]:
            with self.subTest(q=q): self.assertEqual(question_language(q),expected)
    def test_prompt_and_abstention_both_modes(self):
        for compact in [True,False]:
            for question, language in [('What is diabetes?', 'English'),('Qu’est-ce que le diabète ?', 'French')]:
                cfg=dict(settings(),generation='groq',compact_prompt=compact)
                with patch('healthbot.generation.groq_provider.complete',return_value=json.dumps({'insufficient':True,'statements':[]})) as net:
                    answer,mode,_=generate(question,[{'text':'Document.'}],cfg)
                prompt=net.call_args.args[0]['messages'][0]['content']
                self.assertIn('RESPONSE LANGUAGE: '+language,prompt)
                self.assertNotIn('en français',prompt)
                self.assertEqual(mode,'abstain')
                self.assertTrue(answer.startswith('The available') if language=='English' else answer.startswith('Les passages'))
    def test_local_answers_skip_api(self):
        cfg=dict(settings(),retriever='lexical',reranking_enabled=False)
        with patch('healthbot.engine.Retriever'),patch('healthbot.engine.IntentModel'):
            bot=HealthBot(cfg)
        with patch('healthbot.engine.generate') as gen:
            self.assertTrue(bot.ask('Hello')['answer'].startswith('Hello'))
            result=bot.ask('What was my last blood glucose value?')
            self.assertTrue(result['answer'].startswith('I do not have access'))
            self.assertEqual(result['sources'],[])
            gen.assert_not_called()
    def test_excerpts_not_translated(self):
        cfg=dict(settings(),generation='extractive')
        answer,mode,_=generate('What is diabetes?',[{'text':'Texte français original.'}],cfg)
        self.assertEqual(answer,'[1] Texte français original.')
