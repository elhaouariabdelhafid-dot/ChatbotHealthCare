"""Lancer depuis la racine : python scripts/fix_session_routes.py."""
from pathlib import Path
from datetime import datetime
import json
import shutil

root=Path.cwd()
engine=root/'healthbot/engine.py'
if not engine.exists(): raise SystemExit('Lance depuis ~/ChatBotHealthCare.')
text=engine.read_text(encoding='utf-8')
if 'from .local_routes import local_reply' in text:
    raise SystemExit('Correction déjà installée ; aucun changement.')
needle="        elif route=='personal':"
if text.count(needle)!=1 or text.count('from . import groq_provider')!=1:
    raise SystemExit('Structure différente : aucun fichier modifié.')
files={'healthbot/local_routes.py': '"""Routes déterministes limitées ; aucune donnée personnelle déduite du corpus."""\nimport re\nimport unicodedata\n\ndef normalize(text):\n    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))\n\ndef local_reply(question):\n    q = normalize(question)\n    greeting = re.sub(r"[^a-z ]", "", q).strip()\n    if greeting in {"bnj", "bjr", "bonjour", "bonsoir", "salut", "hello", "hi", "coucou", "merci", "merci beaucoup"}:\n        return {"status": "answered", "mode": "local", "answer": "Bonjour ! Je peux expliquer des termes de santé à partir des documents disponibles. Quelle est ta question ?"}\n    personal = bool(re.search(r"\\b(ma|mon|mes|my|moi|ai[- ]je|did i|was i)\\b", q))\n    record = bool(re.search(r"\\b(dossier|compte rendu|rendez.vous|record|report|appointment)\\b", q))\n    measure = bool(re.search(r"\\b(glycemie|tension|hemoglobine|groupe sanguin|irm|vaccin|allergie|blood|glucose|hemoglobin|vaccine|vaccination|allergy|scan)\\b", q))\n    past = bool(re.search(r"\\b(dernier|derniere|derniers|dernieres|last|latest|recent|recentes|recently|mesuree|mesure|recu|recue|received|indique|inscrit|inscrite|etait|was|resultat|result|valeur|value)\\b", q))\n    if personal and (record or (measure and past)):\n        return {"status": "abstain", "mode": "local", "answer": "Je n’ai pas accès à ton dossier médical ni à tes résultats personnels. Je ne peux donc pas retrouver cette information. Consulte ton compte rendu ou le professionnel qui te suit. Tu peux me demander une explication générale d’un terme médical."}\n    return None\n', 'tests/test_local_routes.py': 'import unittest\nfrom unittest.mock import patch\nfrom healthbot.local_routes import local_reply\nfrom healthbot.engine import HealthBot\nfrom healthbot.settings import settings\n\nclass LocalRoutesTests(unittest.TestCase):\n    def test_simple_greetings_only(self):\n        self.assertEqual(local_reply("bnj")[\'mode\'], \'local\')\n        self.assertIsNone(local_reply("Bonjour, qu’est-ce que le diabète ?"))\n    def test_personal_results(self):\n        for q in ["Quelle était ma dernière glycémie ?", "Quelle allergie est inscrite dans mon dossier hospitalier ?", "Quel vaccin ai-je reçu lors de ma dernière consultation ?", "What was the exact hemoglobin value in my most recent blood test?"]:\n            with self.subTest(q=q): self.assertEqual(local_reply(q)[\'status\'], \'abstain\')\n    def test_general_questions_preserved(self):\n        for q in ["Qu’est-ce que la glycémie ?", "Comment mesure-t-on la tension ?", "What is blood sugar?", "Bonjour, ma glycémie est élevée, que signifie ce terme ?"]:\n            with self.subTest(q=q): self.assertIsNone(local_reply(q))\n    def test_routes_skip_search_and_generation(self):\n        cfg=dict(settings(),retriever=\'lexical\',reranking_enabled=False)\n        with patch(\'healthbot.engine.Retriever\') as retriever, patch(\'healthbot.engine.IntentModel\') as intent:\n            retriever.return_value.hash=\'test\'\n            intent.return_value.predict.return_value={}\n            bot=HealthBot(cfg)\n        with patch(\'healthbot.engine.generate\') as gen, patch(\'healthbot.engine.safety_route\', return_value=None):\n            for q in [\'bnj\',\'Quelle était ma dernière glycémie ?\']:\n                result=bot.ask(q)\n                self.assertEqual(result[\'mode\'],\'local\')\n                self.assertEqual(result[\'sources\'],[])\n            retriever.return_value.search.assert_not_called()\n            gen.assert_not_called()\n    def test_emergency_keeps_priority(self):\n        cfg=dict(settings(),retriever=\'lexical\',reranking_enabled=False)\n        with patch(\'healthbot.engine.Retriever\'), patch(\'healthbot.engine.IntentModel\'):\n            bot=HealthBot(cfg)\n        with patch(\'healthbot.engine.safety_route\',return_value=\'safety\'):\n            self.assertEqual(bot.ask(\'Quelle était ma dernière glycémie ?\')[\'status\'],\'safety\')\n'}
for name in files:
    if (root/name).exists(): raise SystemExit('Fichier déjà présent : '+name)
configs={}
for name in ['config.json','config.groq.json']:
    p=root/name
    if p.exists():
        cfg=json.loads(p.read_text(encoding='utf-8'))
        if cfg.get('generation')=='groq':
            cfg['groq_max_completion_tokens']=2048
            configs[name]=json.dumps(cfg,ensure_ascii=False,indent=2)+'\n'
backup=root/'artifacts'/('backup_session_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
backup.mkdir(parents=True)
for name in ['healthbot/engine.py',*configs]:
    dst=backup/name
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/name,dst)
text=text.replace('from . import groq_provider','from . import groq_provider\nfrom .local_routes import local_reply',1)
text=text.replace(needle,"        elif (local := local_reply(question)) is not None:\n            result.update(local)\n"+needle,1)
engine.write_text(text.rstrip()+'\n',encoding='utf-8')
for name,content in {**files,**configs}.items():
    p=root/name
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(content,encoding='utf-8')
print('Routes locales installées ; plafond de sortie Groq : 2048 tokens.')
print('Quota TPM conservé. Réponses tronquées toujours refusées.')
print('Sauvegarde :',backup)
print('Redémarre FastAPI après les tests.')
