import io,json,threading,unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from urllib.error import HTTPError
from healthbot import groq_provider as gp
from healthbot.generation import generate

class GroqTests(unittest.TestCase):
 def setUp(self):
  gp.ADMISSION=gp.Admission();gp.TRACE.set(None)
  self.cfg=gp.configure_groq({'generation':'groq','ollama_model':'unused'})
  self.hits=[{'text':'La cause est inconnue.','title':'Cause'}]
  p=patch.dict('os.environ',{'GROQ_API_KEY':'test-secret'});p.start();self.addCleanup(p.stop)
 def response(self,refs=None,insufficient=False,finish='stop'):
  content=json.dumps({'insufficient':insufficient,'statements':[] if insufficient else [{'text':'La cause est inconnue.','sources':refs or [1]}]})
  return io.BytesIO(json.dumps({'choices':[{'finish_reason':finish,'message':{'content':content}}],'usage':{'total_tokens':100}}).encode())
 def ask(self):return generate('Cause ?',self.hits,self.cfg)
 def test_success_and_secret_redaction(self):
  with patch.object(gp,'urlopen',return_value=self.response()) as net:
   a,m,w=self.ask()
  self.assertEqual(m,'groq');self.assertIsNone(w);self.assertIn('[1]',a)
  self.assertNotIn('test-secret',json.dumps(gp.snapshot()))
  self.assertEqual(net.call_args.args[0].get_header('User-agent'),'ChatBotHealthCare/1.0')
 def test_abstention(self):
  with patch.object(gp,'urlopen',return_value=self.response(insufficient=True)):
   self.assertEqual(self.ask()[1],'abstain')
 def test_invalid_citations(self):
  with patch.object(gp,'urlopen',return_value=self.response(refs=[3])):
   self.assertEqual(self.ask()[1],'extractive')
  self.assertEqual(gp.snapshot()['status'],'invalid_response')
 def test_missing_key(self):
  with patch.dict('os.environ',{'GROQ_API_KEY':''}),patch.object(gp,'urlopen') as net:
   self.assertEqual(self.ask()[1],'extractive');net.assert_not_called()
  self.assertEqual(gp.snapshot()['status'],'missing_key')
 def test_429_and_cooldown(self):
  e=HTTPError('',429,'',{'Retry-After':'12'},io.BytesIO(b'secret'))
  with patch.object(gp,'urlopen',side_effect=e) as net:
   self.ask();self.assertEqual(gp.snapshot()['status'],'rate_limited')
   self.ask();self.assertEqual(gp.snapshot()['status'],'cooldown');self.assertEqual(net.call_count,1)
 def test_503_and_cooldown(self):
  with patch.object(gp,'urlopen',side_effect=HTTPError('',503,'',{},None)) as net:
   self.ask();self.ask();self.assertEqual(net.call_count,1)
 def test_timeout_releases_slot(self):
  with patch.object(gp,'urlopen',side_effect=TimeoutError):self.ask()
  self.assertEqual(gp.snapshot()['status'],'timeout');self.assertEqual(gp.ADMISSION.active,0)
 def test_malformed_and_truncated(self):
  for response in [io.BytesIO(b'not json'),self.response(finish='length')]:
   gp.ADMISSION=gp.Admission()
   with patch.object(gp,'urlopen',return_value=response):self.assertEqual(self.ask()[1],'extractive')
 def test_usage_reconciliation(self):
  cfg=dict(self.cfg,groq_tpm=100)
  a=gp.ADMISSION.reserve(60,cfg)
  with self.assertRaises(gp.GroqUnavailable):gp.ADMISSION.reserve(60,cfg)
  gp.ADMISSION.finish(a,10)
  b=gp.ADMISSION.reserve(60,cfg);gp.ADMISSION.finish(b,10)
  self.assertEqual(sum(e[1] for e in gp.ADMISSION.events),20)
 def test_concurrent_admission(self):
  barrier=threading.Barrier(5)
  cfg=dict(self.cfg,groq_tpm=100000,groq_max_concurrent=2)
  def worker():
   barrier.wait()
   try:return gp.ADMISSION.reserve(1,cfg)
   except gp.GroqUnavailable:return None
  with ThreadPoolExecutor(max_workers=5) as pool:results=list(pool.map(lambda _:worker(),range(5)))
  admitted=[x for x in results if x is not None]
  self.assertEqual(len(admitted),2)
  for event in admitted:gp.ADMISSION.finish(event,1)
  self.assertEqual(gp.ADMISSION.active,0)
 def test_invalid_configuration(self):
  for key in ['groq_tpm','groq_rpm','groq_timeout_seconds','groq_max_concurrent']:
   with self.assertRaises(ValueError):gp.configure_groq({key:0})
 def test_window_expiration(self):
  cfg=dict(self.cfg,groq_tpm=100)
  with patch.object(gp.time,'monotonic',return_value=100):
   event=gp.ADMISSION.reserve(60,cfg);gp.ADMISSION.finish(event)
  with patch.object(gp.time,'monotonic',return_value=161):
   event=gp.ADMISSION.reserve(60,cfg);gp.ADMISSION.finish(event)
  self.assertEqual(len(gp.ADMISSION.events),1)
 def test_engine_marks_degraded_and_preserves_sources(self):
  from healthbot.engine import HealthBot
  from healthbot.settings import settings
  hit=dict(id='a:0',document_id='a',title='A',text='Cause inconnue.',source_url='https://example.org',language='fr',accepted=True)
  cfg=dict(settings(),generation='groq',retriever='lexical',reranking_enabled=False)
  with patch('healthbot.engine.Retriever') as retriever,patch('healthbot.engine.IntentModel') as intent:
   retriever.return_value.hash='test';retriever.return_value.search.return_value=[hit]
   intent.return_value.predict.return_value={'label':'definition','score':1}
   bot=HealthBot(cfg)
  with patch.dict('os.environ',{'GROQ_API_KEY':''}):result=bot.ask('Définition diabète')
  self.assertEqual(result['status'],'degraded');self.assertEqual(result['sources'][0]['id'],'a:0')
  self.assertEqual(result['generation_trace']['status'],'missing_key')
  with patch('healthbot.engine.safety_route',return_value='safety'):
   result=bot.ask('Question santé')
  self.assertNotIn('generation_trace',result)
