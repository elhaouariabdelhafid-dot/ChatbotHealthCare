from functools import lru_cache
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from .engine import HealthBot
from .settings import ROOT

app=FastAPI(title='ChatBotHealthCare',version='1.0.0',description='Prototype documentaire local. Pas un service médical.')
app.mount('/static',StaticFiles(directory=ROOT/'web'),name='static')
@lru_cache(maxsize=1)
def get_bot():return HealthBot()
class Question(BaseModel):
    question:str=Field(min_length=1,max_length=1200)
@app.get('/')
def home():return FileResponse(ROOT/'web/index.html')
@app.get('/health')
def health():return {'status':'ok','version':'1.0.0'}
@app.get('/api/info')
def info():
    bot=get_bot()
    return {'documents':len({x['document_id'] for x in bot.retriever.chunks}),'chunks':len(bot.retriever.chunks),'retriever':bot.cfg['retriever'],'generation':bot.cfg['generation'],'corpus_hash':bot.retriever.hash,'training_examples':bot.intent.training_count}
@app.post('/api/chat')
def chat(body:Question):
    try:return get_bot().ask(body.question)
    except ValueError as e:raise HTTPException(status_code=422,detail=str(e)) from e
