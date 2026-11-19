import json
from pathlib import Path
from .reranking import configure_reranking
from .groq_provider import configure_groq
ROOT = Path(__file__).resolve().parents[1]
def settings(path=None):
    p = Path(path) if path else ROOT / 'config.json'
    cfg = json.loads(p.read_text(encoding='utf-8'))
    for key in ('corpus','intents'):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    if cfg['retriever'] not in ('lexical','dense','hybrid'):
        raise ValueError('retriever doit être lexical, dense ou hybrid')
    if cfg['generation'] not in ('extractive','ollama','groq'):
        raise ValueError('generation doit être extractive, ollama ou groq')
    if not 1 <= cfg['top_k'] <= 5:
        raise ValueError('top_k doit être compris entre 1 et 5')
    return configure_groq(configure_reranking(cfg))

