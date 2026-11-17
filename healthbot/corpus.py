import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

def load_documents(path):
    docs, ids = [], set()
    for line_no, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if not line.strip():
            continue
        d = json.loads(line)
        for field in ('id','title','text','source_url','language','license_status'):
            if not isinstance(d.get(field),str) or not d[field].strip():
                raise ValueError(f'Ligne {line_no}: champ absent/vide {field}')
        if d['id'] in ids:
            raise ValueError('Identifiant dupliqué: '+d['id'])
        ids.add(d['id'])
        if urlparse(d['source_url']).scheme != 'https':
            raise ValueError('Une source HTTPS est requise')
        # Inclusion explicite seulement : les imports restent en attente de revue.
        if d.get('approved') is True:
            docs.append(d)
    if not docs:
        raise ValueError('Aucun document approuvé. Vérifier corpus et approved.')
    return docs

def chunk_documents(docs, size=140, overlap=25):
    if size < 20 or not 0 <= overlap < size:
        raise ValueError('Découpage invalide')
    chunks=[]
    for d in docs:
        words=d['text'].split()
        for n,start in enumerate(range(0,len(words),size-overlap)):
            text=' '.join(words[start:start+size])
            chunks.append({**d,'document_id':d['id'],'id':f"{d['id']}:{n}",'text':text})
            if start+size >= len(words):
                break
    return chunks

def fingerprint(chunks):
    return hashlib.sha256(json.dumps(chunks,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
