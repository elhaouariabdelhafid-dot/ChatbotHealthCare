"""Reranking optionnel des candidats E5 ; aucun score de rejet appris."""
import math
import threading

DEFAULT_MODEL = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'


def configure_reranking(cfg):
    cfg.setdefault('reranking_enabled', False)
    cfg.setdefault('reranking_candidates', 5)
    cfg.setdefault('reranking_model', DEFAULT_MODEL)
    if type(cfg['reranking_enabled']) is not bool:
        raise ValueError('reranking_enabled doit être un booléen JSON.')
    count = cfg['reranking_candidates']
    if type(count) is not int or not 1 <= count <= 50:
        raise ValueError('reranking_candidates doit être un entier entre 1 et 50.')
    if not isinstance(cfg['reranking_model'], str) or not cfg['reranking_model'].strip():
        raise ValueError('reranking_model doit être un identifiant non vide.')
    if cfg['reranking_enabled']:
        if cfg['retriever'] != 'dense':
            raise ValueError('Le reranking intégré est évalué avec retriever=dense uniquement.')
        if type(cfg['top_k']) is not int or not 1 <= cfg['top_k'] <= count:
            raise ValueError('top_k doit être positif et inférieur ou égal à reranking_candidates.')
    return cfg


class PassageReranker:
    """Chargement à la première utilisation ; accès sérialisé pour l'API."""
    def __init__(self, model_name=DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def rerank(self, question, hits):
        if not hits:
            return []
        pairs = [(question, h.get('title', '') + '. ' + h['text']) for h in hits]
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, device='cpu', max_length=384)
            scores = self._model.predict(pairs, batch_size=4, show_progress_bar=False)
        values = [float(s) for s in scores]
        if len(values) != len(hits) or not all(math.isfinite(s) for s in values):
            raise ValueError('Scores mMiniLM invalides.')
        scored = [dict(h, e5_rank=i, reranker_score=s)
                  for i, (h, s) in enumerate(zip(hits, values), 1)]
        # Tri stable ; accepted et les scores E5 ne sont jamais remplacés.
        return sorted(scored, key=lambda h: h['reranker_score'], reverse=True)
