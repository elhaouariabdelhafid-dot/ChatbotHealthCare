# Reranking optionnel intégré

Le moteur HealthBot récupère cinq candidats E5, reclasse les cinq avec
cross-encoder/mmarco-mMiniLMv2-L12-H384-v1, prend les top_k premiers,
puis conserve ceux dont accepted vaut true. Il ne complète pas la liste
avec les candidats suivants quand un passage est rejeté. Le seuil dense
reste inchangé. Les scores mMiniLM ne sont pas des probabilités et ne
remplacent pas les scores E5 pour accepter un passage.

Configuration installée :

```json
{
  "reranking_enabled": true,
  "reranking_candidates": 5,
  "reranking_model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
}
```

La valeur top_k existante est conservée (3 dans la configuration fournie).
Pour comparer sans écrire config.json, utiliser dict(settings(),
reranking_enabled=False) puis HealthBot(cfg). Pour désactiver durablement,
mettre reranking_enabled à false. Le rewriting reste absent du moteur.

mMiniLM est chargé sur CPU à la première recherche ayant au moins un
candidat accepté. Paramètres identiques au test : max_length=384,
batch_size=4, texte = titre + '. ' + passage. Il reste chargé pour les
requêtes suivantes. Aucun appel à Ollama n'est effectué par le reranker.
Les routes sécurité, demande personnelle, langue non prise en charge
et clarification passent avant la recherche et le reranking.

La réponse JSON contient retrieval.reranking_status :
- disabled : option désactivée ; chemin de recherche initial ;
- applied : reclassement réussi ;
- skipped_no_accepted : aucun candidat accepté, modèle non chargé ;
- fallback_e5 : erreur rattrapée, ordre E5 conservé et warning dans retrieval.

rerank_ms inclut le chargement lors du premier appel. candidate_order
trace les identifiants, scores denses, scores du reranker et accepted.
Les citations restent numérotées dans l'ordre réellement transmis au LLM.
Le comportement existant de génération/extraits n'est pas modifié.

## Vérification

```bash
python -m unittest discover -s tests -v
```

Les tests historiques utilisent désormais explicitement le corpus de
 démonstration et désactivent le reranking. Ils ne dépendent plus du corpus
MedQuAD choisi dans config.json. Les nouveaux tests simulent les modèles :
ordre, seuils, stabilité des égalités, erreurs, citations, abstention et routes.
Un test réel reste nécessaire pour la mémoire et la qualité de réponse.

Attention : Retriever.search reste une recherche brute. Les anciens scripts
qui l'appellent directement ne mesurent pas le reranking intégré à HealthBot.
Les métriques de recherche de validation_healthcare.py restent donc celles
 du retriever brut, même si ses réponses --answers utilisent HealthBot.
Ne pas les présenter comme des scores du pipeline reranké.

## Limites des résultats

L'expérience sur 15 questions documentaires a donné Hit@1 14/15 pour E5
et 15/15 après reclassement, sur les mêmes cinq candidats. N05 a fourni une
réponse appuyée par le passage reranké alors que le passage initial conduisait
à une abstention. Ce lot de développement ne constitue pas une validation
clinique ni une garantie de généralisation. Le test de génération N05 avait
un seul passage ; l'intégration conserve top_k=3 pour ne pas changer deux
paramètres à la fois. E5, mMiniLM et Qwen peuvent résider simultanément en
mémoire : leur cohabitation doit être vérifiée sur WSL.
