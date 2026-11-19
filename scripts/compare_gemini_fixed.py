import argparse
import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from healthbot import generation

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()

key = os.environ.get("GEMINI_API_KEY", "").strip()
if not key:
    raise SystemExit("GEMINI_API_KEY absente dans ce terminal.")

source = ROOT / "artifacts/qwen7b_top2_25_20261119_131109.json"
baseline = json.loads(source.read_text(encoding="utf-8"))

current_hash = hashlib.sha256(
    (ROOT / "healthbot/generation.py").read_bytes()
).hexdigest()
expected_hash = baseline["code_sha256"]["healthbot/generation.py"]
if current_hash != expected_hash:
    raise SystemExit(
        "generation.py a changé depuis le test Qwen. "
        "Arrêt pour éviter de comparer des prompts différents."
    )

model = "gemini-2.5-flash"
cfg = dict(baseline["configuration"])
cfg.update(generation="ollama", compact_prompt=False)

cases = baseline["details"]
if args.smoke:
    cases = [r for r in cases if r["id"] == "N02"]

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output = ROOT / "artifacts" / f"gemini_fixed_{stamp}.json"
report = {
    "model": model,
    "baseline_report": source.name,
    "baseline_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "generation_code_sha256": current_hash,
    "smoke_test": args.smoke,
    "completed": False,
    "details": [],
}
observed = {}

def save():
    temp = output.with_suffix(".tmp")
    temp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(output)

def open_with_retry(request):
    observed["attempts"] = []
    for attempt in range(1, 4):
        started = time.perf_counter()
        try:
            response = urlopen(request, timeout=120)
            observed["attempts"].append({
                "attempt": attempt,
                "http_status": response.status,
                "seconds": round(time.perf_counter() - started, 3),
            })
            return response
        except HTTPError as exc:
            observed["attempts"].append({
                "attempt": attempt,
                "http_status": exc.code,
                "seconds": round(time.perf_counter() - started, 3),
            })
            if exc.code not in (500, 502, 503, 504) or attempt == 3:
                raise
            exc.close()
            delay = (5, 15)[attempt - 1]
            print(
                f"Service indisponible : tentative {attempt}/3. "
                f"Nouvel essai dans {delay} secondes.",
                flush=True,
            )
            time.sleep(delay)


def gemini_transport(ollama_request, *unused, **kwargs):
    original = json.loads(ollama_request.data)
    messages = original["messages"]

    body = {
        "systemInstruction": {
            "parts": [{"text": m["content"]} for m in messages
                      if m["role"] == "system"]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages if m["role"] == "user"
        ],
        "generationConfig": {
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",
            "responseJsonSchema": original["format"],
        },
    }
    observed["request_body"] = body

    request = Request(
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model}:generateContent",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": key,
        },
        method="POST",
    )

    started = time.perf_counter()
    try:
        with open_with_retry(request) as response:
            response_data = json.load(response)
    except HTTPError as exc:
        detail = exc.read(20000).decode("utf-8", errors="replace")
        detail = detail.replace(key, "[CLE_MASQUEE]")
        observed["error"] = f"HTTP {exc.code}"
        observed["error_detail"] = detail
        print("Détail de l'erreur Gemini :", detail, flush=True)
        raise OSError(observed["error"]) from None
    except (OSError, ValueError) as exc:
        observed["error"] = type(exc).__name__
        raise OSError("Appel Gemini échoué") from None
    finally:
        observed["api_seconds"] = round(time.perf_counter() - started, 3)

    observed["usage"] = response_data.get("usageMetadata", {})
    observed["model_version"] = response_data.get("modelVersion")
    candidates = response_data.get("candidates", [])
    if not candidates:
        observed["error"] = "Aucun candidat retourné"
        raise ValueError(observed["error"])

    candidate = candidates[0]
    observed["finish_reason"] = candidate.get("finishReason")
    text = "".join(
        p.get("text", "")
        for p in candidate.get("content", {}).get("parts", [])
        if not p.get("thought", False)
    )
    observed["raw_answer"] = text

    if candidate.get("finishReason") != "STOP" or not text.strip():
        observed["error"] = "Réponse bloquée, vide ou interrompue"
        raise ValueError(observed["error"])

    # Réutilise le parseur et les contrôles existants du projet.
    return io.BytesIO(json.dumps({
        "message": {"content": text}
    }).encode("utf-8"))

save()
print("Rapport :", output, flush=True)

for index, row in enumerate(cases, 1):
    print(f"\n[{index}/{len(cases)}] {row['id']}", flush=True)
    passages = row.get("generation_passages", [])
    observed.clear()

    if not passages:
        comparison = {
            "id": row["id"],
            "question": row["question"],
            "category": row["category"],
            "api_called": False,
            "note": "Routage local conservé ; pas un résultat Gemini.",
            "qwen_pipeline_result": row["result"],
        }
    else:
        started = time.perf_counter()
        with patch.object(generation, "urlopen", gemini_transport):
            answer, mode, warning = generation.generate(
                row["question"], passages, cfg
            )

        comparison = {
            "id": row["id"],
            "question": row["question"],
            "category": row["category"],
            "api_called": True,
            "qwen_pipeline_result": row["result"],
            "qwen_pipeline_seconds": row["wall_seconds"],
            "generation_passages": passages,
            "gemini": {
                "answer": answer,
                "mode": "gemini" if mode == "ollama" else mode,
                "warning": warning,
                "generation_seconds": round(time.perf_counter() - started, 3),
                **observed,
            },
        }
        print("Mode :", comparison["gemini"]["mode"], flush=True)
        print("Réponse :", answer, flush=True)
        print("Temps API :", observed.get("api_seconds"), "s", flush=True)
        if warning:
            print("Contrôle existant :", warning, flush=True)

    report["details"].append(comparison)
    save()

    if observed.get("error"):
        print("ARRÊT :", observed["error"], flush=True)
        print("Rapport partiel conservé :", output)
        raise SystemExit(1)

report["completed"] = True
save()
print("\nTERMINÉ — rapport :", output)
