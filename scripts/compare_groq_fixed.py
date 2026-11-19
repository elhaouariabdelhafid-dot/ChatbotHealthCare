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
parser.add_argument("--resume")
args = parser.parse_args()

key = os.environ.get("GROQ_API_KEY", "").strip()
if not key:
    raise SystemExit("GROQ_API_KEY absente dans ce terminal.")

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

model = "openai/gpt-oss-120b"
cfg = dict(baseline["configuration"])
cfg.update(generation="ollama", compact_prompt=False)

cases = baseline["details"]
if args.smoke:
    cases = [r for r in cases if r["id"] == "N02"]

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output = ROOT / "artifacts" / f"groq_fixed_{stamp}.json"
report = {
    "model": model,
    "baseline_report": source.name,
    "baseline_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "generation_code_sha256": current_hash,
    "smoke_test": args.smoke,
    "completed": False,
    "details": [],
}

if args.resume:
    previous_path = Path(args.resume)
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    for field in ("model", "baseline_sha256", "generation_code_sha256"):
        if previous.get(field) != report.get(field):
            raise SystemExit("Reprise incompatible : " + field)
    if previous.get("smoke_test") or args.smoke:
        raise SystemExit("La reprise concerne uniquement le test complet.")

    kept = [
        row for row in previous["details"]
        if not row.get("api_called")
        or (
            not row.get("groq", {}).get("error")
            and row.get("groq", {}).get("mode") in ("groq", "abstain")
            and not row.get("groq", {}).get("warning")
        )
    ]
    report["details"] = kept
    report["resumed_from"] = str(previous_path)
    done_ids = {row["id"] for row in kept}
    cases = [row for row in cases if row["id"] not in done_ids]
    print(f"Reprise : {len(kept)} cas conservés ; {len(cases)} restants.")

observed = {}

def save():
    temp = output.with_suffix(".tmp")
    temp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(output)

last_request_started = 0.0

def open_with_retry(request):
    global last_request_started
    observed["attempts"] = []
    observed["pacing_seconds"] = 0.0

    for attempt in range(1, 4):
        # Espacement minimum entre deux débuts de requête.
        wait = max(0.0, 15.0 - (time.monotonic() - last_request_started))
        if wait:
            print(f"Espacement des appels : {wait:.1f} s", flush=True)
            time.sleep(wait)
            observed["pacing_seconds"] += round(wait, 3)

        last_request_started = time.monotonic()
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
            record = {
                "attempt": attempt,
                "http_status": exc.code,
                "seconds": round(time.perf_counter() - started, 3),
            }
            observed["attempts"].append(record)
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise

            if exc.code == 429:
                try:
                    delay = max(60.0, float(exc.headers.get("Retry-After", "60")))
                except (TypeError, ValueError):
                    delay = 60.0
            else:
                delay = (5.0, 15.0)[attempt - 1]

            record["retry_wait_seconds"] = delay
            code = exc.code
            exc.close()
            print(
                f"HTTP {code} : attente de {delay:.1f} s "
                f"avant tentative {attempt + 1}/3.",
                flush=True,
            )
            time.sleep(delay)


def groq_transport(ollama_request, *unused, **kwargs):
    original = json.loads(ollama_request.data)
    body = {
        "model": model,
        "messages": original["messages"],
        "temperature": 0,
        "max_completion_tokens": 4096,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "healthcare_answer",
                "strict": True,
                "schema": original["format"],
            },
        },
    }
    observed["request_body"] = body

    request = Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "ChatBotHealthCare/1.0",
        },
        method="POST",
    )

    started = time.perf_counter()
    try:
        with open_with_retry(request) as response:
            response_data = json.load(response)
    except HTTPError as exc:
        detail = exc.read(20000).decode("utf-8", errors="replace")
        observed["error"] = f"HTTP {exc.code}"
        observed["error_detail"] = detail.replace(key, "[CLE_MASQUEE]")
        print(observed["error_detail"], flush=True)
        raise OSError(observed["error"]) from None
    except (OSError, ValueError) as exc:
        observed["error"] = type(exc).__name__
        raise OSError("Appel Groq échoué") from None
    finally:
        observed["api_seconds"] = round(time.perf_counter() - started, 3)

    observed["usage"] = response_data.get("usage", {})
    observed["model_version"] = response_data.get("model")
    choices = response_data.get("choices", [])
    if not choices:
        observed["error"] = "Aucune réponse retournée"
        raise ValueError(observed["error"])

    choice = choices[0]
    observed["finish_reason"] = choice.get("finish_reason")
    answer = choice.get("message", {}).get("content") or ""
    observed["raw_answer"] = answer

    if choice.get("finish_reason") != "stop" or not answer.strip():
        observed["error"] = "Réponse vide ou interrompue"
        raise ValueError(observed["error"])

    return io.BytesIO(json.dumps({
        "message": {"content": answer}
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
        with patch.object(generation, "urlopen", groq_transport):
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
            "groq": {
                "answer": answer,
                "mode": "groq" if mode == "ollama" else mode,
                "warning": warning,
                "generation_seconds": round(time.perf_counter() - started, 3),
                **observed,
            },
        }
        print("Mode :", comparison["groq"]["mode"], flush=True)
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
