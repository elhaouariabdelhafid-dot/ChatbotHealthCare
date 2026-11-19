#!/usr/bin/env python3
"""Compare resident vs isolated retrieval. No project files/config are modified.
Run from the project root: python scripts/benchmark_memory.py --mode isolated
The isolated parent imports only the stdlib and healthbot.generation.
"""
import argparse
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

QUESTION = 'Which reflex and muscle findings are listed for hypertrophic neuropathy of Dejerine-Sottas?'


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def memory():
    p = Path('/proc/meminfo')
    if not p.exists():
        return {}
    values = {}
    for line in p.read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemAvailable', 'SwapFree', 'SwapTotal'):
            values[key + '_kB'] = int(value.split()[0])
    return values


def generation_call(question, hits, cfg):
    from healthbot import generation
    metrics = {'memory_before': memory()}
    original = generation.urlopen

    @contextmanager
    def observed(*args, **kwargs):
        try:
            with original(*args, **kwargs) as response:
                raw = response.read(2_000_000)
            data = json.loads(raw)
            for key in ('total_duration', 'load_duration', 'prompt_eval_duration',
                        'eval_duration', 'prompt_eval_count', 'eval_count', 'done_reason'):
                if key in data:
                    metrics[key] = data[key]
            with io.BytesIO(raw) as buffered:
                yield buffered
        except Exception as exc:
            metrics['error_type'] = type(exc).__name__
            raise

    started = time.perf_counter()
    with patch.object(generation, 'urlopen', observed):
        outcome = generation.generate(question, hits, cfg)
    metrics['wall_seconds'] = round(time.perf_counter() - started, 3)
    metrics['memory_after'] = memory()
    metrics['torch_loaded_in_generation_process'] = 'torch' in sys.modules
    return outcome, metrics


def worker(request_path, response_path):
    request = json.loads(Path(request_path).read_text(encoding='utf-8'))
    # Model progress and incidental prints never contaminate the JSON exchange.
    with redirect_stdout(sys.stderr):
        from healthbot.settings import settings
        from healthbot.engine import HealthBot
        cfg = dict(settings())
        cfg.update(generation='ollama', timeout_seconds=request['timeout'])
        captured = {}

        def intercept(question, hits, config):
            captured.update(question=question, hits=hits, cfg=config)
            if request['mode'] == 'baseline':
                outcome, metrics = generation_call(question, hits, config)
                captured['generation_metrics'] = metrics
                return outcome
            return '', 'extractive', None

        started = time.perf_counter()
        bot = HealthBot(cfg)
        init_seconds = time.perf_counter() - started
        with patch('healthbot.engine.generate', intercept):
            result = bot.ask(request['question'])
        output = {'result': result, 'captured': captured,
                  'init_seconds': round(init_seconds, 3), 'memory_worker_end': memory()}
    save(response_path, output)


def run(args):
    root = Path.cwd().resolve()
    if not (root / 'healthbot/engine.py').exists():
        raise SystemExit('Lancer depuis ~/ChatBotHealthCare.')
    sys.path.insert(0, str(root))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output = root / 'artifacts' / ('memory_' + args.mode + '_' + stamp + '.json')
    output.parent.mkdir(exist_ok=True)
    report = {'mode': args.mode, 'question': args.question, 'timeout_seconds': args.timeout,
              'memory_before': memory(), 'ollama_state': args.state,
              'note': 'Ollama timings are nanoseconds; wall times are seconds.'}
    report['code_hashes'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                             for name in ['healthbot/engine.py', 'healthbot/generation.py',
                                          'healthbot/retrieval.py', 'healthbot/reranking.py', 'config.json']}
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix='healthbot_memory_') as tmp:
            request, response = Path(tmp)/'request.json', Path(tmp)/'response.json'
            save(request, {'question': args.question, 'timeout': args.timeout, 'mode': args.mode})
            print('Recherche dans un processus séparé...', flush=True)
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            '--worker', str(request), str(response)],
                           cwd=root, check=True, timeout=args.timeout + 600)
            data = json.loads(response.read_text(encoding='utf-8'))
        report['worker_seconds'] = round(time.perf_counter() - started, 3)
        report['memory_after_worker_exit'] = memory()
        report['init_seconds'] = data['init_seconds']
        result, captured = data['result'], data['captured']
        if captured:
            report['selected_passages'] = captured['hits']
            report['effective_config'] = captured['cfg']
            if args.mode == 'isolated':
                print('Processus terminé : E5 et mMiniLM libérés. Appel Qwen...', flush=True)
                (answer, mode, warning), metrics = generation_call(
                    captured['question'], captured['hits'], captured['cfg'])
                result.update(answer=answer, mode=mode,
                              status='abstain' if mode == 'abstain' else 'answered')
                if mode == 'abstain':
                    result['sources'] = []
                if warning:
                    result['warning'] = warning
                result['latency_ms'] = round(result['latency_ms'] + metrics['wall_seconds']*1000, 2)
            else:
                metrics = captured['generation_metrics']
            report['generation_metrics'] = metrics
        report['result'] = result
    except Exception as exc:
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        report['total_wall_seconds'] = round(time.perf_counter() - started, 3)
        save(output, report)
        print('Rapport :', output, flush=True)
    print(json.dumps({'answer': result['answer'], 'mode': result['mode'],
                      'warning': result.get('warning'),
                      'generation': report.get('generation_metrics'),
                      'total_wall_seconds': report['total_wall_seconds']}, ensure_ascii=False, indent=2))


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        sys.path.insert(0, str(Path.cwd()))
        worker(sys.argv[2], sys.argv[3])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['baseline', 'isolated'], default='isolated')
    parser.add_argument('--question', default=QUESTION)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--state', choices=['cold', 'warm', 'unknown'], default='unknown',
                        help='État Ollama avant le test, indiqué par l’opérateur.')
    args = parser.parse_args()
    if not 1 <= args.timeout <= 1800:
        parser.error('timeout doit être compris entre 1 et 1800 secondes')
    run(args)


if __name__ == '__main__':
    main()
