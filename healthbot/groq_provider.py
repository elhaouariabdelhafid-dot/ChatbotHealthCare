"""Groq transport; shared admission control within ONE server process.
No hidden retries, no fallback provider calls, no secret in diagnostics.
"""
import json
import math
import os
import socket
import threading
import time
from collections import deque
from contextvars import ContextVar
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TRACE = ContextVar('groq_trace', default=None)

class GroqUnavailable(Exception):
    def __init__(self, code, retry_after=None):
        self.code = code
        self.retry_after = retry_after
        super().__init__(code)

class Admission:
    def __init__(self):
        self.lock = threading.Lock()
        self.events = deque()
        self.active = 0
        self.blocked_until = 0.0

    def reserve(self, tokens, cfg):
        now = time.monotonic()
        with self.lock:
            while self.events and now - self.events[0][0] >= 60:
                self.events.popleft()
            if now < self.blocked_until:
                raise GroqUnavailable('cooldown', math.ceil(self.blocked_until-now))
            if self.active >= cfg.get('groq_max_concurrent', 2):
                raise GroqUnavailable('busy', 2)
            if (sum(e[1] for e in self.events) + tokens > cfg.get('groq_tpm', 8000) * .9
                    or len(self.events) >= cfg.get('groq_rpm', 30)):
                delay = max(1, math.ceil(60-(now-self.events[0][0]))) if self.events else 60
                raise GroqUnavailable('local_rate_limit', delay)
            event = [now, tokens]
            self.events.append(event)
            self.active += 1
            return event

    def finish(self, event, actual=None):
        with self.lock:
            self.active -= 1
            if type(actual) is int and actual >= 0:
                event[1] = actual

    def block(self, seconds):
        with self.lock:
            self.blocked_until = max(self.blocked_until, time.monotonic()+seconds)

ADMISSION = Admission()

def retry_seconds(headers, default):
    value = headers.get('Retry-After') if headers else None
    try:
        return max(1, math.ceil(float(value)))
    except (ValueError, TypeError, OverflowError):
        try:
            return max(1, math.ceil(parsedate_to_datetime(value).timestamp()-time.time()))
        except (ValueError, TypeError, OverflowError):
            return default

def configure_groq(cfg):
    cfg = dict(cfg)
    defaults = {'groq_model':'openai/gpt-oss-120b', 'groq_timeout_seconds':20,
                'groq_max_completion_tokens':4096, 'groq_tpm':8000,
                'groq_rpm':30, 'groq_max_concurrent':2}
    for key, default in defaults.items():
        cfg.setdefault(key, default)
    if not isinstance(cfg['groq_model'], str) or not cfg['groq_model'].strip():
        raise ValueError('groq_model invalide')
    for key in defaults:
        if key != 'groq_model' and (type(cfg[key]) is not int or cfg[key] <= 0):
            raise ValueError(key+' doit être un entier positif')
    if cfg['groq_timeout_seconds'] > 60:
        raise ValueError('groq_timeout_seconds doit être <= 60')
    return cfg

def complete(payload, cfg):
    cfg = configure_groq(cfg)
    trace = {'provider':'groq', 'model':cfg['groq_model'], 'status':'started',
             'api_called':False, 'attempts':0}
    TRACE.set(trace)
    key = os.environ.get('GROQ_API_KEY', '').strip()
    if not key:
        trace['status']='missing_key'
        raise GroqUnavailable('missing_key')
    body = {'model':cfg['groq_model'], 'messages':payload['messages'],
            'temperature':0, 'stream':False,
            'max_completion_tokens':cfg['groq_max_completion_tokens'],
            'response_format':{'type':'json_schema', 'json_schema':{
                'name':'healthcare_answer', 'strict':True, 'schema':payload['format']}}}
    # Conservative heuristic, not an exact tokenizer. Actual usage reconciles it.
    reservation = math.ceil(len(json.dumps(body['messages'],ensure_ascii=False).encode())/3) + cfg['groq_max_completion_tokens']
    trace['reserved_tokens_estimate'] = reservation
    event = None
    actual = None
    start = time.perf_counter()
    try:
        event = ADMISSION.reserve(reservation, cfg)
        req = Request('https://api.groq.com/openai/v1/chat/completions',
                      data=json.dumps(body).encode(), method='POST', headers={
                          'Authorization':'Bearer '+key, 'Content-Type':'application/json',
                          'Accept':'application/json', 'User-Agent':'ChatBotHealthCare/1.0'})
        trace.update(api_called=True, attempts=1)
        with urlopen(req, timeout=cfg['groq_timeout_seconds']) as response:
            result = json.loads(response.read(2_000_000))
        usage = result.get('usage', {})
        actual = usage.get('total_tokens')
        trace['usage'] = {k:usage[k] for k in ('prompt_tokens','completion_tokens','total_tokens') if k in usage}
        choice = result['choices'][0]
        answer = choice['message'].get('content')
        if choice.get('finish_reason') != 'stop' or not isinstance(answer,str) or not answer.strip():
            raise GroqUnavailable('incomplete_response')
        trace['status']='received'
        return answer.strip()
    except HTTPError as exc:
        if exc.code == 429:
            delay=retry_seconds(exc.headers,60)
            ADMISSION.block(delay)
            error=GroqUnavailable('rate_limited',delay)
        elif exc.code >= 500:
            delay=retry_seconds(exc.headers,30)
            ADMISSION.block(delay)
            error=GroqUnavailable('provider_unavailable',delay)
        elif exc.code in (401,403):
            ADMISSION.block(60)
            error=GroqUnavailable('access_denied',60)
        elif exc.code == 402:
            ADMISSION.block(300)
            error=GroqUnavailable('billing_required',300)
        else:
            error=GroqUnavailable('request_rejected')
        exc.close()
        trace['status']=error.code
        trace['retry_after_seconds']=error.retry_after
        raise error from None
    except GroqUnavailable as exc:
        trace['status']=exc.code
        trace['retry_after_seconds']=exc.retry_after
        raise
    except (TimeoutError,socket.timeout):
        trace['status']='timeout'
        raise GroqUnavailable('timeout') from None
    except (OSError,URLError):
        trace['status']='network_error'
        raise GroqUnavailable('network_error') from None
    except (ValueError,KeyError,TypeError,IndexError):
        trace['status']='invalid_response'
        raise GroqUnavailable('invalid_response') from None
    finally:
        trace['transport_ms']=round((time.perf_counter()-start)*1000,2)
        if event is not None:
            ADMISSION.finish(event, actual)

def snapshot():
    return dict(TRACE.get() or {})

def set_status(status):
    trace = snapshot()
    trace['status'] = status
    TRACE.set(trace)
