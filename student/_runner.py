"""16线程运行学生pipeline。统一计数、模型调用、日志与最终返回值记录。"""
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent
MODEL = 'Qwen/Qwen3-4B'
CONTRACT = 'lab1-pipeline-v2'
PARAMS = {'model': MODEL, 'temperature': 0.6, 'top_p': 0.95, 'top_k': 20,
          'min_p': 0.0, 'stream': False,
          'chat_template_kwargs': {'enable_thinking': True}}
MAX_RPC = 2_000_000

def sha(data):
    return hashlib.sha256(data).hexdigest()

def stamp():
    return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:6]

def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf8').splitlines() if x.strip()]

def load_data(path):
    rows = read_jsonl(path)
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('数据为空或题号重复')
    for r in rows:
        pair_view(r)
    return rows

def pair_view(row):
    return {k: {f: row[k][f] for f in ('title', 'abstract', 'date')} for k in ('paper_a', 'paper_b')}

def validate_messages(messages):
    if not isinstance(messages, list) or not messages or len(messages) > 100:
        raise ValueError('messages须为非空消息列表，最多100条')
    for m in messages:
        if not isinstance(m, dict) or set(m) != {'role', 'content'} or m['role'] not in {'system', 'user', 'assistant'} or not isinstance(m['content'], str):
            raise ValueError('每条消息须包含role和字符串content')
        if '/no_think' in m['content'].lower():
            raise ValueError('课程固定thinking模式，不允许/no_think')
    if sum(len(m['content'].encode('utf8')) for m in messages) > 20000:
        raise ValueError('每次请求的消息文本合计最多20,000 UTF-8字节')

def normalize_message(msg, thinking=True):
    raw = msg.get('content') or ''
    reasoning = msg.get('reasoning') or msg.get('reasoning_content') or ''
    content = raw
    # Older vLLM returns Qwen3's reasoning and answer in one content field.
    # Preserve that raw field while exposing the same interface as new vLLM.
    if not reasoning and 'reasoning' not in msg and 'reasoning_content' not in msg:
        if '</think>' in raw and (thinking or raw.lstrip().startswith('<think>')):
            reasoning, content = raw.split('</think>', 1)
            reasoning = reasoning.removeprefix('<think>').strip()
            content = content.lstrip('\n')
        elif thinking:
            reasoning, content = raw.removeprefix('<think>').strip(), ''
    return content, reasoning, raw


def deployment_metadata(base, key, mock):
    root = ROOT if (ROOT/'deploy').exists() else ROOT.parent
    profile, version = ('mock', 'mock') if mock else ('unknown', 'unavailable')
    if not mock:
        try:
            req = urllib.request.Request(base.rstrip('/').removesuffix('/v1') + '/version',
                                         headers={'Authorization': 'Bearer ' + key})
            with urllib.request.urlopen(req, timeout=15) as response:
                version = json.load(response).get('version', 'unknown')
        except (OSError, ValueError):
            pass
        for name in ('4090', '5090'):
            spec = json.loads((root/'deploy/profiles'/name/'environment.json').read_text(encoding='utf8'))
            if version == spec['vllm']:
                profile = name
                break
    fingerprint = (profile + ':' + str(version)).encode()
    if profile in ('4090', '5090'):
        for filename in ('environment.json', 'requirements.lock', 'vllm.yaml'):
            fingerprint += (root/'deploy/profiles'/profile/filename).read_bytes()
    return {'deployment_profile': profile, 'server_version': version,
            'deployment_sha256': sha(fingerprint)}


def request_model(messages, base, key, params=None, timeout=3600):
    body = dict(PARAMS if params is None else params)
    body['messages'] = messages
    req = urllib.request.Request(base.rstrip('/') + '/chat/completions',
          data=json.dumps(body).encode('utf8'), method='POST',
          headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        obj = json.load(response)
    if obj.get('model') != MODEL:
        raise ValueError('服务返回的模型名与课程模型不一致')
    choice = obj['choices'][0]
    msg = choice['message']
    content, reasoning, raw = normalize_message(msg, body.get('chat_template_kwargs', {}).get('enable_thinking', True))
    return {'content': content, 'reasoning': reasoning, 'raw_content': raw,
            'reasoning_content': reasoning, 'finish_reason': choice.get('finish_reason'),
            'usage': obj.get('usage') or {}}

class BudgetExceeded(RuntimeError):
    pass

class Broker:
    def __init__(self, folder, limit, base, key, mock=False, request_timeout=3600):
        self.folder, self.limit, self.base, self.key = Path(folder), limit, base, key
        self.mock, self.request_timeout = mock, request_timeout
        self.used = 0
        self.lock = threading.Lock()

    def append(self, name, record):
        with self.lock:
            with (self.folder/name).open('a', encoding='utf8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())

    def chat(self, rid, local_index, messages, remaining):
        validate_messages(messages)
        # 先原子预扣并落盘，再发HTTP。错误、超时、断线也占一次。
        with self.lock:
            if self.used >= self.limit:
                raise BudgetExceeded('本次运行的5×n总调用预算已耗尽')
            self.used += 1
            call_id = self.used
            with (self.folder/'calls.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps({'event': 'reserved', 'call_id': call_id, 'id': rid,
                    'local_index': local_index, 'messages': messages}, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        began = time.monotonic()
        try:
            if self.mock:
                result = {'content': '{"answer":"A"}', 'reasoning': 'MOCK：仅验证调用流程。',
                          'reasoning_content': 'MOCK：仅验证调用流程。', 'finish_reason': 'stop', 'usage': {}}
            else:
                params = dict(PARAMS)
                params['seed'] = int(sha(f'{rid}:{local_index}'.encode())[:8], 16) % 2147483647
                result = request_model(messages, self.base, self.key, params,
                                       timeout=max(0.1, min(self.request_timeout, remaining)))
        except Exception as exc:
            self.append('calls.jsonl', {'event': 'error', 'call_id': call_id, 'id': rid,
                'error': type(exc).__name__, 'seconds': time.monotonic()-began})
            raise RuntimeError('模型请求失败（已计入预算）：' + type(exc).__name__) from None
        self.append('calls.jsonl', {'event': 'completed', 'call_id': call_id, 'id': rid,
                    'response': result, 'seconds': round(time.monotonic()-began, 3)})
        return result

def run_question(row, examples, source, broker, timeout):
    began = time.monotonic()
    deadline = began + timeout
    env = {k: os.environ[k] for k in ('SystemRoot', 'WINDIR', 'PATH', 'TEMP', 'TMP', 'LANG') if k in os.environ}
    answer, error, local_calls = None, None, 0
    with tempfile.TemporaryDirectory(prefix='worker_', dir=broker.folder) as cwd:
        process = subprocess.Popen([sys.executable, '-I', '-X', 'utf8', str(ROOT/'_worker.py')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding='utf8', cwd=cwd, env=env)
        events = queue.Queue(maxsize=8)
        def reader():
            try:
                while True:
                    line = process.stdout.readline(MAX_RPC + 1)
                    if not line:
                        events.put(None)
                        return
                    if len(line) > MAX_RPC:
                        events.put({'type': 'error', 'error': 'worker输出过长'})
                        return
                    events.put(json.loads(line))
            except Exception:
                events.put({'type': 'error', 'error': 'worker通信格式错误'})
        threading.Thread(target=reader, daemon=True).start()
        try:
            payload = {'source': source, 'pair': pair_view(row), 'examples': examples,
                       'seed': int(sha(row['id'].encode())[:8], 16)}
            process.stdin.write(json.dumps(payload, ensure_ascii=False) + '\n')
            process.stdin.flush()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('题目执行超时')
                event = events.get(timeout=remaining)
                if event is None:
                    raise RuntimeError('学生进程未返回结果')
                if event.get('type') == 'call':
                    local_calls += 1
                    try:
                        reply = broker.chat(row['id'], local_calls, event.get('messages'), remaining)
                        response = {'ok': True, 'response': reply}
                    except BudgetExceeded:
                        raise
                    except Exception as exc:
                        response = {'ok': False, 'error': str(exc)[:1000]}
                    process.stdin.write(json.dumps(response, ensure_ascii=False) + '\n')
                    process.stdin.flush()
                elif event.get('type') == 'result':
                    value = event.get('value')
                    if type(value) is str and value in ('A', 'B'):
                        answer = value
                    else:
                        error = 'predict必须返回严格的字符串A或B'
                    break
                else:
                    raise RuntimeError(event.get('error', '未知worker事件'))
        except Exception as exc:
            error = '题目执行超时' if isinstance(exc, queue.Empty) else str(exc)[:1000]
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdin.close()
            process.stdout.close()
    return {'id': row['id'], 'prediction': answer, 'error': error,
            'seconds': round(time.monotonic()-began, 3)}

def run(args):
    if not 1 <= args.workers <= 64 or args.limit is not None and args.limit < 1 or args.question_timeout <= 0 or args.request_timeout <= 0:
        raise ValueError('workers须为1至64，limit与超时须为正数')
    data_path, solution = Path(args.data).resolve(), Path(args.solution).resolve()
    rows = load_data(data_path)
    labels = Path(args.labels).resolve() if args.labels else data_path.with_name(data_path.stem + '_labels.json')
    if args.labels and not labels.is_file():
        raise ValueError('指定的标签文件不存在')
    if labels.exists():
        gold = json.loads(labels.read_text(encoding='utf8'))
        if len(gold) != len(rows) or {g['id'] for g in gold} != {r['id'] for r in rows} or any(g['answer'] not in ('A','B') for g in gold):
            raise ValueError('标签文件与完整测试集不匹配')
    selected = rows[:args.limit] if args.limit else rows
    examples_path = ROOT/'data/examples.jsonl'
    examples = load_data(examples_path)
    examples = [{**pair_view(x), 'id': x['id'], 'answer': x['answer']} for x in examples]
    source = solution.read_text(encoding='utf8')
    from _worker import check_source
    check_source(source)
    base = os.environ.get('COURSE_API_BASE', 'http://127.0.0.1:8000/v1').strip()
    key = os.environ.get('COURSE_API_KEY', 'lab-local').strip()
    if not args.mock:
        from check_server import check
        check(base, key)
    folder = Path(args.output).resolve()/(stamp() + '_' + solution.stem)
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'solution.py').write_text(source, encoding='utf8')
    shutil.copy2(data_path, folder/'dataset.jsonl')
    deployment = deployment_metadata(base, key, args.mock)
    config = {'contract': CONTRACT, 'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
       'model': MODEL, 'params': PARAMS, 'mock': args.mock, 'workers': args.workers,
       'dataset_n': len(rows), 'selected_n': len(selected), 'budget_limit': 5*len(selected),
       'data_sha256': sha(data_path.read_bytes()), 'source_sha256': sha(source.encode('utf8')),
       'examples_sha256': sha(examples_path.read_bytes()), **deployment,
       'dataset_name': data_path.stem, 'question_timeout': args.question_timeout, 'request_timeout': args.request_timeout}
    (folder/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf8')
    for name in ('predictions.jsonl', 'calls.jsonl'):
        (folder/name).touch(exist_ok=False)
    broker = Broker(folder, config['budget_limit'], base, key, args.mock, args.request_timeout)
    began = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_question, row, examples, source, broker, args.question_timeout):row for row in selected}
        for future in concurrent.futures.as_completed(futures):
            try:
                record = future.result()
            except Exception as exc:
                record = {'id': futures[future]['id'], 'prediction': None, 'error': type(exc).__name__ + ': ' + str(exc)[:300], 'seconds': 0}
            broker.append('predictions.jsonl', record)
            print(f"{record['id']}: {record['prediction'] or record['error']}", flush=True)
    summary = {'completed': len(selected), 'budget_used': broker.used, 'budget_limit': broker.limit,
               'external_calls': 0 if args.mock else broker.used, 'wall_seconds': round(time.monotonic()-began, 3)}
    (folder/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    from evaluate import finish
    finish(folder, labels if labels.exists() else None, root=Path(args.output), open_browser=not args.no_open)
    print(f"已保存：{folder}\n调用预算 {broker.used}/{broker.limit}；外部模型请求 {summary['external_calls']} 次")
    return folder

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--solution', default='solution.py')
    p.add_argument('--data', default=str(ROOT/'data/public.jsonl'))
    p.add_argument('--labels', help='可选；默认寻找同目录的<数据名>_labels.json')
    p.add_argument('--output', default='runs', help='记录根目录；每次创建带时间戳的新子目录')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--limit', type=int, help='仅运行前n题，预算为5n，标记为不完整测试')
    p.add_argument('--mock', action='store_true', help='本地模拟模型，仅验证pipeline')
    p.add_argument('--no-open', action='store_true', help='无桌面服务器仅保存HTML')
    p.add_argument('--question-timeout', type=float, default=21600)
    p.add_argument('--request-timeout', type=float, default=3600)
    return p

