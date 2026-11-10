"""框架内部：每题独立进程，模型请求通过父进程计数与转发。"""
import ast
import builtins
import collections
import functools
import itertools
import json
import math
import random
import re
import statistics
import sys
import types
import os


def emit(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=False) + '\n')
    sys.stdout.flush()


def facade(module, names):
    return types.SimpleNamespace(**{name: getattr(module, name) for name in names})


def environment(seed):
    rng = random.Random(seed)
    modules = {
        'json': facade(json, ['loads', 'dumps', 'JSONDecodeError']),
        're': facade(re, ['search', 'findall', 'finditer', 'fullmatch', 'match', 'sub', 'split', 'compile', 'escape', 'I', 'M', 'S', 'IGNORECASE', 'MULTILINE', 'DOTALL']),
        'math': facade(math, [n for n in dir(math) if not n.startswith('_')]),
        'statistics': facade(statistics, ['mean', 'median', 'mode', 'multimode', 'pstdev', 'stdev']),
        'collections': facade(collections, ['Counter', 'defaultdict', 'deque']),
        'itertools': facade(itertools, ['chain', 'islice', 'combinations', 'permutations', 'product']),
        'functools': facade(functools, ['reduce']),
        'random': facade(rng, ['random', 'randint', 'randrange', 'choice', 'choices', 'sample', 'shuffle', 'uniform']),
    }

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in modules:
            raise ImportError('请使用README列出的标准库：' + name)
        return modules[name]

    safe = {n: getattr(builtins, n) for n in (
        'abs all any bool dict enumerate filter float int isinstance len list map max min next '
        'range reversed round set slice sorted str sum tuple zip iter Exception ValueError TypeError '
        'RuntimeError KeyError IndexError StopIteration ZeroDivisionError'
    ).split()}
    safe['__import__'] = safe_import
    return {'__builtins__': safe, '__name__': 'student_solution'}


def check_source(source):
    if len(source.encode('utf8')) > 65536:
        raise ValueError('solution.py 最多64 KiB')
    tree = ast.parse(source, filename='solution.py')
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.Global, ast.Nonlocal)):
            raise ValueError('不支持的语法：' + type(node).__name__)
        if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
            raise ValueError('不允许访问内部属性')
        if isinstance(node, ast.Name) and node.id.startswith('__'):
            raise ValueError('不允许访问内部名称')
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            raise ValueError('不支持装饰器')
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'predict']
    if len(funcs) != 1:
        raise ValueError('须定义一个 predict(pair, examples, llm) 函数')
    f = funcs[0].args
    if [a.arg for a in f.args] != ['pair', 'examples', 'llm'] or f.vararg or f.kwarg or f.kwonlyargs or f.posonlyargs or f.defaults:
        raise ValueError('函数签名须为 predict(pair, examples, llm)')
    return compile(tree, 'solution.py', 'exec')


class Client:
    def chat(self, messages):
        emit({'type': 'call', 'messages': messages})
        reply = json.loads(sys.stdin.readline())
        if not reply.get('ok'):
            raise RuntimeError(reply.get('error', '模型请求失败'))
        result = reply['response']
        return {'content': result['content'], 'reasoning': result.get('reasoning', '')}


def main():
    payload = json.loads(sys.stdin.readline())
    if os.name == 'posix':
        import resource
        limit = 512 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    try:
        namespace = environment(payload['seed'])
        exec(check_source(payload['source']), namespace)
        answer = namespace['predict'](payload['pair'], payload['examples'], Client())
        emit({'type': 'result', 'value': answer})
    except BaseException as exc:
        emit({'type': 'error', 'error': type(exc).__name__ + ': ' + str(exc)[:1000]})


if __name__ == '__main__':
    main()
