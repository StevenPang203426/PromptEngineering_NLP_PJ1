"""检查已启动的课程模型服务；默认只读元数据，不生成答案。"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from _runner import MODEL


def check(base, key):
    base = base.rstrip('/')
    parts = urlsplit(base)
    if parts.scheme not in ('http', 'https') or not parts.netloc or not parts.path.endswith('/v1'):
        raise ValueError('COURSE_API_BASE 必须是以 /v1 结尾的 http(s) 地址')
    headers = {'Authorization': 'Bearer ' + key}
    # Health and model-list calls do not trigger model generation.
    with urllib.request.urlopen(urllib.request.Request(base[:-3] + '/health', headers=headers), timeout=15) as r:
        if r.status != 200:
            raise ValueError('服务尚未就绪')
    with urllib.request.urlopen(urllib.request.Request(base + '/models', headers=headers), timeout=15) as r:
        info = json.load(r)
    if MODEL not in [m.get('id') for m in info.get('data', [])]:
        raise ValueError('服务中的模型名称不是课程指定的 Qwen/Qwen3-4B')
    print('服务就绪：health = OK')
    print('模型名称：' + MODEL)
    print('本次检查未发起模型生成请求。')
    return info


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    a = p.parse_args()
    try:
        check(os.environ.get('COURSE_API_BASE', 'http://127.0.0.1:8000/v1'),
              os.environ.get('COURSE_API_KEY', 'lab-local'))
    except (urllib.error.URLError, ValueError, KeyError, TimeoutError) as e:
        print('检查失败：请确认服务已启动、地址与端口正确、密钥一致。', file=sys.stderr)
        print('若使用GPU服务器，客户端须在同一服务器运行，或先建立SSH转发。', file=sys.stderr)
        if isinstance(e, ValueError):print(str(e), file=sys.stderr)
        sys.exit(1)
