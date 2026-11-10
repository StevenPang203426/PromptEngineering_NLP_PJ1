"""One-command setup for the course's isolated Linux GPU environment."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

DEPLOY = Path(__file__).resolve().parent
ROOT = DEPLOY.parent
RUNTIME = ROOT / '.lab-runtime'
SPEC = None
PROFILE_DIR = None
MODEL = json.loads((DEPLOY / 'model-manifest.json').read_text(encoding='utf-8'))

def say(message):
    print(message, flush=True)

def run(args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, **kwargs)

def release_version(value):
    # Python packaging treats 13.0.3 and 13.0.3.0 as the same release.
    parts = value.split('+', 1)[0].split('.')
    while parts and parts[-1] == '0':
        parts.pop()
    return parts

def activate_environment(python):
    os.environ['VIRTUAL_ENV'] = str(python.parent.parent)
    os.environ['PATH'] = str(python.parent) + os.pathsep + os.environ.get('PATH', '')
    os.environ.pop('PYTHONHOME', None)
    if SPEC['profile'] == '4090':
        # The CUDA 12.8 runtime is provided by the pinned PyTorch wheels.
        # Do not inherit the image's unrelated CUDA 11.8 toolkit.
        for name in ('CUDA_HOME', 'CUDA_PATH', 'CUDACXX'):
            os.environ.pop(name, None)
        return None
    # Use the CUDA compiler shipped with the locked environment, not a host toolkit.
    code = ('import importlib.metadata as m; d=m.distribution("nvidia-cuda-nvcc"); '
            'print(next(d.locate_file(f) for f in d.files if str(f).endswith("/bin/nvcc")))')
    nvcc = Path(run([python, '-c', code], capture_output=True, text=True).stdout.strip())
    cuda = nvcc.parent.parent
    # CUDA wheels contain the versioned runtime; native linkers need its SDK alias.
    runtime_library = cuda / 'lib/libcudart.so.13'
    if not runtime_library.is_file():
        raise RuntimeError('固定环境中缺少 CUDA 13 运行库')
    linker_alias = cuda / 'lib/libcudart.so'
    if not linker_alias.exists():
        linker_alias.symlink_to(runtime_library.name)
    if not (cuda / 'lib64').exists():
        (cuda / 'lib64').symlink_to('lib', target_is_directory=True)
    os.environ['VIRTUAL_ENV'] = str(python.parent.parent)
    os.environ['CUDA_HOME'] = str(cuda)
    os.environ['CUDA_PATH'] = str(cuda)
    os.environ['CUDACXX'] = str(nvcc)
    for name, directories in [('PATH', [python.parent, nvcc.parent]),
                              ('LD_LIBRARY_PATH', [cuda / 'lib']),
                              ('LIBRARY_PATH', [cuda / 'lib'])]:
        old = os.environ.get(name, '')
        os.environ[name] = os.pathsep.join([str(x) for x in directories] + ([old] if old else []))
    os.environ.pop('PYTHONHOME', None)
    return run([nvcc, '--version'], capture_output=True, text=True).stdout.strip()

def choose_profile(requested):
    global SPEC, PROFILE_DIR
    if requested == 'auto':
        if not shutil.which('nvidia-smi'):
            raise RuntimeError('无法识别GPU。请在GPU服务器运行，或用 --profile 4090/5090 --describe 查看配置。')
        selected = os.environ.get('COURSE_GPU', os.environ.get('CUDA_VISIBLE_DEVICES', '0')).split(',')[0]
        name = run(['nvidia-smi', '--id=' + selected, '--query-gpu=name', '--format=csv,noheader'],
                   capture_output=True, text=True).stdout.strip()
        requested = next((p for p in ('4090', '5090') if p in name), None)
        if not requested:
            raise RuntimeError('未识别为4090或5090；请由助教确认硬件后显式指定 --profile。')
    PROFILE_DIR = DEPLOY / 'profiles' / requested
    SPEC = json.loads((PROFILE_DIR / 'environment.json').read_text(encoding='utf-8'))
    return requested


def preflight():
    if sys.platform != 'linux' or platform.machine() not in ('x86_64', 'AMD64'):
        raise RuntimeError('请在 Linux x86_64 GPU 服务器运行部署脚本。个人电脑只需运行评测客户端。')
    if not shutil.which('nvidia-smi'):
        raise RuntimeError('请选择已安装 NVIDIA 驱动的 GPU 开发镜像。')
    info = run(['nvidia-smi', '--query-gpu=driver_version,name,memory.total',
                '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
    issues = []
    minimum = tuple(map(int, SPEC['minimum_driver'].split('.')))
    selected = os.environ.get('COURSE_GPU', os.environ.get('CUDA_VISIBLE_DEVICES', '0')).split(',')[0]
    selected_info = run(['nvidia-smi', '--id=' + selected, '--query-gpu=driver_version,name',
                         '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
    driver, gpu = selected_info.split(',', 1)
    version = tuple(map(int, driver.strip().split('.')))
    if version + (0,) * (len(minimum)-len(version)) < minimum:
        issues.append(f"{SPEC['profile']}配置需要驱动至少{SPEC['minimum_driver']}，当前为{driver.strip()}")
    if '5090' in gpu and SPEC['profile'] == '4090':
        issues.append('5090不能使用4090的CUDA 12.8配置，请改为 --profile 5090')
    if not (shutil.which('gcc') or shutil.which('cc')):
        issues.append('镜像缺少C编译器，请使用GPU开发镜像或由管理员安装build-essential')
    if not RUNTIME.exists() and shutil.disk_usage(ROOT).free < 30 * 1024**3:
        issues.append(f'首次安装需至少30 GiB剩余空间，当前仅{shutil.disk_usage(ROOT).free/1024**3:.1f} GiB；建议100 GB系统盘')
    marker = RUNTIME / 'installed.json'
    if marker.is_file():
        installed = json.loads(marker.read_text(encoding='utf-8'))
        if installed['packages']['vllm'] != SPEC['vllm']:
            issues.append('当前目录已有另一套环境。请在新的课程目录部署，避免混用两个版本')
    if issues:
        raise RuntimeError('\n- ' + '\n- '.join(issues))
    say('GPU：\n' + info)

def bootstrap_uv():
    tools = RUNTIME / 'tools'
    tools.mkdir(parents=True, exist_ok=True)
    uv = tools / 'uv'
    if uv.is_file():
        version = run([uv, '--version'], capture_output=True, text=True).stdout.split()[1]
        if version == SPEC['uv']:
            return uv
        raise RuntimeError('已有安装工具版本不符，请将 .lab-runtime/tools 改名后重新运行。')
    wheel = tools / 'uv.whl'
    source = SPEC['uv_wheel']
    suffix = source['url'].split('/packages/', 1)[1]
    for url in ['https://mirrors.aliyun.com/pypi/packages/' + suffix, 'https://pypi.tuna.tsinghua.edu.cn/packages/' + suffix, source['url']]:
        try:
            with urllib.request.urlopen(url, timeout=60) as response, wheel.open('wb') as out:
                shutil.copyfileobj(response, out)
            if hashlib.sha256(wheel.read_bytes()).hexdigest() != source['sha256']:
                raise RuntimeError('安装工具下载校验失败')
            break
        except Exception as exc:
            say('安装工具下载重试：' + str(exc))
    else:
        raise RuntimeError('无法下载 uv，请检查服务器到 PyPI 的网络。')
    with zipfile.ZipFile(wheel) as archive:
        members = [x for x in archive.infolist() if not x.is_dir() and x.filename.rsplit('/', 1)[-1] == 'uv']
        if len(members) != 1:
            raise RuntimeError('安装工具归档内容异常')
        with archive.open(members[0]) as source_file, uv.open('wb') as out:
            shutil.copyfileobj(source_file, out)
    uv.chmod(0o755)
    wheel.unlink()
    return uv

def prepare_python(uv):
    os.environ['UV_PYTHON_INSTALL_DIR'] = str(RUNTIME / 'python')
    os.environ.setdefault('UV_HTTP_TIMEOUT', '180')
    os.environ.setdefault('UV_CONCURRENT_DOWNLOADS', '4')
    run([uv, 'python', 'install', SPEC['python'], '--no-bin'])
    python = run([uv, 'python', 'find', '--managed-python', SPEC['python']],
                 capture_output=True, text=True).stdout.strip()
    env_python = RUNTIME / 'venv/bin/python'
    if not env_python.is_file():
        run([uv, 'venv', '--python', python, RUNTIME / 'venv'])
    lock = PROFILE_DIR / 'requirements.lock'
    if not lock.is_file():
        raise RuntimeError('缺少所选配置的requirements.lock，请重新解压完整课程包。')
    recipe = hashlib.sha256(lock.read_bytes() + (PROFILE_DIR / 'environment.json').read_bytes()).hexdigest()
    marker = RUNTIME / 'installed.json'
    index = os.environ.get('COURSE_PYPI_INDEX')
    indexes = ['--index-url', index] if index else [
        '--index-url', 'https://pypi.tuna.tsinghua.edu.cn/simple',
        '--extra-index-url', 'https://mirrors.aliyun.com/pypi/simple',
        '--index-strategy', 'unsafe-best-match']
    # Exact versions and hashes stay locked; the second mirror fills missing releases.
    run([uv, 'pip', 'sync', '--python', env_python, '--require-hashes', *indexes, lock])
    nvcc_version = activate_environment(env_python)
    packages = ['vllm', 'torch', 'modelscope', 'transformers']
    if 'cuda-toolkit' in SPEC:
        packages.append('cuda-toolkit')
    code = ('import json,sys,importlib.metadata as m,torch; '
            'print(json.dumps({"python":sys.version.split()[0],'
            '"packages":{k:m.version(k) for k in '
            + repr(packages) + '},'
            '"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0)}))')
    actual = json.loads(run([env_python, '-c', code], capture_output=True, text=True).stdout)
    if actual['python'] != SPEC['python']:
        raise RuntimeError('Python 版本与课程配置不符')
    for name, version in actual['packages'].items():
        if release_version(version) != release_version(SPEC[name]):
            raise RuntimeError(f'{name} 版本与课程配置不符：{version}')
    if actual['cuda'] != SPEC['cuda']:
        raise RuntimeError(f"PyTorch CUDA版本不符：{actual['cuda']}，预期{SPEC['cuda']}")
    actual.update(profile=SPEC['profile'], recipe_sha256=recipe, nvcc=nvcc_version, checked_at=datetime.now(timezone.utc).isoformat())
    marker.write_text(json.dumps(actual, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    say('环境已校验：Python ' + actual['python'] + '，vLLM ' + actual['packages']['vllm'])
    return env_python

def verify_model(model_dir):
    for item in MODEL['files']:
        path = model_dir / item['path']
        if not path.is_file() or path.stat().st_size != item['size']:
            return False
        digest = hashlib.sha256() if item['algorithm'] == 'sha256' else hashlib.sha1()
        if item['algorithm'] == 'git-sha1':
            digest.update(f'blob {path.stat().st_size}\0'.encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != item['digest']:
            return False
    return True

def prepare_model(python):
    default = Path.home() / '.cache/nlp-lab' / ('Qwen3-4B-' + MODEL['revision'][:12])
    model_dir = Path(os.environ.get('COURSE_MODEL_DIR', str(default))).expanduser().resolve()
    say('校验模型文件：' + str(model_dir))
    if not verify_model(model_dir):
        if 'COURSE_MODEL_DIR' in os.environ:
            raise RuntimeError('指定的 COURSE_MODEL_DIR 与课程权重校验值不符。请取消该变量，让脚本下载标准模型。')
        say('从 ModelScope 下载 Qwen3-4B，已下载文件会复用。')
        run([python, '-c', 'from modelscope import snapshot_download; import sys; '
             'snapshot_download(model_id=sys.argv[1], local_dir=sys.argv[2])', MODEL['model'], model_dir])
        if not verify_model(model_dir):
            raise RuntimeError('模型文件与课程固定版本不一致，请保留下载目录并联系助教。')
    (RUNTIME / 'model-verification.json').write_text(json.dumps({
        'model': MODEL['model'], 'revision': MODEL['revision'], 'all_match': True,
        'checked_at': datetime.now(timezone.utc).isoformat(), 'files': len(MODEL['files'])}, indent=2) + '\n')
    say('模型权重与分词器均通过固定版本校验。')
    return model_dir

def runtime_config(model_dir, port):
    config = (PROFILE_DIR / 'vllm.yaml').read_text(encoding='utf-8')
    config, count = re.subn(r'^model:.*$', lambda _: 'model: ' + json.dumps(str(model_dir)), config, flags=re.M)
    if count != 1:
        raise RuntimeError('deploy/profiles/<所选显卡>/vllm.yaml 的 model 配置异常')
    config = re.sub(r'^(?:revision|tokenizer-revision|tokenizer):.*\n?', '', config, flags=re.M)
    config = re.sub(r'^port:.*$', f'port: {port}', config, flags=re.M)
    config += '\ntokenizer: ' + json.dumps(str(model_dir)) + '\n'
    path = RUNTIME / 'vllm.yaml'
    path.write_text(config, encoding='utf-8')
    path.chmod(0o600)
    return path

def main():
    parser = argparse.ArgumentParser(description='一键准备课程环境、下载并校验模型；加 --start 启动 vLLM。')
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--profile', choices=['auto', '4090', '5090'], default=os.environ.get('COURSE_DEPLOY_PROFILE', 'auto'))
    parser.add_argument('--describe', action='store_true', help='只显示所选配置，不安装或运行模型')
    parser.add_argument('--check', action='store_true', help='只检查硬件、驱动与磁盘，不安装')
    args = parser.parse_args()
    choose_profile(args.profile)
    say(f"部署配置：{SPEC['profile']} / CUDA {SPEC['cuda']} / vLLM {SPEC['vllm']}")
    if args.describe:
        say(json.dumps(SPEC, ensure_ascii=False, indent=2))
        return
    preflight()
    if args.check:
        say('部署前检查通过。')
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('COURSE_GPU', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    RUNTIME.mkdir(parents=True, exist_ok=True)
    uv = bootstrap_uv()
    say('准备固定版本 Python 和推理依赖，首次运行需要下载。')
    python = prepare_python(uv)
    model_dir = prepare_model(python)
    port = int(os.environ.get('COURSE_PORT', '8000'))
    if not 1 <= port <= 65535:
        raise RuntimeError('COURSE_PORT 须在 1—65535 之间')
    config = runtime_config(model_dir, port)
    say('准备完成。另开终端进入相同目录，执行：source .lab-runtime/venv/bin/activate')
    if args.start:
        os.environ.update(VLLM_NO_USAGE_STATS='1', HF_HUB_DISABLE_TELEMETRY='1',
                          HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        server = RUNTIME / 'venv/bin/vllm'
        say(f'启动 Qwen3-4B thinking：http://127.0.0.1:{port}/v1')
        os.execv(str(server), [str(server), 'serve', '--config', str(config)])

if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print('部署未完成：' + str(exc), file=sys.stderr)
        sys.exit(1)
