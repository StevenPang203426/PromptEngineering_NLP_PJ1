"""运行solution.py，自动评估并保存结果。平时直接执行：python run.py。"""
import argparse
from types import SimpleNamespace
from _runner import ROOT, run


def options(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--limit', type=int, help='先试前几题，例如 --limit 3；不填则运行全部题目')
    p.add_argument('--data', default=str(ROOT/'data/public.jsonl'), help='测试集文件；默认公开集')
    a = p.parse_args(argv)
    if a.limit is not None and a.limit < 1:
        p.error('--limit须为正整数')
    return SimpleNamespace(solution=str(ROOT/'solution.py'), data=a.data,
        labels=None, output=str(ROOT/'runs'), workers=16, limit=a.limit,
        mock=False, no_open=True, question_timeout=21600, request_timeout=3600)


if __name__ == '__main__':
    try:
        run(options())
    except (OSError, ValueError) as exc:
        raise SystemExit('运行未完成：'+str(exc)+'\n请确认模型启动窗口仍在运行，并检查solution.py。')
