"""生成各方法的独立解答文件，并通过未改动的课程运行器运行。"""
import argparse
import json
import re
from types import SimpleNamespace
from _runner import ROOT, run

METHODS = ("m1", "m2", "m3", "m3r", "m4", "m5", "m6")
# M4 和 M6 可以由 M5 的调用记录离线得到，批量运行时不重复消耗模型调用。
DEFAULT_RUNS = ("m1", "m2", "m3", "m3r", "m5")
METHOD_LINE = r'^METHOD = "m[1-6]r?"$'


def build():
    """按 METHOD 行生成 experiments/ 下各方法的独立文件，返回方法到路径的映射。"""
    source = (ROOT / "solution.py").read_text(encoding="utf8")
    if len(re.findall(METHOD_LINE, source, re.M)) != 1:
        raise ValueError("Expected exactly one METHOD setting in solution.py")
    folder = ROOT / "experiments"
    folder.mkdir(exist_ok=True)
    paths = {}
    for method in METHODS:
        path = folder / (method + ".py")
        path.write_text(re.sub(METHOD_LINE, 'METHOD = "' + method + '"', source, flags=re.M),
                        encoding="utf8", newline="\n")
        paths[method] = path
    return paths


def ensure_dev_labels():
    """从示例池生成开发集标签文件并返回路径，标签只来自 examples.jsonl。"""
    rows = [json.loads(s) for s in (ROOT / "data/examples.jsonl").read_text(encoding="utf8").splitlines() if s.strip()]
    path = ROOT / "data/examples_labels.json"
    path.write_text(json.dumps([{"id": r["id"], "answer": r["answer"]} for r in rows], ensure_ascii=False, indent=2),
                    encoding="utf8", newline="\n")
    return path


def invalid_count(folder):
    """统计一次运行中出错或返回值不是 A/B 的题数。"""
    rows = [json.loads(s) for s in (folder / "predictions.jsonl").read_text(encoding="utf8").splitlines()]
    return sum(bool(r.get("error")) or r.get("prediction") not in ("A", "B") for r in rows)


def main():
    """解析命令行，生成方法文件并依次运行；有无效返回时记录并继续后面的方法。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=("all",) + METHODS, default="all",
                   help="all 运行 m1、m2、m3、m3r、m5；m4 和 m6 用 analyze_votes.py 从 m5 记录派生")
    p.add_argument("--limit", type=int)
    p.add_argument("--dev", action="store_true", help="在示例池上调试流程（当前题会从示例中排除）")
    p.add_argument("--mock", action="store_true", help="Simulation only, not model accuracy")
    p.add_argument("--build-only", action="store_true")
    a = p.parse_args()
    if a.limit is not None and a.limit < 1:
        p.error("--limit must be positive")
    paths = build()
    if a.build_only:
        print("Built standalone experiments/" + ", ".join(m + ".py" for m in METHODS))
        return
    data = ROOT / ("data/examples.jsonl" if a.dev else "data/public.jsonl")
    labels = str(ensure_dev_labels()) if a.dev else None
    output = ROOT / (("runs_mock" if a.mock else "runs") + ("_dev" if a.dev else ""))
    finished = []
    for method in DEFAULT_RUNS if a.method == "all" else (a.method,):
        folder = run(SimpleNamespace(solution=str(paths[method]), data=str(data),
            labels=labels, output=str(output), workers=16, limit=a.limit, mock=a.mock,
            no_open=True, question_timeout=21600, request_timeout=3600))
        finished.append((method, folder, invalid_count(folder)))
    print("\n本次运行：")
    for method, folder, bad in finished:
        print(f"{method}: 无效返回 {bad} 题，目录 {folder}")
    if any(bad for _, _, bad in finished):
        print("存在无效返回，请打开对应目录的 HTML 查看错误记录。")
    print("离线分析：python analyze_votes.py " + " ".join(str(folder) for _, folder, _ in finished))


if __name__ == "__main__":
    main()
