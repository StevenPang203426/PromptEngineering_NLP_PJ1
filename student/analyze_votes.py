"""从运行记录离线计算投票指标：派生 M4/M5/M6、A/B 翻转率和方法间配对比较，不调用模型。"""
import argparse
import json
import math
import runpy
from collections import defaultdict
from pathlib import Path
from _runner import ROOT, read_jsonl

PREFIX = "Predict for this paper pair:\n"
VOTING = ("m5", "m6")


def load_calls(folder):
    """读取 calls.jsonl，按题号返回按调用次序排列的 (消息, 响应) 列表，失败调用的响应为 None。"""
    reserved, responses = {}, {}
    for event in read_jsonl(folder / "calls.jsonl"):
        if event["event"] == "reserved":
            reserved[event["call_id"]] = event
        elif event["event"] == "completed":
            responses[event["call_id"]] = event["response"]
    grouped = defaultdict(list)
    for call_id, event in reserved.items():
        grouped[event["id"]].append((event["local_index"], event["messages"], responses.get(call_id)))
    return {rid: [(m, r) for _, m, r in sorted(items, key=lambda item: item[0])]
            for rid, items in grouped.items()}


def extract_votes(pair, calls, code):
    """解析一题的全部调用，返回每张有效票的顺序、模型原始答案和映射回原顺序的答案。"""
    original = PREFIX + code["pair_text"](pair)
    swapped = PREFIX + code["pair_text"](code["flip_pair"](pair))
    votes = []
    for messages, response in calls:
        content = messages[-1]["content"]
        if not content.startswith((original, swapped)):
            raise ValueError("调用记录中的题目与数据集不一致")
        is_swapped = content.startswith(swapped)
        raw = code["parse_answer"](response.get("content", "")) if response else None
        if raw:
            votes.append({"swapped": is_swapped, "raw": raw,
                          "paper": code["other_label"](raw) if is_swapped else raw})
    return votes


def derive_methods(pair, votes, examples, code):
    """用一题的投票推导 M4、M5、M6 的答案，无法给出答案时记为 None。"""
    papers = [vote["paper"] for vote in votes]
    originals = [vote["paper"] for vote in votes if not vote["swapped"]]
    result = {"m4": originals[0] if originals else None}
    for method in VOTING:
        try:
            result[method] = code["decide"](pair, examples, papers, method)
        except ValueError:
            result[method] = None
    return result


def vote_stats(all_votes):
    """统计 A/B 翻转率、平票率和模型选择位置 A 的比例。"""
    compared = flips = ties = voted = picks = picks_a = 0
    for votes in all_votes.values():
        first_original = next((v for v in votes if not v["swapped"]), None)
        first_swapped = next((v for v in votes if v["swapped"]), None)
        if first_original and first_swapped:
            compared += 1
            flips += first_original["paper"] != first_swapped["paper"]
        if votes:
            voted += 1
            ties += sum(v["paper"] == "A" for v in votes) * 2 == len(votes)
        picks += len(votes)
        picks_a += sum(v["raw"] == "A" for v in votes)
    return {"compared": compared, "flips": flips, "voted": voted, "ties": ties,
            "picks": picks, "picks_a": picks_a}


def load_run(folder, examples):
    """载入一次运行，返回其方法、预测和统计；投票运行会额外派生 M4/M5/M6。"""
    folder = Path(folder).resolve()
    config = json.loads((folder / "config.json").read_text(encoding="utf8"))
    code = runpy.run_path(str(folder / "solution.py"))
    method = code["METHOD"]
    actual = {r["id"]: r.get("prediction") for r in read_jsonl(folder / "predictions.jsonl")}
    run = {"folder": folder, "config": config, "method": method, "methods": {method: actual},
           "stats": None, "mismatches": 0}
    if method in VOTING:
        calls = load_calls(folder)
        rows = {r["id"]: r for r in read_jsonl(folder / "dataset.jsonl")}
        votes, derived = {}, defaultdict(dict)
        for rid in actual:
            pair = {k: rows[rid][k] for k in ("paper_a", "paper_b")}
            votes[rid] = extract_votes(pair, calls.get(rid, []), code)
            for name, answer in derive_methods(pair, votes[rid], examples, code).items():
                derived[name][rid] = answer
        run["mismatches"] = sum(derived[method][rid] != actual[rid] for rid in actual)
        run["methods"] = {**derived, method: actual}
        run["stats"] = vote_stats(votes)
    return run


def merge_runs(runs):
    """合并多次运行的预测：实际运行优先于派生结果，同一方法的多次实际运行视为错误。"""
    if len({run["config"]["data_sha256"] for run in runs}) != 1:
        raise ValueError("不能跨数据集比较，请只传同一数据集的运行目录")
    merged, actual = {}, set()
    for run in runs:
        if run["method"] in actual:
            raise ValueError("方法 " + run["method"] + " 出现了多次实际运行，请只保留一份")
        actual.add(run["method"])
    for run in runs:
        for name, predictions in run["methods"].items():
            if name == run["method"] or name not in actual:
                merged[name] = predictions
    return merged


def load_gold(path):
    """读取标签文件，返回题号到答案的映射。"""
    return {g["id"]: g["answer"] for g in json.loads(Path(path).read_text(encoding="utf8"))}


def score(predictions, gold):
    """返回一组预测的正确数、题数和无效数。"""
    ids = [rid for rid in predictions if rid in gold]
    correct = sum(predictions[rid] == gold[rid] for rid in ids)
    invalid = sum(predictions[rid] not in ("A", "B") for rid in ids)
    return correct, len(ids), invalid


def mcnemar_p(left_only, right_only):
    """计算配对比较的精确 McNemar 双侧 p 值。"""
    total = left_only + right_only
    if not total:
        return 1.0
    tail = sum(math.comb(total, i) for i in range(min(left_only, right_only) + 1))
    return min(1.0, 2 * tail / 2 ** total)


def paired(reference, other, gold):
    """比较两组预测，返回只有参考方法答对的题数和只有另一方法答对的题数。"""
    ids = [rid for rid in reference if rid in other and rid in gold]
    only_reference = sum(reference[rid] == gold[rid] and other[rid] != gold[rid] for rid in ids)
    only_other = sum(reference[rid] != gold[rid] and other[rid] == gold[rid] for rid in ids)
    return only_reference, only_other


def render(runs, merged, gold, reference):
    """把准确率、投票统计和配对比较渲染成 Markdown 文本。"""
    lines = []
    if any(run["config"].get("mock") for run in runs):
        lines += ["> 警告：包含 MOCK 模拟运行，以下数字不是模型成绩。", ""]
    lines += ["| 方法 | 正确/总数 | 准确率 | 无效 |", "|---|---:|---:|---:|"]
    for name in sorted(merged):
        correct, total, invalid = score(merged[name], gold)
        lines.append(f"| {name} | {correct}/{total} | {correct / max(total, 1):.1%} | {invalid} |")
    for run in runs:
        stats = run["stats"]
        if stats:
            lines += ["", f"投票统计（{run['folder'].name}）：",
                      f"- A/B 翻转：{stats['flips']}/{stats['compared']}",
                      f"- 平票：{stats['ties']}/{stats['voted']}",
                      f"- 模型选位置 A：{stats['picks_a']}/{stats['picks']}",
                      f"- 派生结果与实际返回不一致：{run['mismatches']} 题"]
    if reference in merged:
        lines += ["", f"与 {reference} 的逐题比较（精确 McNemar）：", "",
                  f"| 方法 | 仅 {reference} 对 | 仅该方法对 | p |", "|---|---:|---:|---:|"]
        for name in sorted(merged):
            if name != reference:
                left, right = paired(merged[reference], merged[name], gold)
                lines.append(f"| {name} | {left} | {right} | {mcnemar_p(left, right):.3f} |")
    return "\n".join(lines)


def main():
    """解析命令行，载入运行目录并打印分析结果。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folders", nargs="+", help="同一数据集上的运行目录，可同时传多个方法")
    p.add_argument("--labels", help="标签文件；默认 data/<数据集名>_labels.json")
    p.add_argument("--reference", default="m2", help="配对比较的参考方法")
    a = p.parse_args()
    examples = read_jsonl(ROOT / "data/examples.jsonl")
    runs = [load_run(folder, examples) for folder in a.folders]
    merged = merge_runs(runs)
    labels = a.labels or ROOT / "data" / (runs[0]["config"]["dataset_name"] + "_labels.json")
    print(render(runs, merged, load_gold(labels), a.reference))


if __name__ == "__main__":
    main()
