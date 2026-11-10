"""离线分析工具的测试：用脚本化回答伪造运行记录，不访问模型，也不读取公开标签。"""
import json
import runpy
import shutil
import tempfile
import unittest
from pathlib import Path
import analyze_votes as av
from _runner import ROOT, pair_view, read_jsonl
from experiment import build


def reply(answer):
    """构造一条只含最终回答的模型响应。"""
    return {"content": json.dumps({"answer": answer}), "reasoning": ""}


class ScriptedLLM:
    """按脚本回答，并按框架格式记录调用事件的假模型。"""

    def __init__(self, replies, rid, events, counter):
        """保存脚本、题号以及共享的事件列表和全局调用计数。"""
        self.replies, self.rid, self.events, self.counter = iter(replies), rid, events, counter
        self.local = 0

    def chat(self, messages):
        """写入 reserved 和 completed 事件，并返回下一条脚本回答。"""
        self.local += 1
        self.counter[0] += 1
        call_id = self.counter[0]
        self.events.append({"event": "reserved", "call_id": call_id, "id": self.rid,
                            "local_index": self.local, "messages": messages})
        response = next(self.replies)
        if response is None:
            self.events.append({"event": "error", "call_id": call_id, "id": self.rid, "error": "RuntimeError"})
            raise RuntimeError("scripted failure")
        self.events.append({"event": "completed", "call_id": call_id, "id": self.rid, "response": response})
        return response


class AnalyzeVotesTests(unittest.TestCase):
    """验证派生方法、投票统计、合并规则和配对检验。"""

    @classmethod
    def setUpClass(cls):
        """生成方法文件，载入示例池，并准备临时目录。"""
        cls.paths = build()
        cls.examples = read_jsonl(ROOT / "data/examples.jsonl")
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        """删除临时目录。"""
        cls.temp.cleanup()

    def make_run(self, name, method, script, data_sha="same"):
        """用真实 predict 跑脚本回答，并按框架格式写出完整的运行目录。"""
        folder = self.root / name
        folder.mkdir()
        shutil.copy(self.paths[method], folder / "solution.py")
        code = runpy.run_path(str(self.paths[method]))
        rows = self.examples[:len(script)]
        events, predictions, counter = [], [], [0]
        for row, replies in zip(rows, script):
            llm = ScriptedLLM(replies, row["id"], events, counter)
            try:
                predictions.append({"id": row["id"], "prediction": code["predict"](pair_view(row), self.examples, llm)})
            except ValueError as exc:
                predictions.append({"id": row["id"], "prediction": None, "error": str(exc)})
        (folder / "config.json").write_text(json.dumps({"dataset_name": "examples", "data_sha256": data_sha, "mock": False}), encoding="utf8")
        for filename, records in (("calls.jsonl", events), ("predictions.jsonl", predictions), ("dataset.jsonl", rows)):
            (folder / filename).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf8")
        return folder, code, rows

    def test_vote_run_derives_methods_and_stats(self):
        """一次 M6 运行能派生 M4/M5/M6，统计正确，且与实际返回一致。"""
        script = [[reply(x) for x in row] for row in ("AAAA", "ABAB", "BAAB", "BBBB")]
        folder, code, rows = self.make_run("vote", "m6", script)
        run = av.load_run(folder, self.examples)
        ids = [r["id"] for r in rows]
        self.assertEqual([run["methods"]["m4"][i] for i in ids], ["A", "A", "B", "B"])
        self.assertEqual([run["methods"]["m5"][i] for i in ids], ["A", "A", "B", "B"])
        expected = [code["lexical_answer"](pair_view(rows[0]), self.examples), "A",
                    code["lexical_answer"](pair_view(rows[2]), self.examples),
                    code["lexical_answer"](pair_view(rows[3]), self.examples)]
        self.assertEqual([run["methods"]["m6"][i] for i in ids], expected)
        self.assertEqual(run["mismatches"], 0)
        self.assertEqual(run["stats"], {"compared": 4, "flips": 2, "voted": 4, "ties": 3, "picks": 16, "picks_a": 8})

    def test_failed_calls_drop_votes_only(self):
        """失败或无法解析的调用只减少票数，全部失败时派生结果为 None。"""
        script = [[None, reply("A"), reply("B"), reply("A"), reply("B")],
                  [{"content": "unsure", "reasoning": ""}] * 5]
        folder, _, rows = self.make_run("partial", "m5", script)
        run = av.load_run(folder, self.examples)
        self.assertEqual(run["methods"]["m5"][rows[0]["id"]], "A")
        self.assertIsNone(run["methods"]["m5"][rows[1]["id"]])
        self.assertIsNone(run["methods"]["m4"][rows[1]["id"]])
        self.assertEqual(run["mismatches"], 0)

    def test_single_call_run_has_no_derived_methods(self):
        """单次调用运行只报告自身方法。"""
        folder, _, _ = self.make_run("single", "m2", [[reply("A")]])
        run = av.load_run(folder, self.examples)
        self.assertEqual(set(run["methods"]), {"m2"})
        self.assertIsNone(run["stats"])

    def test_merge_prefers_actual_runs_and_rejects_conflicts(self):
        """实际运行优先于派生结果，重复方法或不同数据集会报错。"""
        script = [[reply(x) for x in "ABAB"]]
        vote_folder, _, rows = self.make_run("merge_vote", "m5", script)
        real_folder, _, _ = self.make_run("merge_real", "m4", [[reply("B")]])
        vote, real = av.load_run(vote_folder, self.examples), av.load_run(real_folder, self.examples)
        for order in ([vote, real], [real, vote]):
            self.assertEqual(av.merge_runs(order)["m4"][rows[0]["id"]], "B")
        with self.assertRaises(ValueError):
            av.merge_runs([real, real])
        other_folder, _, _ = self.make_run("other_data", "m1", [[reply("A")]], data_sha="different")
        with self.assertRaises(ValueError):
            av.merge_runs([real, av.load_run(other_folder, self.examples)])

    def test_mcnemar_and_scoring(self):
        """精确 McNemar 与配对计数、评分结果符合手算值。"""
        self.assertAlmostEqual(av.mcnemar_p(0, 5), 0.0625)
        self.assertEqual(av.mcnemar_p(0, 0), 1.0)
        gold = {"x": "A", "y": "B", "z": "A"}
        left = {"x": "A", "y": "A", "z": None}
        right = {"x": "B", "y": "B", "z": "A"}
        self.assertEqual(av.paired(left, right, gold), (1, 2))
        self.assertEqual(av.score(left, gold), (1, 3, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
