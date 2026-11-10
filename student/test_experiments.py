"""离线正确性测试：不访问模型，也不声称任何准确率。"""
import json
import unittest
from _runner import ROOT, validate_messages, pair_view
from _worker import check_source, environment
from experiment import METHODS, build

SINGLE = ("m1", "m2", "m3", "m3r", "m4")
VOTING = ("m5", "m6")
SHOTS = ("m3", "m3r")
PREFIX = "Predict for this paper pair:\n"


def reply(answer):
    """构造一条只含最终回答的模型响应。"""
    return {"content": json.dumps({"answer": answer})}


class FakeLLM:
    """按预设序列回答的假模型，同时校验并记录每次请求。"""

    def __init__(self, replies):
        """保存预设回答和调用记录。"""
        self.replies = iter(replies)
        self.calls = []

    def chat(self, messages):
        """校验消息并返回下一条预设回答，预设为异常时抛出。"""
        validate_messages(messages)
        self.calls.append(messages)
        item = next(self.replies)
        if isinstance(item, Exception):
            raise item
        return item


class ExperimentTests(unittest.TestCase):
    """覆盖提示构造、解析、预算、检索、投票和词项分类器。"""

    @classmethod
    def setUpClass(cls):
        """生成全部方法文件，在课程沙箱中执行，并载入示例池和公开题输入。"""
        cls.paths = build()
        cls.variants = {method: cls.load(method) for method in METHODS}
        cls.examples = [json.loads(s) for s in (ROOT / "data/examples.jsonl").read_text(encoding="utf8").splitlines()]
        cls.public = [pair_view(json.loads(s)) for s in (ROOT / "data/public.jsonl").read_text(encoding="utf8").splitlines()]

    @classmethod
    def load(cls, method, seed=42):
        """用固定种子在课程沙箱中执行某方法的文件，返回其命名空间。"""
        ns = environment(seed)
        exec(check_source(cls.paths[method].read_text(encoding="utf8")), ns)
        return ns

    def shot_ids(self, ns, messages):
        """把消息中的示例还原成示例 ID，翻转过的示例也能识别。"""
        lookup = {}
        for e in self.examples:
            lookup[ns["pair_text"](e)] = e["id"]
            lookup[ns["pair_text"](ns["flip_pair"](e))] = e["id"]
        return [lookup[m["content"].split("\n", 1)[1]] for m in messages[1:-1:2]]

    def test_all_public_prompts_fit_and_use_only_examples(self):
        """公开题提示都不超限，示例只来自示例池，答案在翻转后依然正确。"""
        base = self.variants["m1"]
        allowed = {}
        for e in self.examples:
            allowed[base["pair_text"](e)] = e["answer"]
            flipped = base["flip_pair"](e)
            allowed[base["pair_text"](flipped)] = flipped["answer"]
        for method, ns in self.variants.items():
            for pair in self.public:
                messages = ns["make_messages"](pair, self.examples)
                validate_messages(messages)
                self.assertLessEqual(ns["message_size"](messages), 18000)
                self.assertTrue(messages[-1]["content"].endswith(ns["pair_text"](pair)))
                shots = messages[1:-1]
                if method not in SHOTS:
                    self.assertEqual(shots, [])
                    continue
                self.assertLessEqual(len(shots), 8)
                for i in range(0, len(shots), 2):
                    material = shots[i]["content"].split("\n", 1)[1]
                    self.assertIn(material, allowed)
                    self.assertEqual(json.loads(shots[i + 1]["content"])["answer"], allowed[material])

    def test_system_prompt_per_method(self):
        """各方法使用计划中的系统提示。"""
        for method in ("m1",):
            self.assertNotIn("Compare both papers", self.variants[method]["system_prompt"]())
        for method in ("m2", "m3", "m3r"):
            self.assertIn("Compare both papers", self.variants[method]["system_prompt"]())
        for method in ("m4", "m5", "m6"):
            prompt = self.variants[method]["system_prompt"]()
            self.assertIn("Size and growth", prompt)
            self.assertNotIn("Compare both papers", prompt)

    def test_normal_answers(self):
        """单次调用方案在正常回答时只调用一次。"""
        for method in SINGLE:
            ns = self.variants[method]
            for answer in ("A", "B"):
                llm = FakeLLM([reply(answer)])
                self.assertEqual(ns["predict"](self.public[0], self.examples, llm), answer)
                self.assertEqual(len(llm.calls), 1)

    def test_parser_rejects_ambiguous_or_reasoning_text(self):
        """解析器拒绝含糊文本，接受代码块和文中唯一的答案 JSON。"""
        parse = self.variants["m1"]["parse_answer"]
        for value in ("A or B", "", '{"answer":"C"}', "Paper A is better", "null", "[]",
                      '{"answer":"A"} or {"answer":"B"}'):
            self.assertIsNone(parse(value))
        self.assertEqual(parse('```json\n{"answer":"B"}\n```'), "B")
        self.assertEqual(parse('Final decision: {"answer": "A"}'), "A")

    def test_retry_ignores_reasoning_and_handles_transport_error(self):
        """重试时忽略思考文本，并能跨过传输错误。"""
        for method in SINGLE:
            ns = self.variants[method]
            llm = FakeLLM([{"content": "", "reasoning": "Answer A"}, RuntimeError("offline"), {"content": "B"}])
            self.assertEqual(ns["predict"](self.public[0], self.examples, llm), "B")
            self.assertEqual(len(llm.calls), 3)

    def test_failure_is_bounded_and_explicit(self):
        """单次调用方案三次失败后明确报错，不伪装成功。"""
        for method in SINGLE:
            ns = self.variants[method]
            llm = FakeLLM([{"content": "uncertain"}] * 3)
            with self.assertRaises(ValueError):
                ns["predict"](self.public[0], self.examples, llm)
            self.assertEqual(len(llm.calls), 3)

    def test_flip_preserves_fact(self):
        """翻转会交换论文并翻转答案，翻转两次还原。"""
        ns = self.variants["m3"]
        example = self.examples[0]
        flipped = ns["flip_pair"](example)
        self.assertEqual(flipped["paper_a"], example["paper_b"])
        self.assertNotEqual(flipped["answer"], example["answer"])
        self.assertEqual(ns["flip_pair"](flipped), example)

    def test_retrieval_deterministic_and_order_invariant(self):
        """检索结果与 A/B 顺序和示例池顺序无关，标签固定为 A/B/A/B。"""
        ns = self.variants["m3"]
        pair = self.public[0]
        swapped = {"paper_a": pair["paper_b"], "paper_b": pair["paper_a"]}
        first = [e["id"] for e in ns["ranked_examples"](pair, self.examples)]
        second = [e["id"] for e in ns["ranked_examples"](swapped, list(reversed(self.examples)))]
        self.assertEqual(first, second)
        messages = ns["make_messages"](pair, self.examples)
        labels = [json.loads(m["content"])["answer"] for m in messages if m["role"] == "assistant"]
        self.assertEqual(labels, ["A", "B", "A", "B"])

    def test_m3_keeps_true_top_four_when_they_fit(self):
        """真实前 4 名放得下时，M3 选中的必须正好是它们，且按相似度排列。"""
        ns = self.variants["m3"]
        kept = 0
        for pair in self.public:
            messages = ns["make_messages"](pair, self.examples)
            top = ns["ranked_examples"](pair, self.examples)[:4]
            extra = [{"role": "user", "content": "Labeled example:\n" + ns["pair_text"](e)} for e in top]
            answers = [{"role": "assistant", "content": '{"answer": "A"}'} for _ in top]
            if ns["message_size"]([messages[0], messages[-1]] + extra + answers) <= 18000:
                self.assertEqual(self.shot_ids(ns, messages), [e["id"] for e in top])
                kept += 1
        self.assertGreaterEqual(kept, 90)

    def test_m3r_random_control(self):
        """M3r 的选例不随示例池顺序变化，并且与 M3 的选例不同。"""
        pair = self.public[0]
        first = self.load("m3r", seed=7)
        second = self.load("m3r", seed=7)
        a = self.shot_ids(first, first["make_messages"](pair, self.examples))
        b = self.shot_ids(second, second["make_messages"](pair, list(reversed(self.examples))))
        self.assertEqual(a, b)
        differ = 0
        m3 = self.variants["m3"]
        m3r = self.variants["m3r"]
        for item in self.public[:20]:
            differ += self.shot_ids(m3, m3["make_messages"](item, self.examples)) != \
                self.shot_ids(m3r, m3r["make_messages"](item, self.examples))
        self.assertGreaterEqual(differ, 15)

    def test_self_pair_excluded_in_both_orders(self):
        """在开发集上，无论 A/B 顺序如何，当前题都不会进入示例或分类器。"""
        sample = self.examples[0]
        pair = pair_view(sample)
        for method in SHOTS:
            ns = self.variants[method]
            for view in (pair, ns["flip_pair"](pair)):
                self.assertNotIn(sample["id"], self.shot_ids(ns, ns["make_messages"](view, self.examples)))
        self.assertIsNone(self.variants["m6"]["lexical_answer"](pair, [sample]))

    def test_empty_and_oversized_example_pools(self):
        """示例池为空或示例都放不下时，退化为只含当前题的请求。"""
        for method in SHOTS:
            ns = self.variants[method]
            self.assertEqual(len(ns["make_messages"](self.public[0], [])), 2)
            big = json.loads(json.dumps(self.examples))
            for e in big:
                e["paper_a"]["abstract"] = "超长摘要" * 6000
            messages = ns["make_messages"](self.public[0], big)
            self.assertEqual(len(messages), 2)
            validate_messages(messages)

    def test_oversized_current_pair_rejected_without_call(self):
        """当前题超限时直接报错，不发起任何模型调用。"""
        pair = json.loads(json.dumps(self.public[0]))
        pair["paper_a"]["abstract"] = "中文" * 10000
        for ns in self.variants.values():
            llm = FakeLLM([])
            with self.assertRaises(ValueError):
                ns["predict"](pair, self.examples, llm)
            self.assertEqual(llm.calls, [])

    def test_vote_alternates_orders_and_maps_answers_back(self):
        """投票按原序、交换交替调用，交换后的答案会映射回原顺序。"""
        ns = self.variants["m5"]
        pair = self.public[0]
        llm = FakeLLM([reply("A"), reply("B"), reply("B"), reply("B")])
        self.assertEqual(ns["predict"](pair, self.examples, llm), "A")
        self.assertEqual(len(llm.calls), 4)
        original = PREFIX + ns["pair_text"](pair)
        swapped = PREFIX + ns["pair_text"](ns["flip_pair"](pair))
        self.assertEqual([c[-1]["content"] for c in llm.calls], [original, swapped, original, swapped])

    def test_m5_tie_uses_first_vote(self):
        """M5 平票时使用第一票。"""
        llm = FakeLLM([reply("B")] * 4)
        self.assertEqual(self.variants["m5"]["predict"](self.public[0], self.examples, llm), "B")

    def test_m6_tie_uses_lexical_classifier(self):
        """M6 平票时使用词项分类器的结果。"""
        ns = self.variants["m6"]
        pair = self.public[0]
        expected = ns["lexical_answer"](pair, self.examples)
        for raw in ("A", "B"):
            llm = FakeLLM([reply(raw)] * 4)
            self.assertEqual(ns["predict"](pair, self.examples, llm), expected)

    def test_m6_majority_is_not_overridden(self):
        """M6 有多数票时不调用词项分类器的结论。"""
        ns = self.variants["m6"]
        pair = self.public[0]
        target = "A" if ns["lexical_answer"](pair, self.examples) == "B" else "B"
        raw = (target, ns["other_label"](target), target, ns["other_label"](target))
        self.assertEqual(ns["predict"](pair, self.examples, FakeLLM([reply(r) for r in raw])), target)

    def test_vote_retry_stays_within_budget(self):
        """全部失败时投票方案最多调用 5 次，M5 报错，M6 用词项分类器兜底。"""
        pair = self.public[0]
        llm = FakeLLM([{"content": "uncertain"}] * 5)
        with self.assertRaises(ValueError):
            self.variants["m5"]["predict"](pair, self.examples, llm)
        self.assertEqual(len(llm.calls), 5)
        llm = FakeLLM([{"content": "uncertain"}] * 5)
        ns = self.variants["m6"]
        self.assertEqual(ns["predict"](pair, self.examples, llm), ns["lexical_answer"](pair, self.examples))
        self.assertEqual(len(llm.calls), 5)

    def test_vote_uses_one_retry_then_continues(self):
        """第一次失败重试后，其余顺序仍各调用一次，总数为 5。"""
        pair = self.public[0]
        llm = FakeLLM([RuntimeError("offline"), reply("A"), reply("B"), reply("A"), reply("B")])
        self.assertEqual(self.variants["m5"]["predict"](pair, self.examples, llm), "A")
        self.assertEqual(len(llm.calls), 5)
        self.assertIn("Return only", llm.calls[1][-1]["content"])

    def test_solve_known_system(self):
        """高斯消元能解出已知的线性方程组。"""
        result = self.variants["m6"]["solve"]([[2.0, 1.0], [1.0, 3.0]], [3.0, 5.0])
        self.assertAlmostEqual(result[0], 0.8)
        self.assertAlmostEqual(result[1], 1.4)

    def test_lexical_answer_is_order_equivariant(self):
        """交换两篇论文后，词项分类器的答案随之翻转。"""
        ns = self.variants["m6"]
        for pair in self.public[:5]:
            self.assertNotEqual(ns["lexical_answer"](pair, self.examples),
                                ns["lexical_answer"](ns["flip_pair"](pair), self.examples))


if __name__ == "__main__":
    unittest.main(verbosity=2)
