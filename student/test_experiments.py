"""Offline correctness tests; no model access and no accuracy claims."""
import json
import unittest
from _runner import ROOT, validate_messages, pair_view
from _worker import check_source, environment
from experiment import build


class FakeLLM:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def chat(self, messages):
        validate_messages(messages)
        self.calls.append(messages)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.variants = {}
        for method, path in build().items():
            ns = environment(42)
            exec(check_source(path.read_text(encoding="utf8")), ns)
            cls.variants[method] = ns
        cls.examples = [json.loads(s) for s in (ROOT / "data/examples.jsonl").read_text(encoding="utf8").splitlines()]
        cls.public = [pair_view(json.loads(s)) for s in (ROOT / "data/public.jsonl").read_text(encoding="utf8").splitlines()]

    def test_all_public_prompts_fit_and_use_only_examples(self):
        allowed = {self.variants['m1']['pair_text'](e): e['answer'] for e in self.examples}
        for method, ns in self.variants.items():
            for pair in self.public:
                messages = ns['make_messages'](pair, self.examples)
                validate_messages(messages)
                self.assertLessEqual(ns['message_size'](messages), 18000)
                self.assertTrue(messages[-1]['content'].endswith(ns['pair_text'](pair)))
                shots = messages[1:-1]
                if method != 'm3':
                    self.assertEqual(shots, [])
                else:
                    self.assertLessEqual(len(shots), 8)
                    for i in range(0, len(shots), 2):
                        material = shots[i]['content'].split('\n', 1)[1]
                        self.assertIn(material, allowed)
                        self.assertEqual(json.loads(shots[i+1]['content'])['answer'], allowed[material])

    def test_normal_answers(self):
        for ns in self.variants.values():
            for answer in ('A', 'B'):
                llm = FakeLLM([{'content': json.dumps({'answer': answer})}])
                self.assertEqual(ns['predict'](self.public[0], self.examples, llm), answer)
                self.assertEqual(len(llm.calls), 1)

    def test_parser_rejects_ambiguous_or_reasoning_text(self):
        parse = self.variants['m1']['parse_answer']
        for value in ('A or B', '', '{"answer":"C"}', 'Paper A is better', 'null', '[]'):
            self.assertIsNone(parse(value))
        self.assertEqual(parse('```json\n{"answer":"B"}\n```'), 'B')

    def test_retry_ignores_reasoning_and_handles_transport_error(self):
        for ns in self.variants.values():
            llm = FakeLLM([{'content': '', 'reasoning': 'Answer A'}, RuntimeError('offline'), {'content': 'B'}])
            self.assertEqual(ns['predict'](self.public[0], self.examples, llm), 'B')
            self.assertEqual(len(llm.calls), 3)

    def test_failure_is_bounded_and_explicit(self):
        for ns in self.variants.values():
            llm = FakeLLM([{'content': 'uncertain'}] * 3)
            with self.assertRaises(ValueError):
                ns['predict'](self.public[0], self.examples, llm)
            self.assertEqual(len(llm.calls), 3)

    def test_retrieval_deterministic_and_order_invariant(self):
        ns = self.variants['m3']
        pair = self.public[0]
        swapped = {'paper_a': pair['paper_b'], 'paper_b': pair['paper_a']}
        first = [e['id'] for e in ns['ranked_examples'](pair, self.examples)]
        second = [e['id'] for e in ns['ranked_examples'](swapped, list(reversed(self.examples)))]
        self.assertEqual(first, second)
        messages = ns['make_messages'](pair, self.examples)
        labels = [json.loads(m['content'])['answer'] for m in messages if m['role'] == 'assistant']
        self.assertEqual(labels, ['A', 'B', 'A', 'B'])

    def test_empty_and_oversized_example_pools(self):
        ns = self.variants['m3']
        self.assertEqual(len(ns['make_messages'](self.public[0], [])), 2)
        big = json.loads(json.dumps(self.examples))
        for e in big:
            e['paper_a']['abstract'] = '超长摘要' * 6000
        messages = ns['make_messages'](self.public[0], big)
        self.assertEqual(len(messages), 2)
        validate_messages(messages)

    def test_oversized_current_pair_rejected_without_call(self):
        pair = json.loads(json.dumps(self.public[0]))
        pair['paper_a']['abstract'] = '中文' * 10000
        for ns in self.variants.values():
            llm = FakeLLM([])
            with self.assertRaises(ValueError):
                ns['predict'](pair, self.examples, llm)
            self.assertEqual(llm.calls, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
