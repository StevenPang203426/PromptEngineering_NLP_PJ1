"""Build standalone variants and run through the unchanged course runner."""
import argparse
import json
import re
from types import SimpleNamespace
from _runner import ROOT, run

METHODS = ("m1", "m2", "m3")


def build():
    source = (ROOT / "solution.py").read_text(encoding="utf8")
    if len(re.findall(r'^METHOD = "m[123]"$', source, re.M)) != 1:
        raise ValueError("Expected exactly one METHOD setting in solution.py")
    folder = ROOT / "experiments"
    folder.mkdir(exist_ok=True)
    paths = {}
    for method in METHODS:
        path = folder / (method + ".py")
        path.write_text(re.sub(r'^METHOD = "m[123]"$', 'METHOD = "' + method + '"', source, flags=re.M), encoding="utf8")
        paths[method] = path
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=("all",) + METHODS, default="all")
    p.add_argument("--limit", type=int)
    p.add_argument("--mock", action="store_true", help="Simulation only, not model accuracy")
    p.add_argument("--build-only", action="store_true")
    a = p.parse_args()
    if a.limit is not None and a.limit < 1:
        p.error("--limit must be positive")
    paths = build()
    if a.build_only:
        print("Built standalone experiments/m1.py, m2.py, m3.py")
        return
    output = ROOT / ("runs_mock" if a.mock else "runs")
    for method in METHODS if a.method == "all" else (a.method,):
        folder = run(SimpleNamespace(solution=str(paths[method]), data=str(ROOT / "data/public.jsonl"),
            labels=None, output=str(output), workers=16, limit=a.limit, mock=a.mock,
            no_open=True, question_timeout=21600, request_timeout=3600))
        rows = [json.loads(s) for s in (folder / "predictions.jsonl").read_text(encoding="utf8").splitlines()]
        if any(r.get("error") or r.get("prediction") not in ("A", "B") for r in rows):
            raise SystemExit("Invalid predictions detected; inspect " + str(folder))


if __name__ == "__main__":
    main()
