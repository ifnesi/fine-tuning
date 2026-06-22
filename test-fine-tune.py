#!/usr/bin/env python3
import re
import os
import time
import json
import requests
import argparse

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestCase:
    id: str
    prompt: str
    expected: str
    kind: str


def load_test_cases(questions_file: str) -> list[TestCase]:
    """Load test cases from a JSONL file."""
    test_cases = []
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                test_cases.append(TestCase(
                    id=data['id'],
                    kind=data['kind'],
                    expected=data['expected'],
                    prompt=data['prompt']
                ))
    return test_cases


def normalize_answer(text: str, kind: str) -> str:
    cleaned = text.strip().upper()
    if kind == "true_false":
        if cleaned.startswith("TRUE"):
            return "TRUE"
        if cleaned.startswith("FALSE"):
            return "FALSE"
        m = re.search(r"\b(TRUE|FALSE)\b", cleaned)
        return m.group(1) if m else cleaned.splitlines()[0][:20]
    m = re.search(r"\b([ABCD])\b", cleaned)
    return m.group(1) if m else cleaned.splitlines()[0][:20]


def run_case(case: TestCase, ollama_url: str, model: str, timeout: int, show_feedback: bool = True) -> dict:
    payload = {
        "model": model,
        "prompt": case.prompt,
        "stream": False,
        "options": {
            "temperature": 0,
        },
    }
    started = time.perf_counter()
    try:
        resp = requests.post(ollama_url, json=payload, timeout=timeout)
        elapsed = time.perf_counter() - started
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("response", "")
        normalized = normalize_answer(raw, case.kind)
        passed = normalized == case.expected
        
        # Show real-time feedback
        if show_feedback:
            status = "✓ CORRECT" if passed else "✗ INCORRECT"
            color = "\033[92m" if passed else "\033[91m"  # Green or Red
            reset = "\033[0m"
            print(f"{color}{status}{reset} [{case.id}] Expected: {case.expected}, Got: {normalized} ({elapsed:.2f}s)")
        
        return {
            "id": case.id,
            "kind": case.kind,
            "expected": case.expected,
            "raw_response": raw.strip(),
            "normalized_response": normalized,
            "passed": passed,
            "latency_seconds": round(elapsed, 3),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
            "eval_duration_ns": data.get("eval_duration"),
            "error": None,
        }
    except Exception as e:
        elapsed = time.perf_counter() - started
        
        # Show error feedback
        if show_feedback:
            print(f"\033[91m✗ ERROR\033[0m [{case.id}] {str(e)[:50]}... ({elapsed:.2f}s)")
        
        return {
            "id": case.id,
            "kind": case.kind,
            "expected": case.expected,
            "raw_response": "",
            "normalized_response": "",
            "passed": False,
            "latency_seconds": round(elapsed, 3),
            "prompt_eval_count": None,
            "eval_count": None,
            "eval_duration_ns": None,
            "error": str(e),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description='Test fine-tuned model with questions from JSONL file')
    parser.add_argument(
        '--ollama-url',
        type=str,
        default='http://localhost:11434/api/generate',
        help='Ollama API URL (default: http://localhost:11434/api/generate)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gemma-3-270m-nist',
        help='Model name to test (default: gemma-3-270m-nist)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=120,
        help='Request timeout in seconds (default: 120)'
    )
    parser.add_argument(
        '--questions',
        type=str,
        default=os.path.join('dataset', 'test-nist-rmf.jsonl'),
        help='Path to questions JSONL file (default: dataset/test-nist-rmf.jsonl)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='ollama_inference_test_results.json',
        help='Output file path (default: ollama_inference_test_results.json)'
    )
    
    args = parser.parse_args()
    
    # Load test cases from JSONL file
    test_cases = load_test_cases(args.questions)
    print(f"Loaded {len(test_cases)} test questions from {args.questions}")
    print(f"Testing model: {args.model}")
    print(f"{'='*60}\n")
    
    # Run tests with real-time feedback
    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] Testing {case.id}...", end=" ")
        result = run_case(case, args.ollama_url, args.model, args.timeout, show_feedback=True)
        results.append(result)
    
    passed = sum(1 for r in results if r["passed"])
    print(f"\n{'='*60}")

    by_kind: dict[str, dict] = {}
    for r in results:
        k = r["kind"]
        if k not in by_kind:
            by_kind[k] = {"total": 0, "passed": 0}
        by_kind[k]["total"] += 1
        if r["passed"]:
            by_kind[k]["passed"] += 1
    kind_accuracy = {
        k: round(v["passed"] / v["total"], 3)
        for k, v in by_kind.items()
    }

    latencies = [r["latency_seconds"] for r in results if r["error"] is None]
    avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else None

    summary = {
        "model": args.model,
        "ollama_url": args.ollama_url,
        "questions_file": args.questions,
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "accuracy": round(passed / len(results), 3) if results else 0.0,
        "accuracy_by_kind": kind_accuracy,
        "avg_latency_seconds": avg_latency,
        "results": results,
    }
    
    # Ensure output directory exists
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    print(f"\nTest Results:")
    print(f"Total: {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Accuracy: {summary['accuracy']:.1%}")
    for kind, acc in summary["accuracy_by_kind"].items():
        kind_total = by_kind[kind]["total"]
        kind_passed = by_kind[kind]["passed"]
        print(f"  {kind}: {acc:.1%} ({kind_passed}/{kind_total})")
    if avg_latency is not None:
        print(f"Avg latency: {avg_latency:.3f}s")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
