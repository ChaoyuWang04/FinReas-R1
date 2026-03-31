"""Utilities for two-pass reasoning-chain generation.

These helpers support the optional reasoning-distillation workflow described in
 the project notes:
1. process a first-pass judge model's outputs,
2. isolate incorrect cases,
3. prepare a second-pass correction batch.

The outputs are JSON / JSONL artifacts under ``data/reasoning_chains/`` and are
meant to feed SFT-style distillation rather than the default customer-service
GRPO path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Sequence


def process_first_pass(input_mapping_path: Path, batch_results_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Join first-pass batch outputs with the original sample metadata.

    Returns two lists:
    - ``correct``: samples whose final ``[[A]]`` / ``[[B]]`` tag matched label
    - ``incorrect``: samples that should be sent to second-pass correction
    """
    input_rows = json.loads(input_mapping_path.read_text())
    data_by_id = {
        item["custom_id"]: {
            "context_messages": item["context_messages"],
            "winner": item["winner"],
        }
        for item in input_rows
    }

    results = [json.loads(line) for line in batch_results_path.read_text().splitlines() if line.strip()]
    processed: list[dict[str, Any]] = []
    for item in results:
        sample_id = item["custom_id"]
        response_text = item["result"]["message"]["content"][0]["text"]
        winner = data_by_id[sample_id]["winner"]
        prediction_tail = response_text[-100:]
        is_correct = (
            winner == "model_a" and "[[A]]" in prediction_tail
        ) or (
            winner == "model_b" and "[[B]]" in prediction_tail
        )
        processed.append(
            {
                "custom_id": sample_id,
                "context_messages": data_by_id[sample_id]["context_messages"],
                "winner": winner,
                "sft_response": response_text,
                "correct": is_correct,
            }
        )

    return {
        "correct": [item for item in processed if item["correct"]],
        "incorrect": [item for item in processed if not item["correct"]],
    }


async def run_openai_second_pass(input_json_path: Path, output_jsonl_path: Path, api_key: str, model: str) -> None:
    """Submit a prepared OpenAI batch file and download the completed results."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    batch_file = await client.files.create(file=input_json_path, purpose="batch")
    job = await client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    while job.status not in {"completed", "failed", "cancelled"}:
        await asyncio.sleep(10)
        job = await client.batches.retrieve(job.id)

    if job.status != "completed":
        raise RuntimeError(f"OpenAI batch ended with status {job.status}")

    del model  # The model is embedded in the input file body.
    output_bytes = (await client.files.content(job.output_file_id)).content
    output_jsonl_path.write_bytes(output_bytes)


def build_second_pass_requests(incorrect_samples: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    """Create second-pass batch requests that force the known-correct verdict."""
    suffix_a = (
        "\n\nFor this sample case, the correct verdict is [[A]]. Please produce a high-quality reasoning trace that explains why Chatbot A outperforms Chatbot B."
    )
    suffix_b = (
        "\n\nFor this sample case, the correct verdict is [[B]]. Please produce a high-quality reasoning trace that explains why Chatbot B outperforms Chatbot A."
    )
    tasks = []
    for item in incorrect_samples:
        messages = json.loads(json.dumps(item["context_messages"]))
        suffix = suffix_a if item["winner"] == "model_a" else suffix_b
        messages[-1]["content"] += f"{suffix} The previous incorrect judgement was: {item['sft_response']}"
        tasks.append(
            {
                "custom_id": item["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": messages,
                },
            }
        )
    return tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reasoning-chain generation utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser("process-first-pass")
    process_parser.add_argument("--input-mapping", required=True)
    process_parser.add_argument("--batch-results", required=True)
    process_parser.add_argument("--correct-output", required=True)
    process_parser.add_argument("--incorrect-output", required=True)

    second_pass_parser = subparsers.add_parser("prepare-second-pass")
    second_pass_parser.add_argument("--incorrect-input", required=True)
    second_pass_parser.add_argument("--output-jsonl", required=True)
    second_pass_parser.add_argument("--model", default="o3-2025-04-16")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for first-pass processing and second-pass preparation."""
    args = build_parser().parse_args(argv)
    if args.command == "process-first-pass":
        processed = process_first_pass(Path(args.input_mapping), Path(args.batch_results))
        Path(args.correct_output).write_text(json.dumps(processed["correct"], ensure_ascii=False, indent=2))
        Path(args.incorrect_output).write_text(json.dumps(processed["incorrect"], ensure_ascii=False, indent=2))
        print(f"correct={len(processed['correct'])} incorrect={len(processed['incorrect'])}")
        return 0

    incorrect_samples = json.loads(Path(args.incorrect_input).read_text())
    requests = build_second_pass_requests(incorrect_samples, model=args.model)
    with Path(args.output_jsonl).open("w", encoding="utf-8") as handle:
        for item in requests:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"prepared={len(requests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
