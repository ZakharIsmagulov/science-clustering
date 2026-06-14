import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TypeVar
from openai import OpenAI
from pydantic import BaseModel, ValidationError


ModelT = TypeVar("ModelT", bound=BaseModel)

DEFAULT_MODEL = "gpt-5-mini"
CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}


@dataclass(frozen=True)
class BatchParseStats:
    total: int
    succeeded: int
    failed: int


class BatchNotReadyError(RuntimeError):
    pass


class BatchFailedError(RuntimeError):
    pass


def get_chatgpt_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


def _json_schema_name(response_model: type[BaseModel]) -> str:
    return response_model.__name__.replace(" ", "_")


def _make_schema_openai_strict(schema: dict[str, Any]) -> dict[str, Any]:
    schema = dict(schema)

    def visit(node: Any) -> Any:
        if isinstance(node, dict):
            node = {k: visit(v) for k, v in node.items()}
            node.pop("default", None)
            node_type = node.get("type")
            has_properties = isinstance(node.get("properties"), dict)

            if node_type == "object" or has_properties:
                node["additionalProperties"] = False

                if has_properties:
                    node["required"] = list(node["properties"].keys())

            return node

        if isinstance(node, list):
            return [visit(item) for item in node]

        return node

    return visit(schema)


def build_response_format(
    response_model: type[BaseModel],
    *,
    name: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    schema = response_model.model_json_schema()

    if strict:
        schema = _make_schema_openai_strict(schema)

    return {
        "type": "json_schema",
        "json_schema": {
            "name": name or _json_schema_name(response_model),
            "schema": schema,
            "strict": strict,
        },
    }


def make_batch_line(
    system_prompt: str,
    user_prompt: str,
    response_model: type[BaseModel],
    custom_id: str,
    model: str = DEFAULT_MODEL,
    schema_name: str | None = None,
    strict_schema: bool = True,
) -> dict[str, Any]:
    body = {
        "model": model,
        "response_format": build_response_format(
            response_model,
            name=schema_name,
            strict=strict_schema,
        ),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": CHAT_COMPLETIONS_ENDPOINT,
        "body": body,
    }


def write_jsonl(rows: Iterable[dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def push_batch_to_gpt(input_path: str) -> str:
    client = get_chatgpt_client()
    input_path = Path(input_path)

    with input_path.open("rb") as f:
        batch_file = client.files.create(
            file=f,
            purpose="batch",
        )

    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint=CHAT_COMPLETIONS_ENDPOINT,
        completion_window="24h",
    )

    return batch.id


def save_batch_output(
    batch_id: str,
    output_path: str = "res/llm/batch_output.jsonl",
    error_output_path: str = "res/llm/failed_from_gpt.jsonl",
) -> bool:
    """
    Returns False while the batch is still running.
    Saves output and error files when available.
    Raises BatchFailedError for terminal non-completed batches.
    """
    client = get_chatgpt_client()
    batch = client.batches.retrieve(batch_id)

    if batch.status not in TERMINAL_BATCH_STATUSES:
        return False

    if batch.output_file_id is not None:
        content = client.files.content(batch.output_file_id).content

        with Path(output_path).open("wb") as f:
            f.write(content)

    if batch.error_file_id is not None:
        content = client.files.content(batch.error_file_id).content

        with Path(error_output_path).open("wb") as f:
            f.write(content)

    if batch.status != "completed":
        raise BatchFailedError(
            f"Batch ended with status={batch.status}; "
            f"errors={getattr(batch, 'errors', None)}"
        )

    if batch.output_file_id is None:
        raise BatchFailedError("Batch completed, but output_file_id is missing")

    return True


def wait_for_batch_output(
    batch_id: str,
    output_path: str = "res/llm/batch_output.jsonl",
    error_output_path: str = "res/llm/failed_from_gpt.jsonl",
    poll_interval_seconds: int = 5 * 60,
    max_attempts: int = 288,
    logger_name: str = "logger"
):
    logger = logging.getLogger(logger_name)

    for attempt in range(max_attempts):
        if save_batch_output(
            batch_id,
            output_path,
            error_output_path=error_output_path,
        ):
            return

        logger.info(f"Batch {batch_id} is not ready yet; attempt={attempt}")
        time.sleep(poll_interval_seconds)

    raise BatchNotReadyError(
        f"Batch polling timeout: batch_id={batch_id}, attempts={max_attempts}"
    )


def _extract_message_content(row: dict[str, Any]) -> str:
    return row["response"]["body"]["choices"][0]["message"]["content"]


def parse_batch_results(
    results_path: str,
    success_path: str,
    failed_path: str,
    response_model: type[ModelT],
) -> BatchParseStats:
    success: dict[str, ModelT] = {}
    failed: dict[str, dict[str, Any]] = {}
    total = 0

    with Path(results_path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            total += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                failed[f"line-{line_number}"] = {
                    "error": "Invalid JSONL row",
                    "details": str(e),
                }
                continue

            cid = row.get("custom_id") or f"line-{line_number}"
            resp = row.get("response") or {}
            status_code = resp.get("status_code", 0)

            if status_code != 200:
                failed[cid] = {
                    "error": f"HTTP {status_code}",
                    "details": (resp.get("body") or {}).get("error") or row.get("error"),
                }
                continue

            try:
                content = _extract_message_content(row)
                success[cid] = response_model.model_validate_json(content)

            except ValidationError as e:
                failed[cid] = {
                    "error": "Pydantic validation failed",
                    "details": e.errors(),
                }

            except Exception as e:
                failed[cid] = {
                    "error": "Malformed successful response",
                    "details": str(e),
                }

    with Path(success_path).open("w", encoding="utf-8") as f:
        for cid, obj in success.items():
            f.write(
                json.dumps(
                    {
                        "custom_id": cid,
                        "data": obj.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with Path(failed_path).open("w", encoding="utf-8") as f:
        for cid, err in failed.items():
            f.write(
                json.dumps(
                    {
                        "custom_id": cid,
                        **err,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return BatchParseStats(
        total=total,
        succeeded=len(success),
        failed=len(failed),
    )


def process_batch(
    response_model: type[ModelT],
    input_file: str = "res/llm/requests.jsonl",
    results_path: str = "res/llm/results.jsonl",
    success_path: str = "res/llm/success.jsonl",
    failed_path: str = "res/llm/failed.jsonl",
    error_output_path: str = "res/llm/failed_from_gpt.jsonl",
    poll_interval_seconds: int = 5 * 60,
    max_attempts: int = 288,
    logger_name: str = "logger"
) -> BatchParseStats:
    logger = logging.getLogger(logger_name)

    batch_id = push_batch_to_gpt(input_file)
    logger.info(f"Created batch: {batch_id}")

    wait_for_batch_output(
        batch_id,
        results_path,
        error_output_path=error_output_path,
        poll_interval_seconds=poll_interval_seconds,
        max_attempts=max_attempts,
    )

    stats = parse_batch_results(
        results_path=results_path,
        success_path=success_path,
        failed_path=failed_path,
        response_model=response_model,
    )

    logger.info(f"Batch parsed: total={stats.total}, succeeded={stats.succeeded}, failed={stats.failed}")

    return stats
