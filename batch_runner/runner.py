"""Submit prompts to the Claude Message Batches API and collect the results.

Batches run asynchronously at 50% of standard API prices. Most finish within an
hour; any request not processed within 24 hours expires.

Typical use:

    python -m batch_runner run prompts.jsonl --out results.jsonl

See README.md in this directory for the input format and every subcommand.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_MAX_TOKENS = 16000

# API limits: 100,000 requests or 256 MB per batch. Stay a little under the
# byte limit because the size we measure is our own JSON encoding.
MAX_REQUESTS_PER_BATCH = 100_000
MAX_BATCH_BYTES = 250 * 1024 * 1024

CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# Parameters the Batches API rejects inside a request.
UNSUPPORTED_PARAMS = {"stream", "speed", "fallbacks"}

# Fields of an input record that are not Messages API params.
RECORD_META_FIELDS = {"custom_id", "id", "prompt"}


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #


@dataclass
class Defaults:
    """Request settings applied to every input record that doesn't set its own."""

    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    system: str | None = None
    effort: str | None = None
    cache_system: bool = False
    cache_ttl: str = "1h"


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read input records from a .jsonl or .csv file.

    JSONL: one object per line. Either `prompt` (a string) or `messages` (a
    Messages API list) is required; any other Messages API param (`system`,
    `model`, `max_tokens`, `tools`, `output_config`, ...) is passed through.
    CSV: needs a `prompt` column; `custom_id`/`id`, `system` and `model` columns
    are optional.
    """
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            return [{k: v for k, v in row.items() if v not in (None, "")} for row in csv.DictReader(f)]

    records = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {e}") from e
            if not isinstance(record, dict):
                raise SystemExit(f"{path}:{lineno}: each line must be a JSON object")
            records.append(record)
    return records


def make_custom_id(raw: Any, index: int, seen: set[str]) -> str:
    """Return a unique custom_id matching ^[a-zA-Z0-9_-]{1,64}$."""
    if raw is None or str(raw) == "":
        cid = f"req-{index:06d}"
    else:
        cid = re.sub(r"[^a-zA-Z0-9_-]", "_", str(raw))[:64]
    base, n = cid, 1
    while cid in seen:
        suffix = f"-{n}"
        cid = base[: 64 - len(suffix)] + suffix
        n += 1
    seen.add(cid)
    return cid


def build_params(record: dict[str, Any], defaults: Defaults) -> dict[str, Any]:
    """Turn one input record into Messages API params for a batch request."""
    params = {k: v for k, v in record.items() if k not in RECORD_META_FIELDS}

    if "messages" not in params:
        if "prompt" not in record:
            raise ValueError("record needs either 'prompt' or 'messages'")
        params["messages"] = [{"role": "user", "content": str(record["prompt"])}]

    bad = UNSUPPORTED_PARAMS & params.keys()
    if bad:
        raise ValueError(f"not supported in batches: {', '.join(sorted(bad))}")

    params.setdefault("model", defaults.model)
    params["max_tokens"] = int(params.get("max_tokens", defaults.max_tokens))
    if params["max_tokens"] < 1:
        raise ValueError("max_tokens must be at least 1 in a batch")

    if "system" not in params and defaults.system:
        params["system"] = defaults.system
    if defaults.cache_system and isinstance(params.get("system"), str):
        # Batches can take longer than the default 5-minute cache lifetime, so
        # the docs recommend the 1-hour TTL for shared context.
        params["system"] = [
            {
                "type": "text",
                "text": params["system"],
                "cache_control": {"type": "ephemeral", "ttl": defaults.cache_ttl},
            }
        ]

    if defaults.effort:
        output_config = dict(params.get("output_config") or {})
        output_config.setdefault("effort", defaults.effort)
        params["output_config"] = output_config

    return params


def build_requests(records: Iterable[dict[str, Any]], defaults: Defaults) -> list[Request]:
    requests: list[Request] = []
    seen: set[str] = set()
    errors: list[str] = []
    for i, record in enumerate(records):
        cid = make_custom_id(record.get("custom_id", record.get("id")), i, seen)
        try:
            params = build_params(record, defaults)
        except ValueError as e:
            errors.append(f"record {i} ({cid}): {e}")
            continue
        requests.append(Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**params)))
    if errors:
        raise SystemExit("Invalid input:\n  " + "\n  ".join(errors))
    return requests


def chunk_requests(requests: list[Request]) -> Iterator[list[Request]]:
    """Split requests into batches that respect the count and size limits."""
    chunk: list[Request] = []
    size = 0
    for req in requests:
        req_size = len(json.dumps(req, ensure_ascii=False).encode("utf-8")) + 2
        if req_size > MAX_BATCH_BYTES:
            raise SystemExit(f"request {req['custom_id']} alone exceeds the batch size limit")
        if chunk and (len(chunk) >= MAX_REQUESTS_PER_BATCH or size + req_size > MAX_BATCH_BYTES):
            yield chunk
            chunk, size = [], 0
        chunk.append(req)
        size += req_size
    if chunk:
        yield chunk


# --------------------------------------------------------------------------- #
# State file: remembers batch ids and the original requests (for retries)
# --------------------------------------------------------------------------- #


@dataclass
class State:
    path: Path
    batch_ids: list[str] = field(default_factory=list)
    requests: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "State":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(path=path, batch_ids=data["batch_ids"], requests=data.get("requests", {}))

    def save(self) -> None:
        self.path.write_text(
            json.dumps({"batch_ids": self.batch_ids, "requests": self.requests}, ensure_ascii=False),
            encoding="utf-8",
        )


# --------------------------------------------------------------------------- #
# API operations
# --------------------------------------------------------------------------- #


def submit(client: anthropic.Anthropic, requests: list[Request], state: State) -> list[str]:
    for req in requests:
        state.requests[req["custom_id"]] = dict(req["params"])
    new_ids = []
    for chunk in chunk_requests(requests):
        batch = client.messages.batches.create(requests=chunk)
        new_ids.append(batch.id)
        state.batch_ids.append(batch.id)
        state.save()  # save after every batch so a crash doesn't lose ids
        log(f"Created {batch.id} with {len(chunk)} requests (expires {batch.expires_at})")
    return new_ids


def wait(client: anthropic.Anthropic, batch_ids: list[str], interval: float) -> None:
    pending = list(batch_ids)
    while pending:
        still = []
        for bid in pending:
            batch = client.messages.batches.retrieve(bid)
            c = batch.request_counts
            log(
                f"{bid}: {batch.processing_status} | processing={c.processing} "
                f"succeeded={c.succeeded} errored={c.errored} "
                f"canceled={c.canceled} expired={c.expired}"
            )
            if batch.processing_status != "ended":
                still.append(bid)
        pending = still
        if pending:
            time.sleep(interval)


def classify(result: Any) -> dict[str, Any]:
    """Flatten one batch result into a JSON-serializable row."""
    outcome = result.result
    row: dict[str, Any] = {"custom_id": result.custom_id, "status": outcome.type}
    if outcome.type == "succeeded":
        msg = outcome.message
        row["text"] = "".join(b.text for b in msg.content if b.type == "text")
        row["stop_reason"] = msg.stop_reason
        row["model"] = msg.model
        row["usage"] = msg.usage.model_dump(exclude_none=True)
        if msg.stop_reason == "refusal" and getattr(msg, "stop_details", None):
            row["stop_details"] = msg.stop_details.model_dump(exclude_none=True)
        if msg.stop_reason == "pause_turn":
            # Server-tool loop didn't finish; keep the content so it can be continued.
            row["content"] = [b.model_dump(exclude_none=True) for b in msg.content]
    elif outcome.type == "errored":
        err = outcome.error.error
        row["error_type"] = err.type
        row["error_message"] = err.message
        # invalid_request_error needs the request fixed; anything else can be retried as is.
        row["retryable"] = err.type != "invalid_request_error"
    elif outcome.type == "expired":
        row["retryable"] = True
    return row


def collect(
    client: anthropic.Anthropic,
    state: State,
    out_path: Path,
    retry_path: Path | None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    retry_rows = []
    with out_path.open("w", encoding="utf-8") as out:
        for bid in state.batch_ids:
            for result in client.messages.batches.results(bid):
                row = classify(result)
                row["batch_id"] = bid
                counts[row["status"]] = counts.get(row["status"], 0) + 1
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                if row.get("retryable") and row["custom_id"] in state.requests:
                    retry_rows.append({"custom_id": row["custom_id"], **state.requests[row["custom_id"]]})

    if retry_path is not None:
        if retry_rows:
            with retry_path.open("w", encoding="utf-8") as f:
                for r in retry_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            log(f"Wrote {len(retry_rows)} retryable requests to {retry_path}")
        elif retry_path.exists():
            retry_path.unlink()
    return counts


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def add_request_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", type=Path, help="prompts file (.jsonl or .csv)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"default model (default: {DEFAULT_MODEL})")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--system", help="system prompt text for every request")
    p.add_argument("--system-file", type=Path, help="read the system prompt from a file")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument(
        "--cache-system",
        action="store_true",
        help="mark the system prompt for prompt caching (1-hour TTL)",
    )
    p.add_argument("--state", type=Path, help="state file (default: <input>.batch.json)")
    p.add_argument("--dry-run", action="store_true", help="validate and print a summary, don't submit")


def add_result_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", type=Path, help="results file (default: <state>.results.jsonl)")
    p.add_argument("--retry-file", type=Path, help="where to write retryable requests (default: <state>.retry.jsonl)")


def state_path_for(args: argparse.Namespace) -> Path:
    if args.state:
        return args.state
    return args.input.with_suffix(args.input.suffix + ".batch.json")


def defaults_from(args: argparse.Namespace) -> Defaults:
    system = args.system
    if args.system_file:
        system = args.system_file.read_text(encoding="utf-8")
    return Defaults(
        model=args.model,
        max_tokens=args.max_tokens,
        system=system,
        effort=args.effort,
        cache_system=args.cache_system,
    )


def result_paths(args: argparse.Namespace, state_path: Path) -> tuple[Path, Path]:
    stem = state_path.with_suffix("")
    out = args.out or stem.with_suffix(".results.jsonl")
    retry = args.retry_file or stem.with_suffix(".retry.jsonl")
    return out, retry


def cmd_submit(args: argparse.Namespace, client: anthropic.Anthropic | None, *, then_collect: bool) -> int:
    requests = build_requests(read_records(args.input), defaults_from(args))
    if not requests:
        raise SystemExit("No requests in input file")
    chunks = list(chunk_requests(requests))
    log(f"{len(requests)} requests -> {len(chunks)} batch(es)")
    if args.dry_run:
        print(json.dumps(requests[0], indent=2, ensure_ascii=False))
        return 0

    state_path = state_path_for(args)
    if state_path.exists():
        raise SystemExit(
            f"{state_path} already exists (earlier submission). Use `status`/`results` "
            "with --state, or pass a different --state."
        )
    assert client is not None
    state = State(path=state_path)
    ids = submit(client, requests, state)
    log(f"State saved to {state_path}")
    if not then_collect:
        print("\n".join(ids))
        return 0

    wait(client, ids, args.interval)
    out, retry = result_paths(args, state_path)
    counts = collect(client, state, out, retry)
    log(f"Results written to {out}: {counts}")
    return 0


def cmd_status(args: argparse.Namespace, client: anthropic.Anthropic) -> int:
    state = State.load(args.state)
    for bid in state.batch_ids:
        print(client.messages.batches.retrieve(bid).model_dump_json(indent=2))
    return 0


def cmd_results(args: argparse.Namespace, client: anthropic.Anthropic) -> int:
    state = State.load(args.state)
    if args.wait:
        wait(client, state.batch_ids, args.interval)
    for bid in state.batch_ids:
        status = client.messages.batches.retrieve(bid).processing_status
        if status != "ended":
            raise SystemExit(f"{bid} is still {status}; rerun later or pass --wait")
    out, retry = result_paths(args, args.state)
    counts = collect(client, state, out, retry)
    log(f"Results written to {out}: {counts}")
    return 0


def cmd_cancel(args: argparse.Namespace, client: anthropic.Anthropic) -> int:
    state = State.load(args.state)
    for bid in state.batch_ids:
        batch = client.messages.batches.cancel(bid)
        log(f"{bid}: {batch.processing_status}")
    return 0


def main(argv: list[str] | None = None, client: anthropic.Anthropic | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_runner", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="submit, wait for completion, and write results")
    add_request_options(p_run)
    add_result_options(p_run)
    p_run.add_argument("--interval", type=float, default=60, help="seconds between status checks")

    p_submit = sub.add_parser("submit", help="submit and exit; prints batch ids")
    add_request_options(p_submit)

    p_status = sub.add_parser("status", help="show the batch(es) in a state file")
    p_status.add_argument("--state", type=Path, required=True)

    p_results = sub.add_parser("results", help="download results for a state file")
    p_results.add_argument("--state", type=Path, required=True)
    p_results.add_argument("--wait", action="store_true", help="wait until processing ends first")
    p_results.add_argument("--interval", type=float, default=60)
    add_result_options(p_results)

    p_cancel = sub.add_parser("cancel", help="cancel the batch(es) in a state file")
    p_cancel.add_argument("--state", type=Path, required=True)

    args = parser.parse_args(argv)

    if client is None and not getattr(args, "dry_run", False):
        client = anthropic.Anthropic()

    if args.command == "run":
        return cmd_submit(args, client, then_collect=True)
    if args.command == "submit":
        return cmd_submit(args, client, then_collect=False)
    if args.command == "status":
        return cmd_status(args, client)
    if args.command == "results":
        return cmd_results(args, client)
    if args.command == "cancel":
        return cmd_cancel(args, client)
    parser.error(f"unknown command {args.command}")
    return 2
