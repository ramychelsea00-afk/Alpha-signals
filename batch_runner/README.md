# Batch runner

Sends a file of prompts to Claude through the
[Message Batches API](https://platform.claude.com/docs/en/build-with-claude/batch-processing)
and writes one result per prompt. Batches cost 50% of standard API prices and
usually finish within an hour; any request not processed within 24 hours expires.

## Setup

```bash
pip install -r batch_runner/requirements.txt
export ANTHROPIC_API_KEY=...   # or sign in with `ant auth login`
```

## Input

**JSONL**, one object per line. `prompt` (a string) or `messages` (a Messages API
list) is required. `custom_id` is optional; any other Messages API parameter
(`system`, `model`, `max_tokens`, `tools`, `output_config`, ...) is passed
through for that request.

```jsonl
{"custom_id": "greeting", "prompt": "Say hello in three languages."}
{"custom_id": "classify-1", "prompt": "Classify: great product!", "model": "claude-haiku-5-5", "max_tokens": 1024}
```

**CSV** with a `prompt` column, plus optional `id`/`custom_id`, `system`, `model`.

Missing ids become `req-000000`, `req-000001`, ...; ids are cleaned up to match
the API's `^[a-zA-Z0-9_-]{1,64}$` and made unique. `stream`, `speed`,
`fallbacks` and `max_tokens: 0` are rejected up front because batches don't
accept them.

## Usage

```bash
# Submit, wait, and write results in one go
python -m batch_runner run prompts.jsonl --out results.jsonl

# Check the input without sending anything
python -m batch_runner run prompts.jsonl --dry-run

# Or submit now and fetch later
python -m batch_runner submit prompts.jsonl            # writes prompts.jsonl.batch.json
python -m batch_runner status  --state prompts.jsonl.batch.json
python -m batch_runner results --state prompts.jsonl.batch.json [--wait]
python -m batch_runner cancel  --state prompts.jsonl.batch.json
```

Options for `run` / `submit`:

| Option | Default | Meaning |
|---|---|---|
| `--model` | `claude-opus-5-5` | Model for records that don't set one |
| `--max-tokens` | `16000` | `max_tokens` for records that don't set one |
| `--system` / `--system-file` | none | System prompt for records that don't set one |
| `--cache-system` | off | Cache the system prompt with a 1-hour TTL (worth it when it's long and shared) |
| `--effort` | model default | `output_config.effort`: `low`, `medium`, `high`, `xhigh`, `max` |
| `--interval` | `60` | Seconds between status checks (`run` only) |
| `--state` | `<input>.batch.json` | Where batch ids and requests are saved |

Inputs larger than one batch allows (100,000 requests or 256 MB) are split
into several batches automatically; the state file tracks all of them.

## Output

`results.jsonl`, one line per request (results come back in any order, so match
on `custom_id`):

| `status` | Fields |
|---|---|
| `succeeded` | `text`, `stop_reason`, `model`, `usage` (plus `stop_details` on a refusal, `content` on `pause_turn`) |
| `errored` | `error_type`, `error_message`, `retryable` |
| `expired` | `retryable: true` |
| `canceled` | — |

Server errors and expired requests are also written to `<state>.retry.jsonl`
in the same input format, so you can resubmit them with:

```bash
python -m batch_runner run prompts.jsonl.batch.retry.jsonl --state retry.batch.json
```

`invalid_request_error` results are not retried: the request has to be fixed first.

## Notes

- Results can be downloaded for 29 days after a batch is created.
- Batch data is stored server side, so it isn't covered by zero data retention.
- Run the tests with `python -m pytest tests`.
