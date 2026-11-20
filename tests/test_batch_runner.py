import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from anthropic.types.messages import MessageBatchIndividualResponse

from batch_runner import runner


def make_result(custom_id, kind, text="ok", error_type="api_error"):
    if kind == "succeeded":
        result = {
            "type": "succeeded",
            "message": {
                "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 5},
            },
        }
    elif kind == "errored":
        result = {"type": "errored", "error": {"type": "error", "error": {"type": error_type, "message": "boom"}}}
    else:
        result = {"type": kind}
    return MessageBatchIndividualResponse.model_validate({"custom_id": custom_id, "result": result})


class FakeBatches:
    def __init__(self, outcomes):
        self.outcomes = outcomes  # custom_id -> (kind, extra)
        self.created = []
        self.polls = 0

    def _batch(self, bid, status):
        counts = SimpleNamespace(processing=0, succeeded=0, errored=0, canceled=0, expired=0)
        return SimpleNamespace(id=bid, processing_status=status, request_counts=counts, expires_at="later")

    def create(self, requests):
        self.created.append(list(requests))
        return self._batch(f"msgbatch_{len(self.created)}", "in_progress")

    def retrieve(self, bid):
        self.polls += 1
        return self._batch(bid, "ended" if self.polls > 1 else "in_progress")

    def results(self, bid):
        idx = int(bid.split("_")[1]) - 1
        for req in reversed(self.created[idx]):  # out of order on purpose
            kind, extra = self.outcomes.get(req["custom_id"], ("succeeded", {}))
            yield make_result(req["custom_id"], kind, **extra)


def fake_client(outcomes=None):
    batches = FakeBatches(outcomes or {})
    return SimpleNamespace(messages=SimpleNamespace(batches=batches)), batches


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def test_run_end_to_end_with_retries(tmp_path):
    inp = tmp_path / "in.jsonl"
    write_jsonl(inp, [
        {"custom_id": "a", "prompt": "hi"},
        {"custom_id": "b", "prompt": "server error"},
        {"custom_id": "c", "prompt": "bad request"},
        {"custom_id": "d", "prompt": "too slow"},
    ])
    client, batches = fake_client({
        "a": ("succeeded", {"text": "hello"}),
        "b": ("errored", {"error_type": "overloaded_error"}),
        "c": ("errored", {"error_type": "invalid_request_error"}),
        "d": ("expired", {}),
    })
    out = tmp_path / "out.jsonl"
    retry = tmp_path / "retry.jsonl"
    rc = runner.main(["run", str(inp), "--out", str(out), "--retry-file", str(retry), "--interval", "0"], client=client)
    assert rc == 0

    rows = {r["custom_id"]: r for r in read_jsonl(out)}
    assert rows["a"]["text"] == "hello" and rows["a"]["usage"]["output_tokens"] == 5
    assert rows["b"]["retryable"] is True
    assert rows["c"]["retryable"] is False and rows["c"]["error_type"] == "invalid_request_error"
    assert rows["d"]["status"] == "expired"

    retried = {r["custom_id"]: r for r in read_jsonl(retry)}
    assert set(retried) == {"b", "d"}
    assert retried["b"]["messages"][0]["content"] == "server error"
    # retry file is valid input for another run
    assert len(runner.build_requests(runner.read_records(retry), runner.Defaults())) == 2


def test_submit_then_results(tmp_path):
    inp = tmp_path / "in.csv"
    inp.write_text("id,prompt\nq1,one\nq2,two\n")
    client, _ = fake_client()
    runner.main(["submit", str(inp)], client=client)
    state = tmp_path / "in.csv.batch.json"
    assert json.loads(state.read_text())["batch_ids"] == ["msgbatch_1"]
    with pytest.raises(SystemExit):  # refuses to overwrite earlier submission
        runner.main(["submit", str(inp)], client=client)
    runner.main(["results", "--state", str(state), "--wait", "--interval", "0"], client=client)
    assert {r["custom_id"] for r in read_jsonl(tmp_path / "in.csv.results.jsonl")} == {"q1", "q2"}


def test_custom_ids_sanitized_and_unique():
    seen = set()
    assert runner.make_custom_id("a b/c", 0, seen) == "a_b_c"
    assert runner.make_custom_id("a b/c", 1, seen) == "a_b_c-1"
    assert runner.make_custom_id(None, 7, seen) == "req-000007"
    long_id = runner.make_custom_id("x" * 100, 2, seen)
    assert runner.CUSTOM_ID_RE.match(long_id)


@pytest.mark.parametrize("record", [
    {"prompt": "x", "stream": True},
    {"prompt": "x", "max_tokens": 0},
    {"system": "no prompt"},
])
def test_invalid_records_rejected(record):
    with pytest.raises(SystemExit):
        runner.build_requests([record], runner.Defaults())


def test_chunking_by_count(monkeypatch):
    monkeypatch.setattr(runner, "MAX_REQUESTS_PER_BATCH", 2)
    reqs = runner.build_requests([{"prompt": str(i)} for i in range(5)], runner.Defaults())
    assert [len(c) for c in runner.chunk_requests(reqs)] == [2, 2, 1]


def test_effort_and_cache_defaults():
    params = runner.build_params(
        {"prompt": "x", "output_config": {"effort": "high"}},
        runner.Defaults(system="sys", effort="low", cache_system=True),
    )
    assert params["output_config"]["effort"] == "high"  # record wins
    assert params["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
