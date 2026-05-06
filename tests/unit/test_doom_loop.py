"""Tests for the doom-loop detector — repeated/cycling tool call patterns."""

from dataclasses import dataclass

from agent.core.doom_loop import (
    ToolCallSignature,
    _hash_args,
    _normalize_args,
    check_for_doom_loop,
    detect_identical_consecutive,
    detect_repeating_sequence,
    extract_recent_tool_signatures,
)


# ── Lightweight stand-ins so we don't need the litellm message classes ──


@dataclass
class _Fn:
    name: str
    arguments: str


@dataclass
class _ToolCall:
    function: _Fn


@dataclass
class _Msg:
    role: str
    tool_calls: list | None = None


def _assistant_call(name: str, args: str) -> _Msg:
    return _Msg(role="assistant", tool_calls=[_ToolCall(_Fn(name, args))])


# ── _normalize_args / _hash_args ────────────────────────────────────────


def test_normalize_args_collapses_key_order():
    a = '{"path": "/foo", "query": "bar"}'
    b = '{"query": "bar", "path": "/foo"}'
    assert _normalize_args(a) == _normalize_args(b)


def test_normalize_args_collapses_whitespace():
    a = '{"path": "/foo", "query": "bar"}'
    b = '{"path":"/foo","query":"bar"}'
    assert _normalize_args(a) == _normalize_args(b)


def test_normalize_args_preserves_value_difference():
    a = '{"path": "/foo"}'
    b = '{"path": "/bar"}'
    assert _normalize_args(a) != _normalize_args(b)


def test_normalize_args_preserves_nested_structure():
    a = '{"a": {"x": 1, "y": 2}, "b": [3, 4]}'
    b = '{"b": [3, 4], "a": {"y": 2, "x": 1}}'
    assert _normalize_args(a) == _normalize_args(b)


def test_normalize_args_array_order_is_significant():
    # Lists are positional — different orderings should NOT collapse.
    a = '{"items": [1, 2, 3]}'
    b = '{"items": [3, 2, 1]}'
    assert _normalize_args(a) != _normalize_args(b)


def test_normalize_args_falls_back_for_invalid_json():
    # Some providers occasionally pass a bare string; we shouldn't raise.
    assert _normalize_args("not json") == "not json"
    assert _normalize_args("{broken") == "{broken"


def test_normalize_args_handles_empty_string():
    assert _normalize_args("") == ""


def test_hash_args_collapses_semantically_identical_calls():
    # The headline regression: pre-fix these hashed differently and the
    # doom-loop detector silently missed identical-consecutive calls.
    a = '{"path": "/foo", "query": "bar"}'
    b = '{"query": "bar", "path": "/foo"}'
    assert _hash_args(a) == _hash_args(b)


def test_hash_args_still_differs_on_real_argument_change():
    assert _hash_args('{"path": "/a"}') != _hash_args('{"path": "/b"}')


# ── extract_recent_tool_signatures ──────────────────────────────────────


def test_extract_recent_signatures_collapses_reordered_keys():
    """Three calls with reordered keys should produce identical signatures."""
    msgs = [
        _assistant_call("read", '{"path": "/foo", "limit": 100}'),
        _assistant_call("read", '{"limit": 100, "path": "/foo"}'),
        _assistant_call("read", '{"path":"/foo","limit":100}'),
    ]
    sigs = extract_recent_tool_signatures(msgs)
    assert len(sigs) == 3
    assert sigs[0] == sigs[1] == sigs[2]


def test_extract_skips_non_assistant_messages():
    msgs = [
        _Msg(role="user", tool_calls=None),
        _assistant_call("read", '{"path": "/x"}'),
        _Msg(role="tool", tool_calls=None),
    ]
    sigs = extract_recent_tool_signatures(msgs)
    assert len(sigs) == 1
    assert sigs[0].name == "read"


def test_extract_skips_assistant_without_tool_calls():
    msgs = [_Msg(role="assistant", tool_calls=None)]
    assert extract_recent_tool_signatures(msgs) == []


# ── detect_identical_consecutive ────────────────────────────────────────


def _sig(name: str, args: str = "{}") -> ToolCallSignature:
    return ToolCallSignature(name=name, args_hash=_hash_args(args))


def test_identical_consecutive_fires_at_threshold():
    sigs = [_sig("read", '{"p": 1}')] * 3
    assert detect_identical_consecutive(sigs, threshold=3) == "read"


def test_identical_consecutive_stays_silent_below_threshold():
    sigs = [_sig("read", '{"p": 1}')] * 2
    assert detect_identical_consecutive(sigs, threshold=3) is None


def test_identical_consecutive_resets_on_break():
    # A, A, B, A, A — never 3 in a row.
    sigs = [
        _sig("read", '{"p": 1}'),
        _sig("read", '{"p": 1}'),
        _sig("read", '{"p": 2}'),
        _sig("read", '{"p": 1}'),
        _sig("read", '{"p": 1}'),
    ]
    assert detect_identical_consecutive(sigs, threshold=3) is None


def test_identical_consecutive_catches_reordered_args_after_normalization():
    """Regression for the bug: same call with shuffled keys must collapse."""
    msgs = [
        _assistant_call("research", '{"task": "find paper", "depth": 3}'),
        _assistant_call("research", '{"depth": 3, "task": "find paper"}'),
        _assistant_call("research", '{"task":"find paper","depth":3}'),
    ]
    sigs = extract_recent_tool_signatures(msgs)
    assert detect_identical_consecutive(sigs, threshold=3) == "research"


# ── detect_repeating_sequence ───────────────────────────────────────────


def test_repeating_sequence_catches_alternating_pair():
    sigs = [_sig("a"), _sig("b")] * 3
    pattern = detect_repeating_sequence(sigs)
    assert pattern is not None
    assert [s.name for s in pattern] == ["a", "b"]


def test_repeating_sequence_misses_when_pattern_breaks():
    sigs = [_sig("a"), _sig("b"), _sig("a"), _sig("c")]
    assert detect_repeating_sequence(sigs) is None


def test_repeating_sequence_normalizes_args_inside_pattern():
    """Cycle [research, read, research, read, ...] survives key reordering."""
    msgs = [
        _assistant_call("research", '{"q": "x", "n": 1}'),
        _assistant_call("read", '{"path": "/a"}'),
        _assistant_call("research", '{"n": 1, "q": "x"}'),
        _assistant_call("read", '{"path":"/a"}'),
        _assistant_call("research", '{"q":"x","n":1}'),
        _assistant_call("read", '{"path": "/a"}'),
    ]
    sigs = extract_recent_tool_signatures(msgs)
    pattern = detect_repeating_sequence(sigs)
    assert pattern is not None
    assert [s.name for s in pattern] == ["research", "read"]


# ── check_for_doom_loop ─────────────────────────────────────────────────


def test_check_for_doom_loop_quiet_below_minimum_signatures():
    msgs = [_assistant_call("read", '{"p": 1}'), _assistant_call("read", '{"p": 1}')]
    assert check_for_doom_loop(msgs) is None


def test_check_for_doom_loop_returns_corrective_prompt_for_identical_run():
    msgs = [_assistant_call("read", '{"p": 1}')] * 3
    out = check_for_doom_loop(msgs)
    assert out is not None
    assert "REPETITION GUARD" in out
    assert "'read'" in out


def test_check_for_doom_loop_returns_corrective_prompt_for_cycle():
    msgs = []
    for _ in range(3):
        msgs.append(_assistant_call("a", "{}"))
        msgs.append(_assistant_call("b", "{}"))
    out = check_for_doom_loop(msgs)
    assert out is not None
    assert "REPETITION GUARD" in out
    assert "a → b" in out


def test_check_for_doom_loop_quiet_when_args_meaningfully_differ():
    """Same tool, three different arg values — not a loop."""
    msgs = [
        _assistant_call("read", '{"path": "/a.py"}'),
        _assistant_call("read", '{"path": "/b.py"}'),
        _assistant_call("read", '{"path": "/c.py"}'),
    ]
    assert check_for_doom_loop(msgs) is None
