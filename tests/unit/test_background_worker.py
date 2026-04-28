"""Tests for durable background worker helpers."""

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from background_worker import operation_from_run  # noqa: E402
from agent.core.session import OpType  # noqa: E402


def test_operation_from_user_input_run():
    operation = operation_from_run(
        {
            "operation": {
                "type": "user_input",
                "payload": {"text": "build a demo"},
            }
        }
    )

    assert operation.op_type == OpType.USER_INPUT
    assert operation.data == {"text": "build a demo"}


def test_operation_from_approval_run():
    approvals = [{"tool_call_id": "call_1", "approved": True}]

    operation = operation_from_run(
        {
            "operation": {
                "type": "exec_approval",
                "payload": {"approvals": approvals},
            }
        }
    )

    assert operation.op_type == OpType.EXEC_APPROVAL
    assert operation.data == {"approvals": approvals}


def test_operation_from_unknown_run_rejects_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported background run operation"):
        operation_from_run({"operation": {"type": "truncate", "payload": {}}})
