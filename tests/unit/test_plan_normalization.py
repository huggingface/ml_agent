"""Tests for Hugging Face plan normalization."""

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import dependencies  # noqa: E402


def test_oauth_is_pro_flag_takes_priority_over_user_type():
    assert dependencies._normalize_user_plan({"type": "user", "isPro": True}) == "pro"


@pytest.mark.parametrize(
    "payload",
    [
        {"is_pro": True},
        {"accountType": "pro"},
        {"plan": "HF Pro"},
        {"subscription": "hf_pro"},
        {"accountType": "team"},
        {"plan": "enterprise"},
        {"tier": "promotional"},
    ],
)
def test_non_ispro_signals_stay_free(payload):
    assert dependencies._normalize_user_plan(payload) == "free"


def test_free_user_with_free_org_stays_free():
    whoami = {
        "name": "alice",
        "type": "user",
        "orgs": [{"name": "oss-friends", "plan": "free"}],
    }

    assert dependencies._normalize_user_plan(whoami) == "free"


def test_user_with_paid_org_without_personal_pro_stays_free():
    whoami = {
        "name": "alice",
        "type": "user",
        "orgs": [{"name": "team-a", "plan": "team"}],
    }

    assert dependencies._normalize_user_plan(whoami) == "free"


@pytest.mark.parametrize("payload", [None, [], {"type": "user"}, {"plan": "free"}])
def test_unknown_or_malformed_payload_defaults_to_free(payload):
    assert dependencies._normalize_user_plan(payload) == "free"
