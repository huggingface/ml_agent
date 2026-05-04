from agent.core.hf_access import is_billing_error, jobs_access_from_whoami


def test_personal_user_lists_username_namespace():
    access = jobs_access_from_whoami(
        {
            "name": "alice",
            "orgs": [],
        }
    )
    assert access.username == "alice"
    assert access.org_names == []
    assert access.eligible_namespaces == ["alice"]
    assert access.default_namespace == "alice"


def test_user_with_orgs_lists_all_namespaces_regardless_of_plan():
    # Plan/tier is ignored — credits live on the namespace itself, so any
    # org the user belongs to is eligible.  We sort orgs alphabetically and
    # always put the personal namespace first so the picker default is the
    # user's own account.
    access = jobs_access_from_whoami(
        {
            "name": "alice",
            "orgs": [
                {"name": "team-a", "plan": "team"},
                {"name": "oss-friends", "plan": "free"},
            ],
        }
    )
    assert access.username == "alice"
    assert access.org_names == ["oss-friends", "team-a"]
    assert access.eligible_namespaces == ["alice", "oss-friends", "team-a"]
    assert access.default_namespace == "alice"


def test_free_user_without_org_still_eligible_under_personal_namespace():
    # Pro is no longer required — the user is offered their personal
    # namespace; whether they actually have credits is decided at job
    # creation time when HF returns a 402 / billing error.
    access = jobs_access_from_whoami(
        {
            "name": "alice",
            "orgs": [],
        }
    )
    assert access.eligible_namespaces == ["alice"]
    assert access.default_namespace == "alice"


def test_org_only_token_falls_back_to_first_org():
    access = jobs_access_from_whoami(
        {
            "name": None,
            "orgs": [{"name": "team-a"}, {"name": "team-b"}],
        }
    )
    assert access.username is None
    assert access.eligible_namespaces == ["team-a", "team-b"]
    assert access.default_namespace == "team-a"


def test_is_billing_error_detects_402_and_credit_phrasing():
    assert is_billing_error("402 Payment Required")
    assert is_billing_error("Insufficient credits on namespace foo")
    assert is_billing_error("This namespace requires credits to run jobs")
    assert is_billing_error("Out of credit, please add billing")
    assert not is_billing_error("Internal server error")
    assert not is_billing_error("")
