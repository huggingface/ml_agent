from agent.core.hf_access import jobs_access_from_whoami


def test_personal_pro_prefers_username_namespace():
    access = jobs_access_from_whoami({
        "name": "alice",
        "plan": "pro",
        "orgs": [],
    })
    assert access.plan == "pro"
    assert access.eligible_namespaces == ["alice"]
    assert access.default_namespace == "alice"


def test_free_user_with_paid_org_uses_org_namespace():
    access = jobs_access_from_whoami({
        "name": "alice",
        "plan": "free",
        "orgs": [
            {"name": "team-a", "plan": "team"},
            {"name": "oss-friends", "plan": "free"},
        ],
    })
    assert access.plan == "org"
    assert access.personal_can_run_jobs is False
    assert access.eligible_namespaces == ["team-a"]
    assert access.default_namespace is None


def test_free_user_without_paid_org_cannot_run_jobs():
    access = jobs_access_from_whoami({
        "name": "alice",
        "plan": "free",
        "orgs": [{"name": "community", "plan": "free"}],
    })
    assert access.plan == "free"
    assert access.can_run_jobs is False
    assert access.eligible_namespaces == []
    assert access.default_namespace is None
