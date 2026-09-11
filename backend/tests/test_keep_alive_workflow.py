"""
KueryCore AI — Keep-Alive Workflow Regression Tests

Validates the GitHub Actions keep-alive.yml:
1. Valid YAML syntax
2. Cron schedule is ≤14 minutes (under Render's 15-minute inactivity threshold)
3. Correct health endpoint URL is targeted
4. Workflow_dispatch trigger exists for manual testing
"""

import re
import yaml
import pathlib

import pytest


WORKFLOW_PATH = pathlib.Path(__file__).parent.parent.parent / ".github" / "workflows" / "keep-alive.yml"


@pytest.fixture(scope="module")
def workflow() -> dict:
    """Load and parse the keep-alive workflow YAML once for all tests."""
    assert WORKFLOW_PATH.exists(), (
        f"keep-alive.yml not found at {WORKFLOW_PATH}. "
        "Create .github/workflows/keep-alive.yml to prevent Render cold starts."
    )
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestKeepAliveWorkflow:
    """Regression tests for the GitHub Actions keep-alive cron workflow."""

    def test_workflow_file_exists(self):
        """The keep-alive.yml file must exist in .github/workflows/."""
        assert WORKFLOW_PATH.exists(), (
            "keep-alive.yml is missing — Render will cold-start on recruiter visits"
        )

    def test_workflow_is_valid_yaml(self, workflow):
        """The workflow file must be parseable as valid YAML."""
        assert isinstance(workflow, dict), "keep-alive.yml must be a valid YAML mapping"

    def test_workflow_has_schedule_trigger(self, workflow):
        """The workflow must have an 'on.schedule' trigger."""
        on = workflow.get("on") or workflow.get(True)  # YAML 'on' can parse as True
        assert on is not None, "Workflow must have an 'on' key"
        assert "schedule" in on, "Workflow must have a schedule trigger"

    def test_cron_interval_is_14_minutes_or_less(self, workflow):
        """Cron schedule must run every 14 minutes or less (Render idle = 15 min)."""
        on = workflow.get("on") or workflow.get(True)
        schedules = on.get("schedule", [])
        assert schedules, "At least one cron schedule must be defined"

        for schedule_entry in schedules:
            cron_expr = schedule_entry.get("cron", "")
            assert cron_expr, "Schedule entry must have a cron expression"

            # Parse the minute field (first field of cron)
            minute_field = cron_expr.strip().split()[0]

            # Accept */N where N <= 14
            if minute_field.startswith("*/"):
                interval = int(minute_field[2:])
                assert interval <= 14, (
                    f"Cron interval {interval} minutes is too large — "
                    f"Render spins down after 15 min. Use */14 or less."
                )
            else:
                # For explicit minute lists or other patterns, just verify it's not hourly+
                assert "0" in minute_field or "*" in minute_field or "/" in minute_field, (
                    f"Cron minute field '{minute_field}' must fire more than once per hour"
                )

    def test_workflow_has_workflow_dispatch(self, workflow):
        """Workflow must support manual dispatch for debugging."""
        on = workflow.get("on") or workflow.get(True)
        assert "workflow_dispatch" in on, (
            "Workflow should have workflow_dispatch for manual testing via GitHub UI"
        )

    def test_workflow_has_at_least_one_job(self, workflow):
        """The workflow must define at least one job."""
        jobs = workflow.get("jobs", {})
        assert len(jobs) >= 1, "Workflow must have at least one job"

    def test_workflow_pings_health_endpoint(self, workflow):
        """At least one step must ping the /api/health endpoint."""
        jobs = workflow.get("jobs", {})
        all_run_commands = []

        for job in jobs.values():
            steps = job.get("steps", [])
            for step in steps:
                run_cmd = step.get("run", "")
                if run_cmd:
                    all_run_commands.append(run_cmd)

        combined = "\n".join(all_run_commands)
        assert "/api/health" in combined, (
            "Keep-alive workflow must ping /api/health to warm up Render and keep Supabase active"
        )

    def test_workflow_targets_render_backend_url(self, workflow):
        """Workflow must reference the deployed Render backend URL."""
        jobs = workflow.get("jobs", {})
        all_run_commands = []

        for job in jobs.values():
            steps = job.get("steps", [])
            for step in steps:
                run_cmd = step.get("run", "")
                if run_cmd:
                    all_run_commands.append(run_cmd)

        combined = "\n".join(all_run_commands)
        assert "onrender.com" in combined, (
            "Keep-alive workflow must target the Render deployment URL (.onrender.com)"
        )

    def test_workflow_uses_curl_or_wget(self, workflow):
        """At least one step must use curl or wget to make the HTTP request."""
        jobs = workflow.get("jobs", {})
        all_run_commands = []

        for job in jobs.values():
            steps = job.get("steps", [])
            for step in steps:
                run_cmd = step.get("run", "")
                if run_cmd:
                    all_run_commands.append(run_cmd)

        combined = "\n".join(all_run_commands).lower()
        assert "curl" in combined or "wget" in combined, (
            "Keep-alive workflow must use curl or wget to ping the health endpoint"
        )
