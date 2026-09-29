#!/usr/bin/env python3

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_URL = "https://api.github.com"
THEROCK_REF = "main"


@dataclass(frozen=True)
class Workflow:
    name: str
    file: str
    artifact_group: str
    runner: str
    context: str


WORKFLOWS = (
    Workflow(
        name="Linux",
        file="multi_arch_build_portable_linux_pytorch_wheels_ci.yml",
        artifact_group="gfx94X-dcgpu",
        runner="aws-linux-scale-rocm-prod",
        context="TheRock/Linux wheel CI",
    ),
    Workflow(
        name="Windows",
        file="multi_arch_build_windows_pytorch_wheels_ci.yml",
        artifact_group="gfx1151",
        runner="aws-windows-scale-rocm-prod-mix",
        context="TheRock/Windows wheel CI",
    ),
)


def parse_github_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def build_workflow_inputs(
    workflow: Workflow,
    *,
    sha: str,
    rocm_version: str,
    package_index_url: str,
) -> dict[str, str]:
    return {
        "artifact_group": workflow.artifact_group,
        "python_version": "3.12",
        "pytorch_git_ref": sha,
        "rocm_package_index_url": package_index_url,
        "rocm_version": rocm_version,
        "cache_type": "sccache",
        "build_runs_on": workflow.runner,
    }


def select_dispatched_run(
    runs: list[dict[str, Any]],
    *,
    sha: str,
    dispatched_at: datetime,
    excluded_ids: set[int] | None = None,
) -> dict[str, Any] | None:
    excluded_ids = excluded_ids or set()
    candidates = [
        run
        for run in runs
        if run["id"] not in excluded_ids
        and sha in run.get("display_title", "")
        and parse_github_time(run["created_at"]) >= dispatched_at
    ]
    # Never monitor or cancel a same-SHA run if another dispatch is ambiguous.
    return candidates[0] if len(candidates) == 1 else None


def status_state(conclusion: str | None) -> str:
    if conclusion == "success":
        return "success"
    if conclusion in {"cancelled", "skipped"}:
        return "error"
    return "failure"


def all_success(results: dict[str, str | None]) -> bool:
    return len(results) == len(WORKFLOWS) and all(
        result == "success" for result in results.values()
    )


class GitHubClient:
    def __init__(
        self,
        token: str,
        api_url: str = API_URL,
        token_provider: Callable[[bool], str] | None = None,
    ):
        self.token = token
        self.api_url = api_url.rstrip("/")
        self.token_provider = token_provider

    def request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Any:
        return self._request(method, path, payload, retry_auth=True)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        *,
        retry_auth: bool,
    ) -> Any:
        if self.token_provider:
            self.token = self.token_provider(False)
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            f"{self.api_url}{path}",
            data=data,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "rocm-pytorch-therock-ci",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read()
                return json.loads(body) if body else None
        except urllib.error.HTTPError as error:
            if error.code == 401 and retry_auth and self.token_provider:
                self.token = self.token_provider(True)
                return self._request(method, path, payload, retry_auth=False)
            detail = error.read().decode(errors="replace")
            raise RuntimeError(
                f"GitHub API {method} {path} failed: {error.code} {detail}"
            ) from error

    def dispatch(
        self,
        repository: str,
        workflow_file: str,
        *,
        ref: str,
        inputs: dict[str, str],
    ) -> None:
        workflow_path = urllib.parse.quote(workflow_file, safe="")
        self.request(
            "POST",
            f"/repos/{repository}/actions/workflows/{workflow_path}/dispatches",
            {"ref": ref, "inputs": inputs},
        )

    def workflow_runs(
        self, repository: str, workflow_file: str
    ) -> list[dict[str, Any]]:
        workflow_path = urllib.parse.quote(workflow_file, safe="")
        result = self.request(
            "GET",
            f"/repos/{repository}/actions/workflows/{workflow_path}/runs"
            "?event=workflow_dispatch&branch=main&per_page=50",
        )
        return result["workflow_runs"]

    def run(self, repository: str, run_id: int) -> dict[str, Any]:
        return self.request("GET", f"/repos/{repository}/actions/runs/{run_id}")

    def cancel_run(self, repository: str, run_id: int) -> None:
        self.request("POST", f"/repos/{repository}/actions/runs/{run_id}/cancel")

    def set_status(
        self,
        repository: str,
        sha: str,
        *,
        state: str,
        context: str,
        description: str,
        target_url: str | None = None,
    ) -> None:
        payload = {
            "state": state,
            "context": context,
            "description": description[:140],
        }
        if target_url:
            payload["target_url"] = target_url
        self.request("POST", f"/repos/{repository}/statuses/{sha}", payload)


def base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


class AppTokenProvider:
    def __init__(
        self,
        app_id: str,
        private_key: str,
        repository: str,
        initial_token: str,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.app_id = app_id
        self.private_key = private_key
        self.repository = repository
        self.token = initial_token
        self.monotonic = monotonic
        self.refreshed_at = monotonic()

    def __call__(self, force: bool = False) -> str:
        if not force and self.monotonic() - self.refreshed_at < 2_700:
            return self.token
        self.token = self._create_token()
        self.refreshed_at = self.monotonic()
        return self.token

    def _create_token(self) -> str:
        now = int(time.time())
        header = base64url(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
        claims = base64url(
            json.dumps({"iat": now - 60, "exp": now + 540, "iss": self.app_id}).encode()
        )
        unsigned = f"{header}.{claims}".encode()
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as key_file:
            key_file.write(self.private_key)
            key_file.flush()
            signature = subprocess.run(
                ["openssl", "dgst", "-sha256", "-sign", key_file.name],
                input=unsigned,
                capture_output=True,
                check=True,
            ).stdout
        jwt = f"{unsigned.decode()}.{base64url(signature)}"
        app = GitHubClient(jwt)
        installation = app.request("GET", f"/repos/{self.repository}/installation")
        token = app.request(
            "POST",
            f"/app/installations/{installation['id']}/access_tokens",
            {
                "repositories": [self.repository.split("/", 1)[1]],
                "permissions": {"actions": "write"},
            },
        )
        return token["token"]


def wait_for_run(
    client: GitHubClient,
    repository: str,
    workflow: Workflow,
    *,
    sha: str,
    dispatched_at: datetime,
    timeout_seconds: int,
    poll_interval: int,
    excluded_ids: set[int] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        run = select_dispatched_run(
            client.workflow_runs(repository, workflow.file),
            sha=sha,
            dispatched_at=dispatched_at,
            excluded_ids=excluded_ids,
        )
        if run is not None:
            return run
        sleep(poll_interval)
    raise TimeoutError(f"Timed out finding the dispatched {workflow.name} run")


def append_summary(lines: list[str]) -> None:
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a") as summary:
            summary.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dispatch and monitor TheRock wheel CI for a ROCm/pytorch PR"
    )
    parser.add_argument("--sha", required=True)
    parser.add_argument("--source-repository", default="ROCm/pytorch")
    parser.add_argument("--therock-repository", default="ROCm/TheRock")
    parser.add_argument("--rocm-version", required=True)
    parser.add_argument("--package-index-url", required=True)
    parser.add_argument("--poll-interval", type=int, default=20)
    parser.add_argument("--discovery-timeout", type=int, default=300)
    parser.add_argument("--run-timeout", type=int, default=20_400)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        for workflow in WORKFLOWS:
            inputs = build_workflow_inputs(
                workflow,
                sha=args.sha,
                rocm_version=args.rocm_version,
                package_index_url=args.package_index_url,
            )
            print(
                json.dumps(
                    {"workflow": workflow.file, "inputs": inputs}, sort_keys=True
                )
            )
        return 0

    therock_token = os.environ["THEROCK_TOKEN"]
    status_token = os.environ["GITHUB_TOKEN"]
    token_provider = AppTokenProvider(
        os.environ["THEROCK_APP_ID"],
        os.environ["THEROCK_APP_PRIVATE_KEY"],
        args.therock_repository,
        therock_token,
    )
    therock = GitHubClient(therock_token, token_provider=token_provider)
    source = GitHubClient(status_token)
    active_runs: dict[str, int] = {}
    run_urls: dict[str, str] = {}
    dispatched: dict[str, Workflow] = {}
    dispatch_times: dict[str, datetime] = {}
    excluded_ids: dict[str, set[int]] = {}
    finalized: set[str] = set()
    completed = False

    def set_unfinished_statuses(description: str) -> None:
        for workflow in WORKFLOWS:
            if workflow.name in finalized:
                continue
            try:
                source.set_status(
                    args.source_repository,
                    args.sha,
                    state="error",
                    context=workflow.context,
                    description=description,
                    target_url=run_urls.get(workflow.name),
                )
            except RuntimeError as error:
                print(f"::warning::{error}", file=sys.stderr)

    def recover_and_cancel_runs() -> None:
        for name, workflow in dispatched.items():
            try:
                run_id = active_runs.get(name)
                if run_id is None:
                    run = select_dispatched_run(
                        therock.workflow_runs(args.therock_repository, workflow.file),
                        sha=args.sha,
                        dispatched_at=dispatch_times[name],
                        excluded_ids=excluded_ids[name],
                    )
                    run_id = run["id"] if run else None
                if run_id is None:
                    continue
                run = therock.run(args.therock_repository, run_id)
                if run["status"] != "completed":
                    try:
                        therock.cancel_run(args.therock_repository, run_id)
                    except RuntimeError:
                        # Completion may race the cancellation request.
                        if (
                            therock.run(args.therock_repository, run_id)["status"]
                            != "completed"
                        ):
                            raise
            except RuntimeError as error:
                print(f"::warning::{error}", file=sys.stderr)

    def cancel_active_runs(_signal: int, _frame: Any) -> None:
        if completed:
            return
        recover_and_cancel_runs()
        set_unfinished_statuses("TheRock wheel CI monitor was cancelled")
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, cancel_active_runs)
    signal.signal(signal.SIGINT, cancel_active_runs)

    try:
        for workflow in WORKFLOWS:
            source.set_status(
                args.source_repository,
                args.sha,
                state="pending",
                context=workflow.context,
                description=f"Dispatching TheRock {workflow.name} wheel build",
            )
            existing_runs = therock.workflow_runs(
                args.therock_repository, workflow.file
            )
            excluded_ids[workflow.name] = {run["id"] for run in existing_runs}
            dispatched_at = datetime.now(timezone.utc).replace(microsecond=0)
            dispatch_times[workflow.name] = dispatched_at
            dispatched[workflow.name] = workflow
            therock.dispatch(
                args.therock_repository,
                workflow.file,
                ref=THEROCK_REF,
                inputs=build_workflow_inputs(
                    workflow,
                    sha=args.sha,
                    rocm_version=args.rocm_version,
                    package_index_url=args.package_index_url,
                ),
            )
            run = wait_for_run(
                therock,
                args.therock_repository,
                workflow,
                sha=args.sha,
                dispatched_at=dispatched_at,
                timeout_seconds=args.discovery_timeout,
                poll_interval=args.poll_interval,
                excluded_ids=excluded_ids[workflow.name],
            )
            active_runs[workflow.name] = run["id"]
            run_urls[workflow.name] = run["html_url"]
            source.set_status(
                args.source_repository,
                args.sha,
                state="pending",
                context=workflow.context,
                description=f"TheRock {workflow.name} wheel build is running",
                target_url=run["html_url"],
            )

        deadline = time.monotonic() + args.run_timeout
        pending = {workflow.name: workflow for workflow in WORKFLOWS}
        results: dict[str, str | None] = {}
        while pending and time.monotonic() < deadline:
            for name, workflow in list(pending.items()):
                run = therock.run(args.therock_repository, active_runs[name])
                if run["status"] != "completed":
                    continue
                conclusion = run.get("conclusion")
                results[name] = conclusion
                source.set_status(
                    args.source_repository,
                    args.sha,
                    state=status_state(conclusion),
                    context=workflow.context,
                    description=f"TheRock {workflow.name} wheel build: {conclusion}",
                    target_url=run["html_url"],
                )
                finalized.add(name)
                del pending[name]
            if pending:
                time.sleep(args.poll_interval)

        if pending:
            for name, workflow in pending.items():
                run = therock.run(args.therock_repository, active_runs[name])
                if run["status"] == "completed":
                    conclusion = run.get("conclusion")
                    results[name] = conclusion
                    state = status_state(conclusion)
                    description = f"TheRock {workflow.name} wheel build: {conclusion}"
                else:
                    try:
                        therock.cancel_run(args.therock_repository, active_runs[name])
                    except RuntimeError:
                        run = therock.run(args.therock_repository, active_runs[name])
                        if run["status"] != "completed":
                            raise
                    if run["status"] == "completed":
                        conclusion = run.get("conclusion")
                        results[name] = conclusion
                        state = status_state(conclusion)
                        description = (
                            f"TheRock {workflow.name} wheel build: {conclusion}"
                        )
                    else:
                        results[name] = "timed_out"
                        state = "error"
                        description = f"TheRock {workflow.name} wheel build timed out"
                source.set_status(
                    args.source_repository,
                    args.sha,
                    state=state,
                    context=workflow.context,
                    description=description,
                    target_url=run_urls[name],
                )
                finalized.add(name)

        append_summary(
            [
                "## TheRock PR wheel CI",
                "",
                *[
                    f"- [{workflow.name}]({run_urls.get(workflow.name, '')}): "
                    f"`{results.get(workflow.name, 'not found')}`"
                    for workflow in WORKFLOWS
                ],
            ]
        )
        completed = True
        return 0 if all_success(results) else 1
    except Exception as error:
        recover_and_cancel_runs()
        set_unfinished_statuses(f"TheRock dispatch failed: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
