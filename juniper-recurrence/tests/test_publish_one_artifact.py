"""Pin the one-artifact publish: PyPI takes exactly the files TestPyPI took, and each run checks it by sha256.

The owner's directive of 2026-11-15: "deploy to pypi should send the same artifact that was already sent to testpypi.
an identical hash check, of the artifacts after testpypi deploy and prior to pypi deploy, might also be helpful here to
prevent this from slipping through the cracks again." juniper-cascor's publisher rebuilt the package in its PyPI job,
so PyPI served a second build of its 0.13.0 and 0.14.0 releases. This repository's three publishers already built once
and handed both publish jobs one artifact, and every release so far is byte-identical on the two indexes, but nothing
checked it: both uploads set ``skip-existing``, which silently keeps an earlier file under the same name, the artifact
expired after 7 days while the ``pypi`` gate may wait 30, and "Re-run all jobs" re-ran the build, whose artifact every
later download takes (download-artifact picks the newest of a name). That is how juniper-ml's run 38045867684 published
a second build of juniper-ci-tools 0.9.1 to PyPI.

``publish-recurrence-app.yml``, ``publish-recurrence-client.yml`` and ``publish-recurrence-model.yml`` now carry
juniper-cascor's one-artifact template (juniper-cascor#763 and #767). The build writes ``SHA256SUMS`` beside the dist
and uploads both as one artifact, kept 90 days; a re-run attempt reuses it rather than build again; each publish job
checks the files against ``SHA256SUMS`` before it uploads; TestPyPI's JSON must list exactly those sha256 after the
TestPyPI upload and again before the PyPI one, and PyPI's after; only the uploading jobs hold ``id-token: write``; and a
dispatch from the release tag with ``republish_from_testpypi`` runs the republish job, the first rung of the owner's
recovery ladder, which copies TestPyPI's files for the version to PyPI and builds nothing. The template steps read only
``PUBLISH_PACKAGE`` and ``PUBLISH_DIST_DIR`` from the workflow ``env``, a step's ``INDEX_URL``, and the runner's own
variables.

The static cases pin the wiring: one build, in a job without an environment or ``id-token``; the job chain and the two
job names juniper-ml's release-train ceremony reads; per-job permissions; the reuse step first in the build job and
guarding every later step; each upload preceded by the dist check; TestPyPI's JSON checked after the TestPyPI upload
and again before the PyPI one, and PyPI's after; no ``skip-existing``; an artifact that outlives the 30-day ``pypi``
gate; the republish job's trigger, gate, steps and name; what each workflow kept (its triggers, concurrency, tag guard,
build commands and TestPyPI install check); every copy of a template step byte-identical across the three; and the app
lane running this file on a pull request that edits only a publish workflow. The rehearsals run each template step's
own Python, extracted from ``publish-recurrence-app.yml``, in a scratch workspace,
against a stand-in index and a stand-in GitHub API on 127.0.0.1: juniper-cascor's rehearsals, which use a flat
``dist``, and this repository's own ``<package>/dist`` layout for each of the three publishers.

Ported from juniper-cascor's ``src/tests/unit/test_publish_one_artifact.py`` (``main`` at ``675884888f5a``). The copies'
identity with juniper-cascor's template is checked across repositories by juniper-ml's
``util/ad-hoc/2026-11-15_publish_port_template_identity.py``; this file pins them identical to one another.

Project: juniper-recurrence
Author: Paul Calnon
Created: 2026-11-15
"""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import re
import shutil
import subprocess  # nosec B404 - runs the workflow's OWN extracted step scripts with a fixed argv
import sys
import threading
import tomllib
from pathlib import Path

import pytest
import yaml

# No marker: ci-recurrence-app.yml runs every test under tests/ (``python -m pytest``, no ``-m``), and ``--strict-markers``
# admits only the registered ``performance`` marker. PyYAML is a runtime dependency of this package, so it is importable
# wherever this suite runs; nothing here skips for want of it. ci-recurrence-app.yml's change filter names the three
# publish workflows, so a pull request that edits only one of them runs this file.

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
#: workflow -> (PUBLISH_PACKAGE, PUBLISH_DIST_DIR, TestPyPI job, PyPI job); each also has a ``build`` and a ``republish`` job.
PUBLISHERS = {
    "publish-recurrence-app.yml": ("juniper-recurrence", "juniper-recurrence/dist", "publish-testpypi", "publish-pypi"),
    "publish-recurrence-client.yml": ("juniper-recurrence-client", "juniper-recurrence-client/dist", "publish-testpypi", "publish-pypi"),
    "publish-recurrence-model.yml": ("juniper-recurrence-model", "juniper-recurrence-model/dist", "publish-testpypi", "publish-pypi"),
}
#: workflow -> (top-level permissions, the PyPI job's). The build, TestPyPI and republish jobs hold the same in all three.
PERMISSIONS = {
    "publish-recurrence-app.yml": ({"contents": "read"}, {"contents": "read", "id-token": "write"}),
    "publish-recurrence-client.yml": ({"contents": "read"}, {"contents": "read", "id-token": "write"}),
    "publish-recurrence-model.yml": ({"contents": "read"}, {"contents": "read", "id-token": "write"}),
}
#: workflow -> (its concurrency group, the module its TestPyPI install check imports): what each publisher kept.
KEPT = {
    "publish-recurrence-app.yml": ("publish-recurrence-app-${{ github.ref_name }}", "juniper_recurrence"),
    "publish-recurrence-client.yml": ("publish-recurrence-client-${{ github.ref_name }}", "juniper_recurrence_client"),
    "publish-recurrence-model.yml": ("publish-recurrence-model-${{ github.ref_name }}", "juniper_recurrence_model"),
}
#: The workflow the rehearsals extract the template steps from; TestTemplate pins the other two copies equal to it.
TEMPLATE_SOURCE = "publish-recurrence-app.yml"
REPUBLISH = "republish"
REPUBLISH_NAME = "Republish the tested files (dispatch only)"
REPUBLISH_IF = "github.event_name == 'workflow_dispatch' && inputs.republish_from_testpypi"
REUSE = "Reuse an earlier attempt's artifact"
REUSE_IF = "steps.reuse.outputs.reuse != 'true'"
RECORD = "Record SHA256SUMS"
UPLOAD = "Upload dist artifacts"
DOWNLOAD = "Download dist artifacts"
VERIFY_DIST = "Verify the dist against SHA256SUMS"
TESTPYPI_AFTER = "Verify TestPyPI serves exactly these files"
TESTPYPI_BEFORE = "Verify TestPyPI still serves exactly these files"
PYPI_AFTER = "Verify PyPI serves exactly these files"
FIND = "Find the release run's artifact"
FETCH_ORIGINAL = "Download the release run's artifact"
FETCH = "Download TestPyPI's files and verify them"
RELEASE_GATE = "Require a GitHub Release for this tag"
INSTALL_CHECK = "Verify install from TestPyPI"
INDEX_CHECKS = {TESTPYPI_AFTER: "https://test.pypi.org", TESTPYPI_BEFORE: "https://test.pypi.org", PYPI_AFTER: "https://pypi.org"}
#: Everything a template step may read from its environment: the two parameters, a step's own inputs, the runner's variables.
TEMPLATE_ENV = {
    "PUBLISH_PACKAGE",
    "PUBLISH_DIST_DIR",
    "INDEX_URL",
    "HASH_CHECK_ATTEMPTS",
    "HASH_CHECK_INTERVAL",
    "VERSION",
    "RELEASE_RUN_SUMS",
    "GH_TOKEN",
    "GITHUB_WORKSPACE",
    "GITHUB_API_URL",
    "GITHUB_REPOSITORY",
    "GITHUB_RUN_ID",
    "GITHUB_RUN_ATTEMPT",
    "GITHUB_OUTPUT",
    "GITHUB_REF",
    "GITHUB_REF_NAME",
    "GITHUB_REF_TYPE",
    "GITHUB_WORKFLOW_REF",
}
PUBLISH_ACTION = "pypa/gh-action-pypi-publish@"
BUILD_COMMAND = re.compile(r"\bpython3?\s+-m\s+build\b|\bpip3?\s+wheel\b|\buv\s+build\b|\bsetup\.py\s+(?:sdist|bdist_wheel)\b")
#: GitHub lets a deployment wait this many days for its reviewer: the PyPI job may start that late, and needs the artifact.
GATE_APPROVAL_DAYS = 30
ARTIFACT = "${{ env.PUBLISH_PACKAGE }}-dist"
DIST_INPUT = "${{ env.PUBLISH_DIST_DIR }}/"
WHEEL = "juniper_recurrence-9.9.9-py3-none-any.whl"
SDIST = "juniper_recurrence-9.9.9.tar.gz"
BUILT = {WHEEL: b"wheel bytes", SDIST: b"sdist bytes"}
REPOSITORY = "pcalnon/juniper-recurrence"
#: The variable the reuse and find steps read the runner's API credential from, the expression that fills it, and the
#: stand-in value the rehearsals pass. Bandit's B105 reads a string under a name like the first as a hardcoded password.
API_CREDENTIAL_ENV = "GH_TOKEN"
RUNNER_CREDENTIAL = "${{ github.token }}"
FAKE_API_CREDENTIAL = "stand-in-credential"


def _json_path(package: str = "juniper-recurrence") -> str:
    return f"/pypi/{package}/9.9.9/json"


JSON_PATH = _json_path()


def _workflow(name: str) -> dict:
    data = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    # PyYAML (YAML 1.1) reads the bare `on:` key as boolean True.
    data["on"] = data.pop(True, data.get("on"))
    return data


def _names(job: dict) -> list[str]:
    return [str(s.get("name") or s.get("uses") or "") for s in job.get("steps") or []]


def _index(job: dict, name: str) -> int:
    names = _names(job)
    assert names.count(name) == 1, f"expected exactly one step named {name!r}, found {names.count(name)} in {names}"
    return names.index(name)


def _step(job: dict, name: str) -> dict:
    return job["steps"][_index(job, name)]


def _uploads(job: dict) -> list[int]:
    return [i for i, s in enumerate(job.get("steps") or []) if str(s.get("uses") or "").startswith(PUBLISH_ACTION)]


def _environment(job: dict) -> str:
    env = job.get("environment")
    return str(env.get("name") if isinstance(env, dict) else env)


def _assert_no_checkout_between(job: dict, start: int, end: int) -> None:
    """A checkout empties the workspace, so one between the download and the last template step would lose the files."""
    between = [str(s.get("uses") or "") for s in job["steps"][start:end]]
    assert not [u for u in between if u.startswith("actions/checkout@")], f"a checkout runs between steps {start} and {end}: {between}"


def _template_scripts() -> dict[str, str]:
    """The template steps' Python, as ``publish-recurrence-app.yml`` runs it; every other copy must equal these."""
    jobs = _workflow(TEMPLATE_SOURCE)["jobs"]
    return {
        REUSE: _step(jobs["build"], REUSE)["run"],
        RECORD: _step(jobs["build"], RECORD)["run"],
        VERIFY_DIST: _step(jobs["publish-testpypi"], VERIFY_DIST)["run"],
        "index": _step(jobs["publish-testpypi"], TESTPYPI_AFTER)["run"],
        FIND: _step(jobs[REPUBLISH], FIND)["run"],
        FETCH: _step(jobs[REPUBLISH], FETCH)["run"],
    }


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _built(package: str) -> dict[str, bytes]:
    """A wheel and an sdist of ``package`` 9.9.9, named as the build backend names them."""
    module = package.replace("-", "_")
    return {f"{module}-9.9.9-py3-none-any.whl": f"{package} wheel".encode(), f"{module}-9.9.9.tar.gz": f"{package} sdist".encode()}


def _script(kind: str, tmp_path: Path) -> Path:
    path = tmp_path / f"{re.sub(r'[^a-z]+', '_', kind.lower())}.py"
    path.write_text(_template_scripts()[kind], encoding="utf-8")
    return path


def _run(script: Path, workspace: Path, cwd: Path | None = None, **env: str) -> subprocess.CompletedProcess[str]:
    """Run an extracted step as the runner does (``python3 {0}``), with a minimal environment rather than a copy of ours."""
    child = {"PATH": os.environ.get("PATH", ""), "GITHUB_WORKSPACE": str(workspace), "PUBLISH_PACKAGE": "juniper-recurrence", "PUBLISH_DIST_DIR": "dist"}
    child.update(env)
    return subprocess.run([sys.executable, "-I", str(script)], cwd=cwd or workspace, env=child, capture_output=True, text=True, timeout=120, check=False)  # nosec B603 - fixed argv: this interpreter and an extracted step


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "dist").mkdir(parents=True)
    for name, data in BUILT.items():
        (root / "dist" / name).write_bytes(data)
    return root


class _Stand(http.server.ThreadingHTTPServer):
    """A stand-in for an index's JSON API and file host, or for the GitHub API: each GET of a path gets that path's next
    scripted answer (the last one repeats), and an unscripted path gets a 404. Every request's path and headers are kept."""

    def __init__(self, routes: dict[str, list[tuple[int, object]]]) -> None:
        super().__init__(("127.0.0.1", 0), _StandHandler)
        self.routes = {path: list(answers) for path, answers in routes.items()}
        self.requests: list[tuple[str, dict[str, str]]] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def paths(self, path: str | None = None) -> list[str]:
        return [p for p, _ in self.requests if path is None or p == path]


class _StandHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler dispatches a GET to exactly this name
        stand = self.server
        stand.requests.append((self.path, dict(self.headers.items())))
        answers = stand.routes.get(self.path) or [(404, {"message": "Not Found"})]
        status, body = answers.pop(0) if len(answers) > 1 else answers[0]
        payload = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        """Quiet: the cases read the requests the step made, not the server's log."""
        return


@pytest.fixture
def stand():
    started: list[_Stand] = []

    def start(routes: dict[str, list[tuple[int, object]]]) -> _Stand:
        server = _Stand(routes)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append(server)
        return server

    yield start
    for server in started:
        server.shutdown()
        server.server_close()


def _release(files: dict[str, str], base: str = "https://test-files.example", package: str = "juniper-recurrence") -> dict:
    return {"info": {"name": package, "version": "9.9.9"}, "urls": [{"filename": name, "url": f"{base}/files/{name}", "digests": {"sha256": digest, "md5": "0" * 32}} for name, digest in files.items()]}


NOT_FOUND = (404, {"message": "Not Found"})
SERVED = {name: _sha(data) for name, data in BUILT.items()}


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Wiring: one build, one artifact, a check before every upload, and the names the release train reads
# ─────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("workflow", sorted(PUBLISHERS))
class TestWiring:
    def test_the_template_parameters_name_this_package(self, workflow):
        package, dist_dir, _, _ = PUBLISHERS[workflow]
        assert _workflow(workflow)["env"] == {"PUBLISH_PACKAGE": package, "PUBLISH_DIST_DIR": dist_dir}

    def test_the_template_parameters_are_the_packages_own_name_and_the_builds_output_directory(self, workflow):
        package, dist_dir, _, _ = PUBLISHERS[workflow]
        # `python -m build` writes to dist/ under the build job's working directory, the package's own directory.
        working = _workflow(workflow)["jobs"]["build"]["defaults"]["run"]["working-directory"]
        assert dist_dir == f"{working}/dist"
        assert tomllib.loads((REPO / working / "pyproject.toml").read_text(encoding="utf-8"))["project"]["name"] == package

    def test_only_the_build_job_builds_and_it_holds_no_environment(self, workflow):
        jobs = _workflow(workflow)["jobs"]
        builders = sorted(job_id for job_id, job in jobs.items() if any(BUILD_COMMAND.search(str(s.get("run") or "")) for s in job.get("steps") or []))
        assert builders == ["build"]
        assert jobs["build"]["name"] == "Build and Validate"
        assert "environment" not in jobs["build"]
        assert not _uploads(jobs["build"])

    def test_the_publish_jobs_chain_on_the_build_under_the_names_the_release_train_reads(self, workflow):
        _, _, test_id, prod_id = PUBLISHERS[workflow]
        jobs = _workflow(workflow)["jobs"]
        assert sorted(jobs) == sorted(["build", test_id, prod_id, REPUBLISH])
        assert jobs[test_id]["needs"] == "build"
        assert jobs[prod_id]["needs"] == test_id
        # juniper-ml's util/release_train/ceremony.py tells the two jobs apart by name ("testpypi" / "pypi").
        assert jobs[test_id]["name"] == "Publish to TestPyPI"
        assert jobs[prod_id]["name"] == "Publish to PyPI"
        assert _environment(jobs[test_id]) == "testpypi"
        assert _environment(jobs[prod_id]) == "pypi"

    def test_only_the_jobs_that_upload_hold_id_token_write(self, workflow):
        wf = _workflow(workflow)
        jobs = wf["jobs"]
        top, prod = PERMISSIONS[workflow]
        assert wf["permissions"] == top
        uploading = sorted(job_id for job_id, job in jobs.items() if _uploads(job))
        minting = sorted(job_id for job_id, job in jobs.items() if (job.get("permissions") or {}).get("id-token") == "write")
        assert uploading == minting == sorted(PUBLISHERS[workflow][2:] + (REPUBLISH,))
        assert jobs["build"]["permissions"] == {"contents": "read", "actions": "read"}
        assert jobs[PUBLISHERS[workflow][2]]["permissions"] == {"contents": "read", "id-token": "write"}
        assert jobs[PUBLISHERS[workflow][3]]["permissions"] == prod
        assert jobs[REPUBLISH]["permissions"] == {"actions": "read", "id-token": "write"}

    def test_a_re_run_reuses_the_first_attempts_artifact_instead_of_building(self, workflow):
        build = _workflow(workflow)["jobs"]["build"]
        reuse = build["steps"][0]
        assert reuse.get("name") == REUSE
        assert reuse.get("id") == "reuse"
        assert reuse.get("if") == "github.run_attempt != '1'"
        # It runs before the checkout, so the job's default directory (the package's, which the checkout creates) does
        # not exist yet.
        assert reuse.get("working-directory") == "${{ github.workspace }}"
        assert reuse.get("shell") == "python3 {0}"
        assert reuse.get("env") == {API_CREDENTIAL_ENV: RUNNER_CREDENTIAL}
        # Every later step is skipped once the reuse step finds the artifact, except the release-convention gate, which
        # runs on `push` alone, an event no publisher subscribes to.
        for step in build["steps"][1:]:
            if step.get("if") == "github.event_name == 'push'":
                continue
            assert step.get("if") == REUSE_IF, f"{step.get('name') or step.get('uses')} would run on a re-run that reuses the artifact"
        for name in (RECORD, UPLOAD):
            assert _step(build, name).get("if") == REUSE_IF

    def test_the_testpypi_job_checks_the_dist_uploads_then_checks_testpypi_before_the_install_check(self, workflow):
        _, _, test_id, _ = PUBLISHERS[workflow]
        job = _workflow(workflow)["jobs"][test_id]
        uploads = _uploads(job)
        assert len(uploads) == 1
        assert job["steps"][uploads[0]]["with"]["repository-url"] == "https://test.pypi.org/legacy/"
        assert _index(job, DOWNLOAD) < _index(job, VERIFY_DIST) < uploads[0] < _index(job, TESTPYPI_AFTER)
        installs = [i for i, s in enumerate(job["steps"]) if "pip install" in str(s.get("run") or "")]
        assert installs, "the TestPyPI install check is gone"
        assert min(installs) > _index(job, TESTPYPI_AFTER), "the install check must come after the sha256 check that vouches for what it installs"
        _assert_no_checkout_between(job, _index(job, DOWNLOAD), _index(job, TESTPYPI_AFTER))

    def test_the_pypi_job_rechecks_testpypi_before_its_upload_and_pypi_after(self, workflow):
        _, _, _, prod_id = PUBLISHERS[workflow]
        job = _workflow(workflow)["jobs"][prod_id]
        uploads = _uploads(job)
        assert len(uploads) == 1
        assert "repository-url" not in (job["steps"][uploads[0]].get("with") or {})
        assert _index(job, DOWNLOAD) < _index(job, VERIFY_DIST) < _index(job, TESTPYPI_BEFORE) < uploads[0] < _index(job, PYPI_AFTER)
        _assert_no_checkout_between(job, _index(job, DOWNLOAD), _index(job, PYPI_AFTER))

    def test_the_republish_job_runs_only_on_its_dispatch_behind_the_pypi_gate_and_builds_nothing(self, workflow):
        wf = _workflow(workflow)
        job = wf["jobs"][REPUBLISH]
        assert job["name"] == REPUBLISH_NAME
        # The ceremony counts a job as the TestPyPI or the PyPI job by those substrings; this one must be neither.
        assert "pypi" not in job["name"].lower()
        assert job["if"] == REPUBLISH_IF
        assert "needs" not in job
        assert _environment(job) == "pypi"
        assert wf["on"]["workflow_dispatch"]["inputs"] == {"republish_from_testpypi": {"description": "Copy TestPyPI's files for this tag's version to PyPI (run from the release tag; nothing is built)", "type": "boolean", "default": False}}
        uploads = _uploads(job)
        assert len(uploads) == 1
        assert job["steps"][uploads[0]].get("with") == {"packages-dir": DIST_INPUT}
        assert _index(job, FIND) < _index(job, FETCH_ORIGINAL) < _index(job, FETCH) < _index(job, VERIFY_DIST) < uploads[0] < _index(job, PYPI_AFTER)
        _assert_no_checkout_between(job, 0, len(job["steps"]))
        assert _step(job, FIND).get("id") == "original"
        assert _step(job, FIND).get("env") == {API_CREDENTIAL_ENV: RUNNER_CREDENTIAL}
        assert _step(job, FETCH_ORIGINAL)["if"] == "steps.original.outputs.run-id != ''"
        assert _step(job, FETCH_ORIGINAL)["with"] == {"name": ARTIFACT, "run-id": "${{ steps.original.outputs.run-id }}", "github-token": RUNNER_CREDENTIAL, "path": "${{ runner.temp }}/release-run"}
        assert _step(job, FETCH)["env"] == {"INDEX_URL": "https://test.pypi.org", "VERSION": "${{ steps.original.outputs.version }}", "RELEASE_RUN_SUMS": "${{ runner.temp }}/release-run/SHA256SUMS"}
        assert _step(job, PYPI_AFTER)["env"] == {"INDEX_URL": "https://pypi.org"}

    def test_every_upload_takes_the_downloaded_dist_and_none_skips_an_existing_file(self, workflow):
        for job in _workflow(workflow)["jobs"].values():
            for i in _uploads(job):
                inputs = job["steps"][i].get("with") or {}
                assert inputs.get("packages-dir") == DIST_INPUT
                # skip-existing would keep an earlier file under the same name, and the install check would test it.
                assert "skip-existing" not in inputs

    def test_one_artifact_carries_the_dist_and_sha256sums_and_outlives_the_pypi_gate(self, workflow):
        _, _, test_id, prod_id = PUBLISHERS[workflow]
        jobs = _workflow(workflow)["jobs"]
        upload = _step(jobs["build"], UPLOAD)["with"]
        assert upload["name"] == ARTIFACT
        assert [p.strip() for p in upload["path"].splitlines() if p.strip()] == [DIST_INPUT, "SHA256SUMS"]
        assert upload["if-no-files-found"] == "error"
        assert int(upload["retention-days"]) > GATE_APPROVAL_DAYS
        assert _index(jobs["build"], RECORD) < _index(jobs["build"], UPLOAD)
        for job_id in (test_id, prod_id):
            assert _step(jobs[job_id], DOWNLOAD)["with"] == {"name": ARTIFACT, "path": "${{ github.workspace }}"}

    def test_each_json_check_reads_its_own_index_and_keeps_the_default_bounds(self, workflow):
        _, _, test_id, prod_id = PUBLISHERS[workflow]
        jobs = _workflow(workflow)["jobs"]
        for job_id, name in ((test_id, TESTPYPI_AFTER), (prod_id, TESTPYPI_BEFORE), (prod_id, PYPI_AFTER)):
            assert _step(jobs[job_id], name)["env"] == {"INDEX_URL": INDEX_CHECKS[name]}


@pytest.mark.parametrize("workflow", sorted(PUBLISHERS))
class TestUnchanged:
    # The permissions are not here: they changed, and TestWiring pins them per job.
    def test_it_still_publishes_on_a_published_release_under_its_name_and_concurrency(self, workflow):
        package = PUBLISHERS[workflow][0]
        group, _ = KEPT[workflow]
        wf = _workflow(workflow)
        assert wf["name"] == f"Publish {package} to PyPI"
        assert set(wf["on"]) == {"release", "workflow_dispatch"}
        assert wf["on"]["release"] == {"types": ["published"]}
        assert wf["concurrency"] == {"group": group, "cancel-in-progress": False}

    def test_it_builds_on_its_own_release_tag_or_a_plain_dispatch(self, workflow):
        package, _, test_id, prod_id = PUBLISHERS[workflow]
        jobs = _workflow(workflow)["jobs"]
        guard = jobs["build"]["if"]
        # A plain dispatch still re-fires the whole publish; the republish dispatch runs the republish job alone.
        assert guard == f"(github.event_name == 'release' && startsWith(github.event.release.tag_name, '{package}-v')) || (github.event_name == 'workflow_dispatch' && !inputs.republish_from_testpypi)"
        # The three packages share the `juniper-recurrence-` prefix, so the `-v` is what keeps a sibling's release out.
        prefix = re.search(r"startsWith\(github\.event\.release\.tag_name, '([^']+)'\)", guard)
        assert prefix, guard
        assert sorted(other for other, _, _, _ in PUBLISHERS.values() if f"{other}-v1.2.3".startswith(prefix.group(1))) == [package]
        # The publish jobs follow the build; only it reads the event.
        assert "if" not in jobs[test_id]
        assert "if" not in jobs[prod_id]

    def test_it_builds_in_the_package_directory_with_the_commands_it_used_before(self, workflow):
        package = PUBLISHERS[workflow][0]
        build = _workflow(workflow)["jobs"]["build"]
        assert build["defaults"] == {"run": {"working-directory": package}}
        runs = [str(s.get("run") or "").strip() for s in build["steps"]]
        assert runs.index("pip install --upgrade build twine") < runs.index("python -m build --sdist --wheel") < runs.index("twine check dist/*")
        setup = [s for s in build["steps"] if str(s.get("uses") or "").startswith("actions/setup-python@")]
        assert len(setup) == 1
        assert setup[0]["with"] == {"python-version": "3.12"}
        # The release-convention gate runs on `push` alone, an event no publisher subscribes to; it is kept as it was.
        assert _step(build, RELEASE_GATE)["if"] == "github.event_name == 'push'"
        assert _index(build, REUSE) < _index(build, RELEASE_GATE)

    def test_the_install_check_still_installs_from_testpypi_alone_after_a_sparse_checkout_of_the_version(self, workflow):
        package, _, test_id, _ = PUBLISHERS[workflow]
        _, module = KEPT[workflow]
        job = _workflow(workflow)["jobs"][test_id]
        checkouts = [i for i, s in enumerate(job["steps"]) if str(s.get("uses") or "").startswith("actions/checkout@")]
        assert len(checkouts) == 1
        # The sparse checkout empties the workspace, so it must follow the JSON check, the job's last template step.
        assert _index(job, TESTPYPI_AFTER) < checkouts[0] < _index(job, INSTALL_CHECK)
        assert job["steps"][checkouts[0]]["with"] == {"sparse-checkout": f"{package}/{module}/_version.py\n"}
        run = _step(job, INSTALL_CHECK)["run"]
        assert f"{package}/{module}/_version.py" in run
        assert "--no-deps" in run
        assert "--index-url https://test.pypi.org/simple/" in run
        assert "extra-index-url" not in run
        assert f'"{package}==${{version}}"' in run
        assert f"import {module};" in run

    def test_the_environments_and_the_release_uploads_are_as_they_were(self, workflow):
        package, _, test_id, prod_id = PUBLISHERS[workflow]
        jobs = _workflow(workflow)["jobs"]
        assert jobs[test_id]["environment"] == {"name": "testpypi", "url": f"https://test.pypi.org/p/{package}"}
        assert jobs[prod_id]["environment"] == {"name": "pypi", "url": f"https://pypi.org/p/{package}"}
        for job_id in (test_id, prod_id):
            assert jobs[job_id]["steps"][_uploads(jobs[job_id])[0]]["with"].get("verbose") is True


class TestTemplate:
    def test_every_copy_of_a_template_step_is_byte_identical(self):
        copies: dict[str, list[str]] = {REUSE: [], RECORD: [], VERIFY_DIST: [], "index": [], FIND: [], FETCH: []}
        for workflow in PUBLISHERS:
            for job in _workflow(workflow)["jobs"].values():
                for step in job.get("steps") or []:
                    kind = "index" if step.get("name") in INDEX_CHECKS else step.get("name")
                    if kind in copies:
                        assert step.get("shell") == "python3 {0}"
                        copies[kind].append(step["run"])
        # Per workflow: one reuse step and one record step; a dist check before each of three uploads; three JSON checks
        # in the release jobs and one in the republish job; one of each republish step.
        assert {kind: len(runs) for kind, runs in copies.items()} == {REUSE: 3, RECORD: 3, VERIFY_DIST: 9, "index": 12, FIND: 3, FETCH: 3}
        assert {kind: len(set(runs)) for kind, runs in copies.items()} == {REUSE: 1, RECORD: 1, VERIFY_DIST: 1, "index": 1, FIND: 1, FETCH: 1}

    def test_the_republish_job_is_byte_identical_in_every_publisher(self):
        jobs = [_workflow(workflow)["jobs"][REPUBLISH] for workflow in PUBLISHERS]
        assert all(job == jobs[0] for job in jobs)

    def test_the_template_reads_nothing_but_its_parameters(self):
        for kind, run in _template_scripts().items():
            assert "${{" not in run, f"{kind}: an expression would tie the step to one workflow"
            read = set(re.findall(r"os\.environ(?:\.get\(|\[)[\"']([A-Z_]+)[\"']", run))
            assert read, kind
            assert read <= TEMPLATE_ENV, f"{kind} reads {sorted(read - TEMPLATE_ENV)}"

    def test_the_json_check_polls_a_bounded_number_of_times(self):
        run = _template_scripts()["index"]
        attempts = re.search(r'"HASH_CHECK_ATTEMPTS", "(\d+)"', run)
        interval = re.search(r'"HASH_CHECK_INTERVAL", "(\d+(?:\.\d+)?)"', run)
        assert attempts and interval
        # Long enough for a CDN to catch up after an upload, and bounded; each request is bounded too.
        assert 60 <= int(attempts.group(1)) * float(interval.group(1)) <= 900
        assert "urllib.request.urlopen(request, timeout=30)" in run
        for workflow in PUBLISHERS:
            for job in _workflow(workflow)["jobs"].values():
                for step in job.get("steps") or []:
                    assert not {"HASH_CHECK_ATTEMPTS", "HASH_CHECK_INTERVAL"} & set(step.get("env") or {}), f"{workflow}: {step.get('name')} overrides the bounds"


class TestLane:
    """ci-recurrence-app.yml runs this file. On a pull request its tests run only when the `changes` job says the app was
    touched, so the three publish workflows must count as the app's, or a pull request that edits only one of them would
    skip the pin that guards it."""

    def test_a_pull_request_or_push_that_edits_only_a_publish_workflow_runs_this_file(self):
        lane = _workflow("ci-recurrence-app.yml")
        script = next(s["run"] for s in lane["jobs"]["changes"]["steps"] if s.get("id") == "filter")
        pattern = re.search(r"grep -qE '([^']+)'", script)
        assert pattern, script
        for workflow in PUBLISHERS:
            assert re.search(pattern.group(1), f".github/workflows/{workflow}"), f"a PR editing only {workflow} skips this file"
            assert f".github/workflows/{workflow}" in lane["on"]["push"]["paths"]
        assert re.search(pattern.group(1), "juniper-recurrence/tests/test_publish_one_artifact.py")
        # The lane stays scoped to the app: a sibling package's change is not the app's.
        assert not re.search(pattern.group(1), "juniper-recurrence-model/pyproject.toml")
        assert lane["jobs"]["test"]["if"] == "needs.changes.outputs.app == 'true'"


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Rehearsals: the template steps' own Python, run as the runner runs it
# ─────────────────────────────────────────────────────────────────────────────────────────────
class TestRecordSha256sums:
    def test_it_hashes_every_distribution_in_sha256sum_format(self, workspace, tmp_path):
        proc = _run(_script(RECORD, tmp_path), workspace)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert (workspace / "SHA256SUMS").read_text(encoding="utf-8") == "".join(f"{_sha(data)}  {name}\n" for name, data in sorted(BUILT.items()))

    def test_sha256sum_checks_what_it_writes(self, workspace, tmp_path):
        binary = shutil.which("sha256sum")
        if binary is None:
            pytest.skip("no sha256sum on PATH")
        assert _run(_script(RECORD, tmp_path), workspace).returncode == 0
        check = subprocess.run([binary, "--check", "--strict", str(workspace / "SHA256SUMS")], cwd=workspace / "dist", capture_output=True, text=True, timeout=60, check=False)  # nosec B603 - fixed argv
        assert check.returncode == 0, check.stdout + check.stderr

    @pytest.mark.parametrize("stray", ["notes.txt", "juniper_recurrence-9.9.9.zip"])
    def test_it_refuses_anything_but_wheels_and_sdists(self, workspace, tmp_path, stray):
        (workspace / "dist" / stray).write_bytes(b"x")
        proc = _run(_script(RECORD, tmp_path), workspace)
        assert proc.returncode == 1
        assert "::error::" in proc.stdout
        assert not (workspace / "SHA256SUMS").exists()

    def test_it_refuses_an_empty_or_a_missing_dist_directory(self, tmp_path):
        root = tmp_path / "empty"
        (root / "dist").mkdir(parents=True)
        script = _script(RECORD, tmp_path)
        assert _run(script, root).returncode == 1
        assert _run(script, root, PUBLISH_DIST_DIR="nowhere").returncode == 1

    def test_it_leaves_out_dotfiles_as_the_upload_does(self, workspace, tmp_path):
        (workspace / "dist" / ".gitignore").write_text("*\n", encoding="utf-8")
        assert _run(_script(RECORD, tmp_path), workspace).returncode == 0
        assert ".gitignore" not in (workspace / "SHA256SUMS").read_text(encoding="utf-8")

    @pytest.mark.parametrize("workflow", sorted(PUBLISHERS))
    def test_a_nested_dist_directory_resolves_from_the_workspace_not_the_working_directory(self, tmp_path, workflow):
        package, dist_dir, _, _ = PUBLISHERS[workflow]
        # Each build job runs its steps in the package directory (defaults.run.working-directory), the record step too.
        working = _workflow(workflow)["jobs"]["build"]["defaults"]["run"]["working-directory"]
        root = tmp_path / "workspace"
        (root / dist_dir).mkdir(parents=True)
        wheel = f"{package.replace('-', '_')}-1.0.0-py3-none-any.whl"
        (root / dist_dir / wheel).write_bytes(b"package wheel")
        proc = _run(_script(RECORD, tmp_path), root, cwd=root / working, PUBLISH_PACKAGE=package, PUBLISH_DIST_DIR=dist_dir)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert (root / "SHA256SUMS").read_text(encoding="utf-8") == f"{_sha(b'package wheel')}  {wheel}\n"


class TestVerifyDist:
    @staticmethod
    def _recorded(workspace: Path, tmp_path: Path) -> Path:
        assert _run(_script(RECORD, tmp_path), workspace).returncode == 0
        return _script(VERIFY_DIST, tmp_path)

    def test_it_passes_on_the_files_the_build_recorded(self, workspace, tmp_path):
        proc = _run(self._recorded(workspace, tmp_path), workspace)
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_it_ignores_dotfiles_as_the_upload_does(self, workspace, tmp_path):
        script = self._recorded(workspace, tmp_path)
        (workspace / "dist" / ".gitignore").write_text("*\n", encoding="utf-8")
        proc = _run(script, workspace)
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_it_fails_when_a_file_changed(self, workspace, tmp_path):
        script = self._recorded(workspace, tmp_path)
        (workspace / "dist" / WHEEL).write_bytes(b"another build")
        assert _run(script, workspace).returncode == 1

    def test_it_fails_on_a_file_the_build_did_not_record(self, workspace, tmp_path):
        script = self._recorded(workspace, tmp_path)
        (workspace / "dist" / "juniper_recurrence-9.9.9-py2.py3-none-any.whl").write_bytes(b"x")
        assert _run(script, workspace).returncode == 1

    def test_it_fails_when_a_recorded_file_is_missing(self, workspace, tmp_path):
        script = self._recorded(workspace, tmp_path)
        (workspace / "dist" / SDIST).unlink()
        assert _run(script, workspace).returncode == 1

    @pytest.mark.parametrize(
        "sums",
        ["", f"nothex  {WHEEL}\n", f"{'a' * 64} {WHEEL}\n", f"{'a' * 64}  {WHEEL}\n{'a' * 64}  {WHEEL}\n"],
        ids=["empty", "not-a-sha256", "one-space", "listed-twice"],
    )
    def test_it_fails_on_a_sha256sums_it_cannot_read(self, workspace, tmp_path, sums):
        (workspace / "SHA256SUMS").write_text(sums, encoding="utf-8")
        assert _run(_script(VERIFY_DIST, tmp_path), workspace).returncode == 1

    @pytest.mark.parametrize("workflow", sorted(PUBLISHERS))
    def test_it_checks_this_repositorys_nested_dist_directory(self, tmp_path, workflow):
        package, dist_dir, _, _ = PUBLISHERS[workflow]
        root = tmp_path / "workspace"
        (root / dist_dir).mkdir(parents=True)
        for name, data in _built(package).items():
            (root / dist_dir / name).write_bytes(data)
        assert _run(_script(RECORD, tmp_path), root, PUBLISH_PACKAGE=package, PUBLISH_DIST_DIR=dist_dir).returncode == 0
        script = _script(VERIFY_DIST, tmp_path)
        proc = _run(script, root, PUBLISH_PACKAGE=package, PUBLISH_DIST_DIR=dist_dir)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        # A flat dist directory, which this layout does not have, holds none of the recorded files.
        assert _run(script, root, PUBLISH_PACKAGE=package, PUBLISH_DIST_DIR="dist").returncode == 1


class TestIndexCheck:
    @staticmethod
    def _ready(workspace: Path, tmp_path: Path) -> Path:
        assert _run(_script(RECORD, tmp_path), workspace).returncode == 0
        return _script("index", tmp_path)

    @staticmethod
    def _check(script: Path, workspace: Path, server: _Stand, attempts: int = 4, **env: str) -> subprocess.CompletedProcess[str]:
        return _run(script, workspace, INDEX_URL=server.url, HASH_CHECK_ATTEMPTS=str(attempts), HASH_CHECK_INTERVAL="0", **env)

    def test_an_exact_match_passes_at_the_first_answer(self, workspace, tmp_path, stand):
        server = stand({JSON_PATH: [(200, _release(SERVED))]})
        proc = self._check(self._ready(workspace, tmp_path), workspace, server)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert server.paths() == [JSON_PATH]

    def test_it_polls_through_a_version_the_index_does_not_serve_yet(self, workspace, tmp_path, stand):
        server = stand({JSON_PATH: [NOT_FOUND, NOT_FOUND, (200, _release(SERVED))]})
        proc = self._check(self._ready(workspace, tmp_path), workspace, server)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert len(server.paths()) == 3

    def test_it_polls_through_a_file_the_index_does_not_list_yet(self, workspace, tmp_path, stand):
        server = stand({JSON_PATH: [(200, _release({WHEEL: SERVED[WHEEL]})), (200, _release(SERVED))]})
        proc = self._check(self._ready(workspace, tmp_path), workspace, server)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert len(server.paths()) == 2

    def test_it_polls_through_an_answer_it_cannot_read(self, workspace, tmp_path, stand):
        server = stand({JSON_PATH: [(200, b"<html>busy</html>"), (503, b""), (200, {"info": {}}), (200, _release(SERVED))]})
        proc = self._check(self._ready(workspace, tmp_path), workspace, server, attempts=5)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert len(server.paths()) == 4

    def test_an_index_that_lists_no_files_never_passes(self, workspace, tmp_path, stand):
        server = stand({JSON_PATH: [(200, _release({}))]})
        proc = self._check(self._ready(workspace, tmp_path), workspace, server, attempts=3)
        assert proc.returncode == 1
        assert "after 3 attempts" in proc.stdout
        assert len(server.paths()) == 3

    def test_a_file_with_another_sha256_fails_at_once(self, workspace, tmp_path, stand):
        server = stand({JSON_PATH: [(200, _release({**SERVED, WHEEL: _sha(b"another build")}))]})
        proc = self._check(self._ready(workspace, tmp_path), workspace, server)
        assert proc.returncode == 1
        assert "::error::" in proc.stdout
        assert len(server.paths()) == 1

    def test_a_file_this_run_did_not_build_fails_at_once(self, workspace, tmp_path, stand):
        server = stand({JSON_PATH: [(200, _release({**SERVED, "juniper_recurrence-9.9.9-py2.py3-none-any.whl": "b" * 64}))]})
        proc = self._check(self._ready(workspace, tmp_path), workspace, server)
        assert proc.returncode == 1
        assert len(server.paths()) == 1

    def test_it_gives_up_after_its_bounded_attempts(self, workspace, tmp_path, stand):
        server = stand({JSON_PATH: [NOT_FOUND]})
        proc = self._check(self._ready(workspace, tmp_path), workspace, server, attempts=3)
        assert proc.returncode == 1
        assert "after 3 attempts" in proc.stdout
        assert len(server.paths()) == 3

    # The three packages here share the `juniper-recurrence` prefix, so a sibling's files are the likeliest stray.
    @pytest.mark.parametrize(
        "extra",
        ["juniper_recurrence_model-9.9.9-py3-none-any.whl", "juniper_recurrence_client-9.9.9.tar.gz", "juniper_recurrence-9.9.8.tar.gz", "juniper_recurrence-9.9.9.zip"],
        ids=["a-siblings-wheel", "a-siblings-sdist", "another-version", "not-a-distribution"],
    )
    def test_it_refuses_a_sha256sums_it_cannot_vouch_for_before_asking_the_index(self, workspace, tmp_path, stand, extra):
        script = self._ready(workspace, tmp_path)
        with (workspace / "SHA256SUMS").open("a", encoding="utf-8") as sums:
            sums.write(f"{'c' * 64}  {extra}\n")
        server = stand({JSON_PATH: [(200, _release(SERVED))]})
        proc = self._check(script, workspace, server)
        assert proc.returncode == 1
        assert "::error::" in proc.stdout
        assert server.paths() == []

    @pytest.mark.parametrize("workflow", sorted(PUBLISHERS))
    def test_each_publishers_package_reads_its_own_json_for_its_own_files(self, tmp_path, stand, workflow):
        package, dist_dir, _, _ = PUBLISHERS[workflow]
        root = tmp_path / "workspace"
        (root / dist_dir).mkdir(parents=True)
        files = _built(package)
        for name, data in files.items():
            (root / dist_dir / name).write_bytes(data)
        params = {"PUBLISH_PACKAGE": package, "PUBLISH_DIST_DIR": dist_dir}
        assert _run(_script(RECORD, tmp_path), root, **params).returncode == 0
        server = stand({_json_path(package): [(200, _release({name: _sha(data) for name, data in files.items()}, package=package))]})
        proc = self._check(_script("index", tmp_path), root, server, **params)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert server.paths() == [_json_path(package)]


def _github_env(tmp_path: Path, server: _Stand, **extra: str) -> dict[str, str]:
    output = tmp_path / "github_output"
    output.touch()
    env = {"GITHUB_API_URL": server.url, "GITHUB_REPOSITORY": REPOSITORY, API_CREDENTIAL_ENV: FAKE_API_CREDENTIAL, "GITHUB_OUTPUT": str(output)}
    env.update(extra)
    return env


def _outputs(tmp_path: Path) -> dict[str, str]:
    text = (tmp_path / "github_output").read_text(encoding="utf-8")
    return dict(line.split("=", 1) for line in text.splitlines() if line)


RUN_ARTIFACTS = f"/repos/{REPOSITORY}/actions/runs/4242/artifacts?name=juniper-recurrence-dist&per_page=100"


class TestReuseStep:
    @staticmethod
    def _env(tmp_path: Path, server: _Stand) -> dict[str, str]:
        return _github_env(tmp_path, server, GITHUB_RUN_ID="4242", GITHUB_RUN_ATTEMPT="2")

    def test_an_unexpired_artifact_from_an_earlier_attempt_is_reused(self, workspace, tmp_path, stand):
        server = stand({RUN_ARTIFACTS: [(200, {"artifacts": [{"id": 7, "name": "juniper-recurrence-dist", "expired": False}]})]})
        proc = _run(_script(REUSE, tmp_path), workspace, **self._env(tmp_path, server))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert _outputs(tmp_path) == {"reuse": "true"}
        path, headers = server.requests[0]
        assert path == RUN_ARTIFACTS
        assert headers.get("Authorization") == f"Bearer {FAKE_API_CREDENTIAL}"

    @pytest.mark.parametrize(
        "artifacts",
        [[], [{"id": 7, "name": "juniper-recurrence-dist", "expired": True}], [{"id": 8, "name": "coverage-report", "expired": False}]],
        ids=["none", "expired", "another-name"],
    )
    def test_without_a_usable_artifact_the_attempt_builds(self, workspace, tmp_path, stand, artifacts):
        server = stand({RUN_ARTIFACTS: [(200, {"artifacts": artifacts})]})
        proc = _run(_script(REUSE, tmp_path), workspace, **self._env(tmp_path, server))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert _outputs(tmp_path) == {}

    def test_an_api_failure_fails_the_job_rather_than_rebuild(self, workspace, tmp_path, stand):
        server = stand({RUN_ARTIFACTS: [(500, {"message": "boom"})]})
        proc = _run(_script(REUSE, tmp_path), workspace, **self._env(tmp_path, server))
        assert proc.returncode != 0
        assert _outputs(tmp_path) == {}


def _runs_path(tag: str, workflow: str = "publish-recurrence-app.yml") -> str:
    return f"/repos/{REPOSITORY}/actions/workflows/{workflow}/runs?event=release&branch={tag}&per_page=30"


def _artifacts_path(run_id: int, package: str = "juniper-recurrence") -> str:
    return f"/repos/{REPOSITORY}/actions/runs/{run_id}/artifacts?name={package}-dist&per_page=100"


class TestFindReleaseRun:
    @staticmethod
    def _env(tmp_path: Path, server: _Stand, tag: str = "juniper-recurrence-v9.9.9", ref_type: str = "tag", workflow: str = "publish-recurrence-app.yml") -> dict[str, str]:
        return _github_env(tmp_path, server, GITHUB_REF_NAME=tag, GITHUB_REF_TYPE=ref_type, GITHUB_REF=f"refs/{'tags' if ref_type == 'tag' else 'heads'}/{tag}", GITHUB_WORKFLOW_REF=f"{REPOSITORY}/.github/workflows/{workflow}@refs/tags/{tag}")

    # This repository tags `<package>-v<version>`; the template also reads a bare `v<version>`, juniper-cascor's app tag.
    @pytest.mark.parametrize("tag", ["juniper-recurrence-v9.9.9", "v9.9.9"])
    def test_the_newest_release_run_that_still_holds_the_artifact_is_found(self, workspace, tmp_path, stand, tag):
        server = stand(
            {
                _runs_path(tag): [(200, {"workflow_runs": [{"id": 111, "head_branch": tag}, {"id": 333, "head_branch": tag}, {"id": 222, "head_branch": tag}]})],
                _artifacts_path(333): [(200, {"artifacts": [{"id": 1, "name": "juniper-recurrence-dist", "expired": True}]})],
                _artifacts_path(222): [(200, {"artifacts": [{"id": 2, "name": "juniper-recurrence-dist", "expired": False}]})],
                _artifacts_path(111): [(200, {"artifacts": [{"id": 3, "name": "juniper-recurrence-dist", "expired": False}]})],
            }
        )
        proc = _run(_script(FIND, tmp_path), workspace, **self._env(tmp_path, server, tag=tag))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert _outputs(tmp_path) == {"version": "9.9.9", "run-id": "222"}
        assert _artifacts_path(111) not in server.paths()

    @pytest.mark.parametrize("workflow", sorted(PUBLISHERS))
    def test_each_publishers_tag_names_its_version(self, workspace, tmp_path, stand, workflow):
        package = PUBLISHERS[workflow][0]
        tag = f"{package}-v1.2.3"
        server = stand({_runs_path(tag, workflow): [(200, {"workflow_runs": []})]})
        proc = _run(_script(FIND, tmp_path), workspace, PUBLISH_PACKAGE=package, **self._env(tmp_path, server, tag=tag, workflow=workflow))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert _outputs(tmp_path) == {"version": "1.2.3", "run-id": ""}
        assert "::warning::" in proc.stdout

    @pytest.mark.parametrize(
        ("package", "tag", "ref_type"),
        [
            ("juniper-recurrence", "v9.9.9", "branch"),
            ("juniper-recurrence", "juniper-recurrence-v9.9.9", "branch"),
            ("juniper-recurrence", "main", "branch"),
            ("juniper-recurrence", "juniper-recurrence-model-v1.2.3", "tag"),
            ("juniper-recurrence", "juniper-recurrence-client-v1.2.3", "tag"),
            ("juniper-recurrence-model", "juniper-recurrence-v1.2.3", "tag"),
            ("juniper-recurrence-client", "juniper-recurrence-model-v1.2.3", "tag"),
        ],
        ids=["branch-named-like-a-v-tag", "branch-named-like-this-packages-tag", "branch", "the-models-tag-for-the-app", "the-clients-tag-for-the-app", "the-apps-tag-for-the-model", "the-models-tag-for-the-client"],
    )
    def test_it_refuses_anything_but_a_release_tag_of_this_package(self, workspace, tmp_path, stand, package, tag, ref_type):
        server = stand({})
        proc = _run(_script(FIND, tmp_path), workspace, PUBLISH_PACKAGE=package, **self._env(tmp_path, server, tag=tag, ref_type=ref_type))
        assert proc.returncode == 1
        assert "::error::" in proc.stdout
        assert server.requests == []
        assert _outputs(tmp_path) == {}

    def test_an_api_failure_fails_the_job_rather_than_skip_the_check(self, workspace, tmp_path, stand):
        server = stand({_runs_path("juniper-recurrence-v9.9.9"): [(502, {"message": "bad gateway"})]})
        proc = _run(_script(FIND, tmp_path), workspace, **self._env(tmp_path, server))
        assert proc.returncode != 0
        assert _outputs(tmp_path) == {}


class TestFetchTestPyPIFiles:
    @staticmethod
    def _env(server: _Stand, sums: Path) -> dict[str, str]:
        return {"INDEX_URL": server.url, "VERSION": "9.9.9", "RELEASE_RUN_SUMS": str(sums)}

    @staticmethod
    def _routes(server_url: str, files: dict[str, bytes], digests: dict[str, str] | None = None, package: str = "juniper-recurrence") -> dict[str, list[tuple[int, object]]]:
        listed = digests or {name: _sha(data) for name, data in files.items()}
        routes: dict[str, list[tuple[int, object]]] = {_json_path(package): [(200, _release(listed, base=server_url, package=package))]}
        for name, data in files.items():
            routes[f"/files/{name}"] = [(200, data)]
        return routes

    def _serve(self, stand, files: dict[str, bytes], digests: dict[str, str] | None = None, package: str = "juniper-recurrence") -> _Stand:
        server = stand({})
        server.routes.update(self._routes(server.url, files, digests, package))
        return server

    def test_it_downloads_testpypis_files_and_records_them(self, tmp_path, stand):
        root = tmp_path / "workspace"
        root.mkdir()
        server = self._serve(stand, BUILT)
        proc = _run(_script(FETCH, tmp_path), root, **self._env(server, tmp_path / "absent" / "SHA256SUMS"))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert {p.name: p.read_bytes() for p in (root / "dist").iterdir()} == BUILT
        assert (root / "SHA256SUMS").read_text(encoding="utf-8") == "".join(f"{_sha(data)}  {name}\n" for name, data in sorted(BUILT.items()))
        assert "::warning::" in proc.stdout

    def test_a_file_whose_bytes_differ_from_the_index_json_is_refused(self, tmp_path, stand):
        root = tmp_path / "workspace"
        root.mkdir()
        server = self._serve(stand, {**BUILT, WHEEL: b"tampered bytes"}, digests=SERVED)
        proc = _run(_script(FETCH, tmp_path), root, **self._env(server, tmp_path / "absent" / "SHA256SUMS"))
        assert proc.returncode == 1
        assert "hashes to" in proc.stdout
        assert not (root / "SHA256SUMS").exists()

    def test_the_release_runs_sha256sums_is_checked_while_it_exists(self, tmp_path, stand):
        root = tmp_path / "workspace"
        root.mkdir()
        sums = tmp_path / "release-run" / "SHA256SUMS"
        sums.parent.mkdir()
        sums.write_text("".join(f"{_sha(data)}  {name}\n" for name, data in sorted(BUILT.items())), encoding="utf-8")
        server = self._serve(stand, BUILT)
        proc = _run(_script(FETCH, tmp_path), root, **self._env(server, sums))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "::warning::" not in proc.stdout

    def test_testpypi_files_the_release_run_did_not_build_are_refused(self, tmp_path, stand):
        root = tmp_path / "workspace"
        root.mkdir()
        sums = tmp_path / "release-run" / "SHA256SUMS"
        sums.parent.mkdir()
        sums.write_text(f"{_sha(b'the first build')}  {WHEEL}\n{_sha(BUILT[SDIST])}  {SDIST}\n", encoding="utf-8")
        server = self._serve(stand, BUILT)
        proc = _run(_script(FETCH, tmp_path), root, **self._env(server, sums))
        assert proc.returncode == 1
        assert "disagree" in proc.stdout
        assert not [p for p, _ in server.requests if p.startswith("/files/")]

    @pytest.mark.parametrize(
        "listed",
        [{}, {**SERVED, "juniper_recurrence_model-9.9.9-py3-none-any.whl": "d" * 64}, {**SERVED, "juniper_recurrence-9.9.8.tar.gz": "d" * 64}],
        ids=["no-files", "a-siblings-file", "another-version"],
    )
    def test_an_index_listing_it_cannot_vouch_for_is_refused(self, tmp_path, stand, listed):
        root = tmp_path / "workspace"
        root.mkdir()
        server = stand({})
        server.routes[JSON_PATH] = [(200, _release(listed, base=server.url))]
        proc = _run(_script(FETCH, tmp_path), root, **self._env(server, tmp_path / "absent" / "SHA256SUMS"))
        assert proc.returncode == 1
        assert "::error::" in proc.stdout
        assert not [p for p, _ in server.requests if p.startswith("/files/")]

    def test_it_refuses_a_dist_directory_that_already_holds_files(self, workspace, tmp_path, stand):
        server = self._serve(stand, BUILT)
        proc = _run(_script(FETCH, tmp_path), workspace, **self._env(server, tmp_path / "absent" / "SHA256SUMS"))
        assert proc.returncode == 1
        assert server.requests == []

    @pytest.mark.parametrize("workflow", sorted(PUBLISHERS))
    def test_it_writes_this_repositorys_nested_dist_directory(self, tmp_path, stand, workflow):
        package, dist_dir, _, _ = PUBLISHERS[workflow]
        root = tmp_path / "workspace"
        root.mkdir()
        files = _built(package)
        server = self._serve(stand, files, package=package)
        proc = _run(_script(FETCH, tmp_path), root, PUBLISH_PACKAGE=package, PUBLISH_DIST_DIR=dist_dir, **self._env(server, tmp_path / "absent" / "SHA256SUMS"))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert {p.name: p.read_bytes() for p in (root / dist_dir).iterdir()} == files
        assert (root / "SHA256SUMS").read_text(encoding="utf-8") == "".join(f"{_sha(data)}  {name}\n" for name, data in sorted(files.items()))
        # The verify step that follows in the republish job accepts what this one wrote.
        proc = _run(_script(VERIFY_DIST, tmp_path), root, PUBLISH_PACKAGE=package, PUBLISH_DIST_DIR=dist_dir)
        assert proc.returncode == 0, proc.stdout + proc.stderr
