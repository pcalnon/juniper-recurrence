"""Pin the one-artifact publish: PyPI takes exactly the files TestPyPI took, and each run checks it by sha256.

The Overtoad's directive of 2026-11-15: "deploy to pypi should send the same artifact that was already sent to testpypi.
an identical hash check, of the artifacts after testpypi deploy and prior to pypi deploy, might also be helpful here to
prevent this from slipping through the cracks again." juniper-cascor's publisher rebuilt the package in its PyPI job,
so PyPI served a second build of its 0.13.0 and 0.14.0 releases. This repository's three publishers already built once
and handed both publish jobs one artifact, and every release so far is byte-identical on the two indexes, but nothing
checked it: both uploads set ``skip-existing``, which silently keeps an earlier file under the same name, the artifact
expired after 7 days while the ``pypi`` gate may wait 30, and "Re-run all jobs" re-ran the build, whose artifact every
later download takes (download-artifact picks the newest of a name). That is how juniper-ml's run 38045867684 published
a second build of juniper-ci-tools 0.9.1 to PyPI.

``publish-recurrence-app.yml``, ``publish-recurrence-client.yml`` and ``publish-recurrence-model.yml`` carry
juniper-cascor's one-artifact template (juniper-cascor#763 and #767, swept by #777). The build writes ``SHA256SUMS``
beside the dist and uploads both as one artifact, kept 90 days; a re-run attempt reuses it rather than build again; each
publish job checks the files against ``SHA256SUMS`` before it uploads; TestPyPI's JSON must list exactly those sha256
after the TestPyPI upload and again before the PyPI one, and PyPI's after; the install check runs in a job of its own,
which cannot mint a credential, and the PyPI job needs it (ruling 2 of 2026-11-16); only the uploading jobs hold
``id-token: write``; and a dispatch from the release tag with ``republish_from_testpypi`` runs the republish job, the
first rung of the recovery ladder, which copies TestPyPI's files for the version to PyPI and builds nothing, and only
once the tag's newest release run passed its TestPyPI jobs in their latest attempt (ruling 1 of 2026-11-16). The
template steps read only ``PUBLISH_PACKAGE``, ``PUBLISH_DIST_DIR`` and ``PUBLISH_TAG_PREFIX`` from the workflow ``env``,
a step's ``INDEX_URL``, and the runner's own variables.

The static cases pin the wiring: one build, in a job without an environment or ``id-token``; the job chain and the
job names juniper-ml's release-train ceremony reads; per-job permissions; each job's keys, runner and exact steps, with
no ``if:`` but the listed ones and no ``continue-on-error``; each build step's commands; the reuse step first in the
build job and guarding every later step; each upload preceded by the dist check; TestPyPI's JSON checked after the
TestPyPI upload and again before the PyPI one, and PyPI's after; no ``skip-existing`` and no ``attestations: false``;
an artifact kept 90 days, past the 30-day ``pypi`` gate; the install check's job, its exact commands, run from outside
any checkout; the republish job's trigger, gate, steps and name; what each workflow kept (its triggers, concurrency,
tag guard and tag prefix, build commands and install check); every copy of a template step byte-identical across the
three, and each action pinned to one ref; and the app lane running this file on a pull request that edits only a
publish workflow. The rehearsals run each template step's own Python, extracted from ``publish-recurrence-app.yml``, in
a scratch workspace, against a stand-in index and a stand-in GitHub API on 127.0.0.1: juniper-cascor's rehearsals,
which use a flat ``dist``, the republish job's refusal among them, and this repository's own ``<package>/dist`` layout
for each of the three publishers. Each install check's own shell runs too, against stand-in ``pip``, ``python`` and
``sleep`` commands.

Ported from juniper-cascor's ``src/tests/unit/test_publish_one_artifact.py`` (``main`` at ``5e3d9801``, whose publishers
are #777's merge ``f5ed3017``). The copies' identity with juniper-cascor's template is checked across repositories by
juniper-ml's ``util/ad-hoc/2026-11-23_publish_port_template_identity.py``; this file pins them identical to one another.

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
#: workflow -> (PUBLISH_PACKAGE, PUBLISH_DIST_DIR, TestPyPI job, PyPI job); each also has a ``build``, an install check and a ``republish`` job.
PUBLISHERS = {
    "publish-recurrence-app.yml": ("juniper-recurrence", "juniper-recurrence/dist", "publish-testpypi", "publish-pypi"),
    "publish-recurrence-client.yml": ("juniper-recurrence-client", "juniper-recurrence-client/dist", "publish-testpypi", "publish-pypi"),
    "publish-recurrence-model.yml": ("juniper-recurrence-model", "juniper-recurrence-model/dist", "publish-testpypi", "publish-pypi"),
}
#: workflow -> (top-level permissions, the PyPI job's). The PyPI job checks nothing out, so it holds ``id-token: write``
#: alone in all three. The build, TestPyPI, install-check and republish jobs hold the same in all three.
PERMISSIONS = {
    "publish-recurrence-app.yml": ({"contents": "read"}, {"id-token": "write"}),
    "publish-recurrence-client.yml": ({"contents": "read"}, {"id-token": "write"}),
    "publish-recurrence-model.yml": ({"contents": "read"}, {"id-token": "write"}),
}
#: workflow -> PUBLISH_TAG_PREFIX, what the package's release tags start with (the version is the rest). Each is a
#: sub-package of this repository, so each prefix is ``<package>-v``.
TAG_PREFIXES = {"publish-recurrence-app.yml": "juniper-recurrence-v", "publish-recurrence-client.yml": "juniper-recurrence-client-v", "publish-recurrence-model.yml": "juniper-recurrence-model-v"}
#: workflow -> its ``name:``. juniper-ml's release-train monitor drops every run of a workflow whose name does not end in
#: "to PyPI" (``PYPI_PUBLISHER_NAME_SUFFIX`` in ``util/release_train/ceremony.py``), so a renamed publisher fails quietly.
WORKFLOW_NAMES = {"publish-recurrence-app.yml": "Publish juniper-recurrence to PyPI", "publish-recurrence-client.yml": "Publish juniper-recurrence-client to PyPI", "publish-recurrence-model.yml": "Publish juniper-recurrence-model to PyPI"}
#: workflow -> (its concurrency group, the module its TestPyPI install check imports): what each publisher kept.
KEPT = {
    "publish-recurrence-app.yml": ("publish-recurrence-app-${{ github.ref_name }}", "juniper_recurrence"),
    "publish-recurrence-client.yml": ("publish-recurrence-client-${{ github.ref_name }}", "juniper_recurrence_client"),
    "publish-recurrence-model.yml": ("publish-recurrence-model-${{ github.ref_name }}", "juniper_recurrence_model"),
}
#: Ruling 2 of 2026-11-16: the install check runs in a job of its own, which holds no ``id-token``. Its name holds
#: "TestPyPI", so the release-train ceremony counts it with the TestPyPI job and reads its failure as HALT_TESTPYPI.
INSTALL = "verify-testpypi-install"
INSTALL_NAME = "Verify the TestPyPI install"
#: The install check's step name, kept from the TestPyPI job it moved out of (the same in all three).
INSTALL_CHECK = "Verify install from TestPyPI"
#: workflow -> what its install check installs by name from the default index before the import: the runtime
#: dependencies the import needs, which ``--no-deps`` left out. The app's top-level import needs none.
RUNTIME_DEPS = {
    "publish-recurrence-app.yml": [],
    "publish-recurrence-client.yml": ['pip install "requests>=2.28.0" "urllib3>=2.0.0"'],
    "publish-recurrence-model.yml": ['pip install "numpy>=1.24" "juniper-model-core>=0.2.0,<0.4.0"'],
}
#: The workflow the rehearsals extract the template steps from; TestTemplate pins the other two copies equal to it.
TEMPLATE_SOURCE = "publish-recurrence-app.yml"
#: The build jobs' release-convention gate. It runs on ``push`` alone, an event no publisher subscribes to, so it never
#: runs: it is the one build step exempt from the reuse condition, and it is exempt by its name.
RELEASE_GATE = "Require a GitHub Release for this tag"
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
INDEX_CHECKS = {TESTPYPI_AFTER: "https://test.pypi.org", TESTPYPI_BEFORE: "https://test.pypi.org", PYPI_AFTER: "https://pypi.org"}
#: Everything a template step may read from its environment: the three parameters, a step's own inputs, the runner's variables.
TEMPLATE_ENV = {
    "PUBLISH_PACKAGE",
    "PUBLISH_DIST_DIR",
    "PUBLISH_TAG_PREFIX",
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
#: The artifact's retention, as the docs state it: past the gate's 30 days and the 30-day re-run window.
RETENTION_DAYS = 90
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
#: The actions the publishers use, by name: a step is compared with its ``@ref`` dropped, so a Dependabot bump of a pin
#: changes nothing here.
CHECKOUT = "actions/checkout"
SETUP_PYTHON = "actions/setup-python"
UPLOAD_ACTION = "actions/upload-artifact"
DOWNLOAD_ACTION = "actions/download-artifact"
PYPI_PUBLISH = "pypa/gh-action-pypi-publish"
#: Each job's steps, in order, as (name, action). Nothing may be added, dropped or moved unnoticed: neither a rebuild
#: slipped into a publish job nor a step that writes into the dist directory before ``Record SHA256SUMS``.
BUILD_STEPS = [(REUSE, None), (None, CHECKOUT), (RELEASE_GATE, None), (None, SETUP_PYTHON), ("Install build tools", None), ("Build sdist + wheel", None), ("Validate distribution metadata", None), (RECORD, None), (UPLOAD, UPLOAD_ACTION)]
TESTPYPI_STEPS = [(DOWNLOAD, DOWNLOAD_ACTION), (VERIFY_DIST, None), ("Publish to TestPyPI", PYPI_PUBLISH), (TESTPYPI_AFTER, None)]
#: The install-check job: the sparse checkout the package's version is read from, then the check.
INSTALL_JOB_STEPS = [(None, CHECKOUT), (INSTALL_CHECK, None)]
PYPI_STEPS = [(DOWNLOAD, DOWNLOAD_ACTION), (VERIFY_DIST, None), (TESTPYPI_BEFORE, None), ("Publish to PyPI", PYPI_PUBLISH), (PYPI_AFTER, None)]
REPUBLISH_STEPS = [(FIND, None), (FETCH_ORIGINAL, DOWNLOAD_ACTION), (FETCH, None), (VERIFY_DIST, None), ("Publish to PyPI", PYPI_PUBLISH), (PYPI_AFTER, None)]
#: The exact inputs of each publisher's TestPyPI and PyPI upload: no ``skip-existing`` (nor the action's older spelling
#: ``skip_existing``), no ``attestations: false``, and the downloaded dist directory as the only source.
UPLOAD_INPUTS = ({"repository-url": "https://test.pypi.org/legacy/", "packages-dir": DIST_INPUT, "verbose": True}, {"packages-dir": DIST_INPUT, "verbose": True})
#: Every key the workflow and each of its jobs carries. An ``env`` on a job (``PIP_EXTRA_INDEX_URL``,
#: ``HASH_CHECK_ATTEMPTS``), a container, a matrix or another runner would change what a check runs, or against what.
WORKFLOW_KEYS = {"name", "on", "permissions", "concurrency", "env", "jobs"}
BUILD_JOB_KEYS = {"name", "if", "runs-on", "permissions", "defaults", "steps"}
PUBLISH_JOB_KEYS = {"name", "needs", "runs-on", "environment", "permissions", "steps"}
INSTALL_JOB_KEYS = {"name", "needs", "runs-on", "permissions", "steps"}
REPUBLISH_JOB_KEYS = {"name", "if", "runs-on", "environment", "permissions", "steps"}
#: The keys a repository may add, none of which changes what a check runs. The workflow's: ``run-name``, which names the
#: run. A job's: ``timeout-minutes``, which stops it sooner, and a job stopped so fails, so nothing after it runs; the
#: republish job takes one only in all three files at once, since it stays identical in every publisher. No job takes
#: a ``concurrency`` group: by default (``queue: single``) GitHub keeps one pending job per group and cancels the older
#: one when another queues, so a group on the PyPI job could stop a run between its two uploads.
OPTIONAL_WORKFLOW_KEYS = {"run-name"}
OPTIONAL_JOB_KEYS = {"timeout-minutes"}
#: Each build step's command: the build, and the check of what it built.
BUILD_RUNS = {"Install build tools": "pip install --upgrade build twine", "Build sdist + wheel": "python -m build --sdist --wheel", "Validate distribution metadata": "twine check dist/*"}
#: The install checks' bounded retry: this many tries, this many seconds apart.
INSTALL_TRIES = 5
INSTALL_RETRY_SECONDS = 10
LAG_ERROR = "is not installable from TestPyPI after 5 tries (index lag?): Re-run failed jobs; a real failure repeats"


def _json_path(package: str = "juniper-recurrence") -> str:
    return f"/pypi/{package}/9.9.9/json"


JSON_PATH = _json_path()


def _version_file(workflow: str) -> str:
    """The file the install check reads the package's version from, relative to the workspace."""
    return f"{PUBLISHERS[workflow][0]}/{KEPT[workflow][1]}/_version.py"


def _retried_install(package: str, version: str) -> list[str]:
    """The install every check runs: five tries, 10 s apart, and after the fifth failure an error that names index lag,
    rather than an import that fails as if the package were broken."""
    return [
        "for attempt in 1 2 3 4 5; do",
        "if pip install \\",
        "--no-deps \\",
        "--index-url https://test.pypi.org/simple/ \\",
        f'"{package}==${{{version}}}"; then',
        "break",
        "fi",
        'if [ "${attempt}" -eq 5 ]; then',
        f'echo "::error::{package} ${{{version}}} {LAG_ERROR}"',
        "exit 1",
        "fi",
        'echo "TestPyPI index lag — retrying in 10s (attempt ${attempt}/5)"',
        "sleep 10",
        "done",
    ]


def _install_commands(workflow: str) -> list[str]:
    """WORKFLOW's install step's commands, exactly, comment and blank lines dropped: no ``exit 0``, ``trap``, earlier
    install or second version can be added and none dropped, while a comment may change freely. The version is read from
    the sparse checkout by its path under the workspace, after the step has left for the runner's temporary directory."""
    package = PUBLISHERS[workflow][0]
    module = KEPT[workflow][1]
    return [
        "set -euo pipefail",
        'cd "${RUNNER_TEMP}"',
        f"""version=$(grep -E '^__version__' "${{GITHUB_WORKSPACE}}/{_version_file(workflow)}" | head -1 | sed -E 's/.*"([^"]+)".*/\\1/')""",
        f'echo "Verifying {package}==${{version}}"',
        *_retried_install(package, "version"),
        *RUNTIME_DEPS[workflow],
        f"""python -c "import {module}; print('TestPyPI install OK', {module}.__version__)\"""",
    ]


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


def _steps(job: dict) -> list[tuple[str | None, str | None]]:
    """Each step as (its name, the action it uses with the ``@ref`` dropped)."""
    return [(s.get("name"), str(s["uses"]).split("@", 1)[0] if s.get("uses") else None) for s in job.get("steps") or []]


def _by_action(value):
    """VALUE with every ``uses:`` reduced to the action's name, the ``@ref`` dropped."""
    if isinstance(value, dict):
        return {k: (str(v).split("@", 1)[0] if k == "uses" else _by_action(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_by_action(v) for v in value]
    return value


def _allowed_if(job_id: str, index: int, step: dict) -> str | None:
    """The one ``if:`` a step may carry: the reuse step's own, the reuse condition on every later build step but the
    push-only release gate (exempt by its name), and the republish job's download of the release run's artifact. Any
    other ``if:`` could skip a check."""
    if job_id == "build":
        if index == 0:
            return "github.run_attempt != '1'"
        return "github.event_name == 'push'" if step.get("name") == RELEASE_GATE else REUSE_IF
    if job_id == REPUBLISH and step.get("name") == FETCH_ORIGINAL:
        return "steps.original.outputs.run-id != ''"
    return None


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
    child = {"PATH": os.environ.get("PATH", ""), "GITHUB_WORKSPACE": str(workspace), "PUBLISH_PACKAGE": "juniper-recurrence", "PUBLISH_DIST_DIR": "dist", "PUBLISH_TAG_PREFIX": "juniper-recurrence-v"}
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
        wf = _workflow(workflow)
        assert wf["env"] == {"PUBLISH_PACKAGE": package, "PUBLISH_DIST_DIR": dist_dir, "PUBLISH_TAG_PREFIX": TAG_PREFIXES[workflow]}
        # A sub-package publisher's tags are <package>-v<version>.
        assert TAG_PREFIXES[workflow] == f"{package}-v"
        # A job-level if: cannot read env, so the build job's tag guard repeats the prefix, for a release and for a plain
        # dispatch: the two must agree.
        guard = wf["jobs"]["build"]["if"]
        assert f"startsWith(github.event.release.tag_name, '{TAG_PREFIXES[workflow]}')" in guard
        assert f"startsWith(github.ref_name, '{TAG_PREFIXES[workflow]}')" in guard

    def test_the_workflow_name_ends_in_to_pypi_as_the_release_train_monitor_requires(self, workflow):
        name = _workflow(workflow)["name"]
        assert name == WORKFLOW_NAMES[workflow]
        assert name.lower().endswith("to pypi")

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
        assert sorted(jobs) == sorted(["build", test_id, INSTALL, prod_id, REPUBLISH])
        assert jobs[test_id]["needs"] == "build"
        assert jobs[INSTALL]["needs"] == test_id
        # PyPI waits for both TestPyPI jobs: the upload with its checks, and the install check.
        assert jobs[prod_id]["needs"] == [test_id, INSTALL]
        # juniper-ml's util/release_train/ceremony.py tells the jobs apart by name: "testpypi" (both TestPyPI jobs) and "pypi".
        assert jobs[test_id]["name"] == "Publish to TestPyPI"
        assert jobs[INSTALL]["name"] == INSTALL_NAME
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
        # The job that can mint runs the template steps and the upload alone: it checks out nothing and installs nothing.
        assert jobs[PUBLISHERS[workflow][2]]["permissions"] == {"id-token": "write"}
        # The install check runs the package TestPyPI serves: it may read the repository, and nothing more.
        assert jobs[INSTALL]["permissions"] == {"contents": "read"}
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
        # runs on `push` alone, an event no publisher subscribes to. It is exempt by its name: an exemption by its `if:`
        # text would let any build step trade the reuse condition for the push condition.
        for step in build["steps"][1:]:
            if step.get("name") == RELEASE_GATE:
                assert step.get("if") == "github.event_name == 'push'"
                continue
            assert step.get("if") == REUSE_IF, f"{step.get('name') or step.get('uses')} would run on a re-run that reuses the artifact"
        for name in (RECORD, UPLOAD):
            assert _step(build, name).get("if") == REUSE_IF

    def test_the_testpypi_job_checks_the_dist_uploads_then_checks_testpypi_before_the_install_check(self, workflow):
        _, _, test_id, _ = PUBLISHERS[workflow]
        jobs = _workflow(workflow)["jobs"]
        job = jobs[test_id]
        uploads = _uploads(job)
        assert len(uploads) == 1
        assert job["steps"][uploads[0]]["with"]["repository-url"] == "https://test.pypi.org/legacy/"
        assert _index(job, DOWNLOAD) < _index(job, VERIFY_DIST) < uploads[0] < _index(job, TESTPYPI_AFTER)
        # The job ends at the sha256 check that vouches for what the install check installs, in the job that needs it.
        assert _index(job, TESTPYPI_AFTER) == len(job["steps"]) - 1
        assert not [s for s in job["steps"] if "pip install" in str(s.get("run") or "")], "the job that can mint runs an install check"
        installs = [s for s in jobs[INSTALL]["steps"] if "pip install" in str(s.get("run") or "")]
        assert installs, "the TestPyPI install check is gone"
        assert jobs[INSTALL]["needs"] == test_id, "the install check must come after the sha256 check that vouches for what it installs"
        _assert_no_checkout_between(job, 0, len(job["steps"]))

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
        _, _, test_id, prod_id = PUBLISHERS[workflow]
        jobs = _workflow(workflow)["jobs"]
        for job in jobs.values():
            for i in _uploads(job):
                inputs = job["steps"][i].get("with") or {}
                assert inputs.get("packages-dir") == DIST_INPUT
                # skip-existing would keep an earlier file under the same name, and the install check would test it. The
                # action still reads its older spelling too.
                assert "skip-existing" not in inputs
                assert "skip_existing" not in inputs
                # Attestations stay on, the action's default.
                assert inputs.get("attestations", True) is True
        for job_id, inputs in zip((test_id, prod_id), UPLOAD_INPUTS, strict=True):
            assert [jobs[job_id]["steps"][i].get("with") for i in _uploads(jobs[job_id])] == [inputs]

    def test_one_artifact_carries_the_dist_and_sha256sums_and_outlives_the_pypi_gate(self, workflow):
        _, _, test_id, prod_id = PUBLISHERS[workflow]
        jobs = _workflow(workflow)["jobs"]
        upload = _step(jobs["build"], UPLOAD)["with"]
        assert upload["name"] == ARTIFACT
        assert [p.strip() for p in upload["path"].splitlines() if p.strip()] == [DIST_INPUT, "SHA256SUMS"]
        assert upload["if-no-files-found"] == "error"
        # Exactly the documented 90 days, which outlive the gate's 30 days and the 30-day re-run window.
        assert upload["retention-days"] == RETENTION_DAYS
        assert RETENTION_DAYS > GATE_APPROVAL_DAYS
        assert _index(jobs["build"], RECORD) < _index(jobs["build"], UPLOAD)
        for job_id in (test_id, prod_id):
            assert _step(jobs[job_id], DOWNLOAD)["with"] == {"name": ARTIFACT, "path": "${{ github.workspace }}"}

    def test_each_json_check_reads_its_own_index_and_keeps_the_default_bounds(self, workflow):
        _, _, test_id, prod_id = PUBLISHERS[workflow]
        jobs = _workflow(workflow)["jobs"]
        for job_id, name in ((test_id, TESTPYPI_AFTER), (prod_id, TESTPYPI_BEFORE), (prod_id, PYPI_AFTER)):
            assert _step(jobs[job_id], name)["env"] == {"INDEX_URL": INDEX_CHECKS[name]}

    def test_each_job_runs_exactly_its_listed_steps_in_order(self, workflow):
        _, _, test_id, prod_id = PUBLISHERS[workflow]
        jobs = _workflow(workflow)["jobs"]
        want = {"build": BUILD_STEPS, test_id: TESTPYPI_STEPS, INSTALL: INSTALL_JOB_STEPS, prod_id: PYPI_STEPS, REPUBLISH: REPUBLISH_STEPS}
        assert {job_id: _steps(job) for job_id, job in jobs.items()} == want

    def test_no_step_or_job_can_switch_a_check_off(self, workflow):
        for job_id, job in _workflow(workflow)["jobs"].items():
            # A job or step that continues on error is a check that no longer blocks, and an if: can skip one. The build
            # and republish jobs' own if: are pinned to their exact text elsewhere; no other job may carry one (`always()`
            # on the PyPI job would publish after a failed install check).
            assert "continue-on-error" not in job, f"{job_id} would not fail the run"
            if job_id not in ("build", REPUBLISH):
                assert "if" not in job, f"{job_id} carries if: {job.get('if')!r}"
            for index, step in enumerate(job.get("steps") or []):
                label = f"{job_id}: {step.get('name') or step.get('uses')}"
                assert "continue-on-error" not in step, f"{label} would not fail its job"
                assert step.get("if") == _allowed_if(job_id, index, step), f"{label} carries if: {step.get('if')!r}"

    def test_every_job_carries_only_its_listed_keys_and_runs_on_a_hosted_runner(self, workflow):
        # A job-level env (PIP_EXTRA_INDEX_URL, or HASH_CHECK_ATTEMPTS over a JSON check's bounds), a container, a matrix
        # or a self-hosted runner would change what a check runs, or against what. The workflow's run-name and a job's
        # timeout-minutes change no check, so a repository may add them (OPTIONAL_*_KEYS); the republish job takes a
        # timeout only in all three files, since it stays identical in every publisher. A job's own concurrency group is
        # refused: by default (queue: single) it cancels the group's pending job when another queues. The workflow's
        # concurrency group, which never cancels a run in progress, is pinned in TestUnchanged.
        wf = _workflow(workflow)
        _, _, test_id, prod_id = PUBLISHERS[workflow]
        assert WORKFLOW_KEYS <= set(wf) <= WORKFLOW_KEYS | OPTIONAL_WORKFLOW_KEYS, f"workflow keys {sorted(wf)}"
        want = {"build": BUILD_JOB_KEYS, test_id: PUBLISH_JOB_KEYS, INSTALL: INSTALL_JOB_KEYS, prod_id: PUBLISH_JOB_KEYS, REPUBLISH: REPUBLISH_JOB_KEYS}
        assert {job_id: set(job) - OPTIONAL_JOB_KEYS for job_id, job in wf["jobs"].items()} == want
        assert {job_id: job["runs-on"] for job_id, job in wf["jobs"].items()} == dict.fromkeys(want, "ubuntu-latest")

    def test_no_env_points_pip_elsewhere(self, workflow):
        # PIP_INDEX_URL or PIP_EXTRA_INDEX_URL in any env would send the install check, or the build tools, to another index.
        wf = _workflow(workflow)
        envs = [wf.get("env") or {}] + [job.get("env") or {} for job in wf["jobs"].values()] + [step.get("env") or {} for job in wf["jobs"].values() for step in job.get("steps") or []]
        assert not [key for env in envs for key in env if str(key).upper().startswith("PIP_")]

    def test_the_build_steps_run_exactly_their_commands(self, workflow):
        # `|| true` after twine check, or a build without its sdist, would ship what the build job no longer checks.
        build = _workflow(workflow)["jobs"]["build"]
        assert {name: str(_step(build, name)["run"]).strip() for name in BUILD_RUNS} == BUILD_RUNS


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Ruling 2 of 2026-11-16: the install check runs in a job of its own, which cannot mint a credential
# ─────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("workflow", sorted(PUBLISHERS))
class TestInstallCheckJob:
    def test_it_runs_after_the_testpypi_job_with_contents_read_alone_and_pypi_needs_it(self, workflow):
        _, _, test_id, prod_id = PUBLISHERS[workflow]
        jobs = _workflow(workflow)["jobs"]
        job = jobs[INSTALL]
        assert job["name"] == INSTALL_NAME
        # juniper-ml's release-train ceremony counts every job whose name holds "testpypi" as a TestPyPI job: a failure
        # here reads HALT_TESTPYPI, and the train stops.
        assert "testpypi" in job["name"].lower()
        assert job["needs"] == test_id
        assert job["runs-on"] == "ubuntu-latest"
        assert job["permissions"] == {"contents": "read"}
        assert "environment" not in job
        assert "if" not in job
        assert not _uploads(job)
        assert jobs[prod_id]["needs"] == [test_id, INSTALL]

    def test_the_testpypi_job_keeps_only_the_template_steps_and_the_upload(self, workflow):
        _, _, test_id, _ = PUBLISHERS[workflow]
        job = _workflow(workflow)["jobs"][test_id]
        assert job["permissions"] == {"id-token": "write"}
        assert _steps(job) == TESTPYPI_STEPS
        assert not [s for s in job["steps"] if str(s.get("uses") or "").startswith(f"{CHECKOUT}@")]

    def test_it_installs_and_imports_from_outside_any_checkout(self, workflow):
        job = _workflow(workflow)["jobs"][INSTALL]
        assert "defaults" not in job
        step = _step(job, INSTALL_CHECK)
        assert "working-directory" not in step
        commands = [line.strip() for line in step["run"].splitlines() if line.strip() and not line.strip().startswith("#")]
        if commands[0] == "set -euo pipefail":
            commands = commands[1:]
        # `python -c` puts its working directory first on sys.path: an import run inside a checkout could pass without
        # the wheel. So the first command leaves for the runner's temporary directory, and no command moves elsewhere.
        assert commands[0] == 'cd "${RUNNER_TEMP}"'
        assert [c for c in commands if c.split()[0] in ("cd", "pushd", "popd")] == ['cd "${RUNNER_TEMP}"']
        assert [c for c in commands if "pip install" in c]
        assert [c for c in commands if c.startswith("python -c ")]
        # The version is read from a sparse checkout of the package's version file, and nothing runs from it. It keeps no
        # credential in the workspace while the package TestPyPI serves runs.
        checkouts = [s.get("with") for s in job["steps"] if str(s.get("uses") or "").startswith(f"{CHECKOUT}@")]
        assert checkouts == [{"sparse-checkout": f"{_version_file(workflow)}\n", "persist-credentials": False}]

    def test_the_install_step_runs_exactly_its_commands(self, workflow):
        # Exactly these commands, in this order, comment and blank lines dropped: no `exit 0`, trap, earlier install or
        # second version can be added, and none dropped, while a comment may change freely. The step carries its name and
        # script alone: no expression is pasted into the script.
        step = _step(_workflow(workflow)["jobs"][INSTALL], INSTALL_CHECK)
        assert [line.strip() for line in step["run"].splitlines() if line.strip() and not line.strip().startswith("#")] == _install_commands(workflow)
        assert set(step) == {"name", "run"}
        assert "${{" not in step["run"]

    def test_the_install_check_cannot_be_switched_off(self, workflow):
        step = _step(_workflow(workflow)["jobs"][INSTALL], INSTALL_CHECK)
        commands = "\n".join(line for line in step["run"].splitlines() if not line.strip().startswith("#"))
        # A failed install or import must fail the job: no `|| true` (nor any `||`), no `set +e`, the runner's default
        # shell, and neither continue-on-error nor if:.
        assert "||" not in commands
        assert "set +e" not in commands
        assert "shell" not in step
        assert "continue-on-error" not in step
        assert "if" not in step
        # TestPyPI alone, with no pypi.org fallback that a same-name squatter could answer (juniper-ml#384).
        assert "--index-url https://test.pypi.org/simple/" in commands
        assert "--extra-index-url" not in commands


def _run_install(workflow: str, tmp_path: Path, pip_failures: int, python_status: int = 0, decoy: str | None = None, checked_out: bool = True) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    """Run WORKFLOW's install step as the runner runs a ``run:`` without a ``shell:`` (``bash -e {0}``), against stand-in
    ``pip``, ``python`` and ``sleep`` commands that log each call (the command, its directory, its arguments): ``pip``
    fails its first PIP_FAILURES calls, and ``python`` exits with PYTHON_STATUS. The workspace's version file says 1.2.3,
    unless CHECKED_OUT is false, when the sparse checkout brought no such file; DECOY, when given, is the version a copy
    of that file in the runner's temporary directory says."""
    bin_dir, runner_temp, workspace = tmp_path / "bin", tmp_path / "runner-temp", tmp_path / "workspace"
    for directory in (bin_dir, runner_temp, (workspace / _version_file(workflow)).parent):
        directory.mkdir(parents=True)
    if checked_out:
        (workspace / _version_file(workflow)).write_text('"""Stand-in."""\n\n__version__ = "1.2.3"\n', encoding="utf-8")
    if decoy is not None:
        (runner_temp / _version_file(workflow)).parent.mkdir(parents=True)
        (runner_temp / _version_file(workflow)).write_text(f'__version__ = "{decoy}"\n', encoding="utf-8")
    log = tmp_path / "calls.log"
    stubs = {"pip": 'test "$(grep -c "^pip" "$STUB_LOG")" -gt "$STUB_PIP_FAILURES"\n', "python": 'exit "$STUB_PYTHON_STATUS"\n', "sleep": ""}
    for name, body in stubs.items():
        stub = bin_dir / name
        stub.write_text(f'#!/bin/sh\nprintf "%s\\t%s\\t%s\\n" "{name}" "$(pwd -P)" "$*" >> "$STUB_LOG"\n{body}', encoding="utf-8")
        stub.chmod(0o700)
    script = tmp_path / "install.sh"
    script.write_text(_step(_workflow(workflow)["jobs"][INSTALL], INSTALL_CHECK)["run"], encoding="utf-8")
    bash = shutil.which("bash")
    assert bash, "the install checks run under bash"
    env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}", "RUNNER_TEMP": str(runner_temp), "GITHUB_WORKSPACE": str(workspace), "STUB_LOG": str(log), "STUB_PIP_FAILURES": str(pip_failures), "STUB_PYTHON_STATUS": str(python_status)}
    proc = subprocess.run([bash, "-e", str(script)], cwd=workspace, env=env, capture_output=True, encoding="utf-8", timeout=60, check=False)  # nosec B603 - fixed argv: bash and the workflow's own extracted step
    calls = [line.split("\t", 2) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    return proc, calls


def _dependency_installs(workflow: str) -> list[str]:
    """The stand-in pip's log of WORKFLOW's by-name runtime installs: the arguments, unquoted, as ``$*`` joins them."""
    return [command.replace('"', "")[len("pip ") :] for command in RUNTIME_DEPS[workflow]]


class TestInstallCheckScript:
    """Each install step's own shell, run as the runner runs it, against stand-in ``pip``, ``python`` and ``sleep``."""

    @pytest.mark.parametrize("workflow", sorted(PUBLISHERS))
    def test_index_lag_that_outlasts_every_try_fails_naming_the_lag_and_never_imports(self, workflow, tmp_path):
        proc, calls = _run_install(workflow, tmp_path, pip_failures=INSTALL_TRIES)
        assert proc.returncode == 1
        assert f"::error::{PUBLISHERS[workflow][0]} 1.2.3 {LAG_ERROR}" in proc.stdout
        # Neither the runtime dependencies nor the import are tried once the package did not install.
        assert [call[0] for call in calls if call[0] != "sleep"] == ["pip"] * INSTALL_TRIES
        # A wait between tries and none after the last.
        assert [call[2] for call in calls if call[0] == "sleep"] == [str(INSTALL_RETRY_SECONDS)] * (INSTALL_TRIES - 1)

    @pytest.mark.parametrize("workflow", sorted(PUBLISHERS))
    def test_a_try_that_installs_ends_the_retries_and_the_check_runs_from_runner_temp(self, workflow, tmp_path):
        package = PUBLISHERS[workflow][0]
        proc, calls = _run_install(workflow, tmp_path, pip_failures=2)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert [call[2] for call in calls if call[0] == "pip"] == [f"install --no-deps --index-url https://test.pypi.org/simple/ {package}==1.2.3"] * 3 + _dependency_installs(workflow)
        assert [call[0] for call in calls][-1] == "python"
        assert {call[1] for call in calls} == {str((tmp_path / "runner-temp").resolve())}
        assert "::error::" not in proc.stdout

    @pytest.mark.parametrize("workflow", sorted(PUBLISHERS))
    def test_a_check_that_fails_after_the_install_fails_the_step(self, workflow, tmp_path):
        proc, calls = _run_install(workflow, tmp_path, pip_failures=0, python_status=1)
        assert proc.returncode != 0
        assert [call[0] for call in calls if call[0] != "sleep"] == ["pip"] * (1 + len(RUNTIME_DEPS[workflow])) + ["python"]

    @pytest.mark.parametrize("workflow", sorted(PUBLISHERS))
    def test_the_version_is_read_from_the_workspace_not_the_runner_temp_directory(self, workflow, tmp_path):
        # The step leaves for the runner's temporary directory before it reads the version, so it names the sparse
        # checkout's file by its path under the workspace: a file of that name in the temporary directory is not it.
        package = PUBLISHERS[workflow][0]
        proc, calls = _run_install(workflow, tmp_path, pip_failures=0, decoy="6.6.6")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert [call[2] for call in calls if call[0] == "pip"][0] == f"install --no-deps --index-url https://test.pypi.org/simple/ {package}==1.2.3"

    @pytest.mark.parametrize("workflow", sorted(PUBLISHERS))
    def test_without_the_version_file_the_step_fails_before_any_install(self, workflow, tmp_path):
        # `set -euo pipefail`: a version file the sparse checkout did not bring fails the step, rather than ask TestPyPI
        # for an empty version.
        proc, calls = _run_install(workflow, tmp_path, pip_failures=0, checked_out=False)
        assert proc.returncode != 0
        assert calls == []


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
        # A concurrency group may queue its runs, never cancel one in progress: that could stop it between two uploads.
        assert wf["concurrency"] == {"group": group, "cancel-in-progress": False}

    def test_it_builds_on_its_own_release_tag_or_a_plain_dispatch(self, workflow):
        package, _, test_id, prod_id = PUBLISHERS[workflow]
        jobs = _workflow(workflow)["jobs"]
        guard = jobs["build"]["if"]
        # A plain dispatch still re-fires the whole publish, from the package's own tag alone (a sibling's tag would
        # publish this package's files under that tag's run); the republish dispatch runs the republish job alone.
        assert guard == f"(github.event_name == 'release' && startsWith(github.event.release.tag_name, '{package}-v')) || (github.event_name == 'workflow_dispatch' && !inputs.republish_from_testpypi && startsWith(github.ref_name, '{package}-v'))"
        # The three packages share the `juniper-recurrence-` prefix, so the `-v` is what keeps a sibling's release out.
        for prefix in re.findall(r"startsWith\(github\.(?:event\.release\.tag_name|ref_name), '([^']+)'\)", guard):
            assert sorted(other for other, _, _, _ in PUBLISHERS.values() if f"{other}-v1.2.3".startswith(prefix)) == [package]
        # The publish jobs follow the build; only it reads the event.
        assert "if" not in jobs[test_id]
        assert "if" not in jobs[INSTALL]
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
        package, _, _, _ = PUBLISHERS[workflow]
        _, module = KEPT[workflow]
        job = _workflow(workflow)["jobs"][INSTALL]
        checkouts = [i for i, s in enumerate(job["steps"]) if str(s.get("uses") or "").startswith("actions/checkout@")]
        assert len(checkouts) == 1
        # The install job needs the TestPyPI job, so the check runs after its JSON check, as it did in that job.
        assert checkouts[0] < _index(job, INSTALL_CHECK)
        assert job["steps"][checkouts[0]]["with"]["sparse-checkout"] == f"{package}/{module}/_version.py\n"
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
        # Compared with each action's @ref dropped: the next case pins each action to one ref in every publisher, so a
        # partial bump fails there, once, rather than here as a template difference.
        jobs = [_by_action(_workflow(workflow)["jobs"][REPUBLISH]) for workflow in PUBLISHERS]
        assert all(job == jobs[0] for job in jobs)

    def test_each_action_is_pinned_to_one_ref_in_every_publisher(self):
        # Dependabot bumps an action in all three files at once; a hand edit that bumps one copy fails here, wherever it is.
        refs: dict[str, set[str]] = {}
        for workflow in PUBLISHERS:
            for job in _workflow(workflow)["jobs"].values():
                for step in job.get("steps") or []:
                    if step.get("uses"):
                        name, _, ref = str(step["uses"]).partition("@")
                        refs.setdefault(name, set()).add(ref)
        assert {name: len(found) for name, found in refs.items()} == dict.fromkeys(refs, 1), refs

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


def _jobs_path(run_id: int, page: int = 1) -> str:
    return f"/repos/{REPOSITORY}/actions/runs/{run_id}/jobs?filter=all&per_page=100&page={page}"


def _job(job_id: int, name: str, conclusion: str | None, attempt: int = 1) -> dict:
    """A job record as ``GET actions/runs/{id}/jobs?filter=all`` lists it: each attempt that ran a job has its own."""
    return {"id": job_id, "name": name, "run_attempt": attempt, "status": "completed" if conclusion else "in_progress", "conclusion": conclusion}


def _attempt(attempt: int, first_id: int, testpypi: str | None = "success", install: str | None = "success", pypi: str | None = "failure") -> list[dict]:
    """One attempt of a release run: the build, both TestPyPI jobs, the PyPI job, and the republish job, which every
    release run skips. The PyPI job fails by default, as when its gate's 30 days ran out: the case rung 1 is for."""
    jobs = (("Build and Validate", "success"), ("Publish to TestPyPI", testpypi), (INSTALL_NAME, install), ("Publish to PyPI", pypi), (REPUBLISH_NAME, "skipped"))
    return [_job(first_id + i, name, conclusion, attempt) for i, (name, conclusion) in enumerate(jobs)]


def _listing(*records: dict, total: int | None = None) -> tuple[int, dict]:
    return (200, {"total_count": len(records) if total is None else total, "jobs": list(records)})


LIVE_ARTIFACT = [{"id": 7, "name": "juniper-recurrence-dist", "expired": False}]
TAG = "juniper-recurrence-v9.9.9"


class TestFindReleaseRun:
    @staticmethod
    def _env(tmp_path: Path, server: _Stand, tag: str = TAG, ref_type: str = "tag", workflow: str = "publish-recurrence-app.yml") -> dict[str, str]:
        return _github_env(tmp_path, server, GITHUB_REF_NAME=tag, GITHUB_REF_TYPE=ref_type, GITHUB_REF=f"refs/{'tags' if ref_type == 'tag' else 'heads'}/{tag}", GITHUB_WORKFLOW_REF=f"{REPOSITORY}/.github/workflows/{workflow}@refs/tags/{tag}")

    @staticmethod
    def _serve(stand, jobs: list[dict], artifacts: list[dict] | None = None, tag: str = TAG, workflow: str = "publish-recurrence-app.yml", package: str = "juniper-recurrence") -> _Stand:
        """The stand-in GitHub API for TAG's one release run, 4242: its job records and its artifacts."""
        return stand(
            {
                _runs_path(tag, workflow): [(200, {"workflow_runs": [{"id": 4242, "head_branch": tag}]})],
                _jobs_path(4242): [_listing(*jobs)],
                _artifacts_path(4242, package): [(200, {"artifacts": LIVE_ARTIFACT if artifacts is None else artifacts})],
            }
        )

    def test_the_newest_release_run_is_the_one_checked_and_its_artifact_is_found(self, workspace, tmp_path, stand):
        runs = [{"id": 111, "head_branch": TAG}, {"id": 333, "head_branch": TAG}, {"id": 222, "head_branch": TAG}, {"id": 444, "head_branch": "juniper-recurrence-v9.9.99"}]
        server = stand({_runs_path(TAG): [(200, {"workflow_runs": runs})], _jobs_path(333): [_listing(*_attempt(1, 100))], _artifacts_path(333): [(200, {"artifacts": LIVE_ARTIFACT})]})
        proc = _run(_script(FIND, tmp_path), workspace, **self._env(tmp_path, server))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert _outputs(tmp_path) == {"version": "9.9.9", "run-id": "333"}
        # The newest run of THIS tag, whether or not it still holds its artifact: never an older run, nor another tag's.
        assert server.paths() == [_runs_path(TAG), _jobs_path(333), _artifacts_path(333)]
        assert all(headers.get("Authorization") == f"Bearer {FAKE_API_CREDENTIAL}" for _, headers in server.requests)

    def test_a_release_run_whose_artifact_expired_is_still_checked_and_testpypi_is_the_only_record(self, workspace, tmp_path, stand):
        server = self._serve(stand, _attempt(1, 100), artifacts=[{"id": 7, "name": "juniper-recurrence-dist", "expired": True}])
        proc = _run(_script(FIND, tmp_path), workspace, **self._env(tmp_path, server))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert _outputs(tmp_path) == {"version": "9.9.9", "run-id": ""}
        assert "::warning::" in proc.stdout

    def test_without_a_release_run_for_the_tag_nothing_is_republished(self, workspace, tmp_path, stand):
        server = stand({_runs_path(TAG): [(200, {"workflow_runs": [{"id": 5, "head_branch": "juniper-recurrence-v9.9.99"}]})]})
        proc = _run(_script(FIND, tmp_path), workspace, **self._env(tmp_path, server))
        assert proc.returncode == 1
        assert "::error::" in proc.stdout
        assert f"no release run for {TAG}" in proc.stdout
        assert server.paths() == [_runs_path(TAG)]
        assert _outputs(tmp_path) == {}

    @pytest.mark.parametrize(
        ("testpypi", "install"),
        [("failure", "skipped"), ("success", "failure"), ("cancelled", "skipped"), ("success", None)],
        ids=["the-upload-failed", "the-install-check-failed", "the-upload-was-cancelled", "the-install-check-still-runs"],
    )
    def test_a_release_run_whose_testpypi_jobs_did_not_both_pass_is_refused(self, workspace, tmp_path, stand, testpypi, install):
        server = self._serve(stand, _attempt(1, 100, testpypi=testpypi, install=install))
        proc = _run(_script(FIND, tmp_path), workspace, **self._env(tmp_path, server))
        assert proc.returncode == 1
        assert "::error::" in proc.stdout
        # Each TestPyPI job is named with its conclusion, or with its status while it has none.
        assert f"  Publish to TestPyPI: {testpypi} (attempt 1)" in proc.stdout
        assert f"  {INSTALL_NAME}: {install or 'in_progress'} (attempt 1)" in proc.stdout
        assert _artifacts_path(4242) not in server.paths()
        assert _outputs(tmp_path) == {}

    def test_a_job_that_re_run_failed_jobs_carried_over_counts_with_its_earlier_success(self, workspace, tmp_path, stand):
        # Attempt 1 passed both TestPyPI jobs and failed at PyPI; "Re-run failed jobs" ran the PyPI job alone again. This
        # listing leaves out the copies GitHub makes of the jobs it carries over (the next case has them): neither TestPyPI
        # job has a record of attempt 2, and each still counts with its earlier success.
        server = self._serve(stand, _attempt(1, 100) + [_job(203, "Publish to PyPI", "failure", attempt=2)])
        proc = _run(_script(FIND, tmp_path), workspace, **self._env(tmp_path, server))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert _outputs(tmp_path) == {"version": "9.9.9", "run-id": "4242"}

    @pytest.mark.parametrize(("install", "passes"), [("success", True), ("failure", False)], ids=["the-re-run-install-check-passed", "it-failed-again"])
    def test_re_run_failed_jobs_copies_the_jobs_it_carries_over_into_the_new_attempt(self, workspace, tmp_path, stand, install, passes):
        # As GitHub lists it (juniper-canopy run 43135960676): attempt 1's install check failed on lag, and "Re-run failed
        # jobs" gave EVERY job a record in attempt 2, the build and the upload as copies (a new id, run_attempt 2, attempt
        # 1's conclusion), the install check and the PyPI job as new runs.
        first = _attempt(1, 100, install="failure", pypi="skipped")
        second = _attempt(2, 200, install=install, pypi="failure" if passes else "skipped")
        server = self._serve(stand, first + second)
        proc = _run(_script(FIND, tmp_path), workspace, **self._env(tmp_path, server))
        if passes:
            assert proc.returncode == 0, proc.stdout + proc.stderr
            assert _outputs(tmp_path) == {"version": "9.9.9", "run-id": "4242"}
            assert "release run 4242: Publish to TestPyPI concluded success in the latest attempt (2)" in proc.stdout
            assert f"release run 4242: {INSTALL_NAME} concluded success in the latest attempt (2)" in proc.stdout
        else:
            assert proc.returncode == 1
            assert f"  {INSTALL_NAME}: failure (attempt 2)" in proc.stdout
            assert _outputs(tmp_path) == {}

    @pytest.mark.parametrize("newest_first", [False, True], ids=["oldest-first", "newest-first"])
    @pytest.mark.parametrize(("first", "second", "passes"), [("success", "failure", False), ("failure", "success", True)], ids=["the-second-attempt-failed", "the-second-attempt-passed"])
    def test_the_highest_attempt_wins_in_whatever_order_the_listing_gives(self, workspace, tmp_path, stand, newest_first, first, second, passes):
        # GitHub does not document the order of filter=all: the step takes each job's record with the highest
        # (run_attempt, id), not the first or the last one listed.
        records = _attempt(1, 100, testpypi=first, install="success" if first == "success" else "skipped", pypi="skipped") + _attempt(2, 200, testpypi=second, install="success" if second == "success" else "skipped", pypi="skipped")
        server = self._serve(stand, list(reversed(records)) if newest_first else records)
        proc = _run(_script(FIND, tmp_path), workspace, **self._env(tmp_path, server))
        assert (proc.returncode == 0) is passes, proc.stdout + proc.stderr
        if not passes:
            assert "  Publish to TestPyPI: failure (attempt 2)" in proc.stdout

    def test_a_release_run_whose_build_failed_so_both_testpypi_jobs_were_skipped_is_refused(self, workspace, tmp_path, stand):
        # A skipped TestPyPI job never ran its check. GitHub records it as completed, with the conclusion "skipped".
        records = [_job(100, "Build and Validate", "failure"), _job(101, "Publish to TestPyPI", "skipped"), _job(102, INSTALL_NAME, "skipped"), _job(103, "Publish to PyPI", "skipped"), _job(104, REPUBLISH_NAME, "skipped")]
        server = self._serve(stand, records)
        proc = _run(_script(FIND, tmp_path), workspace, **self._env(tmp_path, server))
        assert proc.returncode == 1
        assert "  Publish to TestPyPI: skipped (attempt 1)" in proc.stdout
        assert f"  {INSTALL_NAME}: skipped (attempt 1)" in proc.stdout
        assert _outputs(tmp_path) == {}

    def test_an_older_release_run_that_passed_does_not_vouch_for_a_newer_one_that_failed(self, workspace, tmp_path, stand):
        # Ruling 1 reads the tag's NEWEST release run: a Release published again starts another run, which TestPyPI then
        # refuses, and from then on the tag has no run that vouches for its files.
        runs = [{"id": 4243, "head_branch": TAG}, {"id": 4242, "head_branch": TAG}]
        server = stand(
            {
                _runs_path(TAG): [(200, {"workflow_runs": runs})],
                _jobs_path(4243): [_listing(*_attempt(1, 300, testpypi="failure", install="skipped", pypi="skipped"))],
                _jobs_path(4242): [_listing(*_attempt(1, 100))],
                _artifacts_path(4242): [(200, {"artifacts": LIVE_ARTIFACT})],
            }
        )
        proc = _run(_script(FIND, tmp_path), workspace, **self._env(tmp_path, server))
        assert proc.returncode == 1
        assert "  Publish to TestPyPI: failure (attempt 1)" in proc.stdout
        assert _jobs_path(4242) not in server.paths()
        assert _outputs(tmp_path) == {}

    def test_a_testpypi_job_that_failed_and_then_passed_on_a_re_run_counts_as_passed(self, workspace, tmp_path, stand):
        rerun = [_job(201, "Publish to TestPyPI", "success", attempt=2), _job(202, INSTALL_NAME, "success", attempt=2), _job(203, "Publish to PyPI", "failure", attempt=2)]
        server = self._serve(stand, _attempt(1, 100, testpypi="failure", install="skipped", pypi="skipped") + rerun)
        proc = _run(_script(FIND, tmp_path), workspace, **self._env(tmp_path, server))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert _outputs(tmp_path) == {"version": "9.9.9", "run-id": "4242"}

    def test_re_run_all_jobs_whose_second_attempt_failed_is_refused_although_the_first_passed(self, workspace, tmp_path, stand):
        server = self._serve(stand, _attempt(1, 100) + _attempt(2, 200, testpypi="failure", install="skipped", pypi="skipped"))
        proc = _run(_script(FIND, tmp_path), workspace, **self._env(tmp_path, server))
        assert proc.returncode == 1
        assert "  Publish to TestPyPI: failure (attempt 2)" in proc.stdout
        assert _outputs(tmp_path) == {}

    def test_the_latest_attempt_is_read_across_every_page_of_the_listing(self, workspace, tmp_path, stand):
        # 105 records: 95 other jobs and attempt 1, whose upload failed, on page 1; attempt 2, all green, on page 2.
        page_1 = [_job(1000 + i, f"Unrelated job {i}", "success") for i in range(95)] + _attempt(1, 100, testpypi="failure", install="skipped", pypi="skipped")
        server = self._serve(stand, [])
        server.routes[_jobs_path(4242, 1)] = [_listing(*page_1, total=105)]
        server.routes[_jobs_path(4242, 2)] = [_listing(*_attempt(2, 200), total=105)]
        proc = _run(_script(FIND, tmp_path), workspace, **self._env(tmp_path, server))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert _jobs_path(4242, 2) in server.paths()
        assert _outputs(tmp_path) == {"version": "9.9.9", "run-id": "4242"}

    def test_a_release_run_without_a_testpypi_job_is_refused(self, workspace, tmp_path, stand):
        server = self._serve(stand, [_job(100, "Build and Validate", "success"), _job(101, "Publish to PyPI", "failure")])
        proc = _run(_script(FIND, tmp_path), workspace, **self._env(tmp_path, server))
        assert proc.returncode == 1
        assert "the run has no such job" in proc.stdout
        assert _outputs(tmp_path) == {}

    @pytest.mark.parametrize("workflow", sorted(PUBLISHERS))
    def test_each_publishers_tag_names_its_version(self, workspace, tmp_path, stand, workflow):
        package = PUBLISHERS[workflow][0]
        tag = f"{TAG_PREFIXES[workflow]}1.2.3"
        server = self._serve(stand, _attempt(1, 100), artifacts=[], tag=tag, workflow=workflow, package=package)
        proc = _run(_script(FIND, tmp_path), workspace, PUBLISH_PACKAGE=package, PUBLISH_TAG_PREFIX=TAG_PREFIXES[workflow], **self._env(tmp_path, server, tag=tag, workflow=workflow))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert _outputs(tmp_path) == {"version": "1.2.3", "run-id": ""}
        assert "::warning::" in proc.stdout
        assert server.paths() == [_runs_path(tag, workflow), _jobs_path(4242), _artifacts_path(4242, package)]

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
            ("juniper-recurrence", "juniper-recurrence-v", "tag"),
        ],
        ids=["branch-named-like-a-v-tag", "branch-named-like-this-packages-tag", "branch", "the-models-tag-for-the-app", "the-clients-tag-for-the-app", "the-apps-tag-for-the-model", "the-models-tag-for-the-client", "a-prefix-without-a-version"],
    )
    def test_it_refuses_anything_but_a_release_tag_of_this_package(self, workspace, tmp_path, stand, package, tag, ref_type):
        server = stand({})
        proc = _run(_script(FIND, tmp_path), workspace, PUBLISH_PACKAGE=package, PUBLISH_TAG_PREFIX=f"{package}-v", **self._env(tmp_path, server, tag=tag, ref_type=ref_type))
        assert proc.returncode == 1
        assert "::error::" in proc.stdout
        assert server.requests == []
        assert _outputs(tmp_path) == {}

    @pytest.mark.parametrize("workflow", sorted(PUBLISHERS))
    @pytest.mark.parametrize("form", ["v{version}", "{package}-{version}", "{package}-V{version}"], ids=["the-meta-packages-v-tag", "a-tag-without-the-v", "a-capital-v"])
    def test_a_sub_package_publisher_refuses_every_tag_but_its_own_prefix(self, workspace, tmp_path, stand, workflow, form):
        package = PUBLISHERS[workflow][0]
        tag = form.format(package=package, version="1.2.3")
        server = stand({})
        proc = _run(_script(FIND, tmp_path), workspace, PUBLISH_PACKAGE=package, PUBLISH_TAG_PREFIX=TAG_PREFIXES[workflow], **self._env(tmp_path, server, tag=tag, workflow=workflow))
        assert proc.returncode == 1
        assert "::error::" in proc.stdout
        # Refused before any request: a sub-package's republish never reads the release run of another package's tag.
        assert server.requests == []
        assert _outputs(tmp_path) == {}

    @pytest.mark.parametrize("tag", ["juniper-recurrence-v1.2.3", "v"], ids=["a-sub-package-tag", "a-prefix-without-a-version"])
    def test_a_single_package_publisher_refuses_a_sub_package_tag(self, workspace, tmp_path, stand, tag):
        # The template's other form: a single-package publisher (juniper-cascor's publish.yml) sets PUBLISH_TAG_PREFIX
        # to `v`, and its republish refuses a `<package>-v<version>` tag, before any request, as these publishers refuse
        # a `v<version>` one.
        server = stand({})
        proc = _run(_script(FIND, tmp_path), workspace, PUBLISH_TAG_PREFIX="v", **self._env(tmp_path, server, tag=tag))
        assert proc.returncode == 1
        assert "::error::" in proc.stdout
        assert server.requests == []
        assert _outputs(tmp_path) == {}

    @pytest.mark.parametrize("failing", ["runs", "jobs", "artifacts"])
    def test_an_api_failure_fails_the_job_rather_than_skip_the_check(self, workspace, tmp_path, stand, failing):
        server = self._serve(stand, _attempt(1, 100))
        server.routes[{"runs": _runs_path(TAG), "jobs": _jobs_path(4242), "artifacts": _artifacts_path(4242)}[failing]] = [(502, {"message": "bad gateway"})]
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

    @pytest.mark.parametrize("which", ["the-release-run-lists-more", "testpypi-lists-more"])
    def test_testpypi_and_the_release_runs_sha256sums_must_name_the_same_files(self, tmp_path, stand, which):
        # A file one of them names and the other does not is refused, in either direction, before any file is fetched.
        root = tmp_path / "workspace"
        root.mkdir()
        extra = "juniper_recurrence-9.9.9-py2.py3-none-any.whl"
        recorded = dict(SERVED) | ({extra: _sha(b"another wheel")} if which == "the-release-run-lists-more" else {})
        served = dict(BUILT) | ({extra: b"another wheel"} if which == "testpypi-lists-more" else {})
        sums = tmp_path / "release-run" / "SHA256SUMS"
        sums.parent.mkdir()
        sums.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(recorded.items())), encoding="utf-8")
        server = self._serve(stand, served)
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
