#!/usr/bin/env python3
"""Regression tests for util/run_tests.py, the local test runner: juniper-recurrence's spec, against the real workflows.

Project:     Juniper
Sub-Project: juniper-recurrence
Application: repo tooling tests
Author:      Paul Calnon (authored via Claude Code)
License:     MIT License

util/run_tests.py is a thin spec over the shared engine, ``juniper_ci_tools.run_tests`` (juniper-ci-tools 0.10, which
ci-tooling.yml installs). The engine's own suite -- selection, running, the exit codes, ``--changed``, on fake
checkouts of this repo's shape and two others -- is juniper-ml's ``juniper-ci-tools/tests/test_run_tests.py``. This
file pins what is juniper-recurrence's own, against the real workflows (no test runs):

- every root ``tests/test_*.py`` is a suite some workflow runs with ``python3 -m unittest``, and so a job of the
  unittest lane, taken from the workflow that runs it;
- every ``ci-recurrence-<lane>.yml`` with a ``test`` job is a gating lane in the ``packages`` group, with one pytest
  step, a coverage gate (all but the bench), a PYTHONPATH inside this checkout and its workflow's path filter; the
  bench lane tests this checkout's app from the root and leaves out the ``repository_dispatch`` steps; the model's
  ``test-torch`` job is the optional lane ``model-torch``, outside ``all``;
- the engine is the one CI installs: ci-tooling.yml asks for ``ENGINE_REQUIREMENT``, every juniper-ci-tools install
  the unittest lane merges admits that whole range (the lane's requirements are every install of every job that runs
  a suite, so one that did not would make them unsatisfiable together), and the installed juniper-ci-tools is in it;
- every option ``--help`` lists appears in README.md's "Running tests locally" section.

It uses only the engine's public API (``__all__``: ``discover``, the ``Lane`` / ``Job`` / ``Step`` records,
``RepoSpec``, ``build_parser``, and the command line), so a patch release of the engine, which may change anything
else, cannot break it. A workflow change the runner cannot read fails here, in the PR that makes it (ci-tooling.yml
runs this suite on every pull request), instead of on the next person's machine.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "util" / "run_tests.py"
README = REPO_ROOT / "README.md"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
ENGINE_RANGE = re.compile(r"juniper-ci-tools>=([\d.]+),<([\d.]+)")


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_tests_under_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass resolves annotations through sys.modules
    spec.loader.exec_module(module)
    return module


RT = _load_runner()


def _repo_tmp() -> Path:
    """This checkout's gitignored tmp/: on a CI runner the default temp directory is the system one."""
    path = REPO_ROOT / "tmp"
    path.mkdir(exist_ok=True)
    return path


def _run(*args: str, timeout: float = 300) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(RUNNER), "--color", "never", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=timeout)


def _listed_lanes(stdout: str) -> list[str]:
    """The lanes a ``--list`` names: its lines that start in the first column, but the groups line."""
    return [line.split()[0] for line in stdout.splitlines() if line and not line.startswith((" ", "groups:"))]


def _events(workflow: dict) -> set[str]:
    on = workflow.get("on", workflow.get(True))  # PyYAML reads the key `on` as True
    if isinstance(on, str):
        return {on}
    return {str(event) for event in on} if isinstance(on, (list, dict)) else set()


def _release(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


def _engine_range() -> tuple[tuple[int, ...], tuple[int, ...]]:
    match = ENGINE_RANGE.fullmatch(RT.ENGINE_REQUIREMENT)
    assert match is not None, RT.ENGINE_REQUIREMENT
    return _release(match.group(1)), _release(match.group(2))


# ---------------------------------------------------------------------------------------------------------------
# The real workflows
# ---------------------------------------------------------------------------------------------------------------


class RealWorkflowsTest(unittest.TestCase):
    """The spec reads this checkout's workflows, and reads them the way CI runs them."""

    lanes: dict

    @classmethod
    def setUpClass(cls) -> None:
        cls.lanes = RT.rt.discover(REPO_ROOT, RT.SPEC)

    def test_every_root_suite_is_a_unittest_job(self) -> None:
        on_disk = sorted(f"tests/{path.name}" for path in (REPO_ROOT / "tests").glob("test_*.py"))
        self.assertIn("tests/test_run_tests.py", on_disk)
        self.assertEqual(sorted(job.id for job in self.lanes["unittest"].jobs), on_disk, "a root suite no workflow runs: wire it into one (no package lane collects tests/)")

    def test_each_suite_comes_from_the_workflow_that_runs_it(self) -> None:
        expected: dict[str, str] = {}
        for path in sorted(WORKFLOWS.glob("*.yml")):
            text = path.read_text(encoding="utf-8")
            if not _events(yaml.safe_load(text) or {}) & {"pull_request", "push"}:
                continue
            for suite in re.findall(r"python3? -m unittest(?: -\w+)* (tests/\S+\.py)", text):
                expected.setdefault(suite, path.name)
        self.assertGreaterEqual(len(expected), 3)
        got = {job.id: job.origin for job in self.lanes["unittest"].jobs}
        self.assertEqual(sorted(got), sorted(expected))
        for suite, workflow in expected.items():
            with self.subTest(suite=suite):
                self.assertIn(f".github/workflows/{workflow} > ", got[suite])

    def test_one_gating_lane_per_ci_recurrence_workflow_with_a_test_job(self) -> None:
        expected = sorted(path.stem[len(RT.PACKAGE_PREFIX) :] for path in WORKFLOWS.glob(RT.PACKAGE_WORKFLOWS) if RT.PACKAGE_JOB in (yaml.safe_load(path.read_text(encoding="utf-8")).get("jobs") or {}))
        self.assertEqual(expected, ["app", "bench", "client", "model"])
        gating = [lane for lane in self.lanes.values() if lane.is_package and not lane.optional]
        self.assertEqual(sorted(lane.name for lane in gating), expected)
        directories = {"app": "juniper-recurrence", "bench": "bench", "client": "juniper-recurrence-client", "model": "juniper-recurrence-model"}
        for lane in gating:
            with self.subTest(lane=lane.name):
                self.assertEqual(lane.groups, (RT.PACKAGES_GROUP,))
                labels = [step.label for step in lane.jobs[0].steps]
                self.assertEqual(labels.count("pytest"), 1, labels)
                if lane.name != "bench":
                    self.assertIn("coverage-gate", labels)
                self.assertEqual(lane.package_dir, REPO_ROOT / directories[lane.name])
                self.assertTrue(lane.pythonpath, "no install from this checkout found: the lane would test whatever is installed")
                for path in lane.pythonpath:
                    self.assertTrue((path / "pyproject.toml").is_file(), path)
                    path.relative_to(REPO_ROOT)  # inside this checkout
                self.assertTrue(lane.packages, "no importable package found in the installed directories")
                self.assertTrue(lane.path_filter, "the lane did not read its workflow's changes-job path filter")
                self.assertTrue(re.search(lane.path_filter, f".github/workflows/ci-recurrence-{lane.name}.yml"), lane.path_filter)
                self.assertTrue(re.search(lane.path_filter, f"{directories[lane.name]}/some_file.py"), lane.path_filter)

    def test_the_bench_lane_tests_this_checkouts_app_and_leaves_out_the_dispatch_steps(self) -> None:
        bench = self.lanes["bench"]
        self.assertEqual(bench.pythonpath, [REPO_ROOT / "juniper-recurrence"])
        self.assertEqual(set(bench.packages), {"bench", "juniper_recurrence"})
        self.assertEqual([step.label for step in bench.jobs[0].steps], ["pytest"])
        self.assertEqual(bench.jobs[0].steps[0].cwd, REPO_ROOT)

    def test_the_torch_job_is_an_optional_lane_outside_all(self) -> None:
        self.assertEqual([name for name, lane in self.lanes.items() if lane.optional], ["model-torch"])
        torch = self.lanes["model-torch"]
        self.assertTrue(torch.is_package)
        self.assertIn("test-torch", torch.why)
        proc = _run("--list", "all", timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(_listed_lanes(proc.stdout), ["unittest", "app", "bench", "client", "model"])
        proc = _run("--list", "optional", timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(_listed_lanes(proc.stdout), ["model-torch"])

    def test_list_runs_nothing_and_names_every_suite(self) -> None:
        proc = _run("--list", timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        listed = sorted(line.split()[0] for line in proc.stdout.splitlines() if line.startswith("  tests/test_"))
        self.assertEqual(listed, sorted(job.id for job in self.lanes["unittest"].jobs))

    def test_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="run-tests-", dir=_repo_tmp()) as tmp:
            logs = Path(tmp) / "logs"
            proc = _run("--dry-run", "--log-dir", str(logs), "all", "optional")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("Plan", proc.stdout)
            self.assertIn("-m unittest -v tests/test_run_tests.py", proc.stdout)
            self.assertFalse(logs.exists(), "a dry run created the log directory")


# ---------------------------------------------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------------------------------------------


class EngineTest(unittest.TestCase):
    """The spec runs the engine CI installs, and nothing the unittest lane merges asks for another one."""

    def test_the_spec_is_one_for_the_installed_engine(self) -> None:
        self.assertEqual(RT.rt.__name__, "juniper_ci_tools.run_tests")
        self.assertIsInstance(RT.SPEC, RT.rt.RepoSpec)
        self.assertEqual(Path(RT.SPEC.script).resolve(), RUNNER.resolve())

    def test_the_installed_engine_is_in_the_specs_range(self) -> None:
        low, high = _engine_range()
        have = importlib.metadata.version("juniper-ci-tools")
        self.assertTrue(low <= _release(have) < high, f"juniper-ci-tools {have} is outside {RT.ENGINE_REQUIREMENT}")

    def test_ci_tooling_installs_the_engine(self) -> None:
        job = yaml.safe_load((WORKFLOWS / "ci-tooling.yml").read_text(encoding="utf-8"))["jobs"]["run-tests-suite"]
        installs = "\n".join(str(step.get("run") or "") for step in job["steps"] if str(step.get("name") or "").startswith("Install"))
        self.assertIn(f'"{RT.ENGINE_REQUIREMENT}"', installs)

    def test_every_engine_install_the_unittest_lane_merges_admits_the_engine(self) -> None:
        low, high = _engine_range()
        checked = []
        for path in sorted(WORKFLOWS.glob("*.yml")):
            workflow = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not _events(workflow) & {"pull_request", "push"}:
                continue
            for job_id, job in (workflow.get("jobs") or {}).items():
                steps = job.get("steps") or []
                if not any(re.search(r"-m\s+unittest\b", str(step.get("run") or "")) for step in steps):
                    continue
                for step in steps:
                    if not str(step.get("name") or "").startswith("Install"):
                        continue
                    for line in str(step.get("run") or "").splitlines():
                        for found in ENGINE_RANGE.finditer("" if line.strip().startswith("#") else line):
                            checked.append(f"{path.name} > {job_id}")
                            with self.subTest(job=checked[-1], requirement=found.group(0)):
                                self.assertLessEqual(_release(found.group(1)), low)
                                self.assertGreaterEqual(_release(found.group(2)), high, "this install excludes the engine: widen its ceiling")
        self.assertEqual(checked, ["ci-tooling.yml > run-tests-suite", "sequence-safety.yml > sequence-safety"], "the installs this test reads moved: check it still reads them")


# ---------------------------------------------------------------------------------------------------------------
# The documentation
# ---------------------------------------------------------------------------------------------------------------


class ReadmeDocsTest(unittest.TestCase):
    """README.md documents the runner, and every option --help lists."""

    def test_every_option_is_documented(self) -> None:
        text = README.read_text(encoding="utf-8")
        start = text.find("## Running tests locally")
        self.assertNotEqual(start, -1, "README.md has no 'Running tests locally' section")
        end = text.find("\n## ", start + 1)
        section = text[start : end if end != -1 else len(text)]
        options = sorted({option for action in RT.rt.build_parser(RT.SPEC)._actions for option in action.option_strings if option not in ("-h", "--help")})
        missing = [option for option in options if f"`{option}" not in section and f"{option} " not in section and f"{option}`" not in section]
        self.assertEqual(missing, [], "options --help lists that the README section does not document")
        for selector in ("all", "unittest", "packages", "optional", "model-torch", "--changed", "--failed", "juniper_ci_tools.run_tests", "juniper-ci-tools/tests/test_run_tests.py"):
            self.assertIn(selector, section)


if __name__ == "__main__":
    unittest.main()
