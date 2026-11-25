#!/usr/bin/env python3
"""Run any subset of juniper-recurrence's tests locally, with the commands CI runs.

Project: juniper-recurrence
Sub-Project: developer tooling (local test runs)
Author: Paul Calnon (authored via Claude Code)
Created: 2026-11-25
Status: permanent utility
Related: .github/workflows/ci-recurrence-<lane>.yml and every workflow that runs ``python3 -m unittest`` (the lists
         this reads); tests/test_run_tests.py (this script's suite); README.md § Running tests locally (the operator
         documentation). Ported from juniper-ml's util/run_tests.py (juniper-ml#2629), the first of the per-repo
         copies the Overtoad ruled on 2026-11-25.

Why this exists
---------------
juniper-recurrence is a four-part monorepo, and CI tests each part in its own workflow: ``ci-recurrence-app.yml``,
``-model.yml`` and ``-client.yml`` run pytest in the package's directory with a 90% coverage floor and then a
per-file coverage gate, and ``ci-recurrence-bench.yml`` runs the bench smoke suite from the repository root
against a non-editable install of the app. The model workflow also has an optional torch lane. The root ``tests/``
suites run in no package lane: ``memory-budget.yml`` and ``sequence-safety.yml`` each run one with
``python3 -m unittest``. AGENTS.md's Testing section gives the per-package commands, but not the coverage gates,
the bench's install or the root suites, so a local run checked less than CI does.

This script reads the commands from the workflows on every run, so its lists cannot drift from CI's, and it runs
any subset of them: everything, a lane, a suite, one test method, one package test file, the lanes a change
touches, or the failures of the last run.

How it runs a command
---------------------
- Each suite and command is its own process, as in CI, started in the directory CI starts it in. ``python``,
  ``python3`` and ``pytest`` in a workflow command become the selected interpreter (``--python``; default: the
  one running this script). A console script is looked for next to that interpreter, then on PATH.
- Every child gets stdin from /dev/null and its own session and process group. The group is killed on a timeout,
  on Ctrl-C, and when a step ends with processes still in it. ``DBUS_SESSION_BUS_ADDRESS`` names a socket that
  does not exist, so no test can raise a desktop notification (juniper-ml#2559). ``DISPLAY``,
  ``WAYLAND_DISPLAY``, ``PYTHONPATH`` and ``PYTHONHOME`` are removed, and ``PYTHONUNBUFFERED=1`` keeps a log in
  the order its lines were written. ``TMPDIR`` is the primary checkout's gitignored ``tmp/`` unless it already
  names a directory outside the system temp directory, and is prepended to ``GIT_CEILING_DIRECTORIES``. Nothing
  is ever written under the system temp directory (Overtoad directive, 2026-10-26).
- A package lane runs with ``PYTHONPATH`` set to the directories its workflow installs from this checkout:
  editable (``pip install -e .``) or from a local path (the bench's ``pip install "./juniper-recurrence[...]"``).
  It tests this checkout's code whatever the environment has installed, and the preflight refuses the lane
  (UNAVAILABLE) when the package would still import from somewhere else. A package the workflow installs from
  PyPI stays the installed one, as in CI: the app lane tests against the installed juniper-recurrence-model, not
  this checkout's. Coverage data and JSON reports go to the run directory, never into the checkout
  (``coverage.json`` is not gitignored).
- A step a lane runs only on another event (the bench's ``repository_dispatch`` steps) is left out, and so is a
  step whose name starts with "Report" (the informational Python 3.15 summary).
- A workflow job named ``test-<name>`` beside a lane's ``test`` job (the model's ``test-torch``) is an optional
  lane, ``<lane>-<name>``: CI does not gate on it, so it runs only when named (or with ``optional``).
- The workflows' ``pip install`` commands are never run: this script installs nothing. The preflight names what
  is missing and the command that installs it.
- CI stops at the first failing command; this script runs every selected job and reports every failure. ``-x``
  restores stop-at-first-failure. With ``-j`` above 1, a job that fails is re-run alone once the others are done,
  and a job that passes then is reported as RERUN-PASSED (``--no-rerun`` turns this off).

Run ``python3 util/run_tests.py --help`` for the selectors, options, examples and exit codes.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import difflib
import fnmatch
import itertools
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

if sys.version_info < (3, 11):  # tomllib
    sys.stderr.write("run_tests.py: Python 3.11 or newer is required\n")
    raise SystemExit(2)

import tomllib  # noqa: E402

SCHEMA = 1

WORKFLOWS = Path(".github") / "workflows"
PACKAGE_WORKFLOW_PREFIX = "ci-recurrence-"
PACKAGE_WORKFLOW_GLOB = f"{PACKAGE_WORKFLOW_PREFIX}*.yml"
PACKAGE_JOB = "test"
OPTIONAL_JOB_PREFIX = "test-"
CHANGES_JOB = "changes"
# The events a pull request's CI run has; a step that runs only on another event is left out of a lane.
LOCAL_EVENTS = frozenset({"pull_request", "push", "workflow_dispatch"})

LANE_UNITTEST = "unittest"
GROUP_ALL = "all"
GROUP_PACKAGES = "packages"
GROUP_OPTIONAL = "optional"
GROUPS = (GROUP_ALL, GROUP_PACKAGES, GROUP_OPTIONAL)

DEFAULT_TIMEOUT = 900.0
DEFAULT_TAIL = 30
DEFAULT_BASE = "origin/main"
KILL_GRACE = 5.0
HEARTBEAT = 60.0
NO_TESTS_RC = 5

PASSED = "PASS"
FAIL = "FAIL"
TIMEOUT = "TIMEOUT"
NO_TESTS = "NO-TESTS"
UNAVAILABLE = "UNAVAILABLE"
RERUN_PASSED = "RERUN-PASS"
NOT_RUN = "NOT-RUN"
SKIPPED = "SKIPPED"  # a step only: a coverage gate with coverage off
STATUS_ORDER = (FAIL, TIMEOUT, UNAVAILABLE, NOT_RUN, RERUN_PASSED, NO_TESTS, PASSED)
STATUS_WORDS = {FAIL: "failed", TIMEOUT: "timed out", UNAVAILABLE: "unavailable", NOT_RUN: "not run", RERUN_PASSED: "passed on re-run", NO_TESTS: "no matching tests", PASSED: "passed"}

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3
EXIT_NOTHING_RAN = 5
EXIT_INTERRUPTED = 130

BUS_GUARD = "unix:path=/nonexistent/juniper-run-tests/no-session-bus"
ALWAYS_REMOVED = ("PYTHONPATH", "PYTHONHOME", "DISPLAY", "WAYLAND_DISPLAY")
CLEAN_ENV_PREFIXES = ("CLAUDE", "JUNIPER_", "WTC_")
# The system temp roots this script never lets a child use as TMPDIR (Overtoad directive, 2026-10-26).
SYSTEM_TEMP_ROOTS = ("/tmp", "/var/tmp", "/private/tmp", "/private/var/tmp", "/var/folders", "/private/var/folders")  # tmp-path-literal: allow

STAMP_RE = re.compile(r"\d{8}T\d{6}Z(?:-\d+)?")
LOG_HEADER = "# run_tests:"

_PYTHON_NAME = re.compile(r"python(?:3(?:\.\d+)?)?")
_OPERATOR = re.compile(r"[();<>|&]+")
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")
_SHELL_ONLY = frozenset({"cd", "pushd", "popd", "export", "source", ".", "eval", "exec", "if", "then", "fi", "for", "while", "until", "do", "done", "case", "esac", "unset", "alias", "function", "trap"})
_IGNORED_COMMANDS = frozenset({"echo", "printf", "set", "true", ":"})
_PIP_VALUE_OPTIONS = frozenset(
    {
        "-r",
        "--requirement",
        "-c",
        "--constraint",
        "-i",
        "--index-url",
        "--extra-index-url",
        "-f",
        "--find-links",
        "-t",
        "--target",
        "--prefix",
        "--root",
        "--platform",
        "--python-version",
        "--implementation",
        "--abi",
        "--src",
        "--upgrade-strategy",
        "--progress-bar",
        "--trusted-host",
        "--cache-dir",
        "--log",
        "--report",
        "-C",
        "--config-settings",
        "--global-option",
        "--no-binary",
        "--only-binary",
    }
)
_REQ_NAME = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


class UsageError(Exception):
    """A usage or configuration error: exit 2."""


class _Env(dict):
    """A child environment whose repr never shows a value (tests/redacted_env.py's rule, for this script)."""

    def __repr__(self) -> str:
        return f"<child env: {len(self)} vars (values redacted)>"

    __str__ = __repr__


# ---------------------------------------------------------------------------------------------------------------
# Reading the workflows
# ---------------------------------------------------------------------------------------------------------------


def strip_comment(line: str) -> str:
    """Drop a trailing shell comment: an unquoted ``#`` that starts a word, as bash reads it."""
    quote = ""
    escaped = False
    prev = " "
    for index, char in enumerate(line):
        if escaped:
            escaped = False
        elif char == "\\" and quote != "'":
            escaped = True
        elif quote:
            if char == quote:
                quote = ""
        elif char in "'\"":
            quote = char
        elif char == "#" and prev.isspace():
            return line[:index].rstrip()
        prev = char
    return line


def logical_lines(script: str) -> list[str]:
    """The command lines of a ``run:`` block: comment lines dropped, backslash continuations joined."""
    lines: list[str] = []
    parts: list[str] = []
    for raw in (script or "").splitlines():
        text = raw.strip()
        if not parts and (not text or text.startswith("#")):
            continue
        if text.endswith("\\") and not text.endswith("\\\\"):
            parts.append(text[:-1].strip())
            continue
        parts.append(text)
        lines.append(" ".join(part for part in parts if part))
        parts = []
    if parts:
        lines.append(" ".join(part for part in parts if part))
    return lines


def split_command(line: str, where: str) -> list[str]:
    """Shell words of one command line; operators such as ``&&`` and ``|`` come back as their own tokens."""
    lexer = shlex.shlex(strip_comment(line), posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError as exc:
        raise UsageError(f"{where}: cannot parse {line!r} ({exc})") from None


def shell_need(tokens: list[str]) -> str | None:
    """Why a command needs a shell this script does not provide, or None."""
    for token in tokens:
        if _OPERATOR.fullmatch(token):
            return f"the shell operator {token!r}"
        if "$" in token or "`" in token:
            return f"shell expansion in {token!r}"
    if tokens and (tokens[0] in _SHELL_ONLY or _ASSIGNMENT.fullmatch(tokens[0])):
        return f"the shell construct {tokens[0]!r}"
    return None


def segments(tokens: list[str]) -> list[list[str]]:
    """Split a command's tokens on shell operators (``a && b`` -> ``[a], [b]``)."""
    out: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if _OPERATOR.fullmatch(token):
            if current:
                out.append(current)
            current = []
        else:
            current.append(token)
    if current:
        out.append(current)
    return out


def pip_install_args(tokens: list[str]) -> list[str] | None:
    """The arguments of a ``pip install`` / ``python -m pip install`` command, else None."""
    if len(tokens) >= 2 and tokens[0] in ("pip", "pip3") and tokens[1] == "install":
        return tokens[2:]
    if len(tokens) >= 4 and _PYTHON_NAME.fullmatch(tokens[0]) and tokens[1:4] == ["-m", "pip", "install"]:
        return tokens[4:]
    return None


def parse_pip_install(args: list[str]) -> tuple[list[str], list[str]]:
    """(requirements, editable specs) named by ``pip install`` arguments."""
    requirements: list[str] = []
    editables: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token in ("-e", "--editable"):
            if index + 1 < len(args):
                editables.append(args[index + 1])
            index += 2
        elif token.startswith("--editable="):
            editables.append(token.split("=", 1)[1])
            index += 1
        elif token in _PIP_VALUE_OPTIONS:
            index += 2
        elif token.startswith("-"):
            index += 1
        else:
            requirements.append(token)
            index += 1
    return requirements, editables


def normalize(name: str) -> str:
    """PEP 503 name normalisation."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(text: str) -> str:
    match = _REQ_NAME.match(text)
    return normalize(match.group(1) if match else text)


def is_local_path(spec: str) -> bool:
    """Is a ``pip install`` argument a path in the checkout (``.``, ``./juniper-recurrence[test]``), not a requirement?"""
    text = spec.strip()
    if "://" in text:
        return False
    return text.startswith((".", "/", "~")) or "/" in text.split("[", 1)[0]


def editable_target(spec: str, base: Path) -> tuple[Path, list[str]]:
    """The directory and extras of a ``pip install -e`` spec such as ``".[test]"`` or ``juniper-service-core/.[test]``."""
    match = re.fullmatch(r"(.*?)(?:\[([^\]]*)\])?", spec.strip())
    path_text = (match.group(1) if match else spec) or "."
    extras = [extra.strip() for extra in ((match.group(2) if match else "") or "").split(",") if extra.strip()]
    return Path(os.path.normpath(base / path_text)), extras


def project_requirements(package_dir: Path, extras: list[str]) -> tuple[str | None, list[str]]:
    """A package's project name and the requirements an install with ``extras`` pulls in, from its pyproject.toml."""
    pyproject = package_dir / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise UsageError(f"{pyproject}: {exc}") from None
    project = data.get("project") or {}
    requirements = [str(item) for item in project.get("dependencies") or []]
    optional = project.get("optional-dependencies") or {}
    for extra in extras:
        requirements.extend(str(item) for item in optional.get(extra) or [])
    return project.get("name"), requirements


def import_packages(package_dir: Path) -> dict[str, Path]:
    """The importable top-level packages a source directory provides (``tests`` excluded)."""
    found: dict[str, Path] = {}
    try:
        children = sorted(package_dir.iterdir())
    except OSError:
        return found
    for child in children:
        if child.is_dir() and child.name.isidentifier() and child.name not in ("tests", "test") and (child / "__init__.py").is_file():
            found[child.name] = child
    return found


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def load_workflow(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        raise UsageError("PyYAML is needed to read the workflows: python3 -m pip install 'pyyaml>=6.0'") from None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise UsageError(f"cannot read {path}: {exc}") from None
    if not isinstance(data, dict):
        raise UsageError(f"{path}: not a workflow (no mapping at the top level)")
    return data


_EVENT_TEST = re.compile(r"github\.event_name\s*==\s*'([A-Za-z_]+)'")
_GREP_FILTER = re.compile(r"\bgrep\s+-[A-Za-z]*E[A-Za-z]*\s+'([^']+)'")
_UNITTEST_TEXT = re.compile(r"-m\s+unittest\b")


def workflow_events(workflow: dict) -> set[str]:
    """The events that start a workflow. YAML 1.1 reads the bare key ``on`` as True, so both keys are looked up."""
    on = workflow.get(True, workflow.get("on"))
    if isinstance(on, str):
        return {on}
    if isinstance(on, (list, dict)):
        return {str(event) for event in on}
    return set()


def other_event_only(item: dict) -> str:
    """The event a step's or job's ``if:`` limits it to, when a pull request's run never has it; else ''."""
    condition = str(item.get("if") or "")
    events = set(_EVENT_TEST.findall(condition))
    if events and not events & LOCAL_EVENTS and "!=" not in condition:
        return sorted(events)[0]
    return ""


def changes_filter(job: Any) -> str:
    """The path regex a lane workflow's ``changes`` job greps the PR's changed files with, or ''."""
    if not isinstance(job, dict):
        return ""
    for step in job.get("steps") or []:
        if not isinstance(step, dict):
            continue
        match = _GREP_FILTER.search(str(step.get("run") or ""))
        if match:
            try:
                re.compile(match.group(1))
            except re.error:
                return ""
            return match.group(1)
    return ""


def _step_cwd(checkout: Path, job: dict, step: dict) -> Path:
    default = ((job.get("defaults") or {}).get("run") or {}).get("working-directory") or ""
    return Path(os.path.normpath(checkout / str(step.get("working-directory") or default)))


def _is_unittest(tokens: list[str]) -> bool:
    return len(tokens) >= 3 and bool(_PYTHON_NAME.fullmatch(tokens[0])) and tokens[1:3] == ["-m", "unittest"]


def _is_pytest(tokens: list[str]) -> bool:
    return bool(tokens) and (tokens[0] == "pytest" or (len(tokens) >= 3 and bool(_PYTHON_NAME.fullmatch(tokens[0])) and tokens[1:3] == ["-m", "pytest"]))


def _split_unittest(rest: list[str]) -> tuple[list[str], list[str]]:
    """(options, test names) of the arguments after ``-m unittest``."""
    options: list[str] = []
    targets: list[str] = []
    index = 0
    while index < len(rest):
        token = rest[index]
        if token in ("-k", "-p", "-s", "-t"):
            options.extend(rest[index : index + 2])
            index += 2
        elif token.startswith("-"):
            options.append(token)
            index += 1
        else:
            targets.append(token)
            index += 1
    return options, targets


# ---------------------------------------------------------------------------------------------------------------
# Lanes and jobs
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Step:
    """One command of a job, as the workflow writes it."""

    label: str
    argv: list[str]
    cwd: Path
    coverage_gate: bool = False
    hints: list[str] = dataclasses.field(default_factory=list)  # what the workflow pip-installs for this step


@dataclasses.dataclass
class Job:
    """What one status line reports: a unittest suite, or a lane's whole test job."""

    lane: str
    id: str
    kind: str  # "unittest" | "command" | "pytest"
    steps: list[Step]
    order: int
    unittest_options: list[str] = dataclasses.field(default_factory=list)
    targets: list[str] = dataclasses.field(default_factory=list)  # narrowing: unittest names / pytest node ids
    select: list[str] = dataclasses.field(default_factory=list)  # selector tokens that reproduce it (--failed)
    reason: str = ""  # why --changed or --failed chose it
    origin: str = ""  # the workflow > job > step a unittest suite comes from


@dataclasses.dataclass
class Lane:
    """The unittest lane (every workflow's ``python3 -m unittest`` steps) or a lane workflow's test job."""

    name: str
    source: str
    workflow: Path
    jobs: list[Job]
    requirements: list[str] = dataclasses.field(default_factory=list)
    pythonpath: list[Path] = dataclasses.field(default_factory=list)
    packages: dict[str, Path] = dataclasses.field(default_factory=dict)
    package_dir: Path | None = None
    optional: bool = False  # a test-<name> job CI does not gate on: runs only when named
    path_filter: str = ""  # the regex the workflow's changes job uses to decide whether a PR touches the lane

    @property
    def is_package(self) -> bool:
        return bool(self.jobs) and self.jobs[0].kind == "pytest"


def unittest_jobs(script: str, cwd: Path, source: str, counter: Iterable[int]) -> list[Job]:
    """One job per suite of a step's ``python3 -m unittest`` commands; [] when the step runs none."""
    lines = logical_lines(script)
    if not any(_UNITTEST_TEXT.search(line) for line in lines):
        return []
    numbers = iter(counter)
    jobs: list[Job] = []
    for line in lines:
        tokens = split_command(line, source)
        if not tokens or tokens[0] in _IGNORED_COMMANDS:
            continue
        need = shell_need(tokens)
        if need:
            raise UsageError(f"{source}: {line!r} needs a shell ({need}). util/run_tests.py runs CI's commands without one: teach it this shape in the same change.")
        if not _is_unittest(tokens):
            raise UsageError(f"{source}: {line!r} shares a step with a unittest command and is not one. util/run_tests.py runs a unittest step's suites alone: teach it this shape in the same change.")
        options, targets = _split_unittest(tokens[3:])
        if not targets:
            raise UsageError(f"{source}: {line!r} names no test suite")
        for target in targets:
            step = Step("unittest", [*tokens[:3], *options, target], cwd)
            jobs.append(Job(LANE_UNITTEST, target, "unittest", [step], next(numbers), unittest_options=options, select=[target], origin=source))
    return jobs


def _lane_dir(checkout: Path, job: dict, steps: list[Step]) -> Path | None:
    """A lane's directory: its job's working directory, else the first directory its pytest command names (bench/)."""
    default = ((job.get("defaults") or {}).get("run") or {}).get("working-directory")
    if default:
        return Path(os.path.normpath(checkout / str(default)))
    for step in steps:
        if _is_pytest(step.argv):
            for token in step.argv[3:]:
                if not token.startswith("-") and (step.cwd / token).is_dir():
                    return Path(os.path.normpath(step.cwd / token))
    return None


def package_lane(name: str, workflow_path: Path, job_id: str, job: dict, checkout: Path, counter: Iterable[int], optional: bool = False, path_filter: str = "") -> Lane:
    """A lane workflow's test job: its install steps say what is tested, its other steps are the commands."""
    numbers = iter(counter)
    source = f"{workflow_path.relative_to(checkout).as_posix()} > {job_id}"
    pythonpath: list[Path] = []
    packages: dict[str, Path] = {}
    requirements: list[str] = []
    in_repo: set[str] = {"pip"}
    steps: list[Step] = []
    for step in job.get("steps") or []:
        if not isinstance(step, dict) or "run" not in step:
            continue
        title = str(step.get("name") or "")
        if title.startswith("Report") or other_event_only(step):
            continue
        where = f"{source} > {title or '(unnamed step)'}"
        script = str(step.get("run") or "")
        if "${{" in script:
            raise UsageError(f"{where}: uses a GitHub expression, which only CI can evaluate")
        cwd = _step_cwd(checkout, job, step)
        lines = logical_lines(script)
        if title.startswith("Install"):
            for line in lines:
                for part in segments(split_command(line, where)):
                    args = pip_install_args(part)
                    if args is None:
                        continue
                    reqs, editables = parse_pip_install(args)
                    requirements.extend(req for req in reqs if not is_local_path(req))
                    for spec in [*editables, *(req for req in reqs if is_local_path(req))]:
                        target, extras = editable_target(spec, cwd)
                        if not (target / "pyproject.toml").is_file():
                            raise UsageError(f"{where}: the install of {spec!r} names {target}, which has no pyproject.toml")
                        project, deps = project_requirements(target, extras)
                        if project:
                            in_repo.add(normalize(project))
                        requirements.extend(deps)
                        if target not in pythonpath:
                            pythonpath.append(target)
                        packages.update(import_packages(target))
            continue
        hints: list[str] = []
        for line in lines:
            tokens = split_command(line, where)
            if not tokens or tokens[0] in _IGNORED_COMMANDS:
                continue
            parts = segments(tokens)
            if pip_install_args(parts[0]) is not None:
                for part in parts:
                    args = pip_install_args(part)
                    if args is not None:
                        hints.extend(parse_pip_install(args)[0])
                continue
            need = shell_need(tokens)
            if need:
                raise UsageError(f"{where}: {line!r} needs a shell ({need}). util/run_tests.py runs CI's commands without one: teach it this shape in the same change.")
            gate = any(arg == "--coverage-json" or arg.startswith("--coverage-json=") for arg in tokens)
            label = "pytest" if _is_pytest(tokens) else ("coverage-gate" if gate else Path(tokens[0]).name)
            steps.append(Step(label, tokens, cwd, coverage_gate=gate, hints=list(hints)))
    if not any(_is_pytest(step.argv) for step in steps):
        raise UsageError(f"{source}: found no pytest command (renamed or reshaped?); util/run_tests.py must be updated with it")
    package_dir = _lane_dir(checkout, job, steps)
    if package_dir is not None and package_dir not in pythonpath and package_dir.name.isidentifier() and (package_dir / "__init__.py").is_file():
        packages[package_dir.name] = package_dir  # a lane directory that is itself a package (bench/), imported from the root
    lane_job = Job(name, name, "pytest", steps, next(numbers), select=[name])
    wanted = _dedupe(req for req in requirements if requirement_name(req) not in in_repo)
    return Lane(name, source, workflow_path, [lane_job], wanted, pythonpath, packages, package_dir if package_dir is not None and package_dir.is_dir() else None, optional, path_filter)


def _unittest_lane(checkout: Path, workflows: dict[Path, dict], counter: Iterable[int]) -> Lane:
    """Every ``python3 -m unittest`` suite any workflow runs on a pull request or push, with its jobs' installs."""
    suites: list[Job] = []
    seen: set[str] = set()
    requirements: list[str] = []
    sources: list[str] = []
    for path, workflow in workflows.items():
        if not workflow_events(workflow) & LOCAL_EVENTS:
            continue
        rel = path.relative_to(checkout).as_posix()
        for job_id, job in (workflow.get("jobs") or {}).items():
            if not isinstance(job, dict) or other_event_only(job):
                continue
            found: list[Job] = []
            installs: list[tuple[str, str]] = []
            for step in job.get("steps") or []:
                if not isinstance(step, dict) or "run" not in step or other_event_only(step):
                    continue
                title = str(step.get("name") or "")
                where = f"{rel} > {job_id} > {title or '(unnamed step)'}"
                if title.startswith("Install"):
                    installs.append((str(step.get("run") or ""), where))
                    continue
                found.extend(unittest_jobs(str(step.get("run") or ""), _step_cwd(checkout, job, step), where, counter))
            if not found:
                continue
            sources.append(f"{rel} > {job_id}")
            for script, where in installs:
                for line in logical_lines(script):
                    for part in segments(split_command(line, where)):
                        args = pip_install_args(part)
                        if args is not None:
                            requirements.extend(req for req in parse_pip_install(args)[0] if requirement_name(req) != "pip" and not is_local_path(req))
            for job_ in found:
                if job_.id not in seen:  # a suite two workflows run is one job, from the first
                    seen.add(job_.id)
                    suites.append(job_)
    source = "; ".join(sources) or "no workflow runs python3 -m unittest"
    return Lane(LANE_UNITTEST, source, checkout / WORKFLOWS, suites, _dedupe(requirements))


def discover(checkout: Path) -> dict[str, Lane]:
    """Every lane: the unittest lane, then one lane per ``ci-recurrence-<lane>.yml`` test job (and its test-* jobs)."""
    counter = itertools.count()
    wf_dir = checkout / WORKFLOWS
    workflows = {path: load_workflow(path) for path in sorted([*wf_dir.glob("*.yml"), *wf_dir.glob("*.yaml")])}
    lanes: dict[str, Lane] = {LANE_UNITTEST: _unittest_lane(checkout, workflows, counter)}
    for path in sorted(wf_dir.glob(PACKAGE_WORKFLOW_GLOB)):
        jobs_map = workflows[path].get("jobs") or {}
        if PACKAGE_JOB not in jobs_map:
            continue
        name = path.stem[len(PACKAGE_WORKFLOW_PREFIX) :]
        path_filter = changes_filter(jobs_map.get(CHANGES_JOB))
        wanted = [(name, PACKAGE_JOB, False)] + [(f"{name}-{job_id[len(OPTIONAL_JOB_PREFIX) :]}", job_id, True) for job_id in jobs_map if str(job_id).startswith(OPTIONAL_JOB_PREFIX)]
        for lane_name, job_id, optional in wanted:
            if lane_name in lanes or lane_name in GROUPS:
                raise UsageError(f"{path}: the lane name {lane_name!r} collides with another lane or a group")
            lanes[lane_name] = package_lane(lane_name, path, job_id, jobs_map[job_id], checkout, counter, optional, path_filter)
    return lanes


def package_lanes(lanes: dict[str, Lane], optional: bool = False) -> list[Lane]:
    """The package lanes: the gating ones, or (``optional=True``) the opt-in ones."""
    return [lane for lane in lanes.values() if lane.is_package and lane.optional == optional]


def group_lanes(group: str, lanes: dict[str, Lane]) -> list[str]:
    packages = [lane.name for lane in package_lanes(lanes)]
    if group == GROUP_PACKAGES:
        return packages
    if group == GROUP_OPTIONAL:
        return [lane.name for lane in package_lanes(lanes, optional=True)]
    return [LANE_UNITTEST, *packages]


def lane_aliases(lanes: dict[str, Lane], checkout: Path) -> dict[str, str]:
    """Other names for a gating package lane: its directory (juniper-recurrence-model -> model)."""
    aliases: dict[str, str] = {}
    for lane in package_lanes(lanes):
        if lane.package_dir is not None:
            aliases.setdefault(_display_path(lane.package_dir, checkout), lane.name)
    return aliases


# ---------------------------------------------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------------------------------------------


class Picker:
    """The selected jobs. A whole job wins over a narrowed copy of it; narrowed copies merge their targets."""

    def __init__(self, lanes: dict[str, Lane]) -> None:
        self.lanes = lanes
        self.chosen: dict[tuple[str, str], Job] = {}

    def whole(self, job: Job, reason: str = "") -> None:
        key = (job.lane, job.id)
        old = self.chosen.get(key)
        self.chosen[key] = dataclasses.replace(job, targets=[], select=list(job.select), reason=(old.reason if old and old.reason else reason))

    def narrow(self, job: Job, target: str, token: str, reason: str = "") -> None:
        key = (job.lane, job.id)
        old = self.chosen.get(key)
        if old is not None and not old.targets:
            return  # the whole job is already selected
        if old is None:
            self.chosen[key] = dataclasses.replace(job, targets=[target], select=[token], reason=reason)
            return
        if target not in old.targets:
            old.targets.append(target)
            old.select.append(token)

    def jobs(self) -> list[Job]:
        return sorted(self.chosen.values(), key=lambda job: job.order)


def _has_glob(text: str) -> bool:
    return any(char in text for char in "*?[")


def repo_relative(text: str, checkout: Path, cwd: Path) -> str:
    """A path the user typed, relative to the checkout (tried against the current directory, then the checkout)."""
    path = Path(text).expanduser()
    candidates = [path] if path.is_absolute() else [cwd / path, checkout / path]
    root = checkout.resolve()
    for candidate in candidates:
        if candidate.exists():
            resolved = candidate.resolve()
            try:
                return resolved.relative_to(root).as_posix() or "."
            except ValueError:
                raise UsageError(f"{text}: outside the checkout {checkout}") from None
    raise UsageError(f"{text}: no such file or directory (looked in {cwd} and {checkout})")


def resolve_selector(token: str, lanes: dict[str, Lane], checkout: Path, cwd: Path, picker: Picker) -> None:
    """Add what one selector names to ``picker``; raise UsageError when it names nothing."""
    if token in GROUPS:
        for lane_name in group_lanes(token, lanes):
            for job in lanes[lane_name].jobs:
                picker.whole(job)
        return
    lane_name = token if token in lanes else lane_aliases(lanes, checkout).get(token.rstrip("/"))
    if lane_name is not None:
        for job in lanes[lane_name].jobs:
            picker.whole(job)
        return
    suites = [job for job in lanes[LANE_UNITTEST].jobs if job.kind == "unittest"]
    by_id = {job.id: job for job in suites}
    path_part, separator, node = token.partition("::")
    if not separator and re.fullmatch(r"tests(?:\.[A-Za-z_]\w*)+", token):
        parts = token.split(".")
        job = by_id.get(f"tests/{parts[1]}.py")
        if job is None:
            raise UsageError(f"{token}: tests/{parts[1]}.py is not a suite any workflow runs")
        if len(parts) == 2:
            picker.whole(job)
        else:
            picker.narrow(job, token, f"tests/{parts[1]}.py::" + "::".join(parts[2:]))
        return
    # A bare file name (test_run_tests.py) is a path only when it exists here; otherwise it is a suite name.
    pathlike = bool(separator) or "/" in path_part or (path_part.endswith(".py") and (cwd / path_part).exists())
    if pathlike:
        rel = repo_relative(path_part, checkout, cwd)
        canonical = f"{rel}::{node}" if node else rel
        job = by_id.get(rel)
        if job is not None:
            if node:
                picker.narrow(job, rel[: -len(".py")].replace("/", ".") + "." + node.replace("::", "."), canonical)
            else:
                picker.whole(job)
            return
        for package in package_lanes(lanes):
            if package.package_dir is None:
                continue
            target = checkout / rel
            try:
                target.resolve().relative_to(package.package_dir.resolve())
            except ValueError:
                continue
            if target.resolve() == package.package_dir.resolve() and not node:
                picker.whole(package.jobs[0])
            else:
                picker.narrow(package.jobs[0], str(target) + (f"::{node}" if node else ""), canonical)
            return
        if rel.startswith("tests/") and rel.endswith(".py"):
            raise UsageError(f"{token}: {rel} is not a suite any workflow runs (tests/test_run_tests.py fails until it is wired into one)")
        raise UsageError(f"{token}: not a suite a workflow runs, or a path under a lane's directory")
    if _has_glob(token):
        matched = [job for job in suites if any(fnmatch.fnmatchcase(name, token) for name in (job.id, Path(job.id).stem, Path(job.id).stem[len("test_") :]))]
        if not matched:
            raise UsageError(f"{token!r} matches no unittest suite (--list shows them)")
        for job in matched:
            picker.whole(job)
        return
    stem = token[: -len(".py")] if token.endswith(".py") else token
    by_stem = {Path(job.id).stem: job for job in suites}
    for candidate in (stem, f"test_{stem}"):
        if candidate in by_stem:
            picker.whole(by_stem[candidate])
            return
    names = sorted({*lanes, *GROUPS, *lane_aliases(lanes, checkout), *by_stem, *(name[len("test_") :] for name in by_stem)})
    close = difflib.get_close_matches(token, names, n=4, cutoff=0.6)
    hint = f"; did you mean {', '.join(close)}?" if close else ""
    raise UsageError(f"unknown selector {token!r}{hint} (--list shows every lane and suite; quote a glob: '*budget*')")


def changed_files(checkout: Path, base: str) -> tuple[list[str], str]:
    """Files that differ from the merge base with ``base``: committed, staged, unstaged and untracked."""
    if not _is_checkout_root(checkout):
        raise UsageError(f"--changed needs a git checkout; {checkout} is not one")
    merge_base = _git(checkout, "merge-base", base, "HEAD")
    if not merge_base:
        raise UsageError(f"--changed: no merge base between {base} and HEAD (fetch it, or name another --base)")
    diff = _git(checkout, "diff", "--name-only", merge_base, "--") or ""
    untracked = _git(checkout, "ls-files", "--others", "--exclude-standard") or ""
    names = sorted({name.strip() for name in (diff + "\n" + untracked).splitlines() if name.strip()})
    return names, merge_base


def jobs_for_changes(names: list[str], lanes: dict[str, Lane], checkout: Path) -> list[tuple[Job, str]]:
    """The jobs a set of changed files touches, each with the reason.

    A gating package lane is chosen by its workflow's own path filter (the ``changes`` job's regex), as CI chooses it;
    a lane whose workflow has none, by its directory and workflow file. A unittest suite is chosen when it names a
    changed file: a heuristic, where CI runs every suite on every pull request.
    """
    picks: list[tuple[Job, str]] = []
    suites = [job for job in lanes[LANE_UNITTEST].jobs if job.kind == "unittest"]
    texts: dict[str, str] = {}

    def text(job: Job) -> str:
        if job.id not in texts:
            try:
                texts[job.id] = (checkout / job.id).read_text(encoding="utf-8", errors="replace")
            except OSError:
                texts[job.id] = ""
        return texts[job.id]

    for name in names:
        base = Path(name).name
        needles = {name, base}
        if name.startswith("tests/") and name.endswith(".py") and not base.startswith("test_"):
            needles.add(Path(name).stem)  # a helper module: suites import it by name
        for job in suites:
            if job.id == name:
                picks.append((job, "changed"))
            elif any(needle in text(job) for needle in needles):
                picks.append((job, f"mentions {name}"))
        for lane in package_lanes(lanes):
            if lane.path_filter:
                touched = re.search(lane.path_filter, name) is not None
            else:
                workflow = lane.workflow.relative_to(checkout).as_posix()
                package = lane.package_dir.relative_to(checkout).as_posix() if lane.package_dir else None
                touched = name == workflow or bool(package and (name == package or name.startswith(package + "/")))
            if touched:
                picks.append((lane.jobs[0], f"changed {name}"))
    return picks


def run_dirs(log_root: Path) -> list[Path]:
    try:
        return sorted(path for path in log_root.iterdir() if path.is_dir() and STAMP_RE.fullmatch(path.name) and (path / "summary.json").is_file())
    except OSError:
        return []


def last_summary(log_root: Path) -> tuple[Path, dict] | None:
    for path in reversed(run_dirs(log_root)):
        try:
            return path, json.loads((path / "summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------------------------------------------


def _git(checkout: Path, *args: str) -> str | None:
    env = {key: value for key, value in os.environ.items() if key not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE")}
    try:
        proc = subprocess.run(["git", "-C", str(checkout), *args], capture_output=True, text=True, env=env, check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _is_checkout_root(checkout: Path) -> bool:
    """Is ``checkout`` the top of its own git checkout (not a directory inside some other one)?"""
    top = _git(checkout, "rev-parse", "--show-toplevel")
    return bool(top) and Path(top).resolve() == checkout.resolve()


def primary_checkout(checkout: Path) -> Path:
    """The checkout holding the common .git: the primary of a linked worktree, else ``checkout`` itself."""
    if not _is_checkout_root(checkout):
        return checkout
    common = _git(checkout, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if common and Path(common).name == ".git":
        return Path(common).parent
    return checkout


def git_state(checkout: Path) -> dict[str, Any]:
    if not _is_checkout_root(checkout):
        return {}
    status = _git(checkout, "status", "--porcelain") or ""
    return {
        "branch": _git(checkout, "rev-parse", "--abbrev-ref", "HEAD"),
        "head": _git(checkout, "rev-parse", "HEAD"),
        "uncommitted": len([line for line in status.splitlines() if line.strip()]),
    }


def under_system_temp(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    for root in SYSTEM_TEMP_ROOTS:
        base = Path(root)
        if resolved == base or base in resolved.parents or path == base or base in path.parents:
            return True
    return False


def choose_tmpdir(requested: str | None, primary: Path, create: bool = True) -> tuple[Path, str]:
    """The children's TMPDIR, and a note when the inherited one was replaced."""
    if requested:
        path = Path(requested).expanduser().absolute()
        if under_system_temp(path):
            raise UsageError(f"--tmpdir {requested}: under the system temp directory, which no Juniper test may use (Overtoad directive, 2026-10-26)")
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path, ""
    current = os.environ.get("TMPDIR", "")
    if current and Path(current).is_dir() and not under_system_temp(Path(current)):
        return Path(current), ""
    path = primary / "tmp"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    why = "unset" if not current else ("under the system temp directory" if under_system_temp(Path(current)) else "not a directory")
    return path, f"TMPDIR was {why}; the children get {path}"


def resolve_python(requested: str | None) -> tuple[str, str]:
    """(interpreter path, its version). The path is not resolved, so a venv's interpreter stays the venv's."""
    if requested:
        candidate = requested if os.sep in requested else (shutil.which(requested) or requested)
        python = os.path.abspath(os.path.expanduser(candidate))
    else:
        python = sys.executable
    if not (os.path.isfile(python) and os.access(python, os.X_OK)):
        raise UsageError(f"--python {requested}: not an executable file")
    try:
        proc = subprocess.run([python, "-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])"], capture_output=True, text=True, check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise UsageError(f"{python}: cannot run it ({exc})") from None
    if proc.returncode != 0 or not re.fullmatch(r"\d+\.\d+\.\d+", proc.stdout.strip()):
        raise UsageError(f"{python}: does not run as a Python interpreter")
    return python, proc.stdout.strip()


def activation_dir(python: str) -> Path | None:
    """The interpreter's directory when its ``python3`` is this interpreter (a venv's or conda env's bin/)."""
    directory = Path(python).parent
    sibling = directory / "python3"
    try:
        if sibling.exists() and os.path.samefile(sibling, python):
            return directory
    except OSError:
        return None
    return None


def build_base_env(args: argparse.Namespace, tmpdir: Path, python: str) -> tuple[_Env, dict[str, str], list[str]]:
    """The environment every child starts from: (env, the variables set, the variables removed)."""
    env = _Env(os.environ)
    removed = [key for key in ALWAYS_REMOVED if key in env]
    if args.clean_env:
        removed += sorted(key for key in env if key.startswith(CLEAN_ENV_PREFIXES))
    for key in removed:
        env.pop(key, None)
    ceilings = [part for part in env.get("GIT_CEILING_DIRECTORIES", "").split(os.pathsep) if part]
    if str(tmpdir) not in ceilings:
        ceilings.insert(0, str(tmpdir))
    changes = {
        "DBUS_SESSION_BUS_ADDRESS": BUS_GUARD,
        "PYTHONUNBUFFERED": "1",
        "TMPDIR": str(tmpdir),
        "GIT_CEILING_DIRECTORIES": os.pathsep.join(ceilings),
    }
    if args.python:
        bindir = activation_dir(python)
        path = env.get("PATH", "")
        if bindir is not None and path.split(os.pathsep)[0] != str(bindir):
            changes["PATH"] = str(bindir) + os.pathsep + path
    env.update(changes)
    shown = dict(changes)
    if "PATH" in shown:
        shown["PATH"] = shown["PATH"].split(os.pathsep)[0] + os.pathsep + "..."
    return env, shown, removed


# ---------------------------------------------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------------------------------------------

_PROBE = r"""
import importlib.metadata as md, importlib.util, json, re, sys
spec = json.load(sys.stdin)
out = {"requirements": [], "packages": {}, "packaging": True}
try:
    from packaging.requirements import Requirement
except Exception:
    Requirement = None
    out["packaging"] = False
for text in spec["requirements"]:
    item = {"requirement": text, "status": "ok", "have": None}
    name, wanted = None, None
    if Requirement is not None:
        try:
            req = Requirement(text)
        except Exception:
            req = None
        if req is not None:
            if req.marker is not None and not req.marker.evaluate():
                item["status"] = "not-applicable"
                out["requirements"].append(item)
                continue
            name, wanted = req.name, req.specifier
    if name is None:
        match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", text)
        name = match.group(1) if match else text
    try:
        item["have"] = md.version(name)
    except md.PackageNotFoundError:
        item["status"] = "missing"
    else:
        if wanted is not None and len(wanted) and not wanted.contains(item["have"], prereleases=True):
            item["status"] = "unsatisfied"
    out["requirements"].append(item)
for name in spec["packages"]:
    try:
        found = importlib.util.find_spec(name)
    except Exception as exc:
        out["packages"][name] = {"origin": None, "error": "%s: %s" % (type(exc).__name__, exc)}
        continue
    origin = None
    if found is not None:
        origin = found.origin
        if origin in (None, "namespace") and found.submodule_search_locations:
            origin = list(found.submodule_search_locations)[0]
    out["packages"][name] = {"origin": origin, "error": None}
print(json.dumps(out))
"""


@dataclasses.dataclass
class LaneCheck:
    """The preflight verdict for one lane."""

    lane: str
    refused: str = ""  # non-empty: the lane cannot run (its jobs are UNAVAILABLE)
    notes: list[str] = dataclasses.field(default_factory=list)
    problems: list[dict] = dataclasses.field(default_factory=list)  # requirement entries that are not ok
    install: str = ""


def origin_verdict(name: str, origin: str | None, expected: Path) -> str:
    """'' when ``name`` imports from ``expected`` (this checkout), else why not."""
    if not origin:
        return f"{name} is not importable"
    try:
        Path(origin).resolve().relative_to(expected.resolve())
    except ValueError:
        return f"{name} imports from {origin}, not from this checkout's {expected}"
    return ""


def lane_env(lane: Lane, base_env: _Env) -> tuple[_Env, dict[str, str]]:
    env = _Env(base_env)
    delta: dict[str, str] = {}
    if lane.is_package and lane.pythonpath:
        delta["PYTHONPATH"] = os.pathsep.join(str(path) for path in lane.pythonpath)
    env.update(delta)
    return env, delta


def preflight(lanes: list[Lane], python: str, base_env: _Env) -> dict[str, LaneCheck]:
    """Ask the interpreter what each lane needs: its requirements, and where its packages import from."""

    def check(lane: Lane) -> LaneCheck:
        verdict = LaneCheck(lane.name)
        if not lane.requirements and not lane.packages:
            return verdict
        env, _ = lane_env(lane, base_env)
        cwd = next((step.cwd for step in lane.jobs[0].steps if _is_pytest(step.argv)), lane.jobs[0].steps[0].cwd)
        payload = json.dumps({"requirements": lane.requirements, "packages": sorted(lane.packages)})
        try:
            proc = subprocess.run([python, "-c", _PROBE], input=payload, capture_output=True, text=True, env=env, cwd=cwd, check=False, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            verdict.notes.append(f"preflight could not run ({exc}); nothing was checked")
            return verdict
        try:
            report = json.loads(proc.stdout)
        except ValueError:
            tail = (proc.stderr.strip().splitlines() or ["no output"])[-1]
            verdict.notes.append(f"preflight could not run ({tail}); nothing was checked")
            return verdict
        for name in sorted(lane.packages):
            entry = report["packages"].get(name) or {}
            why = entry.get("error") or origin_verdict(name, entry.get("origin"), lane.packages[name])
            if why:
                verdict.refused = why + " -- the lane would test other code (see README.md, 'Running tests locally')"
                break
        if lane.packages and not verdict.refused:
            verdict.notes.append(", ".join(sorted(lane.packages)) + " from this checkout")
        verdict.problems = [item for item in report["requirements"] if item["status"] in ("missing", "unsatisfied")]
        checked = [item for item in report["requirements"] if item["status"] != "not-applicable"]
        if checked:
            verdict.notes.append(f"{len(checked) - len(verdict.problems)} of {len(checked)} requirements satisfied")
        if not report.get("packaging"):
            verdict.notes.append("versions not compared: packaging is not installed in this interpreter")
        if verdict.problems:
            verdict.install = shlex.join([python, "-m", "pip", "install", *(item["requirement"] for item in verdict.problems)])
        return verdict

    with ThreadPoolExecutor(max_workers=max(1, min(8, len(lanes)))) as pool:
        return {check_.lane: check_ for check_ in pool.map(check, lanes)}


# ---------------------------------------------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Planned:
    """One step, ready to run."""

    label: str
    argv: list[str]
    cwd: Path
    env: _Env
    env_delta: dict[str, str]
    log: Path
    skip: str = ""  # non-empty: not run (a coverage gate with coverage off)
    unavailable: str = ""


def _slug(text: str) -> str:
    """A log file stem: tests/test_x.py -> test_x; util/check.py verify -> util_check.py_verify."""
    text = text[len("tests/") :] if text.startswith("tests/") else text
    text = text[: -len(".py")] if text.endswith(".py") else text
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "job"


def _is_cov_option(arg: str) -> bool:
    return arg == "--cov" or arg.startswith(("--cov=", "--cov-", "--no-cov"))


def resolve_program(argv: list[str], python: str, env: _Env) -> tuple[list[str], str]:
    """The argv with its program made runnable, and why it cannot be when it is missing."""
    program = argv[0]
    if _PYTHON_NAME.fullmatch(program):
        return [python, *argv[1:]], ""
    if program == "pytest":
        return [python, "-m", "pytest", *argv[1:]], ""
    if "/" in program:
        return list(argv), ""
    beside = Path(python).parent / program
    if beside.is_file() and os.access(beside, os.X_OK):
        return [str(beside), *argv[1:]], ""
    found = shutil.which(program, path=env.get("PATH"))
    if found:
        return [found, *argv[1:]], ""
    return list(argv), f"{program} is not installed next to {python} or on PATH"


def pytest_argv(argv: list[str], step: Step, job: Job, coverage: bool, lane_dir: Path, keywords: list[str], passthrough: list[str]) -> list[str]:
    """The workflow's pytest command, adjusted: coverage paths moved or coverage dropped, narrowing, -k, extras."""
    head, rest = argv[:3], argv[3:]
    out: list[str] = []
    skip_next = False
    for index, token in enumerate(rest):
        if skip_next:
            skip_next = False
            continue
        if not coverage and _is_cov_option(token):
            if token in ("--cov", "--cov-report", "--cov-fail-under", "--cov-config", "--cov-context") and index + 1 < len(rest) and not rest[index + 1].startswith("-"):
                skip_next = True
            continue
        if coverage and token.startswith("--cov-report=") and ":" in token.split("=", 1)[1]:
            kind, _, target = token.split("=", 1)[1].partition(":")
            token = f"--cov-report={kind}:{lane_dir / Path(target).name}"
        if job.targets and not token.startswith("-") and (step.cwd / token).exists():
            continue  # the workflow's own test path: the narrowed targets replace it
        out.append(token)
    if keywords:
        out += ["-k", keywords[0] if len(keywords) == 1 else " or ".join(f"({keyword})" for keyword in keywords)]
    return [*head, *out, *passthrough, *job.targets]


def gate_argv(argv: list[str], lane_dir: Path) -> list[str]:
    out: list[str] = []
    replace_next = False
    for arg in argv:
        if replace_next:
            out.append(str(lane_dir / Path(arg).name))
            replace_next = False
        elif arg == "--coverage-json":
            out.append(arg)
            replace_next = True
        elif arg.startswith("--coverage-json="):
            out.append("--coverage-json=" + str(lane_dir / Path(arg.split("=", 1)[1]).name))
        else:
            out.append(arg)
    return out


@dataclasses.dataclass
class Options:
    keywords: list[str]
    passthrough: list[str]
    no_coverage: bool
    python: str
    run_dir: Path


def plan_job(job: Job, lane: Lane, base_env: _Env, opts: Options, suffix: str = "") -> list[Planned]:
    lane_dir = opts.run_dir / job.lane
    env, delta = lane_env(lane, base_env)
    coverage = job.kind == "pytest" and not (opts.no_coverage or job.targets or opts.keywords)
    if coverage:
        delta = {**delta, "COVERAGE_FILE": str(lane_dir / ".coverage")}
        env["COVERAGE_FILE"] = delta["COVERAGE_FILE"]
    planned: list[Planned] = []
    for number, step in enumerate(job.steps, 1):
        name = f"{number}-{_slug(step.label)}" if job.kind == "pytest" else _slug(job.id)
        log = lane_dir / f"{name}{suffix}.log"
        if job.kind == "unittest":
            keyword_args = [part for keyword in opts.keywords for part in ("-k", keyword)]
            argv = [opts.python, "-m", "unittest", *job.unittest_options, *keyword_args, *opts.passthrough, *(job.targets or [job.id])]
            planned.append(Planned(step.label, argv, step.cwd, env, delta, log))
            continue
        argv, missing = resolve_program(step.argv, opts.python, env)
        if missing and step.hints:
            missing += " (the workflow installs it with: pip install " + " ".join(shlex.quote(hint) for hint in step.hints) + ")"
        if job.kind == "pytest" and _is_pytest(step.argv):
            argv = pytest_argv(argv, step, job, coverage, lane_dir, opts.keywords, opts.passthrough)
        elif step.coverage_gate:
            if not coverage:
                why = "--no-coverage" if opts.no_coverage else "a partial run (a path, a node id or -k)"
                planned.append(Planned(step.label, argv, step.cwd, env, delta, log, skip=f"skipped: coverage is off for {why}"))
                continue
            argv = gate_argv(argv, lane_dir)
        planned.append(Planned(step.label, argv, step.cwd, env, delta, log, unavailable=missing))
    return planned


# ---------------------------------------------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------------------------------------------


def _signal_group(pgid: int, sig: int) -> bool:
    """Signal a step's process group. False when there was nothing to signal: the group is gone, or not ours."""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pump(source: Any, log: Any) -> None:
    """Copy a step's output to the terminal and its log as it arrives (stream mode).

    Stops at end of output, or when either end is closed under it (the step's log is closed once the step has ended,
    and a process the step left behind may still hold the pipe until sweep_group() kills it).
    """
    try:
        for chunk in iter(lambda: source.read1(65536), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            log.write(chunk)
    except (OSError, ValueError):
        return  # a closed pipe, terminal or log: the step is over, so there is nothing left to copy


def sweep_group(pgid: int) -> bool:
    """Kill what a finished step left in its process group. True when something was left."""
    if not _group_alive(pgid):
        return False
    time.sleep(0.2)  # a grandchild that just exited is a zombie until its new parent reaps it
    if not _group_alive(pgid):
        return False
    _signal_group(pgid, signal.SIGTERM)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and _group_alive(pgid):
        time.sleep(0.05)
    _signal_group(pgid, signal.SIGKILL)
    return True


_UT_RAN = re.compile(r"^Ran (\d+) tests? in [\d.]+s\s*$", re.M)
_UT_RESULT = re.compile(r"^(OK|FAILED)(?: \(([^)]*)\))?\s*$", re.M)
_UT_PROBLEM = re.compile(r"^(?:FAIL|ERROR): (\S+) \(([^)\s]*)\)", re.M)
_PT_TOTALS = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)\b")
_PT_PROBLEM = re.compile(r"^(?:FAILED|ERROR) (\S+)", re.M)
_COV_FLOOR = re.compile(r"Required test coverage of ([\d.]+)% not reached\. Total coverage: ([\d.]+)%")


def unittest_counts(text: str) -> tuple[dict[str, int], list[str]]:
    counts: dict[str, int] = {}
    ran = _UT_RAN.findall(text)
    if ran:
        counts["ran"] = int(ran[-1])
    results = _UT_RESULT.findall(text)
    if results:
        for part in (results[-1][1] or "").split(","):
            key, _, value = part.strip().partition("=")
            if value.strip().isdigit():
                counts[key.strip().replace(" ", "_")] = int(value)
    failed: list[str] = []
    for name, where in _UT_PROBLEM.findall(text):
        ident = where if where.endswith("." + name) else f"{where}.{name}"
        if ident not in failed:
            failed.append(ident)
    return counts, failed


def pytest_counts(text: str) -> tuple[dict[str, int], list[str]]:
    counts: dict[str, int] = {}
    for line in reversed(text.splitlines()):
        if re.search(r"\bin [\d.]+s\b", line) and (_PT_TOTALS.search(line) or "no tests ran" in line):
            for number, word in _PT_TOTALS.findall(line):
                key = "errors" if word.startswith("error") else word
                counts[key] = counts.get(key, 0) + int(number)
            break
    return counts, _dedupe(_PT_PROBLEM.findall(text))


def describe_counts(kind: str, counts: dict[str, int]) -> str:
    if kind == "unittest" and "ran" in counts:
        bad = [f"{counts[key]} {word}" for key, word in (("failures", "failed"), ("errors", "errors"), ("unexpected_successes", "unexpected successes")) if counts.get(key)]
        text = f"{counts['ran']} tests"
        if bad:
            text = ", ".join(bad) + f" of {counts['ran']}"
        if counts.get("skipped"):
            text += f", {counts['skipped']} skipped"
        return text
    if kind == "pytest" and counts:
        order = ("passed", "failed", "errors", "skipped", "xfailed", "xpassed", "deselected")
        return ", ".join(f"{counts[key]} {key}" for key in order if counts.get(key))
    return ""


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


@dataclasses.dataclass
class Result:
    job: Job
    status: str
    detail: str
    duration: float
    steps: list[dict]
    counts: dict[str, int] = dataclasses.field(default_factory=dict)
    failed_tests: list[str] = dataclasses.field(default_factory=list)
    first_pass: "Result | None" = None

    @property
    def name(self) -> str:
        """The job as the user would select it: a suite, a command, a package, or the narrowed selectors."""
        if self.job.targets:
            return ", ".join(self.job.select)
        return self.job.id

    def logs(self) -> list[str]:
        return [step["log"] for step in self.steps if step.get("log") and step.get("status") not in (SKIPPED, None)]

    def to_json(self) -> dict:
        doc = {
            "lane": self.job.lane,
            "id": self.job.id,
            "kind": self.job.kind,
            "select": self.job.select,
            "targets": self.job.targets,
            "reason": self.job.reason,
            "status": self.status,
            "detail": self.detail,
            "duration_s": round(self.duration, 3),
            "counts": self.counts,
            "failed_tests": self.failed_tests,
            "steps": self.steps,
        }
        if self.first_pass is not None:
            doc["first_pass"] = {"status": self.first_pass.status, "detail": self.first_pass.detail, "steps": self.first_pass.steps}
        return doc


class Engine:
    """Runs steps as process groups, tracks them for Ctrl-C, and turns their outcomes into results."""

    def __init__(self, timeout: float | None, fail_fast: bool, filtered: bool) -> None:
        self.timeout = timeout
        self.fail_fast = fail_fast
        self.filtered = filtered
        self.lock = threading.Lock()
        self.running: dict[int, tuple[subprocess.Popen, str, float]] = {}
        self.interrupted = threading.Event()
        self.halted = threading.Event()

    def interrupt(self) -> None:
        self.interrupted.set()
        with self.lock:
            procs = [proc for proc, _, _ in self.running.values()]
        for proc in procs:
            _signal_group(proc.pid, signal.SIGTERM)

        def escalate() -> None:
            for proc in procs:
                if proc.returncode is None:
                    _signal_group(proc.pid, signal.SIGKILL)

        if procs:
            timer = threading.Timer(KILL_GRACE, escalate)
            timer.daemon = True
            timer.start()

    def snapshot(self) -> list[tuple[str, float]]:
        now = time.monotonic()
        with self.lock:
            return sorted(((label, now - started) for _, label, started in self.running.values()), key=lambda item: -item[1])

    def run_step(self, planned: Planned, label: str, stream: bool) -> dict:
        record: dict[str, Any] = {"label": planned.label, "argv": planned.argv, "cwd": str(planned.cwd), "env": planned.env_delta, "log": str(planned.log), "rc": None, "status": None, "detail": "", "duration_s": 0.0, "leftovers": False}
        if planned.skip:
            record.update(status=SKIPPED, detail=planned.skip, log=None)
            return record
        if planned.unavailable:
            record.update(status=UNAVAILABLE, detail=planned.unavailable, log=None)
            return record
        planned.log.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        expired = threading.Event()
        done = threading.Event()
        with open(planned.log, "ab") as log:  # O_APPEND: the footer lands after the child's own writes
            header = [f"{LOG_HEADER} job={label} step={planned.label}", f"{LOG_HEADER} cwd={planned.cwd}", f"{LOG_HEADER} argv={shlex.join(planned.argv)}"]
            header += [f"{LOG_HEADER} env {key}={value}" for key, value in planned.env_delta.items()]
            header.append(f"{LOG_HEADER} started={dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}")
            log.write(("\n".join(header) + "\n").encode())
            log.flush()
            try:
                proc = subprocess.Popen(planned.argv, cwd=planned.cwd, env=planned.env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE if stream else log, stderr=subprocess.STDOUT, start_new_session=True)
            except OSError as exc:
                record.update(status=UNAVAILABLE, detail=f"cannot start {planned.argv[0]}: {exc.strerror or exc}")
                log.write(f"{LOG_HEADER} {record['detail']}\n".encode())
                return record
            with self.lock:
                self.running[proc.pid] = (proc, label, started)
            timer = None
            if self.timeout:

                def kill_if_running() -> None:
                    if not done.is_set():
                        _signal_group(proc.pid, signal.SIGKILL)

                def expire() -> None:
                    expired.set()
                    _signal_group(proc.pid, signal.SIGTERM)
                    escalate = threading.Timer(KILL_GRACE, kill_if_running)
                    escalate.daemon = True
                    escalate.start()

                timer = threading.Timer(self.timeout, expire)
                timer.daemon = True
                timer.start()
            # Stream mode reads the pipe on its own thread. The step ends when its leader exits, not when the pipe
            # closes: a process the step leaves behind would hold the pipe open, and sweep_group() is what ends it.
            pump = None
            if stream and proc.stdout is not None:
                pump = threading.Thread(target=_pump, args=(proc.stdout, log), daemon=True)
                pump.start()
            try:
                rc = proc.wait()
            except KeyboardInterrupt:  # only the main thread sees it: the -j 1 and stream paths
                self.interrupt()
                rc = proc.wait()
            finally:
                if timer is not None:
                    timer.cancel()
                record["leftovers"] = sweep_group(proc.pid)
                done.set()
                if pump is not None:
                    pump.join(timeout=KILL_GRACE)
                with self.lock:
                    self.running.pop(proc.pid, None)
            duration = time.monotonic() - started
            if self.interrupted.is_set() and rc != 0:
                status, detail = NOT_RUN, "interrupted"
            elif expired.is_set():
                status, detail = TIMEOUT, f"timed out after {fmt_duration(self.timeout or 0)}"
            elif rc == 0:
                status, detail = PASSED, ""
            elif rc == NO_TESTS_RC and self.filtered:
                status, detail = NO_TESTS, "no test matched"
            else:
                status, detail = FAIL, f"exit {rc}"
            record.update(rc=rc, status=status, detail=detail, duration_s=round(duration, 3))
            log.write(f"\n{LOG_HEADER} exit={rc} status={status} duration={duration:.2f}s{' leftovers-killed' if record['leftovers'] else ''}\n".encode())
        return record

    def run_job(self, job: Job, planned: list[Planned], stream: bool, rerun: bool = False) -> Result:
        if not rerun and (self.halted.is_set() or self.interrupted.is_set()):
            return Result(job, NOT_RUN, "interrupted" if self.interrupted.is_set() else "not started: an earlier job failed (-x)", 0.0, [])
        started = time.monotonic()
        records: list[dict] = []
        stopped = ""
        for step in planned:
            if stopped:
                records.append({"label": step.label, "argv": step.argv, "cwd": str(step.cwd), "log": None, "rc": None, "status": NOT_RUN, "detail": stopped, "duration_s": 0.0, "leftovers": False})
                continue
            record = self.run_step(step, job.id if job.kind != "pytest" else job.lane, stream)
            records.append(record)
            if record["status"] not in (PASSED, SKIPPED, NO_TESTS):
                stopped = "interrupted" if self.interrupted.is_set() else "an earlier step did not pass"
        result = summarize(job, records, time.monotonic() - started)
        if result.status in (FAIL, TIMEOUT) and self.fail_fast:
            self.halted.set()
        return result


def summarize(job: Job, records: list[dict], duration: float) -> Result:
    ran = [record for record in records if record["status"] not in (SKIPPED, NOT_RUN)]
    statuses = [record["status"] for record in records]
    for status in (FAIL, TIMEOUT, UNAVAILABLE, NOT_RUN, NO_TESTS):
        if status in statuses:
            break
    else:
        status = PASSED
    counts: dict[str, int] = {}
    failed: list[str] = []
    texts = []
    for record in ran:
        if record.get("log"):
            try:
                texts.append(Path(record["log"]).read_text(encoding="utf-8", errors="replace"))
            except OSError:
                texts.append("")
    text = "\n".join(texts)
    if job.kind == "unittest":
        counts, failed = unittest_counts(text)
    elif job.kind == "pytest":
        counts, failed = pytest_counts(text)
    parts = [describe_counts(job.kind, counts)]
    worst = next((record for record in records if record["status"] == status and status != PASSED), None)
    floor = _COV_FLOOR.search(text) if job.kind == "pytest" else None
    if floor:
        parts.append(f"coverage {floor.group(2)}% is under the {floor.group(1)}% floor")
    if worst is not None:
        if status in (TIMEOUT, UNAVAILABLE, NOT_RUN, NO_TESTS):
            parts.append(worst["detail"])
        elif worst["label"] == "coverage-gate":
            parts.append("the per-file coverage gate failed")
        elif job.kind == "unittest" and "ran" not in counts:
            parts.append(f"{worst['detail']}: no unittest result line, so it died before or while loading (see the log)")
        elif job.kind == "pytest" and not counts:
            parts.append(f"{worst['detail']}: no pytest summary line (see the log)")
        elif job.kind == "command":
            parts.append(worst["detail"])
    if job.kind == "pytest" and status == PASSED:
        gates = [record for record in records if record["label"] == "coverage-gate"]
        if gates and gates[0]["status"] == SKIPPED:
            parts.append("coverage gates off")
        elif gates:
            parts.append("coverage gates passed")
    if job.kind == "command" and status == PASSED:
        parts.append("exit 0")
    if any(record.get("leftovers") for record in records):
        parts.append("killed processes it left behind")
    return Result(job, status, "; ".join(part for part in parts if part), duration, records, counts, failed)


# ---------------------------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------------------------


class Output:
    STYLES = {PASSED: "32", FAIL: "1;31", TIMEOUT: "1;31", UNAVAILABLE: "35", NOT_RUN: "2", RERUN_PASSED: "33", NO_TESTS: "33", "dim": "2", "head": "1", "warn": "33", "bad": "31"}

    def __init__(self, color: bool, quiet: bool) -> None:
        self.color = color
        self.quiet = quiet
        self.lock = threading.Lock()
        self.last = time.monotonic()

    def paint(self, text: str, style: str) -> str:
        code = self.STYLES.get(style)
        return f"\033[{code}m{text}\033[0m" if self.color and code else text

    def line(self, text: str = "") -> None:
        with self.lock:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
            self.last = time.monotonic()

    def error(self, text: str) -> None:
        with self.lock:
            sys.stdout.flush()
            sys.stderr.write(f"run_tests.py: {text}\n")
            sys.stderr.flush()


def tail_lines(path: str | None, count: int) -> list[str]:
    if not path or count <= 0:
        return []
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    body = [line for line in lines if not line.startswith(LOG_HEADER)]
    while body and not body[-1].strip():
        body.pop()
    return body[-count:]


class Reporter:
    def __init__(self, out: Output, total: int, width: int, tail: int, stream: bool) -> None:
        self.out = out
        self.total = total
        self.width = width
        self.tail = tail
        self.stream = stream
        self.count = 0

    def result(self, result: Result) -> None:
        self.count += 1
        if result.status == NO_TESTS or (self.out.quiet and result.status in (PASSED, RERUN_PASSED)):
            return  # -k across many suites: the suites it matched nothing in are counted in the summary only
        digits = len(str(self.total))
        status = self.out.paint(f"{result.status:<11}", result.status)
        self.out.line(f"[{self.count:>{digits}}/{self.total}] {status} {result.name:<{self.width}}  {result.detail}  {self.out.paint(fmt_duration(result.duration), 'dim')}")
        if result.status in (FAIL, TIMEOUT) and not self.stream:
            failing = [step for step in result.steps if step.get("status") in (FAIL, TIMEOUT)]
            for step in failing[:1]:
                for text in tail_lines(step.get("log"), self.tail):
                    self.out.line(self.out.paint("      │ ", "dim") + text)
            for step in failing:
                self.out.line(self.out.paint(f"      log: {step['log']}", "dim"))
        elif result.status == UNAVAILABLE:
            for step in result.steps:
                if step.get("status") == UNAVAILABLE:
                    self.out.line(self.out.paint(f"      {step['detail']}", "dim"))


class Heartbeat(threading.Thread):
    """Every minute without output, name the jobs still running."""

    def __init__(self, engine: Engine, out: Output) -> None:
        super().__init__(daemon=True)
        self.engine = engine
        self.out = out
        self.done = threading.Event()

    def run(self) -> None:
        while not self.done.wait(HEARTBEAT / 4):
            if time.monotonic() - self.out.last < HEARTBEAT:
                continue
            running = self.engine.snapshot()
            if running:
                shown = ", ".join(f"{label} {fmt_duration(age)}" for label, age in running[:4])
                more = f" and {len(running) - 4} more" if len(running) > 4 else ""
                self.out.line(self.out.paint(f"      ... still running ({len(running)}): {shown}{more}", "dim"))


def _display_path(path: Path, checkout: Path) -> str:
    try:
        return path.relative_to(checkout).as_posix() or "."
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------------------------------------------

_DESCRIPTION = """\
Run any subset of juniper-recurrence's tests locally, with the commands CI runs.

The lists come from the workflows on every run: the test job of each
.github/workflows/ci-recurrence-<lane>.yml, and every `python3 -m unittest` step of any workflow.
Every selected job runs (CI stops at the first failure), each in its own process group, and the
results go to one log per step plus summary.json under the primary checkout's tmp/."""

_EPILOG = """\
selectors (any number; default: all):
  all                       the unittest lane and every gating lane
  unittest                  every root tests/ suite a workflow runs with `python3 -m unittest`
  packages                  every gating lane: app, bench, client and model
  <lane>                    one ci-recurrence-<lane>.yml test job, as CI runs it: pytest with
                            its coverage floor, then the per-file coverage gate (app, client,
                            model), or the bench smoke suite from the repo root (bench). The
                            directory names work too: juniper-recurrence, juniper-recurrence-model,
                            juniper-recurrence-client
  optional                  every optional lane: a test-<name> job CI runs but does not gate on
                            (model-torch, the torch MLP readout: needs the model's [torch] extra)
  <suite>                   one unittest suite: tests/test_run_tests.py, test_run_tests.py,
                            test_run_tests or run_tests
  '<glob>'                  every unittest suite whose name matches: '*budget*'
  <suite>::<Class>[::<test>]  one class or test of a suite; tests.test_x.Class.test works too
  <lane path>[::<node>]     a file, directory or node id under a gating lane's directory (for
                            example juniper-recurrence-model/tests/test_lmu_model.py): that lane's
                            pytest on it alone, with the coverage gates off
  --changed adds the lanes and suites a change touches; --failed adds the failures of the last run.

examples:
  python3 util/run_tests.py                            everything, -j auto, past failures
  python3 util/run_tests.py model client               two lanes with their coverage gates
  python3 util/run_tests.py bench                      the bench smoke suite, its output live
  python3 util/run_tests.py juniper-recurrence-model/tests/test_lmu_model.py -- -x -vv
  python3 util/run_tests.py run_tests -k changed       one suite, the tests matching "changed"
  python3 util/run_tests.py --python "$VIRTUAL_ENV/bin/python" app
  python3 util/run_tests.py --changed                  what the branch touches (vs origin/main)
  python3 util/run_tests.py --failed                   the failures of the last run, again
  python3 util/run_tests.py --list optional            what a selector selects; runs nothing

exit codes:
  0    every selected job passed (a job that failed alongside others and passed alone counts,
       and is listed)
  1    at least one job failed or timed out
  2    usage or configuration error: an unknown selector, a workflow it cannot parse
  3    nothing failed, but at least one selected job could not run (UNAVAILABLE)
  5    no test ran: -k matched nothing in any selected job
  130  interrupted; summary.json holds the jobs that finished

Logs: the primary checkout's tmp/run-tests/<checkout name>/<UTC stamp>/ (one log per step,
summary.json, summary.txt; never pruned). Full documentation: README.md,
§ Running tests locally."""


def _jobs_value(text: str) -> int:
    if text == "auto":
        return max(1, min(8, (os.cpu_count() or 2) // 2))
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number or 'auto': {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _nonnegative(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_tests.py", usage="python3 util/run_tests.py [options] [SELECTOR ...] [-- RUNNER-ARGS ...]", description=_DESCRIPTION, epilog=_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("selectors", nargs="*", metavar="SELECTOR", help="what to run (below); default: all")
    what = parser.add_argument_group("choosing")
    what.add_argument("-k", dest="keywords", action="append", default=[], metavar="EXPR", help="run only tests whose name matches EXPR (unittest -k / pytest -k; repeatable: any match). Turns a package lane's coverage gates off")
    what.add_argument("--changed", action="store_true", help="add the jobs a change touches: each gating lane whose workflow's path filter matches a changed file (as CI decides), and every unittest suite that names one. The suites are a heuristic: CI runs every suite on every pull request")
    what.add_argument("--base", default=DEFAULT_BASE, metavar="REF", help=f"what --changed compares against, through the merge base (default {DEFAULT_BASE})")
    what.add_argument("--failed", "--lf", dest="failed", action="store_true", help="add the jobs that failed or timed out in the last recorded run of this checkout")
    what.add_argument("-l", "--list", action="store_true", help="list what the selectors select, by lane, and exit")
    run = parser.add_argument_group("running")
    run.add_argument("-j", "--jobs", type=_jobs_value, default="auto", metavar="N", help="jobs to run at once, or 'auto' (half the CPUs, at most 8; the default). 1 runs them in CI's order")
    run.add_argument("-x", "--fail-fast", action="store_true", help="start nothing new after the first failure, as CI's bash -e does")
    run.add_argument("--timeout", type=_nonnegative, default=DEFAULT_TIMEOUT, metavar="SECONDS", help=f"per step; the step's process group is killed when it expires (default {DEFAULT_TIMEOUT:.0f}; 0: none)")
    run.add_argument("--no-rerun", action="store_true", help="with -j above 1, do not re-run a failed job alone")
    run.add_argument("--no-coverage", action="store_true", help="package lanes: run pytest without coverage and skip the coverage gate")
    run.add_argument("--python", metavar="PATH", help="the interpreter for every command (default: the one running this script). Its directory goes first on PATH when its python3 is this interpreter (a venv's or conda env's bin/)")
    run.add_argument("-n", "--dry-run", action="store_true", help="print the environment, the preflight and every command; run and write nothing")
    env = parser.add_argument_group("environment")
    env.add_argument("--clean-env", action="store_true", help="also remove CLAUDE*, JUNIPER_* and WTC_* variables from the children, as on a CI runner")
    env.add_argument("--tmpdir", metavar="DIR", help="the children's TMPDIR (default: the inherited one, else the primary checkout's tmp/; never the system temp directory)")
    show = parser.add_argument_group("output")
    show.add_argument("-s", "--stream", action="store_true", help="show each step's output live (implies -j 1; the default when one job is selected)")
    show.add_argument("-q", "--quiet", action="store_true", help="print only failures and the summary")
    show.add_argument("--tail", type=int, default=DEFAULT_TAIL, metavar="N", help=f"lines of a failing step's output to print (default {DEFAULT_TAIL}; 0: none)")
    show.add_argument("--color", choices=("auto", "always", "never"), default="auto", help="colour the status words (default auto: a terminal, without NO_COLOR)")
    show.add_argument("--log-dir", metavar="DIR", help="where run directories go (default: the primary checkout's tmp/run-tests/<checkout name>)")
    show.add_argument("--checkout", metavar="DIR", help="the checkout to test (default: the one holding this script)")
    return parser


def _raise_interrupt(signum: int, frame: object) -> None:
    """SIGTERM ends a run the way Ctrl-C does: every running step's group is killed and the summary written."""
    raise KeyboardInterrupt


def split_passthrough(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        index = argv.index("--")
        return argv[:index], argv[index + 1 :]
    return argv, []


def _color(choice: str) -> bool:
    if choice == "always":
        return True
    if choice == "never":
        return False
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"


def print_list(jobs: list[Job], lanes: dict[str, Lane], out: Output, checkout: Path) -> None:
    by_lane: dict[str, list[Job]] = {}
    for job in jobs:
        by_lane.setdefault(job.lane, []).append(job)
    for name, lane_jobs in by_lane.items():
        lane = lanes[name]
        noun = "job" if len(lane_jobs) == 1 else "jobs"
        flag = out.paint("  optional: runs only when named", "warn") if lane.optional else ""
        out.line(out.paint(f"{name}", "head") + f"  {len(lane_jobs)} {noun}  ({lane.source}){flag}")
        for job in lane_jobs:
            extra = ("  only: " + ", ".join(job.targets)) if job.targets else ""
            extra += f"  ({job.reason})" if job.reason else ""
            if job.kind == "pytest":
                if extra:
                    out.line(f"  {extra.strip()}")
                for step in job.steps:
                    out.line(f"  $ {shlex.join(step.argv)}" + out.paint(f"   (in {_display_path(step.cwd, checkout)})", "dim"))
            else:
                out.line(f"  {job.id}{extra}" + (out.paint(f"   ({job.origin})", "dim") if job.origin and not extra else ""))
    packages = " ".join(lane.name for lane in package_lanes(lanes))
    optional = " ".join(lane.name for lane in package_lanes(lanes, optional=True)) or "none"
    out.line(out.paint(f"groups: packages = {packages}; optional = {optional}; all = unittest + packages", "dim"))


def write_summary(path: Path, doc: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def point_latest(log_root: Path, run_dir: Path) -> bool:
    """Point ``<log_root>/latest`` at the run directory. False when the filesystem refuses (no symlinks there): only
    the convenience link is missing, and --failed reads the newest summary.json, never the link."""
    link = log_root / "latest"
    temporary = log_root / f".latest-{os.getpid()}"
    try:
        if temporary.is_symlink() or temporary.exists():
            temporary.unlink()
        os.symlink(run_dir.name, temporary)
        os.replace(temporary, link)
    except OSError:
        return False
    return True


def make_run_dir(log_root: Path) -> Path:
    log_root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for number in range(1, 1000):
        candidate = log_root / (stamp if number == 1 else f"{stamp}-{number}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise UsageError(f"cannot create a run directory under {log_root}")


def summary_lines(results: list[Result], lanes: dict[str, Lane], paint: Callable[[str, str], str], verdict: str, wall: float, workers: int, run_dir: Path | None, changed: bool) -> list[str]:
    lines = ["", paint("Summary", "head")]
    by_lane: dict[str, list[Result]] = {}
    for result in sorted(results, key=lambda item: item.job.order):
        by_lane.setdefault(result.job.lane, []).append(result)
    width = max((len(name) for name in by_lane), default=8)
    for name, lane_results in by_lane.items():
        statuses = Counter(result.status for result in lane_results)
        parts = ", ".join(f"{statuses[status]} {STATUS_WORDS[status]}" for status in STATUS_ORDER if statuses[status])
        totals: Counter = Counter()
        for result in lane_results:
            totals.update(result.counts)
        kind = lane_results[0].job.kind
        tests = ""
        if kind == "unittest" or (name == LANE_UNITTEST and totals.get("ran")):
            tests = f"{totals.get('ran', 0)} tests" + (f", {totals['skipped']} skipped" if totals.get("skipped") else "")
        elif kind == "pytest":
            tests = describe_counts("pytest", dict(totals))
        noun = "job " if len(lane_results) == 1 else "jobs"
        lines.append(f"  {name:<{width}}  {len(lane_results):>4} {noun}  {parts}" + (paint(f"   ({tests})", "dim") if tests else ""))
    problems = [result for result in sorted(results, key=lambda item: item.job.order) if result.status in (FAIL, TIMEOUT, UNAVAILABLE)]
    if problems:
        lines += ["", paint("Problems", "head")]
        for result in problems:
            lines.append(f"  {paint(f'{result.status:<11}', result.status)} {result.name}  {result.detail}")
            for test in result.failed_tests[:10]:
                lines.append(f"      {test}")
            if len(result.failed_tests) > 10:
                lines.append(f"      ... and {len(result.failed_tests) - 10} more")
            for log in [step["log"] for step in result.steps if step.get("status") in (FAIL, TIMEOUT) and step.get("log")]:
                lines.append(paint(f"      log: {log}", "dim"))
            for step in result.steps:
                if step.get("status") == UNAVAILABLE and not result.detail.endswith(step["detail"]):
                    lines.append(paint(f"      {step['detail']}", "dim"))
    reruns = [result for result in results if result.status == RERUN_PASSED]
    if reruns:
        lines += ["", paint("Passed only when re-run alone", "head") + " (each failed alongside other jobs; CI runs one at a time -- interference or a flaky test)"]
        for result in sorted(reruns, key=lambda item: item.job.order):
            first = result.first_pass.detail if result.first_pass else ""
            lines.append(f"  {paint(f'{RERUN_PASSED:<11}', RERUN_PASSED)} {result.name}  first pass: {first}")
    not_run = [result for result in results if result.status == NOT_RUN]
    if not_run:
        lines.append(paint(f"  {len(not_run)} job(s) not run: {not_run[0].detail}", "dim"))
    lines += ["", f"{len(results)} jobs in {fmt_duration(wall)} (-j {workers}): {verdict}"]
    if any(result.status in (FAIL, TIMEOUT) for result in results):
        lines.append(paint("Run the failures again: python3 util/run_tests.py --failed", "dim"))
    if changed:
        lines.append(paint("--changed chose the lanes by their workflows' path filters and the suites by file references; CI runs every suite on every pull request. Run all before trusting green.", "dim"))
    if run_dir is not None:
        lines.append(f"Logs: {run_dir}")
    return lines


def exit_code(results: list[Result], interrupted: bool) -> tuple[int, str]:
    statuses = Counter(result.status for result in results)
    if interrupted:
        return EXIT_INTERRUPTED, "INTERRUPTED"
    bad = statuses[FAIL] + statuses[TIMEOUT]
    if bad:
        return EXIT_FAILED, f"FAILED ({bad} of {len(results)})"
    if statuses[UNAVAILABLE]:
        return EXIT_UNAVAILABLE, f"INCOMPLETE ({statuses[UNAVAILABLE]} unavailable)"
    if results and statuses[NO_TESTS] == len(results):
        return EXIT_NOTHING_RAN, "NO TESTS RAN (-k matched nothing)"
    if statuses[RERUN_PASSED]:
        return EXIT_OK, f"PASSED ({statuses[RERUN_PASSED]} only on a re-run alone)"
    return EXIT_OK, "PASSED"


def main(argv: list[str] | None = None) -> int:
    own, passthrough = split_passthrough(list(sys.argv[1:] if argv is None else argv))
    args = build_parser().parse_args(own)
    if args.stream and args.quiet:
        build_parser().error("--stream and --quiet do not go together")
    out = Output(_color(args.color), args.quiet)
    try:
        return _main(args, passthrough, out)
    except UsageError as exc:
        out.error(str(exc))
        return EXIT_USAGE


def _main(args: argparse.Namespace, passthrough: list[str], out: Output) -> int:
    if os.name != "posix":
        raise UsageError("POSIX only: every step runs in its own process group")
    checkout = Path(args.checkout).expanduser().resolve() if args.checkout else Path(__file__).resolve().parents[1]
    if not any((checkout / WORKFLOWS).glob(PACKAGE_WORKFLOW_GLOB)):
        raise UsageError(f"{checkout} has no {(WORKFLOWS / PACKAGE_WORKFLOW_GLOB).as_posix()}: not a juniper-recurrence checkout (--checkout)")
    lanes = discover(checkout)
    primary = primary_checkout(checkout)
    log_root = Path(args.log_dir).expanduser().absolute() if args.log_dir else primary / "tmp" / "run-tests" / checkout.name
    if under_system_temp(log_root):
        raise UsageError(f"--log-dir {log_root}: under the system temp directory, which no Juniper tool may use")

    picker = Picker(lanes)
    selectors = list(args.selectors) or ([] if (args.changed or args.failed) else [GROUP_ALL])
    cwd = Path.cwd()
    for token in selectors:
        resolve_selector(token, lanes, checkout, cwd, picker)
    notes: list[str] = []
    if args.changed:
        names, merge_base = changed_files(checkout, args.base)
        picks = jobs_for_changes(names, lanes, checkout)
        for job, reason in picks:
            picker.whole(job, reason)
        notes.append(f"--changed: {len(names)} file(s) differ from {args.base} (merge base {merge_base[:10]}); {len({(job.lane, job.id) for job, _ in picks})} job(s) touch them")
    if args.failed:
        found = last_summary(log_root)
        if found is None:
            raise UsageError(f"--failed: no recorded run under {log_root}")
        last_dir, summary = found
        failed = [job for job in summary.get("jobs") or [] if job.get("status") in (FAIL, TIMEOUT)]
        for entry in failed:
            for token in entry.get("select") or []:
                before = set(picker.chosen)
                try:
                    resolve_selector(token, lanes, checkout, checkout, picker)
                except UsageError as exc:
                    notes.append(f"--failed: skipped {token!r} ({exc})")
                    continue
                for key in set(picker.chosen) - before:
                    picker.chosen[key].reason = picker.chosen[key].reason or f"failed in {last_dir.name}"
        notes.append(f"--failed: {len(failed)} job(s) failed in {last_dir.name}")
    jobs = picker.jobs()
    if args.keywords:
        dropped = [job for job in jobs if job.kind == "command"]
        if dropped:
            jobs = [job for job in jobs if job.kind != "command"]
            notes.append(f"-k selects tests by name: skipping {len(dropped)} command job(s) that have none ({', '.join(job.id for job in dropped)})")
    if passthrough and {"unittest", "pytest"} <= {job.kind for job in jobs}:
        raise UsageError("arguments after -- go to one runner: select only unittest suites or only lanes that run pytest")

    if args.list:
        for note in notes:
            out.line(out.paint(note, "dim"))
        print_list(jobs, lanes, out, checkout)
        return EXIT_OK
    if not jobs:
        for note in notes:
            out.line(note)
        out.line("Nothing selected.")
        return EXIT_OK

    python, version = resolve_python(args.python)
    tmpdir, tmp_note = choose_tmpdir(args.tmpdir, primary, create=not args.dry_run)
    base_env, env_set, env_removed = build_base_env(args, tmpdir, python)
    stream = args.stream or (len(jobs) == 1 and not args.quiet)
    workers = 1 if stream else min(args.jobs, len(jobs))
    timeout = args.timeout or None
    run_dir = make_run_dir(log_root) if not args.dry_run else log_root / "<UTC stamp>"
    opts = Options(args.keywords, passthrough, args.no_coverage, python, run_dir)

    selected_lanes = [lanes[name] for name in dict.fromkeys(job.lane for job in jobs)]
    checks = preflight(selected_lanes, python, base_env)
    git = git_state(checkout)
    lane_counts = Counter(job.lane for job in jobs)
    out.line(out.paint("juniper-recurrence tests", "head") + f": {len(jobs)} job(s) -- " + ", ".join(f"{name} {count}" for name, count in lane_counts.items()))
    where = f"  ({git.get('branch')} @ {str(git.get('head'))[:10]}, {git.get('uncommitted')} uncommitted)" if git else ""
    out.line(f"  checkout  {checkout}{where}")
    out.line(f"  python    {version}  {python}")
    out.line(f"  TMPDIR    {tmpdir}" + (out.paint(f"  ({tmp_note})", "warn") if tmp_note else ""))
    changes = ", ".join(f"{key}={value}" for key, value in env_set.items() if key not in ("TMPDIR", "GIT_CEILING_DIRECTORIES"))
    out.line(f"  env       set {changes}; removed {', '.join(env_removed) or 'nothing'}")
    mode = f"-j {workers}, timeout {fmt_duration(timeout) if timeout else 'none'} per step, " + ("stop at the first failure" if args.fail_fast else "past failures")
    if workers > 1 and not args.no_rerun:
        mode += ", failures re-run alone"
    out.line(f"  mode      {mode}")
    out.line(f"  logs      {run_dir}" + (out.paint("  (dry run: nothing is written)", "dim") if args.dry_run else ""))
    for note in notes:
        out.line(out.paint(f"  {note}", "dim"))
    out.line(out.paint("Preflight", "head"))
    for lane in selected_lanes:
        check = checks[lane.name]
        if check.refused:
            out.line(f"  {out.paint('REFUSED', 'bad')}  {lane.name}: {check.refused}")
            continue
        word = out.paint("WARN   ", "warn") if check.problems else out.paint("ok     ", PASSED)
        out.line(f"  {word}  {lane.name}" + (": " + "; ".join(check.notes) if check.notes else ""))
        for item in check.problems:
            have = f" (have {item['have']})" if item.get("have") else ""
            out.line(f"           {item['status']}: {item['requirement']}{have}")
        if check.install:
            out.line(out.paint(f"           install: {check.install}", "dim"))

    plans = {(job.lane, job.id): plan_job(job, lanes[job.lane], base_env, opts) for job in jobs}
    if args.dry_run:
        out.line(out.paint("Plan", "head"))
        for job in jobs:
            out.line(f"  [{job.lane}] {job.id}" + (f"  ({job.reason})" if job.reason else "") + (out.paint("  REFUSED by the preflight", "bad") if checks[job.lane].refused else ""))
            for step in plans[(job.lane, job.id)]:
                if step.skip:
                    out.line(out.paint(f"      {step.label}: {step.skip}", "dim"))
                    continue
                env_text = " ".join(f"{key}={shlex.quote(value)}" for key, value in step.env_delta.items())
                out.line(f"      $ (cd {shlex.quote(_display_path(step.cwd, checkout))} && {env_text + ' ' if env_text else ''}{shlex.join(step.argv)})")
                if step.unavailable:
                    out.line(out.paint(f"        UNAVAILABLE: {step.unavailable}", "bad"))
        return EXIT_OK

    filtered = bool(args.keywords) or any(job.targets for job in jobs)
    engine = Engine(timeout, args.fail_fast, filtered)
    width = min(64, max(len(job.id) + (len(", ".join(job.targets)) + 3 if job.targets and job.kind == "unittest" else 0) for job in jobs))
    reporter = Reporter(out, len(jobs), width, args.tail, stream)
    previous = last_summary(log_root)
    durations: dict[tuple[str, str], float] = {}
    if previous is not None:
        for entry in previous[1].get("jobs") or []:
            durations[(entry.get("lane"), entry.get("id"))] = float(entry.get("duration_s") or 0)
    default_cost = {"pytest": 60.0, "unittest": 10.0, "command": 1.0}
    schedule = sorted(jobs, key=lambda job: (-durations.get((job.lane, job.id), default_cost[job.kind]), job.order)) if workers > 1 else jobs
    out.line(out.paint("Running", "head") + f" {len(jobs)} job(s)")
    started = time.monotonic()
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    results: dict[tuple[str, str], Result] = {}

    def refused(job: Job) -> Result:
        return Result(job, UNAVAILABLE, checks[job.lane].refused, 0.0, [])

    def record(result: Result) -> None:
        results[(result.job.lane, result.job.id)] = result
        reporter.result(result)

    previous_handler = signal.signal(signal.SIGTERM, _raise_interrupt)
    heartbeat = Heartbeat(engine, out) if not stream else None
    if heartbeat is not None:
        heartbeat.start()
    try:
        if workers == 1:
            for job in schedule:
                if engine.interrupted.is_set():
                    break
                record(refused(job) if checks[job.lane].refused else engine.run_job(job, plans[(job.lane, job.id)], stream))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {}
                for job in schedule:
                    if checks[job.lane].refused:
                        record(refused(job))
                    else:
                        futures[pool.submit(engine.run_job, job, plans[(job.lane, job.id)], False)] = job
                collected: set = set()

                def collect(future: Any) -> None:
                    collected.add(future)
                    try:
                        record(future.result())
                    except Exception as exc:  # noqa: BLE001 -- one job's internal error must not lose the run's summary
                        record(Result(futures[future], FAIL, f"run_tests.py internal error: {type(exc).__name__}: {exc}", 0.0, []))

                try:
                    for future in as_completed(futures):
                        collect(future)
                except KeyboardInterrupt:
                    engine.interrupt()
                    for future in futures:
                        future.cancel()
                    for future in futures:
                        if future not in collected and not future.cancelled():
                            collect(future)
            failures = [results[(job.lane, job.id)] for job in jobs if (job.lane, job.id) in results and results[(job.lane, job.id)].status in (FAIL, TIMEOUT)]
            if failures and not args.no_rerun and not engine.interrupted.is_set() and not engine.halted.is_set():
                out.line(out.paint("Re-running", "head") + f" {len(failures)} failed job(s) alone, one at a time")
                reporter.total += len(failures)
                for first in failures:
                    if engine.interrupted.is_set():
                        break
                    job = first.job
                    again = engine.run_job(job, plan_job(job, lanes[job.lane], base_env, opts, suffix=".rerun"), False, rerun=True)
                    if again.status == PASSED:
                        again = dataclasses.replace(again, status=RERUN_PASSED, detail=f"passed alone ({again.detail})" if again.detail else "passed alone")
                    again.first_pass = first
                    record(again)
    except KeyboardInterrupt:
        engine.interrupt()
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        if heartbeat is not None:
            heartbeat.done.set()
    interrupted = engine.interrupted.is_set()
    for job in jobs:
        results.setdefault((job.lane, job.id), Result(job, NOT_RUN, "interrupted" if interrupted else "not started", 0.0, []))
    ordered = [results[(job.lane, job.id)] for job in jobs]
    wall = time.monotonic() - started
    code, verdict = exit_code(ordered, interrupted)
    plain = summary_lines(ordered, lanes, lambda text, style: text, verdict, wall, workers, run_dir, args.changed)
    painted = summary_lines(ordered, lanes, out.paint, out.paint(verdict, PASSED if code == EXIT_OK else FAIL), wall, workers, run_dir, args.changed)
    for text in painted:
        out.line(text)
    doc = {
        "schema": SCHEMA,
        "tool": "util/run_tests.py",
        "started_utc": started_utc,
        "duration_s": round(wall, 3),
        "exit_code": code,
        "verdict": verdict,
        "interrupted": interrupted,
        "checkout": str(checkout),
        "git": git,
        "python": {"executable": python, "version": version},
        "argv": sys.argv[1:],
        "selectors": selectors,
        "keywords": args.keywords,
        "passthrough": passthrough,
        "options": {"jobs": workers, "timeout_s": timeout, "fail_fast": args.fail_fast, "rerun": not args.no_rerun, "coverage": not args.no_coverage, "changed": args.changed, "base": args.base if args.changed else None, "failed": args.failed, "clean_env": args.clean_env},
        "env": {"set": env_set, "removed": env_removed},
        "preflight": {name: dataclasses.asdict(check) for name, check in checks.items()},
        "jobs": [result.to_json() for result in ordered],
    }
    write_summary(run_dir / "summary.json", doc)
    (run_dir / "summary.txt").write_text("\n".join(plain).lstrip("\n") + "\n", encoding="utf-8")
    point_latest(log_root, run_dir)
    return code


if __name__ == "__main__":
    sys.exit(main())
