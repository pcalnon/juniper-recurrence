#!/usr/bin/env python3
"""Run any subset of juniper-recurrence's tests locally, with the commands CI runs.

Project: juniper-recurrence
Sub-Project: developer tooling (local test runs)
Author: Paul Calnon (authored via Claude Code)
Created: 2026-11-25 (a thin spec over the shared engine since 2026-11-28)
Status: permanent utility
Related: ``juniper_ci_tools.run_tests`` in juniper-ci-tools 0.10 (the engine this declares a spec for; its canonical
         home is juniper-ml's ``juniper-ci-tools/`` tree); .github/workflows/ci-recurrence-<lane>.yml and every
         workflow that runs ``python3 -m unittest`` (the lists it reads); tests/test_run_tests.py (this spec against
         the real workflows and the README; the engine's own suite is juniper-ml's
         juniper-ci-tools/tests/test_run_tests.py); README.md § Running tests locally (the operator documentation)

Why this exists
---------------
juniper-recurrence is a four-part monorepo, and CI tests each part in its own workflow: ``ci-recurrence-app.yml``,
``-model.yml`` and ``-client.yml`` run pytest in the package's directory with a 90% coverage floor and then a
per-file coverage gate, and ``ci-recurrence-bench.yml`` runs the bench smoke suite from the repository root
against a non-editable install of the app. The model workflow also has an optional torch lane. The root ``tests/``
suites run in no package lane: ``memory-budget.yml``, ``sequence-safety.yml`` and ``ci-tooling.yml`` each run one
with ``python3 -m unittest``. AGENTS.md's Testing section gives the per-package commands, but not the coverage
gates, the bench's install or the root suites, so a local run checked less than CI does. This script runs any
subset of what CI runs, reading the commands from the workflows on every run, so its lists cannot drift from CI's.

What is juniper-recurrence's own
--------------------------------
Everything that runs is the engine's: ``juniper_ci_tools.run_tests`` (the ``--help`` epilog and README.md say how it
selects, runs and reports). This file declares only what the engine cannot read from the workflows, as its
``RepoSpec``:

- every ``python3 -m unittest`` suite a pull-request workflow runs is a job of the ``unittest`` lane, whose
  requirements are what those jobs' "Install..." steps install;
- the ``test`` job of each ``ci-recurrence-<lane>.yml`` is the ``<lane>`` lane (``packages`` is all four), and a
  ``test-<name>`` job beside it (the model's ``test-torch``) is the optional lane ``<lane>-<name>``: CI runs it but
  does not gate on it, so it runs only when named (or with ``optional``);
- the checkout check: a checkout has ``ci-recurrence-*.yml`` workflows, and no root ``pyproject.toml``;
- the ``--help`` text, which tests/test_run_tests.py pins against README.md.

The engine ships in juniper-ci-tools 0.10 (``ENGINE_REQUIREMENT``), which ``ci-tooling.yml`` installs. Install it
into the environment you run this script with; without it the script prints the command that installs it and exits
3, the engine's UNAVAILABLE (nothing failed, and nothing could run).
"""

from __future__ import annotations

import shlex
import sys

# The juniper-ci-tools releases this spec is written for: ci-tooling.yml, which runs this script's suite, installs
# exactly this (tests/test_run_tests.py checks it).
ENGINE_REQUIREMENT = "juniper-ci-tools>=0.10.0,<0.11.0"

try:
    from juniper_ci_tools import run_tests as rt
except ImportError as exc:  # juniper-ci-tools is missing, or older than 0.10 (no run_tests module)
    _install = shlex.join([sys.executable, "-m", "pip", "install", ENGINE_REQUIREMENT])
    sys.stderr.write(f"run_tests.py: the test runner's engine, juniper_ci_tools.run_tests, cannot be imported ({exc})\n")
    sys.stderr.write(f"  install it: {_install}\n")
    raise SystemExit(3) from None  # the engine's EXIT_UNAVAILABLE

PACKAGE_PREFIX = "ci-recurrence-"
PACKAGE_WORKFLOWS = f"{PACKAGE_PREFIX}*.yml"
PACKAGE_JOB = "test"
OPTIONAL_JOBS = "test-*"  # beside a lane's test job: an optional lane, <lane>-<name>
PACKAGES_GROUP = "packages"

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

SPEC = rt.RepoSpec(
    project="juniper-recurrence",
    script=__file__,
    lanes=(
        rt.unittest_lane(),
        rt.job_lanes(workflows=(PACKAGE_WORKFLOWS,), jobs=(PACKAGE_JOB, OPTIONAL_JOBS), name=rt.workflow_stem(PACKAGE_PREFIX), optional={OPTIONAL_JOBS: "{workflow}'s {job} job, which CI runs beside its test job"}, groups=(PACKAGES_GROUP,)),
    ),
    marker=PACKAGE_WORKFLOWS,
    pyproject=False,
    docs="README.md, 'Running tests locally'",
    description=_DESCRIPTION,
    epilog=_EPILOG,
    unwired_hint="tests/test_run_tests.py fails until it is wired into one",
)


if __name__ == "__main__":
    raise SystemExit(rt.main(SPEC))
