#!/usr/bin/env python3
"""YAML-extraction rehearsal for main-verify.yml G3.1 catch-up BASE resolution.

Flood-remediation P2 gate G3 (ml#873 / §4 item 8): a quoted ``[skip ci]`` in a
merge-commit body skips THIS workflow entirely, so a window of merges can land
un-screened. The ``Resolve catch-up base`` step must reach back to the last tip
KNOWN SCREENED when that tip is an ancestor of HEAD (sweeping the skipped
window), else the last SUCCESSFUL tip, else ``github.event.before``, else
``HEAD^1``.

The base ratchets on SCREENED, not on GREEN (2026-08-23). Resolving it from
run-level ``status=success`` conflated "this window was screened" with "the
screens found nothing", and was the mechanism behind a recurring red ``main``:
a finding froze the base, so every later merge re-screened the same window and
failed on someone else's damage, each red guaranteeing the next. Design of
record: ``notes/JUNIPER_2026-08-23_JUNIPER-ML_MAIN-VERIFY-CATCHUP-BASE-SCREENED-NOT-GREEN-DESIGN.md``.

This unittest extracts the workflow's OWN shell (not a reimplementation) and
drives it over a hermetic git fixture + stub ``gh`` — the same idiom as
``tests/test_release_train_workflow_guard.py`` ModeResolutionMatrixTest.

Neither the workflow YAML nor ``util/sequence_safety/`` is otherwise lint-gated
for this resolver, so this unittest IS the gate.

Ported from juniper-ml (ml#1291 fan-out) by
``util/ad-hoc/port_main_verify_drift_tests.py``. The design of record stays there:
``notes/JUNIPER_2026-08-23_JUNIPER-ML_MAIN-VERIFY-CATCHUP-BASE-SCREENED-NOT-GREEN-DESIGN.md``.

This port differs from BOTH references in one way that matters: where they raise
``SkipTest`` when the workflow or the resolver step cannot be found, this FAILS.
This repo has a ``main-verify.yml``, so a skip would mean the file is misplaced --
and a misplaced guard is silently green, which is the very shape it exists to catch.

Project: juniper-recurrence
Author: Paul Calnon
Created: 2026-08-05
"""

from __future__ import annotations

import os
import re
import subprocess  # nosec B404 - runs the workflow's OWN extracted shell hermetically (fixed argv)
import tempfile
import unittest
from pathlib import Path

import yaml

WORKFLOW_NAME = "main-verify.yml"
STEP_NAME = "Resolve catch-up base"


def _child_env(**overrides: str) -> dict[str, str]:
    """Build a MINIMAL child environment rather than copying ``os.environ``.

    juniper-ml's original passes ``RedactedEnv(os.environ, ...)`` -- a masked copy of
    the real environment -- because a subprocess-driving test leaves its env mapping
    as a frame-local in every failure traceback, and ``--showlocals`` renders it. This
    port does not copy the environment at all, so there is nothing to mask: the
    fixture needs only ``PATH`` (the stub ``gh`` is prepended to it), ``HOME`` for
    git, and an explicit identity. That also removes the juniper-ml-only
    ``tests.redacted_env`` import, which does not exist in this repo.
    """
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),  # nosec B108 - fallback only; git needs a HOME
        "LANG": "C",
    }
    env.update(overrides)
    return env


def _find_repo_root(start: Path) -> Path:
    cur = start
    for _ in range(8):
        if (cur / ".github" / "workflows").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise AssertionError(f"could not locate repo root with .github/workflows from {start}")


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(  # nosec B603 B607 - fixed git argv in temp fixture
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=_child_env(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"),
    )
    if proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr}")
    return proc.stdout.strip()


class CatchUpBaseRehearsalTest(unittest.TestCase):
    """Extract and run the real ``Resolve catch-up base`` shell over the G3.1 matrix."""

    script: str

    @classmethod
    def setUpClass(cls) -> None:
        repo_root = _find_repo_root(Path(__file__).resolve().parent)
        wf = repo_root / ".github" / "workflows" / WORKFLOW_NAME
        if not wf.is_file():
            raise AssertionError(f"{WORKFLOW_NAME} not present at {wf} -- this repo HAS one, so this test is misplaced (a skip here would be silently green)")
        doc = yaml.safe_load(wf.read_text(encoding="utf-8"))
        steps = doc.get("jobs", {}).get("symbol-screen", {}).get("steps", [])
        step = next((s for s in steps if s.get("name") == STEP_NAME or s.get("id") == "base"), None)
        if step is None or "run" not in step:
            raise AssertionError(f"could not locate {STEP_NAME!r} run step in {WORKFLOW_NAME} -- that IS the drift this test guards, so it must fail, not skip")
        cls.script = step["run"]

    def _stage_repo(self, root: Path) -> tuple[str, str, str]:
        """Build A -> B -> C linear history; return (sha_a, sha_b, sha_c=HEAD)."""
        repo = root / "repo"
        repo.mkdir()
        _git(repo, "init")
        _git(repo, "config", "user.email", "t@t")
        _git(repo, "config", "user.name", "t")
        # Avoid signing noise in hermetic fixtures.
        _git(repo, "config", "commit.gpgsign", "false")
        shas: list[str] = []
        for label in ("A", "B", "C"):
            (repo / "f.txt").write_text(f"{label}\n", encoding="utf-8")
            _git(repo, "add", "f.txt")
            _git(repo, "commit", "-m", f"commit {label}")
            shas.append(_git(repo, "rev-parse", "HEAD"))
        return shas[0], shas[1], shas[2]

    def _run_resolver(
        self,
        *,
        repo: Path,
        head_sha: str,
        before: str,
        last_ok: str,
        repo_name: str = "pcalnon/juniper-recurrence",
        completed_runs: list[tuple[str, str]] | None = None,
        verdicts: dict[str, str] | None = None,
    ) -> tuple[str, str, str]:
        """Return (base, reason_line, step_summary).

        ``completed_runs`` / ``verdicts`` drive the TIER 1 (screened) walk added
        2026-08-23: the resolver lists completed runs, then asks each one's jobs
        for the conclusion of the ``Assert screens reached a verdict`` step.
        ``last_ok`` drives the legacy TIER 2 ``status=success`` query. Leaving the
        tier-1 inputs empty exercises tier 2 and below, which is what the five
        pre-existing cases below do.
        """
        completed_runs = completed_runs or []
        verdicts = verdicts or {}
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            script_path = td_path / "resolve.sh"
            script_path.write_text(self.script, encoding="utf-8")
            gh_out = td_path / "gh_output"
            gh_out.write_text("", encoding="utf-8")
            step_summary = td_path / "step_summary"
            step_summary.write_text("", encoding="utf-8")

            # Fixture data for the argument-aware ``gh`` stub below.
            runs_file = td_path / "runs.txt"
            runs_file.write_text("".join(f"{rid} {sha}\n" for rid, sha in completed_runs), encoding="utf-8")
            verdicts_file = td_path / "verdicts.txt"
            verdicts_file.write_text("".join(f"{rid} {conc}\n" for rid, conc in verdicts.items()), encoding="utf-8")

            stub_bin = td_path / "bin"
            stub_bin.mkdir()
            # Argument-aware stub. The resolver makes THREE distinct request shapes and
            # they must not be conflated, or a tier-2 answer masquerades as tier 1:
            #   * …/runs?status=completed…  -> the tier-1 candidate list ("<id> <sha>" lines)
            #   * …/actions/runs/<id>/jobs  -> that run's verdict-step conclusion
            #   * …/runs?status=success…    -> the legacy tier-2 head_sha
            gh = stub_bin / "gh"
            gh.write_text(
                "#!/usr/bin/env bash\n" "set -euo pipefail\n" 'url=""\n' 'for a in "$@"; do\n' '  case "$a" in repos/*) url="$a" ;; esac\n' "done\n" 'case "$url" in\n' f'  *status=completed*) cat "{runs_file}" ;;\n' "  */jobs)\n" '    rid="${url%/jobs}"; rid="${rid##*/}"\n' f'    awk -v id="$rid" \'$1==id{{print $2}}\' "{verdicts_file}"\n' "    ;;\n" f'  *status=success*) printf "%s\\n" "{last_ok}" ;;\n' "  *) : ;;\n" "esac\n",
                encoding="utf-8",
            )
            gh.chmod(0o755)

            env = _child_env()
            env["PATH"] = str(stub_bin) + os.pathsep + env.get("PATH", "")
            env["GH_TOKEN"] = "unused"  # nosec B105 - dummy token for the PATH-stubbed gh, never a real credential
            env["BEFORE"] = before
            env["HEAD_SHA"] = head_sha
            env["REPO"] = repo_name
            env["GITHUB_OUTPUT"] = str(gh_out)
            env["GITHUB_STEP_SUMMARY"] = str(step_summary)

            proc = subprocess.run(  # nosec B603 B607 - workflow shell, fixed argv
                ["bash", str(script_path)],
                cwd=repo,
                capture_output=True,
                text=True,
                env=env,
                check=False,
                timeout=30,
            )
            combined = proc.stdout + proc.stderr
            self.assertEqual(proc.returncode, 0, msg=combined)
            written = gh_out.read_text(encoding="utf-8")
            m = re.search(r"^base=(.*)$", written, re.MULTILINE)
            self.assertIsNotNone(m, f"no base= in GITHUB_OUTPUT:\n{written}\n---\n{combined}")
            reason_m = re.search(r"Post-merge screen base: (.+)$", combined, re.MULTILINE)
            reason = reason_m.group(1).strip() if reason_m else ""
            return m.group(1).strip(), reason, step_summary.read_text(encoding="utf-8")

    def test_ancestor_last_ok_wins_catchup(self) -> None:
        """Successful main-verify tip that is an ancestor of HEAD becomes BASE (sweep)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sha_a, sha_b, sha_c = self._stage_repo(root)
            repo = root / "repo"
            base, reason, summary = self._run_resolver(
                repo=repo,
                head_sha=sha_c,
                before=sha_b,
                last_ok=sha_a,
            )
            self.assertEqual(base, sha_a)
            self.assertIn("catch-up from", reason)
            self.assertIn(sha_a, reason)
            self.assertIn(sha_a, summary)
            self.assertIn(sha_c, summary)

    def test_non_ancestor_last_ok_falls_to_event_before(self) -> None:
        """A tip that is not an ancestor of HEAD must not invent catch-up BASE."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sha_a, sha_b, sha_c = self._stage_repo(root)
            # Divergent tip: orphan commit not on HEAD's ancestry.
            orphan = root / "orphan"
            orphan.mkdir()
            _git(orphan, "init")
            _git(orphan, "config", "user.email", "t@t")
            _git(orphan, "config", "user.name", "t")
            _git(orphan, "config", "commit.gpgsign", "false")
            (orphan / "x").write_text("x\n", encoding="utf-8")
            _git(orphan, "add", "x")
            _git(orphan, "commit", "-m", "orphan")
            foreign = _git(orphan, "rev-parse", "HEAD")
            # Fetch the foreign object into the fixture so rev-parse succeeds but
            # merge-base --is-ancestor fails (not an ancestor of HEAD).
            repo = root / "repo"
            _git(repo, "fetch", str(orphan), "HEAD:refs/heads/foreign")
            base, reason, _summary = self._run_resolver(
                repo=repo,
                head_sha=sha_c,
                before=sha_b,
                last_ok=foreign,
            )
            self.assertEqual(base, sha_b)
            self.assertIn("event.before", reason)
            self.assertIn(sha_b, reason)
            self.assertNotIn("catch-up", reason)
            # silence unused
            self.assertTrue(sha_a)

    def test_zero_before_and_empty_last_ok_uses_head_parent(self) -> None:
        """Force-push / initial / dispatch: zero BEFORE + empty last_ok → HEAD^1."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _sha_a, sha_b, sha_c = self._stage_repo(root)
            repo = root / "repo"
            zeros = "0" * 40
            base, reason, _summary = self._run_resolver(
                repo=repo,
                head_sha=sha_c,
                before=zeros,
                last_ok="",
            )
            self.assertEqual(base, sha_b)
            self.assertIn("HEAD^1 fallback", reason)

    def test_last_ok_equal_head_skips_catchup(self) -> None:
        """last_ok == HEAD must not select itself as BASE (empty screen window)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _sha_a, sha_b, sha_c = self._stage_repo(root)
            repo = root / "repo"
            base, reason, _summary = self._run_resolver(
                repo=repo,
                head_sha=sha_c,
                before=sha_b,
                last_ok=sha_c,
            )
            self.assertEqual(base, sha_b)
            self.assertIn("event.before", reason)
            self.assertNotIn("catch-up", reason)

    def test_null_last_ok_jq_token_falls_through(self) -> None:
        """gh/jq ``null`` string must not be treated as a real SHA."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _sha_a, sha_b, sha_c = self._stage_repo(root)
            repo = root / "repo"
            base, reason, _summary = self._run_resolver(
                repo=repo,
                head_sha=sha_c,
                before=sha_b,
                last_ok="null",
            )
            self.assertEqual(base, sha_b)
            self.assertIn("event.before", reason)

    # ── TIER 1: ratchet on SCREENED, not on GREEN (2026-08-23) ────────────────────────
    # Design of record:
    # notes/JUNIPER_2026-08-23_JUNIPER-ML_MAIN-VERIFY-CATCHUP-BASE-SCREENED-NOT-GREEN-DESIGN.md

    def test_red_screen_run_still_advances_base(self) -> None:
        """THE REGRESSION. A run whose screen FAILED still advances the base.

        This is the defect that made red ``main`` self-perpetuating (4 occurrences,
        2026-08-12 .. 2026-08-21). Run 100 screened tip B and found something, so the
        run is not ``status=success`` and the legacy tier reaches all the way back to
        A -- re-screening B's finding on every later merge and failing innocent commit
        C for damage done at B. Exit 1 IS a verdict: the window was screened, so the
        base must advance to B.

        Fails against the pre-2026-08-23 resolver, which returns A here.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sha_a, sha_b, sha_c = self._stage_repo(root)
            repo = root / "repo"
            base, reason, summary = self._run_resolver(
                repo=repo,
                head_sha=sha_c,
                before=sha_b,
                last_ok=sha_a,  # the last GREEN run, far behind
                completed_runs=[("100", sha_b)],
                verdicts={"100": "success"},  # verdict REACHED, screen was red
            )
            self.assertEqual(base, sha_b, "a screened-but-red tip must become BASE")
            self.assertNotEqual(base, sha_a, "must not fall back to the last GREEN tip")
            self.assertIn("screened-tip catch-up from", reason)
            self.assertIn(sha_b, summary)

    def test_invocation_error_run_does_not_advance_base(self) -> None:
        """Exit >=2 is NOT a verdict: the window is un-screened, so do not advance."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sha_a, sha_b, sha_c = self._stage_repo(root)
            repo = root / "repo"
            base, reason, _summary = self._run_resolver(
                repo=repo,
                head_sha=sha_c,
                before=sha_b,
                last_ok=sha_a,
                completed_runs=[("100", sha_b)],
                verdicts={"100": "failure"},  # the verdict-assert step itself failed
            )
            self.assertEqual(base, sha_a, "an un-screened window must fall through to tier 2")
            self.assertNotIn("screened-tip", reason)

    def test_skipped_verdict_step_does_not_advance_base(self) -> None:
        """A screens step that died leaves the assert `skipped` -- never coverage."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sha_a, sha_b, sha_c = self._stage_repo(root)
            repo = root / "repo"
            base, reason, _summary = self._run_resolver(
                repo=repo,
                head_sha=sha_c,
                before=sha_b,
                last_ok=sha_a,
                completed_runs=[("100", sha_b)],
                verdicts={"100": "skipped"},
            )
            self.assertEqual(base, sha_a)
            self.assertNotIn("screened-tip", reason)

    def test_missing_verdict_step_falls_through_to_legacy_tier(self) -> None:
        """Transition: historical runs carry no verdict step, so tier 2 must still work."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sha_a, sha_b, sha_c = self._stage_repo(root)
            repo = root / "repo"
            base, reason, _summary = self._run_resolver(
                repo=repo,
                head_sha=sha_c,
                before=sha_b,
                last_ok=sha_a,
                completed_runs=[("100", sha_b)],
                verdicts={},  # jq yields "" -- no such step in this run
            )
            self.assertEqual(base, sha_a)
            self.assertIn("catch-up from", reason)
            self.assertNotIn("screened-tip", reason)

    def test_screened_walk_skips_non_ancestor_and_self(self) -> None:
        """The walk continues past a newer tip that is unusable for THIS head.

        Newest-first the candidates are: HEAD itself (empty window), a foreign tip
        from a concurrent/force-pushed branch, then the usable B. A single-shot query
        would abandon catch-up on either of the first two.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sha_a, sha_b, sha_c = self._stage_repo(root)
            orphan = root / "orphan"
            orphan.mkdir()
            _git(orphan, "init")
            _git(orphan, "config", "user.email", "t@t")
            _git(orphan, "config", "user.name", "t")
            _git(orphan, "config", "commit.gpgsign", "false")
            (orphan / "x").write_text("x\n", encoding="utf-8")
            _git(orphan, "add", "x")
            _git(orphan, "commit", "-m", "orphan")
            foreign = _git(orphan, "rev-parse", "HEAD")
            repo = root / "repo"
            _git(repo, "fetch", str(orphan), "HEAD:refs/heads/foreign")
            base, reason, _summary = self._run_resolver(
                repo=repo,
                head_sha=sha_c,
                before=sha_b,
                last_ok=sha_a,
                completed_runs=[("102", sha_c), ("101", foreign), ("100", sha_b)],
                verdicts={"102": "success", "101": "success", "100": "success"},
            )
            self.assertEqual(base, sha_b)
            self.assertIn("screened-tip catch-up from", reason)

    def test_screened_tier_outranks_a_newer_legacy_success(self) -> None:
        """Tier 1 is consulted first even when tier 2 would also answer."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sha_a, sha_b, sha_c = self._stage_repo(root)
            repo = root / "repo"
            base, reason, _summary = self._run_resolver(
                repo=repo,
                head_sha=sha_c,
                before=sha_b,
                last_ok=sha_a,
                completed_runs=[("100", sha_b)],
                verdicts={"100": "success"},
            )
            self.assertEqual(base, sha_b)
            self.assertIn("screened-tip", reason)


class VerdictStepNameDriftTest(unittest.TestCase):
    """The tier-1 signal is an EXACT step name; drift is silent and must be pinned.

    Renaming the step does not fail anything on its own -- the resolver simply matches
    nothing, drops to the legacy tier, and restores the recurring-red defect while every
    check stays green. That is the vacuous-pass shape, so BOTH halves are pinned here:
    the workflow must define the step, and the resolver must grep for the same literal.
    Either assertion alone can drift past the other.
    """

    VERDICT_STEP = "Assert screens reached a verdict"
    SCREEN_JOB = "Symbol & Docs Screen"

    doc: dict
    job: dict

    @classmethod
    def setUpClass(cls) -> None:
        repo_root = _find_repo_root(Path(__file__).resolve().parent)
        wf = repo_root / ".github" / "workflows" / WORKFLOW_NAME
        if not wf.is_file():
            raise AssertionError(f"{WORKFLOW_NAME} not present at {wf} -- this repo HAS one, so this test is misplaced (a skip here would be silently green)")
        cls.doc = yaml.safe_load(wf.read_text(encoding="utf-8"))
        cls.job = cls.doc.get("jobs", {}).get("symbol-screen", {})

    def test_workflow_defines_the_verdict_assert_step(self) -> None:
        """The coverage signal the resolver reads must actually exist."""
        names = [s.get("name") for s in self.job.get("steps", [])]
        self.assertIn(self.VERDICT_STEP, names, f"tier-1 coverage step missing; steps are {names}")

    def test_verdict_step_precedes_the_clean_assert(self) -> None:
        """Coverage must be asserted BEFORE the verdict, or a finding skips it."""
        names = [s.get("name") for s in self.job.get("steps", [])]
        self.assertIn("Assert screens clean", names)
        self.assertLess(names.index(self.VERDICT_STEP), names.index("Assert screens clean"))

    def test_screens_step_does_not_fail_on_findings(self) -> None:
        """The screens step must record exit codes, not exit on them.

        If it exits non-zero on a finding, the coverage step is SKIPPED on exactly the
        runs whose windows most need marking screened -- silently reinstating the bug.
        """
        steps = self.job.get("steps", [])
        screens = next((s for s in steps if s.get("id") == "screens"), None)
        self.assertIsNotNone(screens, "screens step (id: screens) not found")
        run = screens["run"]
        self.assertIn('echo "src=${src}" >> "$GITHUB_OUTPUT"', run)
        self.assertIn('echo "drc=${drc}" >> "$GITHUB_OUTPUT"', run)
        self.assertNotIn("exit 1", run, "screens step must not fail on a finding")

    def test_resolver_greps_the_same_verdict_step_name(self) -> None:
        """The resolver's literal and the step's name must not drift apart."""
        steps = self.job.get("steps", [])
        resolver = next((s for s in steps if s.get("id") == "base"), None)
        self.assertIsNotNone(resolver, "resolver step (id: base) not found")
        run = resolver["run"]
        self.assertIn(self.VERDICT_STEP, run)
        self.assertIn(self.SCREEN_JOB, run)

    def test_resolver_prefers_the_screened_tier_over_legacy_success(self) -> None:
        """Tier 1 must be consulted first, and tier 2 must survive as the fallback.

        Deleting tier 2 would strand the FIRST run after this change (no historical run
        carries the tier-1 step name); deleting tier 1 is the defect itself.
        """
        steps = self.job.get("steps", [])
        resolver = next((s for s in steps if s.get("id") == "base"), None)
        self.assertIsNotNone(resolver)
        run = resolver["run"]
        self.assertIn("status=completed", run, "tier-1 walk over completed runs is missing")
        self.assertIn("status=success", run, "tier-2 legacy fallback is missing")
        self.assertLess(
            run.index('if [ -n "$screened" ]'),
            run.index('elif [ -n "$last_ok" ]'),
            "screened tier must be tested before the legacy success tier",
        )

    def test_screen_job_name_matches_the_workflow_job(self) -> None:
        """The resolver filters jobs by display name; pin it to the real one."""
        self.assertEqual(self.job.get("name"), self.SCREEN_JOB)


if __name__ == "__main__":
    unittest.main()
