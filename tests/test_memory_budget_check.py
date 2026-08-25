#!/usr/bin/env python3
"""
Project:     Juniper
Sub-Project: juniper-recurrence
Application: tests
Author:      Paul Calnon
Version:     0.1.0
License:     MIT License

Tests for ``util/memory_budget_check.py`` (P2 of the shared-session-memory plan).

``util/`` is outside every pre-commit Python hook's scope (flake8/bandit scope to
``scripts/`` + ``tests/``), so this suite IS the gate -- the same gap that left
``tests/test_assert_release_tag.py`` unwired and its vacuous-pass guard unrun.

The load-bearing cases are the ones a well-meaning refactor silently breaks:

* the **no-worsening rule** -- over-ceiling alone must NOT fail; it must also have
  grown, or one bad file on main blocks every unrelated PR and the gate gets
  disabled rather than obeyed;
* the **ratchet never loosening**;
* the **waiver being a loan** -- it suppresses the failure without moving the
  ceiling;
* and the **machinery negative controls**. This repo has a documented class where
  a check's machinery breaks and it reports SUCCESS. A gate that cannot fail is
  not a gate, so each way this one could go blind is pinned to exit 2.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess  # nosec B404 - the checker under test is driven as a subprocess by design
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "util" / "memory_budget_check.py"

_spec = importlib.util.spec_from_file_location("memory_budget_check", MODULE_PATH)
assert _spec and _spec.loader
mbc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mbc)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(  # nosec B603 B607 - fixed git argv into a TemporaryDirectory; no untrusted input
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin", "HOME": str(repo)},
    )


class BudgetFixture:
    """A throwaway git repo with one governed file at a known size."""

    def __init__(self, tmp: Path, base_chars: int, ceiling: int) -> None:
        self.root = tmp
        _git(self.root, "init", "-q", "-b", "main")
        self.governed = self.root / "AGENTS.md"
        self.governed.write_text("x" * base_chars, encoding="utf-8")
        self.budget_path = self.root / "budget.json"
        self.budget_path.write_text(
            json.dumps({"files": {"AGENTS.md": {"ceiling_chars": ceiling}}}),
            encoding="utf-8",
        )
        _git(self.root, "add", "-A")
        _git(self.root, "commit", "-qm", "base")

    def set_size(self, chars: int) -> None:
        self.governed.write_text("x" * chars, encoding="utf-8")

    def set_ceiling(self, chars: int) -> None:
        """Rewrite the ceiling in the WORKING TREE only, leaving HEAD's value
        intact -- which is exactly the shape of a hand-edited raise."""
        self.budget_path.write_text(
            json.dumps({"files": {"AGENTS.md": {"ceiling_chars": chars}}}),
            encoding="utf-8",
        )

    def rows(self, waivers: set[str] | None = None) -> list[dict]:
        budget = mbc.load_budget(self.budget_path)
        return mbc.evaluate(self.root, budget, "HEAD", waivers or set())

    def guard_rows(
        self,
        waivers: set[str] | None = None,
        raise_waivers: set[str] | None = None,
    ) -> list[dict]:
        """Evaluate with the anti-loosening guard wired, as ``main()`` does."""
        budget = mbc.load_budget(self.budget_path)
        return mbc.evaluate(
            self.root,
            budget,
            "HEAD",
            waivers or set(),
            was_ceilings=mbc.base_ceilings(self.root, "budget.json", "HEAD"),
            raise_waivers=raise_waivers or set(),
        )


class NoWorseningRuleTest(unittest.TestCase):
    """Rule 2 (correction C3): over-ceiling fails only if it ALSO grew."""

    def test_under_ceiling_is_ok(self):
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=100, ceiling=200)
            self.assertEqual(fx.rows()[0]["status"], "OK")

    def test_over_ceiling_and_grew_fails(self):
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=100, ceiling=150)
            fx.set_size(300)
            row = fx.rows()[0]
            self.assertEqual(row["status"], "FAIL")
            self.assertTrue(row["over_ceiling"] and row["grew"])

    def test_over_ceiling_but_shrank_passes(self):
        """The load-bearing case: an over-budget file being cleaned up must not
        be blocked, or the gate punishes exactly the work it wants."""
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=500, ceiling=150)
            fx.set_size(400)
            row = fx.rows()[0]
            self.assertTrue(row["over_ceiling"])
            self.assertFalse(row["grew"])
            self.assertEqual(row["status"], "OK")

    def test_over_ceiling_unchanged_passes(self):
        """main already over budget must not block an unrelated PR."""
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=500, ceiling=150)
            self.assertEqual(fx.rows()[0]["status"], "OK")


class WaiverIsALoanTest(unittest.TestCase):
    def test_trailer_waives_the_failure(self):
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=100, ceiling=150)
            fx.set_size(300)
            self.assertEqual(fx.rows({"AGENTS.md"})[0]["status"], "WAIVED")

    def test_waiver_does_not_move_the_ceiling(self):
        """The whole point: the debt is still owed after a waiver."""
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=100, ceiling=150)
            fx.set_size(300)
            fx.rows({"AGENTS.md"})
            reloaded = mbc.load_budget(fx.budget_path)
            self.assertEqual(reloaded["files"]["AGENTS.md"]["ceiling_chars"], 150)

    def test_waiver_for_another_path_does_not_apply(self):
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=100, ceiling=150)
            fx.set_size(300)
            self.assertEqual(fx.rows({"docs/OTHER.md"})[0]["status"], "FAIL")

    def test_trailer_parsing(self):
        self.assertEqual(mbc.read_waivers("body\n\nAllow-Budget-Overrun: AGENTS.md\n"), {"AGENTS.md"})
        self.assertEqual(mbc.read_waivers("no trailer here"), set())
        # The `<path> - <reason>` form the design docs mandate. Accepted since
        # 2026-08-24; before that it parsed as nothing, silently.
        self.assertEqual(
            mbc.read_waivers("body\n\nAllow-Budget-Overrun: AGENTS.md — see notes/inbox/x.md\n"),
            {"AGENTS.md"},
        )

    def test_reason_form_waives_end_to_end(self):
        """Parsing is not the contract -- the WAIVER must actually apply."""
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=100, ceiling=150)
            fx.set_size(300)
            waived = mbc.read_waivers("Allow-Budget-Overrun: AGENTS.md — a stated reason")
            self.assertEqual(fx.rows(waived)[0]["status"], "WAIVED")


class MachineryNegativeControlTest(unittest.TestCase):
    """A gate that cannot fail is not a gate. Each blindness mode must exit 2."""

    def test_missing_governed_file_is_a_hard_failure(self):
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=100, ceiling=200)
            fx.governed.unlink()
            with self.assertRaises(mbc.BudgetError):
                fx.rows()

    def test_empty_governed_set_is_a_hard_failure(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "b.json"
            p.write_text(json.dumps({"files": {}}), encoding="utf-8")
            with self.assertRaises(mbc.BudgetError):
                mbc.load_budget(p)

    def test_unreadable_budget_is_a_hard_failure(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "b.json"
            p.write_text("{not json", encoding="utf-8")
            with self.assertRaises(mbc.BudgetError):
                mbc.load_budget(p)

    def test_absent_budget_is_a_hard_failure(self):
        with TemporaryDirectory() as td:
            with self.assertRaises(mbc.BudgetError):
                mbc.load_budget(Path(td) / "nope.json")

    def test_nonpositive_ceiling_is_a_hard_failure(self):
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=100, ceiling=200)
            fx.budget_path.write_text(
                json.dumps({"files": {"AGENTS.md": {"ceiling_chars": 0}}}),
                encoding="utf-8",
            )
            with self.assertRaises(mbc.BudgetError):
                fx.rows()


class RatchetTest(unittest.TestCase):
    def _run(self, root: Path, budget: Path, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(  # nosec B603 - sys.executable + this repo's own checker, fixed argv
            [sys.executable, str(MODULE_PATH), "--repo-root", str(root), "--budget", str(budget), "--base-ref", "HEAD", *extra],
            capture_output=True,
            text=True,
        )

    def test_ratchet_tightens(self):
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=100, ceiling=500)
            self._run(fx.root, fx.budget_path, "--ratchet")
            self.assertEqual(mbc.load_budget(fx.budget_path)["files"]["AGENTS.md"]["ceiling_chars"], 100)

    def test_ratchet_never_loosens(self):
        """Negative control: a file BELOW its ceiling may tighten it; a file
        ABOVE it must never raise it."""
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=100, ceiling=150)
            fx.set_size(900)
            self._run(fx.root, fx.budget_path, "--ratchet")
            self.assertEqual(mbc.load_budget(fx.budget_path)["files"]["AGENTS.md"]["ceiling_chars"], 150)


class CliExitCodeTest(unittest.TestCase):
    def _run(self, root: Path, budget: Path, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(  # nosec B603 - sys.executable + this repo's own checker, fixed argv
            [sys.executable, str(MODULE_PATH), "--repo-root", str(root), "--budget", str(budget), "--base-ref", "HEAD", *extra],
            capture_output=True,
            text=True,
        )

    def test_exit_zero_when_ok(self):
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=100, ceiling=200)
            self.assertEqual(self._run(fx.root, fx.budget_path).returncode, 0)

    def test_exit_one_when_over_and_grew(self):
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=100, ceiling=150)
            fx.set_size(300)
            self.assertEqual(self._run(fx.root, fx.budget_path).returncode, 1)

    def test_advisory_reports_but_exits_zero(self):
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=100, ceiling=150)
            fx.set_size(300)
            res = self._run(fx.root, fx.budget_path, "--advisory")
            self.assertEqual(res.returncode, 0)
            self.assertIn("ADVISORY", res.stdout)
            self.assertIn("::error::", res.stdout)  # still reports the finding

    def test_exit_two_on_broken_machinery(self):
        with TemporaryDirectory() as td:
            self.assertEqual(self._run(Path(td), Path(td) / "missing.json").returncode, 2)

    def test_json_output_shape(self):
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=100, ceiling=200)
            res = self._run(fx.root, fx.budget_path, "--json")
            row = json.loads(res.stdout)["rows"][0]
            for key in ("path", "chars", "ceiling", "status", "headroom", "delta"):
                self.assertIn(key, row)


class RealRepoTest(unittest.TestCase):
    """Dogfood: the shipped budget must govern a file that actually exists."""

    def test_shipped_budget_is_valid_and_governs_agents_md(self):
        budget = mbc.load_budget(REPO_ROOT / "conf" / "memory_budget.json")
        self.assertIn("AGENTS.md", budget["files"])
        self.assertTrue((REPO_ROOT / "AGENTS.md").is_file())
        ceiling = budget["files"]["AGENTS.md"]["ceiling_chars"]
        self.assertIsInstance(ceiling, int)
        self.assertGreater(ceiling, 0)

    def test_reference_md_is_not_governed(self):
        """docs/REFERENCE.md is the migration DESTINATION; governing it would
        penalise the relocation the plan is asking for."""
        budget = mbc.load_budget(REPO_ROOT / "conf" / "memory_budget.json")
        self.assertNotIn("docs/REFERENCE.md", budget["files"])


class AntiLooseningGuardTest(unittest.TestCase):
    """Rule 4: the ceiling may only move DOWN.

    ``--ratchet`` was downward-only from the start, but until 2026-08-23 nothing
    stopped a hand-edit of ``ceiling_chars`` UPWARD -- one line, and the whole
    ratchet is defeated while CI stays green. On a file that grew ~20x in six
    months under four gates that measured everything except size, that is the
    single edit the design forbids.
    """

    def test_unchanged_ceiling_is_not_flagged(self) -> None:
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=100, ceiling=200)
            row = fx.guard_rows()[0]
            self.assertFalse(row["ceiling_raised"])
            self.assertEqual(row["status"], "OK")

    def test_lowering_the_ceiling_is_always_allowed(self) -> None:
        # The ratchet's whole purpose. Tightening must never trip the guard.
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=100, ceiling=200)
            fx.set_ceiling(150)
            row = fx.guard_rows()[0]
            self.assertFalse(row["ceiling_raised"])
            self.assertEqual(row["status"], "OK")

    def test_raising_the_ceiling_fails(self) -> None:
        # THE regression. Before this guard the row read OK and CI went green.
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=100, ceiling=200)
            fx.set_ceiling(99999)
            row = fx.guard_rows()[0]
            self.assertTrue(row["ceiling_raised"])
            self.assertEqual(row["status"], "FAIL")
            self.assertEqual(row["ceiling_base"], 200)

    def test_a_raise_fails_even_when_the_file_is_far_under_the_new_ceiling(self) -> None:
        # A raise is a policy change on its own terms. It must not be excused by
        # the file happening to fit inside the roomier ceiling it just granted
        # itself -- which is precisely what the raise buys.
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=100, ceiling=200)
            fx.set_ceiling(100000)
            row = fx.guard_rows()[0]
            self.assertFalse(row["over_ceiling"])
            self.assertEqual(row["status"], "FAIL")

    def test_declared_raise_is_allowed_and_labelled(self) -> None:
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=100, ceiling=200)
            fx.set_ceiling(99999)
            row = fx.guard_rows(raise_waivers={"AGENTS.md"})[0]
            self.assertEqual(row["status"], "RAISE-WAIVED")

    def test_the_overrun_waiver_cannot_authorise_a_raise(self) -> None:
        # The separation that makes the guard worth having. An overrun BORROWS
        # against a ceiling that still stands; a raise MOVES it and erases the
        # debt for everyone. If the cheaper trailer authorised the dearer act,
        # every author already reaching for Allow-Budget-Overrun could silently
        # loosen the ratchet instead of paying it.
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=100, ceiling=200)
            fx.set_ceiling(99999)
            row = fx.guard_rows(waivers={"AGENTS.md"})[0]
            self.assertEqual(row["status"], "FAIL")

    def test_a_raise_for_a_different_path_does_not_excuse_this_one(self) -> None:
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=100, ceiling=200)
            fx.set_ceiling(99999)
            row = fx.guard_rows(raise_waivers={"README.md"})[0]
            self.assertEqual(row["status"], "FAIL")

    def test_unresolvable_base_budget_is_not_a_false_positive(self) -> None:
        # First introduction of the budget file has no base to compare against.
        # Failing there would block the commit that creates the gate.
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=100, ceiling=200)
            self.assertIsNone(mbc.base_ceilings(fx.root, "no-such-budget.json", "HEAD"))
            row = fx.guard_rows()[0]
            self.assertFalse(row["ceiling_raised"])

    def test_base_ceilings_reads_the_value_at_the_ref(self) -> None:
        with TemporaryDirectory() as td:
            fx = BudgetFixture(Path(td), base_chars=100, ceiling=200)
            fx.set_ceiling(99999)  # working tree only
            self.assertEqual(mbc.base_ceilings(fx.root, "budget.json", "HEAD"), {"AGENTS.md": 200})


class CeilingRaiseTrailerTest(unittest.TestCase):
    def test_parses_a_bare_trailer(self) -> None:
        self.assertEqual(
            mbc.read_ceiling_raise_waivers("Allow-Ceiling-Raise: AGENTS.md"),
            {"AGENTS.md"},
        )

    def test_the_two_trailers_do_not_cross_match(self) -> None:
        overrun = "Allow-Budget-Overrun: AGENTS.md"
        raise_ = "Allow-Ceiling-Raise: AGENTS.md"
        self.assertEqual(mbc.read_ceiling_raise_waivers(overrun), set())
        self.assertEqual(mbc.read_waivers(raise_), set())

    def test_reason_suffixed_form_is_accepted(self) -> None:
        """Was ``test_reason_suffixed_form_is_not_matched`` -- inverted 2026-08-24.

        The old test PINNED the divergence rather than closing it: the design docs
        mandate `<path> - <reason>` and the regex rejected it, so an author following
        the documentation wrote a trailer that parsed as nothing and got no diagnostic.
        Pinning made the trap deliberate; it did not stop anyone falling into it. Both
        forms are now accepted, and anything that still fails to parse is REPORTED.
        """
        for text in (
            "Allow-Ceiling-Raise: AGENTS.md -- because",
            "Allow-Ceiling-Raise: AGENTS.md — because",
            "Allow-Ceiling-Raise: AGENTS.md - because",
            "Allow-Ceiling-Raise: AGENTS.md – because",
        ):
            with self.subTest(text=text):
                self.assertEqual(mbc.read_ceiling_raise_waivers(text), {"AGENTS.md"})

    def test_reason_form_still_does_not_cross_match(self) -> None:
        """Widening the tail must not let the cheap trailer authorise the dear one."""
        overrun = "Allow-Budget-Overrun: AGENTS.md — reason"
        raise_ = "Allow-Ceiling-Raise: AGENTS.md — reason"
        self.assertEqual(mbc.read_ceiling_raise_waivers(overrun), set())
        self.assertEqual(mbc.read_waivers(raise_), set())


class UnparsedWaiverClaimIsReportedTest(unittest.TestCase):
    """A waiver's entire payload is the commit message, so dropping one MUST be loud.

    Silence is what let the checker and its design doc contradict each other: the
    trailer sits visibly in the message, so the author has no reason to suspect it was
    thrown away, and the run just stays red with no explanation.
    """

    def test_wellformed_claims_are_not_reported(self) -> None:
        for text in (
            "Allow-Budget-Overrun: AGENTS.md",
            "Allow-Budget-Overrun: AGENTS.md — reason",
            "Allow-Ceiling-Raise: docs/REFERENCE.md - reason",
        ):
            with self.subTest(text=text):
                self.assertEqual(mbc.unparsed_waiver_claims(text), [])

    def test_two_paths_on_one_line_are_reported_not_silently_halved(self) -> None:
        """The dangerous near-miss: a greedy tail would take a.md and drop b.md."""
        text = "Allow-Budget-Overrun: a.md b.md"
        self.assertEqual(mbc.read_waivers(text), set())
        self.assertIn(text, mbc.unparsed_waiver_claims(text))

    def test_missing_path_is_reported(self) -> None:
        self.assertEqual(mbc.unparsed_waiver_claims("Allow-Budget-Overrun:"), ["Allow-Budget-Overrun:"])

    def test_non_waiver_prose_is_not_reported(self) -> None:
        self.assertEqual(mbc.unparsed_waiver_claims("discussion of Allow-Budget-Overrun: mid-sentence"), [])


if __name__ == "__main__":
    unittest.main()
