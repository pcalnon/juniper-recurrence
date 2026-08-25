#!/usr/bin/env python3
"""
Project:     Juniper
Sub-Project: juniper-ml
Application: util
Author:      Paul Calnon
Version:     0.1.0
License:     MIT License

Memory-file size budget gate -- P2 of the shared-session-memory plan
(``notes/JUNIPER_2026-08-18_JUNIPER-ML_SHARED-SESSION-MEMORY-PLAN.md``).

Why this exists
---------------
``AGENTS.md`` grew ~20x in six months **while under four active CI gates** --
because every one of them enforces structure or currency and none enforces
size. 172 of 200 main-line merges grew the file; 14 shrank it, by 2,628 bytes
between them. The disease is an ungoverned write path, so a one-time cut is
undone in ~44 days. This is the ratchet that makes a cut durable, and per
correction C1 it must ship BEFORE the cut.

The three rules
---------------
1. **CEILING.** Each governed file has a character ceiling in
   ``conf/memory_budget.json``. Characters, not bytes: the shipped Claude Code
   check compares ``content.length`` (mechanism-facts section 1).

2. **NO-WORSENING** (correction C3, stated by no proposal). A file over its
   ceiling fails only if this change *also makes it bigger*. Without this, one
   over-budget file on ``main`` blocks every unrelated PR until someone fixes
   it -- which is how a gate gets disabled rather than obeyed. A PR that shrinks
   an over-ceiling file always passes.

3. **RATCHET.** ``--ratchet`` rewrites a ceiling **downward only**, never up, so
   the budget can tighten as cleanup lands and can never silently loosen.

The waiver is a LOAN, not a pass
--------------------------------
``Allow-Budget-Overrun: <path>`` in a commit message waives the failure for that
path **without moving the ceiling**, so the debt is still owed and the next
author still sees it. This is the property the house ``Allow-Symbol-Loss:``
idiom lacks. Waivers are always reported, never silent.

Vacuous-pass resistance
-----------------------
This repo has a documented class where a check's machinery breaks and it reports
SUCCESS. Guards here: a governed file that is MISSING is a hard failure (not a
silent skip); an empty governed set is a hard failure; and an unreadable budget
file is a hard failure. ``tests/test_memory_budget_check.py`` carries the
negative controls proving each can still fail.

Usage:
    python util/memory_budget_check.py [--repo-root P] [--budget F]
                                       [--trailers-file F] [--json] [--advisory]
    python util/memory_budget_check.py --ratchet          # tighten to current

Exit: 0 pass (or advisory) / 1 over budget / 2 misuse or broken machinery.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# BOTH forms are accepted: a bare `<path>`, and `<path> — <reason>` with an em/en
# dash or hyphen separator.
#
# They diverged and it cost nothing to notice, because nothing ever hit it. The
# checker accepted ONLY the bare form while the design of record
# (notes/JUNIPER_2026-08-18_JUNIPER-ML_MEMORY-ARCHITECTURE-SYNTHESIS-2.md) mandated
# `<path> — <reason>` and stated the checker FAILS a bare one -- the exact inverse of
# the behaviour. An author following the docs would write a trailer that parsed as
# nothing, get no diagnostic, and stay red with the waiver sitting right there in the
# commit message. Verified 2026-08-24: NO commit on main has ever carried either
# trailer at line start, so there was no de-facto form to preserve and widening breaks
# nothing.
#
# Whether to go further and REQUIRE a reason containing an inbox path (the design's
# "convert a hole into a funnel") is a deliberate, separate decision -- not something
# to ship by accident while repairing a parse bug.
_WAIVER_TAIL = r"\s*(?:[-–—]\s*\S.*)?$"
WAIVER_RE = re.compile(r"^Allow-Budget-Overrun:\s*(?P<path>\S+)" + _WAIVER_TAIL, re.MULTILINE)

# A SECOND, deliberately separate waiver. Raising a ceiling and overrunning one
# are different acts: an overrun BORROWS against a ceiling that still stands,
# while a raise MOVES the ceiling and erases the debt for everyone. They must
# not share a trailer, or the cheaper one silently authorises the dearer.
CEILING_RAISE_RE = re.compile(r"^Allow-Ceiling-Raise:\s*(?P<path>\S+)" + _WAIVER_TAIL, re.MULTILINE)

# Anything CLAIMING to be one of the two trailers, however malformed. The gap between
# this and the strict patterns above is exactly the set of lines a human wrote as a
# waiver and the checker threw away. Silence there is what made the divergence above
# survive: the trailer is IN the commit message, so the author has no reason to doubt
# it. Every unparsed claim is now reported.
CLAIMED_WAIVER_RE = re.compile(r"^(?P<kind>Allow-Budget-Overrun|Allow-Ceiling-Raise):.*$", re.MULTILINE)


class BudgetError(RuntimeError):
    """Machinery failure -- never degrade this to a pass."""


def load_budget(path: Path) -> dict:
    if not path.is_file():
        raise BudgetError(f"budget file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BudgetError(f"budget file unreadable: {path}: {exc}") from exc
    files = data.get("files")
    if not isinstance(files, dict) or not files:
        raise BudgetError(f"budget file declares no governed files: {path}")
    return data


def measure(path: Path) -> int:
    """Characters, not bytes -- the shipped check compares content.length."""
    return len(path.read_text(encoding="utf-8"))


def base_size(repo_root: Path, rel: str, base_ref: str) -> int | None:
    """Size of `rel` at `base_ref`, or None when it cannot be resolved."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{base_ref}:{rel}"],
            capture_output=True, check=False,
        )
    except OSError:
        return None
    if out.returncode != 0:
        return None
    return len(out.stdout.decode("utf-8", errors="replace"))


def read_waivers(trailers: str) -> set[str]:
    return {m.group("path") for m in WAIVER_RE.finditer(trailers or "")}


def read_ceiling_raise_waivers(trailers: str) -> set[str]:
    return {m.group("path") for m in CEILING_RAISE_RE.finditer(trailers or "")}


def unparsed_waiver_claims(trailers: str) -> list[str]:
    """Lines that claim to be a waiver trailer but parse as none.

    A waiver's entire payload IS the commit message, so an author who wrote one has
    every reason to believe it took effect. Dropping it without a word is how the
    checker and its own design doc stayed contradictory for as long as they did.
    Returns the offending lines verbatim so the caller can name them.
    """
    text = trailers or ""
    parsed = {m.group(0).strip() for m in WAIVER_RE.finditer(text)}
    parsed |= {m.group(0).strip() for m in CEILING_RAISE_RE.finditer(text)}
    return [m.group(0).strip() for m in CLAIMED_WAIVER_RE.finditer(text) if m.group(0).strip() not in parsed]


def base_ceilings(repo_root: Path, budget_rel: str, base_ref: str) -> dict[str, int] | None:
    """Every governed file's ceiling as declared at ``base_ref``.

    None when the budget file cannot be resolved there -- which is the genuine
    first-introduction case, not evidence of tampering.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{base_ref}:{budget_rel}"],
            capture_output=True, check=False,
        )
    except OSError:
        return None
    if out.returncode != 0:
        return None
    try:
        data = json.loads(out.stdout.decode("utf-8", errors="replace"))
        files = data["files"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return {
        rel: spec["ceiling_chars"]
        for rel, spec in files.items()
        if isinstance(spec, dict) and isinstance(spec.get("ceiling_chars"), int)
    }


def evaluate(
    repo_root: Path,
    budget: dict,
    base_ref: str,
    waivers: set[str],
    *,
    was_ceilings: dict[str, int] | None = None,
    raise_waivers: set[str] | None = None,
) -> list[dict]:
    """Evaluate every governed file.

    Rule 4, THE ANTI-LOOSENING GUARD. ``--ratchet`` is downward-only, but until
    2026-08-23 nothing stopped a hand-edit of ``ceiling_chars`` UPWARD -- one
    line, and the whole ratchet is defeated while CI stays green. That is the
    single edit the design forbids, on a file that grew ~20x in six months under
    four gates none of which measured size. A raise now FAILS unless the author
    declares it with ``Allow-Ceiling-Raise: <path>``, which is deliberately a
    different trailer from the overrun waiver.
    """
    raise_waivers = raise_waivers or set()
    rows = []
    for rel, spec in sorted(budget["files"].items()):
        ceiling = spec.get("ceiling_chars")
        if not isinstance(ceiling, int) or ceiling <= 0:
            raise BudgetError(f"{rel}: ceiling_chars must be a positive int")

        target = repo_root / rel
        if not target.is_file():
            # A governed file that vanished is the loudest possible signal, not a skip.
            raise BudgetError(f"governed file missing: {rel}")

        now = measure(target)
        was = base_size(repo_root, rel, base_ref)
        over = now > ceiling
        grew = was is not None and now > was

        # Rule 2: over-ceiling alone is not a failure; it must also have grown.
        failing = over and (grew or was is None)
        waived = failing and rel in waivers

        # Rule 4: the ceiling itself may only move DOWN.
        ceiling_was = (was_ceilings or {}).get(rel)
        raised = ceiling_was is not None and ceiling > ceiling_was
        raise_waived = raised and rel in raise_waivers
        if raised and not raise_waived:
            failing = True
            waived = False

        status = "OK"
        if waived:
            status = "WAIVED"
        elif failing:
            status = "FAIL"
        elif raise_waived:
            status = "RAISE-WAIVED"

        rows.append({
            "path": rel, "chars": now, "ceiling": ceiling, "base_chars": was,
            "over_ceiling": over, "grew": grew,
            "ceiling_base": ceiling_was, "ceiling_raised": raised,
            "status": status,
            "headroom": ceiling - now,
            "delta": (now - was) if was is not None else None,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--budget", type=Path, default=None)
    ap.add_argument("--base-ref", default="origin/main")
    ap.add_argument("--trailers-file", type=Path, default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--advisory", action="store_true", help="report, always exit 0")
    ap.add_argument("--ratchet", action="store_true",
                    help="tighten every ceiling to the current size (downward only)")
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    budget_path = args.budget or (repo_root / "conf" / "memory_budget.json")

    try:
        budget = load_budget(budget_path)
        trailers = args.trailers_file.read_text(encoding="utf-8") if args.trailers_file else ""
        # Report any line that claims to be a waiver and is not one, BEFORE evaluating.
        # A dropped waiver otherwise presents as an unexplained red with the trailer
        # visible in the commit message -- the failure mode that let this checker and
        # its design doc contradict each other unnoticed.
        for claim in unparsed_waiver_claims(trailers):
            print(
                f"::warning::malformed waiver trailer IGNORED: {claim!r} -- expected "
                "'Allow-Budget-Overrun: <path>' or 'Allow-Budget-Overrun: <path> - <reason>' "
                "(one path per line)",
                file=sys.stderr,
            )
        waivers = read_waivers(trailers)
        try:
            budget_rel = budget_path.resolve().relative_to(repo_root).as_posix()
        except ValueError:
            budget_rel = None
        was_ceilings = (
            base_ceilings(repo_root, budget_rel, args.base_ref) if budget_rel else None
        )
        rows = evaluate(
            repo_root, budget, args.base_ref, waivers,
            was_ceilings=was_ceilings,
            raise_waivers=read_ceiling_raise_waivers(trailers),
        )
    except BudgetError as exc:
        print(f"::error::memory-budget machinery failure: {exc}", file=sys.stderr)
        return 2

    if args.ratchet:
        tightened = []
        for row in rows:
            if row["chars"] < row["ceiling"]:
                budget["files"][row["path"]]["ceiling_chars"] = row["chars"]
                tightened.append((row["path"], row["ceiling"], row["chars"]))
        budget_path.write_text(json.dumps(budget, indent=2) + "\n", encoding="utf-8")
        for path, old, new in tightened:
            print(f"ratcheted {path}: {old} -> {new}")
        if not tightened:
            print("no ceiling could be tightened (ratchet never loosens)")
        return 0

    if args.json:
        print(json.dumps({"rows": rows}, indent=2))
    else:
        print("=== memory-file size budget ===")
        for r in rows:
            delta = "" if r["delta"] is None else f"  delta={r['delta']:+d}"
            print(f"  [{r['status']:>6}] {r['path']}: {r['chars']} / {r['ceiling']} chars"
                  f"  headroom={r['headroom']}{delta}")
        for r in rows:
            if r["ceiling_raised"] and r["status"] == "FAIL":
                print(f"\n::error::{r['path']}: ceiling RAISED "
                      f"{r['ceiling_base']} -> {r['ceiling']}. The ratchet is downward-only "
                      f"by design -- raising it erases the debt for everyone and is the one "
                      f"edit this gate exists to prevent. Relocate content instead. If the "
                      f"raise is genuinely intended, declare it with a commit trailer "
                      f"'Allow-Ceiling-Raise: {r['path']}' so it is auditable in history.")
                continue
            if r["status"] == "RAISE-WAIVED":
                print(f"\n::warning::{r['path']}: ceiling raised {r['ceiling_base']} -> "
                      f"{r['ceiling']}, declared by trailer. This is a POLICY change, not a "
                      f"loan -- the debt it would have created is now gone.")
                continue
            if r["status"] == "FAIL":
                print(f"\n::error::{r['path']} is over its {r['ceiling']}-char ceiling "
                      f"({r['chars']}) and this change grew it. Relocate content to "
                      f"docs/REFERENCE.md rather than compressing in place; the index "
                      f"row must keep an accurate open/closed status. To defer, add a "
                      f"commit trailer 'Allow-Budget-Overrun: {r['path']}' -- that is a "
                      f"LOAN: the ceiling does not move and the debt blocks the next author.")
            elif r["status"] == "WAIVED":
                print(f"\n::warning::{r['path']} over budget but WAIVED by trailer. "
                      f"Ceiling unchanged at {r['ceiling']}; debt still owed.")

    failed = any(r["status"] == "FAIL" for r in rows)
    if args.advisory and failed:
        print("\nADVISORY MODE — reporting only, not failing the build.")
        return 0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
