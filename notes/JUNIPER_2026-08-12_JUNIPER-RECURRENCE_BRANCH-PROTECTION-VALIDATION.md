# Branch-Protection Validation — juniper-recurrence

**Project**: Juniper
**Sub-Project**: juniper-recurrence
**Author**: Paul Calnon
**License**: MIT License
**Version**: 1.0.0
**Last Updated**: 2026-08-12

---

Records the outcome of the 2026-08-12 fleet ruleset validation.

`main` is governed by an 8-rule ruleset uniform across all 9 publishing repos:
`code_quality`, `code_scanning`, `creation`, `deletion`, `non_fast_forward`,
`pull_request`, `required_signatures`, `required_status_checks`.

Only `required_status_checks` is per-repo — it names this repo's actual CI job
names. The canonical per-repo lists, the derivation method, and the Tier 2
hardening roadmap live in juniper-ml:

`notes/JUNIPER_2026-08-10_JUNIPER-ECOSYSTEM_REQUIRED-STATUS-CHECK-CONTEXT-LISTS.md`

**Operational notes**

- `strict_required_status_checks_policy` is **on** — a PR must be current with
  `main` to merge. Retained deliberately as the anti-storm guarantee.
- `require_last_push_approval` is **off**. With `required_approving_review_count: 0`
  it added no review workflow and made any owner-authored PR unmergeable except by
  admin bypass.
- An unsigned commit anywhere on a PR branch blocks the merge under
  `required_signatures`. Squash-merge does **not** rescue it. Commits made through
  the REST contents API are unsigned; the GraphQL `createCommitOnBranch` mutation
  produces a signed commit.
- If a PR sits at `CLEAN` without merging, **re-arm auto-merge** — a ruleset edit is
  not a PR event, so nothing re-evaluates the queue. Do not admin-merge.

## Package-CI gating (2026-08-12)

This repo publishes **three PyPI packages** behind four path-scoped CI lanes
(`app` / `model` / `client` / `bench`). None of those lanes was a *required* status
check, so a PR breaking a published package could go red and still merge.

The lanes could not simply be required, for two reasons:

1. **The gate never reported on most PRs.** Each lane's `on.pull_request` carried a
   `paths:` filter, so on a PR that did not touch that package the whole workflow —
   gate included — never ran. A required context that never reports blocks the PR
   forever (the failure class that made all 9 fleet repos unmergeable on 2026-08-10).
2. **The context name was ambiguous.** `app`, `client`, and `model` all emitted a job
   named `Required checks`.

**Fix (app lane first, recurrence#107):** path scoping moved from the *workflow* level
to the *job* level. `on.pull_request` is unfiltered; a `changes` job diffs
`origin/<base>...HEAD` and the expensive jobs carry
`if: needs.changes.outputs.app == 'true'`, so CI cost is unchanged. The gate is renamed
**`App required checks`**, runs `if: always()`, and treats `skipped` as a pass — which
is what lets it report on every PR and therefore be required. If the detector itself
fails, the gate fails: reporting a pass without knowing whether the package was touched
would reopen the hole.

`on.push` keeps its path filter — a push has no base ref to diff against.

Remaining: replicate to `model` / `client` / `bench` with uniquely-named gates, then add
all four to the ruleset's required status checks.

## Fleet parity reached (2026-08-13/14)

Replication and ruleset work are complete. This repo now matches the fleet's 8-rule set:

`code_quality`, `code_scanning`, `creation`, `deletion`, `non_fast_forward`,
`pull_request`, `required_signatures`, `required_status_checks`

**Required status checks (8):**

```
Analyze (python)
App required checks
Bench required checks
Client required checks
Documentation links
Guard PR base branch
Model required checks
Pre-commit (all-files)
```

**`code_scanning` is scoped to `CodeQL` ONLY.** Do not add tools to that list unless they
actually upload SARIF for this repo. A configured tool that never uploads is unsatisfiable
and blocks every PR permanently — that is exactly how the 2026-08-10 fleet-union list (7
tools) broke all nine repos. juniper-recurrence uploads CodeQL and nothing else.

**CodeQL** (`.github/workflows/codeql.yml`, recurrence#111) is deliberately **not**
path-filtered: the `code_scanning` rule requires analysis results *for the pull request*, so
it must report on every PR including docs-only ones. `code_scanning` and `code_quality` had
been removed on 2026-08-12 precisely because no analyses existed and both were unsatisfiable.

**Duplicate CI runs** were eliminated fleet-wide on 2026-08-13: all four package lanes
push-trigger on `[main, develop]` only. Topic-branch globs plus `pull_request` meant both
events fired for the same commit and every job ran twice; the concurrency group is keyed on
`github.ref`, which differs between a branch push and a PR, so `cancel-in-progress` never
collapsed the pair.
