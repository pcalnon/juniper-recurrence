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
