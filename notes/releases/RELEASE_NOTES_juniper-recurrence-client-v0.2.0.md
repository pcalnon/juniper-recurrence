# juniper-recurrence-client v0.2.0 Release Notes

**Release Date:** 2026-06-24
**Version:** 0.2.0
**Codename:** DP-3 Readout Selection Forwarding
**Release Type:** MINOR

> Authored from the canonical `juniper-ml/notes/templates/TEMPLATE_RELEASE_NOTES.md`.

---

## Overview

Forwards the service's **DP-3 readout selection** through `JuniperRecurrenceClient.train()` and
`.crossval()`. Callers can now request the `linear` / `rff` / `mlp` readout rungs and `ridge="gcv"`
penalty selection, with every new argument optional and forwarded verbatim in the request body. A batch
of additive, backward-compatible parameters since 0.1.0, so this is a MINOR release.

> **Status:** STABLE — purely additive. All new arguments default to `None` and an unset argument
> produces a byte-identical request body to 0.1.0.

---

## Release Summary

- **Release type:** MINOR (additive, backward-compatible client parameters since 0.1.0)
- **Primary focus:** forwarding the DP-3 `readout` enum + `ridge="gcv"` to the service
- **Breaking changes:** NO (every new parameter is optional; unset ⇒ unchanged request body)

---

## What's New

### `readout` selection forwarded by `train()` / `crossval()`

- **`readout` (P2c + P3)** — widens to `Optional[Literal["linear", "rff", "mlp"]]`.
- **RFF params (P2c)** — `rff_features: Optional[int]`, `rff_gamma: Optional[Union[float, Literal["median"]]]`.
- **MLP params (P3)** — `mlp_hidden` / `mlp_weight_decay` / `mlp_lr` / `mlp_max_epochs` / `mlp_patience`
  (all `Optional`), for the service's torch MLP readout (Rung 2b). The service needs its own `[torch]`
  extra to fulfil `readout="mlp"`; otherwise it returns a 503, surfaced to the caller as the usual error.
- **`ridge="gcv"` (P1)** — `ridge` widens from `Optional[float]` to
  `Optional[Union[float, Literal["gcv"]]]`, requesting the service's closed-form GCV penalty selection.

All values are forwarded verbatim in the request body with no client-side validation change — an omitted
argument is simply not sent.

---

## API Changes

Additive only: the new optional keyword arguments above on `train()` and `crossval()`. No changes to
return shapes, transport, error handling, or any existing argument's behaviour.

---

## Test Results

The client `ci-recurrence-client.yml` suite (the `responses`-mocked forwarding tests, including the new
`readout="mlp"` + `mlp_*` body-passthrough assertions for both `train()` and `crossval()`) is green.

---

## Upgrade Notes

Backward-compatible MINOR release. No migration steps.

```bash
pip install --upgrade juniper-recurrence-client==0.2.0
```

Existing calls are unaffected; pass `readout=` / `mlp_*` / `ridge="gcv"` to use the new readouts.

---

## Known Issues

- `readout="mlp"` is fulfilled only when the **service** has its `[torch]` extra installed; otherwise the
  service returns a 503 (a deployment-capability gap, not a client error).

---

## What's Next

- The juniper-ml meta-package `[recurrence]` extra pin update (`juniper-recurrence-client>=0.2.0`).

---

## Contributors

- Paul Calnon

---

## Version History

| Version | Date       | Description                                                       |
| ------- | ---------- | ----------------------------------------------------------------- |
| 0.2.0   | 2026-06-24 | Forward the DP-3 `readout` enum (`linear`/`rff`/`mlp`) + `ridge="gcv"` |
| 0.1.0   | 2026-06-18 | First `juniper-recurrence-client` release                         |

---

## Links

- [Full Changelog](../../juniper-recurrence-client/CHANGELOG.md)
- [App release (the service edge)](RELEASE_NOTES_juniper-recurrence-v0.2.0.md)
- [DP-3 design-of-record](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_RECURRENCE_DP3_READOUT_SPECTRUM_DESIGN_2026-06-20.md)
