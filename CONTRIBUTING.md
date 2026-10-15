# Contributing

This repo follows the RAMMP module workflow. It is the same flow every
template-born repo uses, so learning it once is enough.

## Pre-commit hooks

Style is enforced automatically before each commit — Python via Ruff, C++ via
clang-format, Dockerfiles via hadolint, workflows via actionlint, plus general
file hygiene.

Run this once after cloning:

```bash
uv tool install pre-commit
pre-commit install
```

Without `uv`: `pip install pre-commit && pre-commit install`.

On first run pre-commit downloads hook environments, which takes a minute. If a
hook fails it blocks the commit and shows what to fix; Ruff, clang-format and
mdformat fix files in place — stage the changes and commit again.

Hook revisions are pinned. Updating them is a deliberate PR
(`pre-commit autoupdate`), never silent drift.

## Branches

- `main` — demo-ready, stable, deployable. Locked; updated periodically from
  `dev` (hotfixes excepted).
- `dev` — staging ground for tested new code. PRs land here.
- `feature/<issue-number>-<brief-description>` — your working branch, forked
  from the latest `dev`. Use `bug/<issue-number>-<brief-description>` for fixes.

## Workflow

1. Open (or claim) a GitHub issue and assign yourself, so two people do not
   build the same thing.
1. `git checkout dev && git pull`
1. `git checkout -b feature/42-add-depth-filter`
1. Develop and test. `make check` and `make smoke` must pass locally.
1. Open a PR into `dev`.
