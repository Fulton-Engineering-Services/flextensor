<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Review Guide

Use this guide for AI and human reviews. Prefer a few high-confidence findings over broad commentary.

## Severity

- **Critical**: correctness bugs, silent failures, security risks, data loss, broken public APIs, invalid tests.
- **Important**: missing failure-path coverage, misleading docs/comments, weak invariants, risky compatibility changes.
- **Suggestion**: simplification or cleanup only when complexity creates real maintenance or correctness risk.

## Do Not Report

- Style, formatting, import order, line length, whitespace, or lint issues already handled by CI, unless they hide a real
  defect.
- Naming, comment-density, or refactor preferences unless the current code is misleading about behavior, units, device
  boundaries, ownership, or lifetime.
- Nits on unchanged lines or generated files unless the changed code makes them newly wrong.

## Review Checks

- **Project rules**: Check changed code against `CLAUDE.md` and nearest directory guidance.
- **Project patterns**: Prefer the nearest existing implementation pattern. Flag new abstractions, config surfaces, or
  workflow conventions when the MR does not explain why existing patterns do not fit.
- **Tests**: Prefer behavioral coverage over line coverage. Look for missing edge cases, negative cases, error paths,
  GPU assumptions, and brittle tests tied to implementation details. Flag skipped, disabled, or weakened tests unless the
  MR explains the tradeoff.
- **Test infrastructure**: Flag hardcoded ports, fixed temp paths, writes into the repository tree, hand-rolled process
  lifecycle, and duplicated setup when shared fixtures or helpers already exist.
- **Errors**: Flag swallowed exceptions, broad catches, hidden fallbacks, missing cleanup, and logs without enough
  context to debug production failures.
- **Docs/comments**: Verify comments, docstrings, examples, and Markdown claims match the code. Remove comments that
  only restate obvious code.
- **Types/API**: New public types and config surfaces should express invariants, validate invalid states at boundaries,
  and preserve compatibility or include migration/deprecation guidance.
- **Stale references**: Check that configs, CI jobs, docs, examples, scripts, imports, and `mkdocs.yml` entries reference
  files, commands, modules, and symbols that exist in the PR or base branch.
- **Simplification**: Prefer explicit, maintainable code. Suggest simplification only when it reduces meaningful
  complexity without changing behavior.

## FlexTensor Focus

- Watch for hidden GPU synchronization, unnecessary CPU/GPU transfer churn, leaked shared-memory resources, and stale
  process handles.
- Integration tests should avoid large models unless the MR explains why a smaller model cannot exercise the behavior and
  the test is gated or profiled for the target CI tier.
- Launch scripts and test harnesses should make ports, temp paths, model paths, and GPU memory budgets injectable instead
  of hardcoding shared resources.
- Treat unsafe model loading, deserialization, shell execution, and credential exposure in logs as high-signal security
  findings.
- Public API or user-visible behavior changes should update docs, examples, and `CHANGELOG.md`, or explain why not.
