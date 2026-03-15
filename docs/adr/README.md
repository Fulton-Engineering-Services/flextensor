<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the FlexTensor project.

## What is an ADR?

An Architecture Decision Record (ADR) captures an important architectural decision made during the development of this project, along with its context and consequences. ADRs help communicate the reasoning behind technical choices to current and future team members.

## Why do we use ADRs?

- **Document rationale**: Capture the "why" behind architectural decisions, not just the "what"
- **Historical context**: Understand decisions made in the past and their constraints
- **Communication**: Share knowledge across team members and with future contributors
- **Learning**: Review past decisions to improve future ones
- **Prevent revisiting**: Avoid reopening settled discussions without new information

## ADR Format

We follow a lightweight format based on [Michael Nygard's ADR template](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions):

- **Title**: Short, descriptive name (e.g., "Use Forward Patching for Module Offloading")
- **Date**: When the decision was made
- **Status**: Proposed, Accepted, Deprecated, Superseded
- **Context**: What factors led to this decision? What constraints existed?
- **Decision**: What we decided to do and why
- **Alternatives Considered**: What other options were evaluated and why were they rejected?
- **Consequences**: What are the positive, negative, and neutral impacts?

## How to Create a New ADR

1. Copy `template.md` to a new file: `NNNN-short-title.md` (e.g., `0002-use-cuda-graphs.md`)
2. Fill in all sections with relevant information
3. Commit the ADR with your code changes or as a separate commit
4. Update the index below

## ADR Index

| Number | Title | Status | Date |
|--------|-------|--------|------|
| [0001](0001-forward-patching-for-module-offloading.md) | Forward Patching for Module Offloading | Accepted | 2026-01-23 |
| [0002](0002-configuration-based-module-patterns.md) | Configuration-based Module Patterns for Offloading | Accepted | 2026-01-23 |
| [0003](0003-custom-type-handlers-for-tensor-processor.md) | Custom Type Handlers for TensorProcessor | Accepted | 2026-02-16 |

## ADR Statuses

- **Proposed**: Under discussion, not yet decided
- **Accepted**: Decision has been made and implemented
- **Deprecated**: Still in effect but discouraged for new work
- **Superseded**: Replaced by a newer ADR (link to the replacement)
