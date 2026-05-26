<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Documentation Guidelines

## Diataxis Framework

| Quadrant | Directory | Nav section |
|----------|-----------|-------------|
| Tutorials | `quick-start.md`, root | Get Started |
| How-to | `how-to/` | Guides |
| Explanation | `explanation/` | Understand |
| Reference | `api/` (auto-generated via mkdocstrings) | Reference |

## Changing Pages

- Update `nav:` in `mkdocs.yml` — unlisted pages won't appear in navigation
- Validate with `uv run --group docs mkdocs build --strict` — catches broken links and missing refs
- Use relative paths for cross-page links (`../how-to/troubleshooting.md`)
- When renaming headings, grep for old anchors and update references
- API cross-refs: `` [`OffloadConfig`][flextensor.OffloadConfig] ``

## Terminology

Follow `api/glossary.md` for canonical definitions. New terms must also be added to `includes/abbreviations.md` (hover tooltips).

## Audience

Docs must be consumable by both humans and AI agents. Write clear, structured prose — avoid ambiguous references, implicit context, or visual-only formatting that agents cannot parse.
