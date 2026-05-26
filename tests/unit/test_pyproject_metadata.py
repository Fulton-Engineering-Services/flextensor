# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pathlib
import re
import sys
from collections.abc import Iterable

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def _dependency_names(dependencies: Iterable[str | dict[str, str]]) -> set[str]:
    names = set()
    for dependency in dependencies:
        if isinstance(dependency, str):
            names.add(re.split(r"[<>=~!;\[]", dependency, maxsplit=1)[0])
        elif "include-group" not in dependency:
            names.add(dependency["name"])
    return names


def test_development_dependencies_are_dependency_groups() -> None:
    pyproject_path = pathlib.Path(__file__).parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text())

    optional_dependencies = pyproject["project"].get("optional-dependencies", {})
    dependency_groups = pyproject["dependency-groups"]

    assert "required-version" in pyproject["tool"]["uv"]

    assert "test" not in optional_dependencies
    assert "docs" not in optional_dependencies
    assert "dev" not in optional_dependencies
    assert "all" not in optional_dependencies

    assert {"pytest", "pytest-cov", "mypy"}.issubset(_dependency_names(dependency_groups["test"]))
    assert {"mkdocs", "mkdocs-material", "mike", "mkdocstrings"}.issubset(_dependency_names(dependency_groups["docs"]))
    assert {"include-group": "test"} in dependency_groups["dev"]
    assert {"build", "pre-commit"}.issubset(_dependency_names(dependency_groups["dev"]))
    assert {"include-group": "test"} in dependency_groups["all"]
    assert {"include-group": "dev"} in dependency_groups["all"]
    assert {"include-group": "docs"} in dependency_groups["all"]
