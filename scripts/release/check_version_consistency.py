#!/usr/bin/env python3
"""Validate AutoAgents release version consistency.

The release process keeps Rust crate versions in the workspace and Python
package versions dynamic, but sibling dependency pins in pyproject.toml and
documentation snippets can drift. This check fails with actionable messages
before a release branch or tag is cut.
"""

from __future__ import annotations

import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]
AUTOAGENTS_PACKAGE_RE = re.compile(r"\bautoagents(?:-[a-z0-9]+)*(?:-py)?==(?P<version>\d+\.\d+\.\d+)")
STALE_DOC_VERSION_RE = re.compile(
    r"\bautoagents(?:-[a-z0-9]+)?\s*=\s*(?:\{[^}\n]*version\s*=\s*)?\"(?P<version>\d+\.\d+\.\d+)\""
)
SECTION_RE = re.compile(r"^\s*\[(?P<name>[^\]]+)\]\s*$")


def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def section_lines(text: str, section: str) -> list[str]:
    lines: list[str] = []
    in_section = False
    for line in text.splitlines():
        match = SECTION_RE.match(line)
        if match:
            current_section = match.group("name").strip()
            if in_section and current_section != section:
                break
            in_section = current_section == section
            continue
        if in_section:
            lines.append(line)
    return lines


def section_has_assignment(lines: list[str], key: str) -> bool:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    return any(pattern.match(line) for line in lines)


def section_value(lines: list[str], key: str) -> str | None:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*(?P<value>.+?)\s*(?:#.*)?$")
    for line in lines:
        match = pattern.match(line)
        if match:
            return match.group("value").strip()
    return None


def quoted_value(lines: list[str], key: str) -> str | None:
    value = section_value(lines, key)
    if value is None:
        return None
    match = re.match(r'"(?P<value>[^"]+)"', value)
    return match.group("value") if match else None


def workspace_version() -> str:
    lines = section_lines(read_text(ROOT / "Cargo.toml"), "workspace.package")
    version = quoted_value(lines, "version")
    if version is None:
        raise SystemExit("error: Cargo.toml is missing workspace.package.version")
    return version


def check_workspace_dependencies(expected: str, errors: list[str]) -> None:
    dependencies = section_lines(read_text(ROOT / "Cargo.toml"), "workspace.dependencies")
    for line in dependencies:
        match = re.match(r"^\s*(?P<name>autoagents[A-Za-z0-9_-]*)\s*=\s*(?P<spec>.+)", line)
        if not match:
            continue
        name = match.group("name")
        spec = match.group("spec")
        if not name.startswith("autoagents"):
            continue
        if not spec.lstrip().startswith("{"):
            errors.append(f"workspace dependency {name} must use a table spec")
            continue
        version_match = re.search(r'\bversion\s*=\s*"(?P<version>\d+\.\d+\.\d+)"', spec)
        version = version_match.group("version") if version_match else None
        if version != expected:
            errors.append(
                f"workspace dependency {name} pins version {version!r}; expected {expected!r}"
            )


def check_package_uses_workspace_version(path: pathlib.Path, errors: list[str]) -> None:
    package = section_lines(read_text(path), "package")
    if not package:
        return
    name = quoted_value(package, "name") or ""
    if not name.startswith("autoagents"):
        return
    if section_value(package, "version.workspace") != "true":
        errors.append(f"{path.relative_to(ROOT)} must use package.version.workspace = true")


def check_python_metadata(expected: str, errors: list[str]) -> None:
    for path in sorted((ROOT / "bindings/python").glob("*/pyproject.toml")):
        text = read_text(path)
        project = section_lines(text, "project")
        rel = path.relative_to(ROOT)

        dynamic = section_value(project, "dynamic") or ""
        if "version" not in dynamic:
            errors.append(f"{rel} must keep project.version dynamic")

        for field in ("authors", "maintainers", "classifiers"):
            if not section_has_assignment(project, field):
                errors.append(f"{rel} is missing project.{field}")
        if not section_lines(text, "project.urls"):
            errors.append(f"{rel} is missing project.urls")

        for match in AUTOAGENTS_PACKAGE_RE.finditer(text):
            if match.group("version") != expected:
                errors.append(
                    f"{rel} dependency {match.group(0)!r} pins {match.group('version')}; expected {expected}"
                )


def check_docs_versions(expected: str, errors: list[str]) -> None:
    doc_paths = sorted(
        {
            *ROOT.glob("README*.md"),
            *(ROOT / "docs/content").rglob("*.md"),
        }
    )
    for path in doc_paths:
        text = path.read_text(encoding="utf-8")
        for match in STALE_DOC_VERSION_RE.finditer(text):
            version = match.group("version")
            if version != expected:
                errors.append(
                    f"{path.relative_to(ROOT)} references AutoAgents version {version}; expected {expected}"
                )


def main() -> int:
    expected = workspace_version()
    errors: list[str] = []

    check_workspace_dependencies(expected, errors)
    for path in sorted(ROOT.glob("crates/*/Cargo.toml")):
        check_package_uses_workspace_version(path, errors)
    for path in sorted(ROOT.glob("bindings/python/*/Cargo.toml")):
        check_package_uses_workspace_version(path, errors)
    check_python_metadata(expected, errors)
    check_docs_versions(expected, errors)

    if errors:
        print("release version consistency check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"release version consistency check passed for {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
