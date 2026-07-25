from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def main() -> int:
    version = read_version()
    errors: list[str] = []

    init_text = (ROOT / "src/dlt_analyzer_pro/__init__.py").read_text(encoding="utf-8")
    init_match = re.search(r'__version__\s*=\s*"([^"]+)"', init_text)
    init_version = init_match.group(1) if init_match else None
    if init_version != version:
        errors.append(f"package version mismatch: {init_version!r} != {version!r}")

    installer_text = (ROOT / "installer/DLTAnalyzerPro.iss").read_text(encoding="utf-8")
    installer_match = re.search(r'#define MyAppVersion "([^"]+)"', installer_text)
    installer_version = installer_match.group(1) if installer_match else None
    if installer_version != version:
        errors.append(f"installer version mismatch: {installer_version!r} != {version!r}")

    expected_installer = f"DLT_Analyzer_Pro_{version}_3Games_Setup_x64"
    if f"OutputBaseFilename={expected_installer}" not in installer_text:
        errors.append(f"installer output filename must be {expected_installer}")

    release_notes = ROOT / f"RELEASE_NOTES_{version}.md"
    if not release_notes.exists():
        errors.append(f"missing release notes: {release_notes.name}")

    workflow_text = (ROOT / ".github/workflows/build-windows.yml").read_text(encoding="utf-8")
    if expected_installer not in workflow_text:
        errors.append("workflow installer filename is inconsistent")

    payload = {
        "version": version,
        "package_version": init_version,
        "installer_version": installer_version,
        "expected_installer": expected_installer,
        "passed": not errors,
        "errors": errors,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
