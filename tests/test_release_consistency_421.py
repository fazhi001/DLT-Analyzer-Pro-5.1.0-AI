from __future__ import annotations

import re
import tomllib
from pathlib import Path


def test_release_versions_are_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as handle:
        version = str(tomllib.load(handle)["project"]["version"])

    init_text = (root / "src/dlt_analyzer_pro/__init__.py").read_text(encoding="utf-8")
    package_version = re.search(r'__version__\s*=\s*"([^"]+)"', init_text).group(1)

    installer_text = (root / "installer/DLTAnalyzerPro.iss").read_text(encoding="utf-8")
    installer_version = re.search(r'#define MyAppVersion "([^"]+)"', installer_text).group(1)

    assert version == "5.1.1"
    assert package_version == version
    assert installer_version == version
    assert f"OutputBaseFilename=DLT_Analyzer_Pro_{version}_3Games_Setup_x64" in installer_text
