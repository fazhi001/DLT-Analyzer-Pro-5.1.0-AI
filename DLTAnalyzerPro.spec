# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

project_root = Path(SPECPATH)

datas = [(str(project_root / "resources"), "resources")]
binaries = []

for package_name in ("xgboost", "lightgbm"):
    binaries += collect_dynamic_libs(package_name)
    datas += collect_data_files(package_name, include_py_files=False)
    try:
        datas += copy_metadata(package_name)
    except Exception:
        pass

hiddenimports = [
    "tkinter",
    "openpyxl",
    "openpyxl.cell._writer",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "sklearn.metrics",
    "sklearn.linear_model",
    "sklearn.linear_model._logistic",
    "sklearn.utils._cython_blas",
    "sklearn.neighbors._partition_nodes",
    "joblib",
    "xgboost",
    "xgboost.core",
    "xgboost.data",
    "xgboost.sklearn",
    "xgboost.training",
    "xgboost.callback",
    "lightgbm",
    "lightgbm.basic",
    "lightgbm.sklearn",
    "lightgbm.engine",
    "lightgbm.callback",
]

excluded_modules = [
    "xgboost.testing",
    "lightgbm.testing",
    "pytest",
    "hypothesis",
    "numpy.tests",
    "scipy.tests",
    "sklearn.tests",
    "pandas.tests",
]

a = Analysis(
    ["main.py"],
    pathex=[str(project_root / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DLTAnalyzerPro",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(project_root / "resources" / "app.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="DLTAnalyzerPro",
)
