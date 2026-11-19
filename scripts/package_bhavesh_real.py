"""Package reproducible source/evidence and a compact ready-model demo.

Large forests stay local and out of Git. The shared tracked dataset is unchanged.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import zipfile
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "output"
PREFIX = "Bhavesh_RealData"
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    "node_modules",
    "raw",
}
SECRET = re.compile(
    rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)


def source_files():
    for folder, dirs, names in os.walk(ROOT):
        dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
        for name in names:
            path = Path(folder) / name
            relative = path.relative_to(ROOT)
            if name in {".git", ".env", "meta.json"} or path.suffix in {
                ".pyc",
                ".joblib",
                ".pkl",
            }:
                continue
            text = relative.as_posix()
            if text in {"results/benchmark.csv", "results/scaling.csv"}:
                continue
            if name.endswith("_progress.csv") or "Demo_Raw" in name:
                continue
            if (
                relative.parts[:2] == ("output", "demo")
                and name != "Bhavesh_Madrid_Demo.webm"
            ):
                continue
            if not path.resolve().is_relative_to(ROOT.resolve()):
                raise ValueError(f"Refusing to package an external path: {relative}")
            yield path, relative


def add_file(archive, path, relative, checks):
    if path.suffix in {".py", ".md", ".txt", ".csv", ".json", ".log"} and SECRET.search(
        path.read_bytes()
    ):
        raise ValueError(f"Potential secret in {relative}")
    archive.write(path, PREFIX + "/" + relative.as_posix())
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    checks.append(digest.hexdigest() + "  " + relative.as_posix())


def add_text(archive, checks, full_path, value):
    data = value.encode("utf-8")
    archive.writestr(full_path, data)
    checks.append(
        hashlib.sha256(data).hexdigest() + "  " + full_path[len(PREFIX) + 1 :]
    )


def main():
    OUT.mkdir(exist_ok=True)
    inputs = list(source_files())
    if not (ROOT / "docs/screenshots/madrid_browser_verification.json").exists():
        raise ValueError("Complete the real UI verification before packaging.")
    browser = json.loads((ROOT / "docs/screenshots/madrid_browser_verification.json").read_text())
    metadata = json.loads((ROOT / "models/meta.json").read_text())
    if browser["status"] != "pass" or browser["run_id"] != metadata["run_id"]:
        raise ValueError("Browser verification and saved model setup must match.")
    for relative in ["output/pdf/Bhavesh_RealData_Report.pdf", "output/pdf/Bhavesh_RealData_Evidence.pdf",
                     "output/demo/Bhavesh_Madrid_Demo.webm"]:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(f"Required deliverable is missing: {relative}")
    for demo in [False, True]:
        name = "Bhavesh_RealData_Demo.zip" if demo else "Bhavesh_RealData_Source.zip"
        checks = []
        with zipfile.ZipFile(
            OUT / name, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5
        ) as archive:
            for path, relative in inputs:
                add_file(archive, path, relative, checks)
            if demo:
                metadata = json.loads((ROOT / "models/meta.json").read_text())
                included = {
                    name: record
                    for name, record in metadata["model_runs"].items()
                    if (ROOT / "models" / f"{name}.joblib").stat().st_size
                    <= 128 * 2**20
                }
                if not {"AdaBoost", "GradientBoosting", "HistGradientBoosting"} <= set(
                    included
                ):
                    raise ValueError(
                        "The compact demo must include all three assigned boosters."
                    )
                metadata["model_runs"] = included
                for model in included:
                    add_file(
                        archive,
                        ROOT / "models" / f"{model}.joblib",
                        Path("models") / f"{model}.joblib",
                        checks,
                    )
                add_text(
                    archive,
                    checks,
                    PREFIX + "/models/meta.json",
                    json.dumps(metadata, indent=2),
                )
                frozen = (
                    ROOT / "results/bhavesh/madrid_h60_development/frozen_benchmark.csv"
                )
                add_file(archive, frozen, Path("results/benchmark.csv"), checks)
                dependencies = [
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "joblib",
                    "streamlit",
                    "xgboost",
                    "lightgbm",
                    "pyarrow",
                ]
                add_text(
                    archive,
                    checks,
                    PREFIX + "/requirements-runtime.txt",
                    "\n".join(f"{p}=={version(p)}" for p in dependencies) + "\n",
                )
                add_text(
                    archive,
                    checks,
                    PREFIX + "/RUN_DEMO.bat",
                    '@echo off\r\ncd /d "%~dp0"\r\nif exist ".venv\\Scripts\\python.exe" (\r\n  ".venv\\Scripts\\python.exe" -m streamlit run app/streamlit_app.py\r\n) else (\r\n  python -m streamlit run app/streamlit_app.py\r\n)\r\npause\r\n',
                )
                add_text(
                    archive,
                    checks,
                    PREFIX + "/DEMO_README.md",
                    """# Compact Madrid demo

This package includes the approved shared input panel, all three assigned
boosters, and other saved models smaller than 128 MiB. Large forests are
omitted from the selector; their measured results remain in the frozen table.
The full working repository retains all 14 artifacts locally.

Use Python 3.13 and install the pinned requirements-runtime.txt in a virtual
environment for compatibility with these joblib artifacts. Then run:

    python -m streamlit run app/streamlit_app.py

On Windows, RUN_DEMO.bat starts the app with the local .venv if present.
Read DATA_ATTRIBUTION.md and docs/BHAVESH_HANDOFF.md for sources, methods and
limits. These are historical one-hour forecasts, not live traffic ingestion.
""",
                )
            archive.writestr(PREFIX + "/CHECKSUMS.sha256", "\n".join(checks) + "\n")
        with zipfile.ZipFile(OUT / name) as archive:
            if archive.testzip() is not None:
                raise ValueError("Archive integrity verification failed.")
        print(f"{OUT / name} ({(OUT / name).stat().st_size / 2**20:.2f} MiB)")


if __name__ == "__main__":
    main()
