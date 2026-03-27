import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_REQUIRED_RELATIVE_PATHS = [
    "abdiff",
    "environments",
    "checkpoints",
    "examples",
]

ENV_SPECS = {
    "abdiff_colabfold": {
        "env_file": "environments/abdiff_colabfold.txt",
        "pip_file": "environments/abdiff_colabfold_pip.txt",
        "imports": ["numpy", "pandas"],
    },
    "abdiff_igfold": {
        "env_file": "environments/abdiff_igfold.txt",
        "pip_file": "environments/abdiff_igfold_pip.txt",
        "imports": ["torch", "numpy"],
    },
    "abdiff_abfold": {
        "env_file": "environments/abdiff_abfold.txt",
        "pip_file": "environments/abdiff_abfold_pip.txt",
        "imports": ["torch", "numpy", "ml_collections", "anarci", "tree", "diffusers"],
    },
    "abdiff_diffusion": {
        "env_file": "environments/abdiff_diffusion.txt",
        "pip_file": "environments/abdiff_diffusion_pip.txt",
        "imports": ["torch", "numpy", "diffusers"],
    },
}


@dataclass
class CheckResult:
    ok: bool
    errors: List[str]
    warnings: List[str]


def _repo_root() -> Path:
    return Path.cwd().resolve()


def _run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def _check_repo_layout(root: Path) -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []

    for rel in REPO_REQUIRED_RELATIVE_PATHS:
        p = root / rel
        if not p.exists():
            errors.append(f"missing directory or file: {rel}")

    for env_name, spec in ENV_SPECS.items():
        env_file = root / spec["env_file"]
        pip_file = root / spec["pip_file"]

        if not env_file.exists():
            errors.append(f"missing conda env definition file: {spec['env_file']}")
        if not pip_file.exists():
            warnings.append(f"missing pip dependency file: {spec['pip_file']} (pip supplement install will be skipped)")

    if not (root / "docs/pipeline.md").exists():
        warnings.append("docs/pipeline.md not found (or empty)")

    return CheckResult(ok=len(errors) == 0, errors=errors, warnings=warnings)


def _check_pip_files_sanity(root: Path) -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []

    for env_name, spec in ENV_SPECS.items():
        pip_file = root / spec["pip_file"]
        if not pip_file.exists():
            continue

        for i, raw in enumerate(pip_file.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("pip install "):
                errors.append(
                    f"{spec['pip_file']}:{i} should not contain shell command `pip install ...`; "
                    f"use requirements format instead, e.g. `abfold @ git+https://...`"
                )

            if " @ file://" in line or line.startswith("file://"):
                warnings.append(
                    f"{spec['pip_file']}:{i} contains local file path dependency (not reproducible across machines): {line}"
                )

            if line.startswith("-e ") and ("file://" in line or line[3:].startswith(("/", ".", ".."))):
                warnings.append(
                    f"{spec['pip_file']}:{i} contains local editable/path dependency (not reproducible across machines): {line}"
                )

    return CheckResult(ok=len(errors) == 0, errors=errors, warnings=warnings)


def _check_conda_available() -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []

    code, out, err = _run(["conda", "--version"])
    if code != 0:
        errors.append("cannot run conda. Please ensure conda is in PATH (e.g. Anaconda/Miniconda is installed).")
        if err.strip():
            warnings.append(f"conda stderr: {err.strip()}")
        return CheckResult(ok=False, errors=errors, warnings=warnings)

    return CheckResult(ok=True, errors=[], warnings=[])


def _conda_envs_json() -> Optional[Dict]:
    code, out, _ = _run(["conda", "env", "list", "--json"])
    if code != 0:
        return None
    try:
        return json.loads(out)
    except Exception:
        return None


def _check_envs_exist(expected_names: List[str]) -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []

    data = _conda_envs_json()
    if data is None or "envs" not in data:
        warnings.append("unable to retrieve env list via `conda env list --json` (skipping env existence check).")
        return CheckResult(ok=True, errors=[], warnings=warnings)

    env_paths = data.get("envs", [])
    env_names = {Path(p).name for p in env_paths if isinstance(p, str)}

    missing = [n for n in expected_names if n not in env_names]
    if missing:
        errors.append("the following conda environments do not exist (run preparation.sh to create): " + ", ".join(missing))

    return CheckResult(ok=len(errors) == 0, errors=errors, warnings=warnings)


def _check_env_imports(expected_names: List[str]) -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []

    data = _conda_envs_json()
    if data is None or "envs" not in data:
        warnings.append("unable to retrieve conda env list (skipping env import check).")
        return CheckResult(ok=True, errors=[], warnings=warnings)

    env_paths = data.get("envs", [])
    env_names = {Path(p).name for p in env_paths if isinstance(p, str)}

    for env_name in expected_names:
        if env_name not in env_names:
            continue

        imports = ENV_SPECS[env_name]["imports"]
        if not imports:
            continue

        py_code = (
            "import importlib, sys\n"
            f"mods = {imports!r}\n"
            "missing = []\n"
            "for m in mods:\n"
            "    try:\n"
            "        importlib.import_module(m)\n"
            "    except Exception:\n"
            "        missing.append(m)\n"
            "if missing:\n"
            "    raise SystemExit(','.join(missing))\n"
        )

        code, out, err = _run(["conda", "run", "-n", env_name, "python", "-c", py_code])
        if code != 0:
            miss = (err.strip() or out.strip() or "unknown").splitlines()[-1]
            errors.append(f"env {env_name} is missing critical import(s): {miss}")

    return CheckResult(ok=len(errors) == 0, errors=errors, warnings=warnings)


def _check_inputs(input_csv: Optional[str], fasta_dir: Optional[str]) -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []
    root = _repo_root()

    if input_csv is not None:
        p = (root / input_csv).resolve() if not os.path.isabs(input_csv) else Path(input_csv)
        if not p.exists():
            errors.append(f"input_csv does not exist: {input_csv}")

    if fasta_dir is not None:
        p = (root / fasta_dir).resolve() if not os.path.isabs(fasta_dir) else Path(fasta_dir)
        if not p.exists():
            errors.append(f"fasta_dir does not exist: {fasta_dir}")
        elif not any(p.glob("*.fasta")):
            warnings.append(f"no .fasta files found in fasta_dir: {fasta_dir}")

    return CheckResult(ok=len(errors) == 0, errors=errors, warnings=warnings)


def _check_weights(mode: str) -> CheckResult:
    """
    Validation only; does not download or install anything.
    - pre mode: missing weights emit warnings
    - post mode: missing weights emit errors
    """
    root = _repo_root()
    errors: List[str] = []
    warnings: List[str] = []

    abfold_ckpt = root / "checkpoints/abfold/checkpoint_ema"
    abdiff_ckpt_dir = root / "checkpoints/abdiff/20250103_1_a_1"

    missing: List[str] = []
    if not abfold_ckpt.exists():
        missing.append("checkpoints/abfold/checkpoint_ema")
    if not abdiff_ckpt_dir.exists():
        missing.append("checkpoints/abdiff/20250103_1_a_1/")

    if missing:
        msg = "missing weights/checkpoints: " + ", ".join(missing)
        if mode == "post":
            errors.append(msg)
        else:
            warnings.append(msg)
        return CheckResult(ok=len(errors) == 0, errors=errors, warnings=warnings)

    if mode == "post":
        need_files = [
            abdiff_ckpt_dir / "model_index.json",
            abdiff_ckpt_dir / "unet" / "config.json",
            abdiff_ckpt_dir / "scheduler" / "scheduler_config.json",
        ]
        missing_files = [str(p.relative_to(root)) for p in need_files if not p.exists()]
        if missing_files:
            errors.append(
                "Diffusers checkpoint structure is incomplete, missing: " + ", ".join(missing_files)
            )

    return CheckResult(ok=len(errors) == 0, errors=errors, warnings=warnings)


def _merge(results: List[CheckResult]) -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []
    ok = True
    for r in results:
        ok = ok and r.ok
        errors.extend(r.errors)
        warnings.extend(r.warnings)
    return CheckResult(ok=ok, errors=errors, warnings=warnings)


def main():
    parser = argparse.ArgumentParser(description="AbDiff repository checker (validation only, no install/download)")
    parser.add_argument(
        "--mode",
        choices=["pre", "post"],
        default="pre",
        help="pre: check before preparation; post: check after preparation (stricter, requires weights)",
    )
    parser.add_argument("--input_csv", default=None, help="input CSV (optional, existence check only)")
    parser.add_argument("--fasta_dir", default=None, help="FASTA directory (optional, existence check only)")
    parser.add_argument(
        "--check_envs",
        action="store_true",
        help="check whether conda environments exist and verify critical imports",
    )
    args = parser.parse_args()

    root = _repo_root()
    if not (root / "abdiff").exists():
        print("Error: please run from the AbDiff repository root (abdiff/ directory should exist here).", file=sys.stderr)
        sys.exit(2)

    expected_envs = list(ENV_SPECS.keys())

    checks: List[CheckResult] = []
    checks.append(_check_repo_layout(root))
    checks.append(_check_pip_files_sanity(root))
    checks.append(_check_conda_available())
    if args.check_envs:
        checks.append(_check_envs_exist(expected_envs))
        checks.append(_check_env_imports(expected_envs))
    checks.append(_check_inputs(args.input_csv, args.fasta_dir))
    checks.append(_check_weights(args.mode))

    res = _merge(checks)

    if res.warnings:
        print("\n[WARN]")
        for w in res.warnings:
            print(f"- {w}")

    if res.errors:
        print("\n[ERROR]")
        for e in res.errors:
            print(f"- {e}")

    if res.ok:
        print("\nAll checks passed.")
        sys.exit(0)
    else:
        print("\nCheck failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()