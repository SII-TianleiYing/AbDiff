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

ENV_FILES = [
    "environments/abdiff_colabfold.txt",
    "environments/abdiff_igfold.txt",
    "environments/abdiff_abfold.txt",
    "environments/abdiff_diffusion.txt",
]

DEFAULT_ENV_NAMES = {
    "colabfold": "abdiff_colabfold",
    "igfold": "abdiff_igfold",
    "abfold": "abdiff_abfold",
    "diffusion": "abdiff_diffusion",
}


@dataclass
class CheckResult:
    ok: bool
    errors: List[str]
    warnings: List[str]


def _repo_root() -> Path:
    # Users are expected to run from repo root; enforce for predictable relative paths.
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
            errors.append(f"缺少目录或文件: {rel}")

    for rel in ENV_FILES:
        p = root / rel
        if not p.exists():
            errors.append(f"缺少 conda 环境定义文件: {rel}")

    # These are intended but may not exist yet in the current snapshot.
    if not (root / "docs/pipeline.md").exists():
        warnings.append("未发现 docs/pipeline.md（或为空）")

    return CheckResult(ok=len(errors) == 0, errors=errors, warnings=warnings)


def _check_conda_available() -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []

    code, out, err = _run(["conda", "--version"])
    if code != 0:
        errors.append("无法运行 conda。请确保 conda 在 PATH 中（例如 Anaconda/Miniconda 已安装）。")
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
        warnings.append("无法通过 `conda env list --json` 获取环境列表（将跳过环境存在性检查）。")
        return CheckResult(ok=True, errors=[], warnings=warnings)

    env_paths = data.get("envs", [])
    env_names = {Path(p).name for p in env_paths if isinstance(p, str)}

    missing = [n for n in expected_names if n not in env_names]
    if missing:
        errors.append("以下 conda 环境不存在（需要 preparation.sh 创建）: " + ", ".join(missing))

    return CheckResult(ok=len(errors) == 0, errors=errors, warnings=warnings)


def _check_inputs(input_csv: Optional[str], fasta_dir: Optional[str]) -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []
    root = _repo_root()

    if input_csv is not None:
        p = (root / input_csv).resolve() if not os.path.isabs(input_csv) else Path(input_csv)
        if not p.exists():
            errors.append(f"input_csv 不存在: {input_csv}")

    if fasta_dir is not None:
        p = (root / fasta_dir).resolve() if not os.path.isabs(fasta_dir) else Path(fasta_dir)
        if not p.exists():
            errors.append(f"fasta_dir 不存在: {fasta_dir}")
        elif not any(p.glob("*.fasta")):
            warnings.append(f"fasta_dir 下未发现 .fasta 文件: {fasta_dir}")

    return CheckResult(ok=len(errors) == 0, errors=errors, warnings=warnings)


def _check_weights(mode: str) -> CheckResult:
    """
    仅做校验，不下载、不安装。
    - pre 模式：允许缺失（给出 warning）
    - post 模式：要求存在（缺失即 error）
    """
    root = _repo_root()
    errors: List[str] = []
    warnings: List[str] = []

    # AbFold checkpoint (repo current file has no .pt suffix)
    abfold_ckpt = root / "checkpoints/abfold/checkpoint_ema"
    # Diffusion checkpoint directory (diffusers format)
    abdiff_ckpt_dir = root / "checkpoints/abdiff/20250103_1_a_1"

    missing: List[str] = []
    if not abfold_ckpt.exists():
        missing.append("checkpoints/abfold/checkpoint_ema")
    if not abdiff_ckpt_dir.exists():
        missing.append("checkpoints/abdiff/20250103_1_a_1/")

    if missing:
        msg = "缺少权重/检查点: " + ", ".join(missing)
        if mode == "post":
            errors.append(msg)
        else:
            warnings.append(msg)
        return CheckResult(ok=len(errors) == 0, errors=errors, warnings=warnings)

    # post 模式下做更严格的结构检查，避免“目录存在但不是 diffusers ckpt”
    if mode == "post":
        need_files = [
            abdiff_ckpt_dir / "model_index.json",
            abdiff_ckpt_dir / "unet" / "config.json",
            abdiff_ckpt_dir / "scheduler" / "scheduler_config.json",
        ]
        missing_files = [str(p.relative_to(root)) for p in need_files if not p.exists()]
        if missing_files:
            errors.append(
                "Diffusers checkpoint 结构不完整，缺少: " + ", ".join(missing_files)
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
    parser = argparse.ArgumentParser(description="AbDiff 仓库检查脚本（只校验，不安装/不下载）")
    parser.add_argument(
        "--mode",
        choices=["pre", "post"],
        default="pre",
        help="pre: preparation 前检查；post: preparation 后检查（更严格，要求权重存在）",
    )
    parser.add_argument("--input_csv", default=None, help="输入 CSV（可选，仅用于校验存在性）")
    parser.add_argument("--fasta_dir", default=None, help="FASTA 目录（可选，仅用于校验存在性）")
    parser.add_argument(
        "--check_envs",
        action="store_true",
        help="检查四个 conda 环境是否已存在（仅检查，不创建）",
    )
    args = parser.parse_args()

    root = _repo_root()
    if not (root / "abdiff").exists():
        print("错误：请在 AbDiff 仓库根目录运行（根目录下应有 abdiff/ 目录）。", file=sys.stderr)
        sys.exit(2)

    checks: List[CheckResult] = []
    checks.append(_check_repo_layout(root))
    checks.append(_check_conda_available())
    if args.check_envs:
        checks.append(_check_envs_exist(list(DEFAULT_ENV_NAMES.values())))
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
        print("\n✅ 检查通过")
        sys.exit(0)
    else:
        print("\n❌ 检查失败")
        sys.exit(1)


if __name__ == "__main__":
    main()

