#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[prep] Repo root: $ROOT_DIR"

echo "[prep] 1) pre-check"
python scripts/check.py --mode pre --check_envs || true

SKIP_CONDA="${SKIP_CONDA:-0}"
if [[ "$SKIP_CONDA" == "1" ]]; then
  echo "[prep] 2) skip conda env checks/creation (SKIP_CONDA=1)"
else
  echo "[prep] 2) check conda environments"
ENV_FILES=(
  "environments/abdiff_colabfold.txt"
  "environments/abdiff_igfold.txt"
  "environments/abdiff_abfold.txt"
  "environments/abdiff_diffusion.txt"
)

for env_file in "${ENV_FILES[@]}"; do
  if [[ ! -f "$env_file" ]]; then
    echo "[prep][ERROR] missing environment file: $env_file" >&2
    exit 1
  fi
done

echo "[prep] 3) create conda environments (if missing)"
for env_file in "${ENV_FILES[@]}"; do
  name="$(python -c "from pathlib import Path; p=Path('$env_file').name; print(p.replace('.txt',''))")"
  if conda env list --json | python -c "import json,sys;d=json.load(sys.stdin);import os;names={os.path.basename(p) for p in d.get('envs',[])};sys.exit(0 if '$name' in names else 1)"; then
    echo "[prep]   - env exists: $name"
  else
    echo "[prep]   - creating env: $name from $env_file"

    if grep -q '^@EXPLICIT' "$env_file"; then
      cache_dir="$(mktemp -d "/tmp/${name}_conda_pkgs.XXXXXX")"
      trap 'rm -rf "$cache_dir"' EXIT
      echo "[prep]     detected @EXPLICIT; use clean CONDA_PKGS_DIRS=$cache_dir"
      CONDA_PKGS_DIRS="$cache_dir" TMPDIR="$cache_dir" \
        conda create --name "$name" --file "$env_file" -y
      rm -rf "$cache_dir"
      trap - EXIT
    else
      conda create --name "$name" --file "$env_file" -y
    fi
  fi
done
fi

echo "[prep] 4) download model weights (placeholder URLs)"
# NOTE: user will fill these URLs later.
ABFOLD_CKPT_URL="${ABFOLD_CKPT_URL:-https://drive.google.com/file/d/1rpL6vZETlX7l9456pn72RGYbqDHTt0Lb/view?usp=drive_link}"
ABDIFF_CKPT_URL="${ABDIFF_CKPT_URL:-https://drive.google.com/file/d/1tvWKiDDIAns_AmWgaBpMFHjfZhxHRHA4/view?usp=drive_link}"

mkdir -p "checkpoints/abfold" "checkpoints/abdiff"
mkdir -p "checkpoints/abdiff/20250103_1_a_1"

download() {
  local url="$1"
  local out="$2"

  if [[ "$url" == __PLACEHOLDER_* ]]; then
    echo "[prep]   - skip download (placeholder URL): $out"
    return 0
  fi

  mkdir -p "$(dirname "$out")"
  rm -f "$out"

  echo "[prep]   - downloading: $url -> $out"

  # Google Drive links: use gdown for robust handling
  if [[ "$url" == *"drive.google.com"* ]]; then
    python - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("gdown") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
PY
    python -m gdown --fuzzy "$url" -O "$out"
  else
    python - "$url" "$out" <<'PY'
import sys
import urllib.request
from pathlib import Path

url = sys.argv[1]
out = Path(sys.argv[2])
out.parent.mkdir(parents=True, exist_ok=True)

req = urllib.request.Request(url, headers={"User-Agent": "AbDiff-preparation/1.0"})
with urllib.request.urlopen(req) as r, out.open("wb") as f:
    while True:
        chunk = r.read(1024 * 1024)
        if not chunk:
            break
        f.write(chunk)

print(f"saved {out} ({out.stat().st_size} bytes)")
PY
  fi

  # Fail fast if Google Drive returned an HTML page instead of a file
  python - "$out" <<'PY'
import sys
from pathlib import Path

p = Path(sys.argv[1])
data = p.read_bytes()[:4096].lstrip()

is_html = (
    data.startswith(b"<!DOCTYPE html")
    or data.startswith(b"<html")
    or b"<title>Google Drive" in data
    or b"google-site-verification" in data
)

if is_html:
    raise SystemExit(
        f"[prep][ERROR] downloaded file is HTML, not the real artifact: {p}\n"
        f"请检查 Google Drive 链接是否设置为“Anyone with the link / Viewer”，"
        f"或者文件是否需要登录权限。"
    )

print(f"saved {p} ({p.stat().st_size} bytes)")
PY
}

# Current repo expects this file path (no .pt suffix in snapshot).
download "$ABFOLD_CKPT_URL" "checkpoints/abfold/checkpoint_ema"

echo "[prep]   - diffusion checkpoint (archive) -> checkpoints/abdiff/20250103_1_a_1/"
# The diffusion checkpoint will be distributed as an archive (planned: .tar).
# We keep a generic filename; extraction is based on file suffix.
# If your URL ends with .tar.gz/.tgz/.zip you can also set ABDIFF_ARCHIVE_PATH externally.
ABDIFF_ARCHIVE_PATH="${ABDIFF_ARCHIVE_PATH:-checkpoints/abdiff/abdiff_diffusers_ckpt.tar}"
download "$ABDIFF_CKPT_URL" "$ABDIFF_ARCHIVE_PATH"

if [[ "$ABDIFF_CKPT_URL" != __PLACEHOLDER_* ]]; then
  echo "[prep]   - extracting $ABDIFF_ARCHIVE_PATH"
  python - "$ABDIFF_ARCHIVE_PATH" "checkpoints/abdiff/20250103_1_a_1" <<'PY'
import sys
import tarfile
import zipfile
from pathlib import Path
import shutil

archive_path = Path(sys.argv[1]).resolve()
target_dir = Path(sys.argv[2]).resolve()
target_dir.mkdir(parents=True, exist_ok=True)

tmp_dir = target_dir.parent / (target_dir.name + "_tmp_extract")
if tmp_dir.exists():
    shutil.rmtree(tmp_dir)
tmp_dir.mkdir(parents=True, exist_ok=True)

suffix = archive_path.name.lower()
if suffix.endswith(".zip"):
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(tmp_dir)
elif suffix.endswith(".tar") or suffix.endswith(".tar.gz") or suffix.endswith(".tgz"):
    mode = "r:*"
    with tarfile.open(archive_path, mode) as tf:
        tf.extractall(tmp_dir)
else:
    raise SystemExit(f"不支持的压缩包格式: {archive_path.name}（请使用 .tar/.tar.gz/.tgz/.zip）")

# Handle two common layouts:
# 1) zip contains model_index.json at root
# 2) zip contains a single top folder that contains model_index.json
def has_expected(p: Path) -> bool:
    return (p / "model_index.json").exists()

src = tmp_dir
children = [p for p in tmp_dir.iterdir()]
if not has_expected(tmp_dir) and len(children) == 1 and children[0].is_dir() and has_expected(children[0]):
    src = children[0]

if not has_expected(src):
    raise SystemExit(
        f"解压后未找到 model_index.json。请确认压缩包内容是 diffusers checkpoint。\n"
        f"extract_dir={tmp_dir}"
    )

# Replace target_dir contents
for p in target_dir.iterdir():
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()

for item in src.iterdir():
    dest = target_dir / item.name
    if item.is_dir():
        shutil.copytree(item, dest)
    else:
        shutil.copy2(item, dest)

shutil.rmtree(tmp_dir)
print(f"extracted to {target_dir}")
PY
fi

echo "[prep] 5) post-check"
python scripts/check.py --mode post --check_envs

echo "[prep] done"

