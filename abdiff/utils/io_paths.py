from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def repo_root(cwd: Optional[str] = None) -> Path:
    """
    仅做路径管理（不做 I/O 副作用）。
    约定：用户从 AbDiff 仓库根目录运行，根目录下应存在 abdiff/。
    """
    p = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()
    return p


@dataclass(frozen=True)
class OutputLayout:
    output_root: Path

    af2_raw_dir: Path
    af2_repr_dir: Path
    igfold_dir: Path
    abfold_emb_dir: Path
    h3_mask_dir: Path
    sample_emb_dir: Path
    struct_dir: Path


def build_output_layout(output_root: str | Path) -> OutputLayout:
    out = Path(output_root)
    return OutputLayout(
        output_root=out,
        af2_raw_dir=out / "AF2_repr_raw",
        af2_repr_dir=out / "AF2_repr",
        igfold_dir=out / "igfold_embedding",
        abfold_emb_dir=out / "abfold_embedding",
        h3_mask_dir=out / "cdr_mask_H3",
        sample_emb_dir=out / "gen_abdiff_embeddings",
        struct_dir=out / "gen_structures",
    )

