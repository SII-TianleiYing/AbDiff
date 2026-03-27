# AbDiff Pipeline — 技术参考文档

[![English](https://img.shields.io/badge/Language-English-blue)](pipeline.md)
[![中文](https://img.shields.io/badge/语言-中文-green)](pipeline.zh.md)

本文档提供 AbDiff 管线的详细技术参考,包括各阶段命令、环境路由、checkpoint 以及常见问题排查。快速入门请参阅：

- [`README.md`](../README.md)（English）
- [`README.zh.md`](../README.zh.md)（中文）

---

## 1. 架构总览

```
input_csv + fasta_dir
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ 阶段 1  AF2/ColabFold         (环境: abdiff_colabfold)       │
│         → AF2_repr_raw/ (.npy) → AF2_repr/ (.pkl)            │
├──────────────────────────────────────────────────────────────┤
│ 阶段 2  IgFold                (环境: abdiff_igfold)          │
│         → igfold_embedding/ (.pt)                            │
├──────────────────────────────────────────────────────────────┤
│ 阶段 3  AbFold Encoder        (环境: abdiff_abfold)          │
│         → abfold_embedding/ (*_pred.pt)                      │
├──────────────────────────────────────────────────────────────┤
│ 阶段 4  CDR-H3 Mask           (环境: abdiff_abfold)          │
│         → cdr_mask_H3/ (.pt)                                 │
├──────────────────────────────────────────────────────────────┤
│ 阶段 5  AbDiff Sampling       (环境: abdiff_diffusion)       │
│         → gen_abdiff_embeddings/ (.pt)                       │
├──────────────────────────────────────────────────────────────┤
│ 阶段 6  Structure Decode      (环境: abdiff_abfold)          │
│         → gen_structures/ (.pdb)                             │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 各阶段命令

所有命令从仓库根目录执行，通过 `conda run -n <env>` 调度：

### 阶段 1 — AF2/ColabFold

```bash
conda run -n abdiff_colabfold python -m abdiff.af2.run_af2 \
  --fasta_csv "$INPUT_CSV" \
  --repr_dir "$OUTPUT_ROOT/AF2_repr_raw" \
  --af2_repr_dir "$OUTPUT_ROOT/AF2_repr"
```

### 阶段 2 — IgFold

```bash
conda run -n abdiff_igfold python -m abdiff.igfold.run_igfold \
  --fasta_dir "$FASTA_DIR" \
  --output_dir "$OUTPUT_ROOT/igfold_embedding"
```

### 阶段 3 — AbFold Encoder

```bash
conda run -n abdiff_abfold python -m abdiff.abfold_encoder.run_abfold_encoder \
  --repr_dir "$OUTPUT_ROOT/AF2_repr" \
  --fasta_dir "$FASTA_DIR" \
  --point_feat_dir "$OUTPUT_ROOT/igfold_embedding" \
  --output_dir "$OUTPUT_ROOT/abfold_embedding" \
  --checkpoint_name "checkpoints/abfold/checkpoint_ema"
```

### 阶段 4 — CDR-H3 Mask

```bash
conda run -n abdiff_abfold python -m abdiff.h3_mask.run_h3_mask \
  --fasta_dir "$FASTA_DIR" \
  --output_dir "$OUTPUT_ROOT/cdr_mask_H3"
```

### 阶段 5 — AbDiff Sampling

```bash
conda run -n abdiff_diffusion python -m abdiff.abdiff_sampling.sample_embedding \
  --cdr_mask_dir "$OUTPUT_ROOT/cdr_mask_H3" \
  --abfold_embedding_dir "$OUTPUT_ROOT/abfold_embedding" \
  --output_dir "$OUTPUT_ROOT/gen_abdiff_embeddings"
```
- **需要 CUDA GPU** — `sample.py` 默认使用 `cuda:0`

### 阶段 6 — Structure Decode

```bash
conda run -n abdiff_abfold python -m abdiff.structure_decode.run_structure_decode \
  --sample_emb_dir "$OUTPUT_ROOT/gen_abdiff_embeddings" \
  --output_dir "$OUTPUT_ROOT/gen_structures" \
  --checkpoint_name "checkpoints/abfold/checkpoint_ema"
```

---

## 3. Checkpoint

### AbFold checkpoint

| 项目 | 值 |
|---|---|
| 路径 | `checkpoints/abfold/checkpoint_ema` |
| 格式 | PyTorch state dict|

### AbDiff diffusion checkpoint

| 项目 | 值 |
|---|---|
| 路径 | `checkpoints/abdiff/20250103_1_a_1/` |
| 格式 | Diffusers pipeline（UNet + scheduler） |
| 分发方式 | `.tar` 压缩包，由 `preparation.sh` 自动解压 |

---

## 4. 常见问题排查

| 现象 | 可能原因 | 解决方案 |
|---|---|---|
| `conda: command not found` | conda 不在 PATH 中 | 安装 Anaconda/Miniconda 并重启终端 |
| `model_index.json not found` | Diffusion checkpoint 缺失或解压失败 | 重新运行 `preparation.sh` 或手动解压 `.tar` 到 `checkpoints/abdiff/20250103_1_a_1/` |
| 阶段 5 报 `CUDA out of memory` | GPU 显存不足 | 使用显存更大的 GPU；阶段 5 需要 CUDA |
| 阶段 5 在 import 时失败 | Checkpoint 目录必须在 import 时存在 | 确保 `checkpoints/abdiff/20250103_1_a_1/` 已填充 |
| 阶段 1 报 `hhsearch not found` | 启用了 ColabFold 模板但缺少 HHsuite | 在 `abdiff_colabfold` 环境中安装 HHsuite，或禁用模板 |
| PDB 结果异常 | AbFold checkpoint 版本不匹配 | 确认 `checkpoints/abfold/checkpoint_ema` 为对应版本的权重 |
