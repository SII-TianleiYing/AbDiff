# AbDiff

[![English](https://img.shields.io/badge/Language-English-blue)](README.md)
[![中文](https://img.shields.io/badge/语言-中文-green)](README.zh.md)

**AbDiff：基于去噪扩散概率模型的抗体构象生成工具**

AbDiff 是一个端到端的抗体三维结构生成管线。它将 AlphaFold2 特征提取、IgFold 嵌入、AbFold 编码器融合、CDR-H3 掩码扩散采样和结构解码整合为统一的工作流，从配对的重/轻链序列出发，生成多样化的抗体构象。

---

## 目录

- [方法概述](#方法概述)
- [仓库结构](#仓库结构)
- [系统要求](#系统要求)
- [安装](#安装)
- [快速开始](#快速开始)
- [使用方法](#使用方法)
- [输出说明](#输出说明)
- [流水线详细文档](#流水线详细文档)
- [引用](#引用)
- [许可证](#许可证)

---

## 方法概述

AbDiff 通过六阶段流水线生成抗体构象：

1. **AF2/ColabFold** — 从抗体序列提取 single/pair 表征
2. **IgFold** — 通过 IgFold 获取结构感知嵌入
3. **AbFold Encoder** — 融合 AF2 表征、IgFold 嵌入与序列特征
4. **CDR-H3 Mask** — 基于 ANARCI 编号识别 CDR-H3 环区
5. **AbDiff Sampling** — 使用去噪扩散模型进行 CDR-H3 区域 inpainting 采样
6. **Structure Decode** — 将采样嵌入解码为三维结构（PDB 格式）

每个阶段运行在独立的 conda 环境中，以隔离不同的依赖。

---

## 仓库结构

```
AbDiff/
├── README.md                  # English README
├── README.zh.md               # 本文件（中文）
├── scripts/
│   ├── preparation.sh         # 准备流程：环境创建及权重下载
│   ├── run_full_pipeline.sh   # 全流程编排脚本
│   ├── example.sh             # 示例脚本
│   └── check.py               # 校验器
├── abdiff/                    # 核心模型
│   ├── af2/                   # 阶段 1: ColabFold/AF2
│   ├── igfold/                # 阶段 2: IgFold
│   ├── abfold_encoder/        # 阶段 3: AbFold 编码器
│   ├── h3_mask/               # 阶段 4: CDR-H3 掩码
│   ├── abdiff_sampling/       # 阶段 5: 扩散采样
│   ├── structure_decode/      # 阶段 6: 结构解码
│   └── utils/                 
├── environments/              # Conda/Pip 环境文件
├── checkpoints/               # 模型权重（通过 preparation.sh 可以自动下载）
├── examples/                  # 示例输入与输出
│   ├── example_input/
│   └── example_output/
└── docs/                      # 详细管线文档
    ├── pipeline.zh.md         # 中文详细说明
    └── pipeline.md            # 英文详细说明
```

> **提示：** 默认设置下，请在仓库根目录 `AbDiff/` 下执行脚本命令。

---

## 系统要求

- **操作系统：** Linux或其它支持系统
- **GPU：** 支持 CUDA 的 NVIDIA 显卡（扩散采样阶段必需）
- **Conda：** Anaconda 或 Miniconda
- **磁盘空间：** 环境约 5 GB，权重约 1 GB

---

## 安装

### 第一步 — 克隆仓库

```bash
git clone https://github.com/SII-TianleiYing/AbDiff.git
cd AbDiff
```

### 第二步 — 运行准备脚本

准备脚本会自动创建 conda 环境、下载模型权重并校验安装结果。

```bash
bash scripts/preparation.sh
```

**模型权重**我们将权重托管在 Google Drive 上，会随环境准备自动下载，请确保可以访问 Google Drive 。

```bash
bash scripts/preparation.sh
```

### 创建的 conda 环境

| 环境名 | 用途 | 关键依赖 |
|---|---|---|
| `abdiff_colabfold` | AF2 表征提取 | ColabFold, JAX, TensorFlow |
| `abdiff_igfold` | IgFold 嵌入 | IgFold, PyTorch |
| `abdiff_abfold` | AbFold 编码器和结构解码 | AbFold, PyTorch, ANARCI |
| `abdiff_diffusion` | 扩散采样 | Diffusers, PyTorch |

---

## 快速开始

运行内置示例：

```bash
bash scripts/example.sh
```

该脚本使用 `examples/example_input/` 作为输入，所有输出写入 `examples/example_output/run_full_pipeline_out/`。

---

## 使用方法

### 输入格式

流水线需要两个输入：

- **`--input_csv`** — ColabFold 输入 CSV 文件（格式参见 ColabFold 文档）
  ```
  id,sequence
  test,<H_chain_seq>:<L_chain_seq>
  ```
- **`--fasta_dir`** — 包含 `.fasta` 文件的目录，每个文件包含配对的重链和轻链序列：
  ```
  >:H
  EVQLVESGGGLVQPGG...
  >:L
  DIQMTQSPSSLSASVG...
  ```

### 运行全流程

```bash
bash scripts/run_full_pipeline.sh \
  --input_csv /path/to/input.csv \
  --fasta_dir /path/to/fasta_dir \
  --output_root ./output
```

| 参数 | 必需 | 默认值 | 说明 |
|---|---|---|---|
| `--input_csv` | 是 | — | ColabFold 输入 CSV |
| `--fasta_dir` | 是 | — | 配对 `.fasta` 文件目录 |
| `--output_root` | 否 | `./output` | 所有输出的根目录 |
| `--abfold_ckpt` | 否 | `checkpoints/abfold/checkpoint_ema` | AbFold checkpoint 路径 |

---

## 输出说明

所有中间产物与最终结果统一组织在 `output_root/` 下：

```
output_root/
├── AF2_repr_raw/              # ColabFold 原始 .npy 文件
├── AF2_repr/                  # 合并的 single+pair 表征（.pkl）
├── igfold_embedding/          # IgFold 结构嵌入（.pt）
├── abfold_embedding/          # AbFold 编码器输出（*_pred.pt）
├── cdr_mask_H3/               # CDR-H3 二值掩码（.pt）
├── gen_abdiff_embeddings/     # 扩散采样嵌入（.pt）
└── gen_structures/            # 最终预测结构（.pdb）
```

`gen_structures/` 中的 PDB 文件即为最终输出。

---

## 管线详细文档

关于每个阶段的具体命令、checkpoint 约定和常见问题排查，请参阅：

- 中文：[`docs/pipeline.zh.md`](docs/pipeline.zh.md)
- English：[`docs/pipeline.md`](docs/pipeline.md)

---

## 引用

如果 AbDiff 对您的研究有帮助，请引用：

```bibtex
@article{abdiff2026,
  title   = {AbDiff: Antibody Conformation Generation Using Denoising Diffusion Probabilistic Models},
  author  = {},
  journal = {},
  year    = {2026}
}
```

---

## 许可证

本项目遵循 [Apache License 2.0](LICENSE)。

