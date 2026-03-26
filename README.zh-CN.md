## AbDiff

AbDiff 是一个抗体构象生成（structure generation）流程，基于扩散模型并编排多个外部工具链（ColabFold/AF2、IgFold、AbFold、AbDiff diffusion）。

本仓库的使用约定是：**所有命令都在 AbDiff 仓库根目录执行**，其中 `abdiff/`、`scripts/`、`environments/`、`checkpoints/` 等目录彼此同级。

如果你需要英文说明，请看 `README.md`。

## 仓库结构（非常重要）

请从仓库根目录开始：

```bash
cd AbDiff
```

关键目录：
- `scripts/`：编排脚本（准备环境 + 跑全流程）
- `environments/`：conda 环境定义（4 个环境）
- `checkpoints/`：模型权重与 diffusion checkpoint
- `examples/`：示例输入/输出
- `docs/`：更详细的内部 pipeline 说明

## conda 环境（4 个）

环境锁定文件都在 `environments/`（`@EXPLICIT` 格式）：
- `environments/abdiff_colabfold.txt` → 环境名 `abdiff_colabfold`
- `environments/abdiff_igfold.txt` → 环境名 `abdiff_igfold`
- `environments/abdiff_abfold.txt` → 环境名 `abdiff_abfold`
- `environments/abdiff_diffusion.txt` → 环境名 `abdiff_diffusion`

## 准备（创建环境 + 下载权重 + 解压 diffusion zip）

你只需要通过环境变量补上 URL：

```bash
export ABFOLD_CKPT_URL="__请填写__"   # 单文件权重
export ABDIFF_CKPT_URL="__请填写__"   # diffusion checkpoint 压缩包（.tar/.tar.gz/.tgz/.zip）
bash scripts/preparation.sh
```

`scripts/preparation.sh` 会执行：
- pre-check（只校验，不安装/不下载）
- 如果缺失则创建 4 个 conda 环境
- 下载：
  - AbFold checkpoint → `checkpoints/abfold/checkpoint_ema`
  - AbDiff diffusion 压缩包 → 自动解压到 `checkpoints/abdiff/20250103_1_a_1/`
- post-check（更严格：会校验 diffusers 目录结构是否完整）

## 跑示例

```bash
bash scripts/example.sh
```

默认使用：
- `examples/example_input/test.csv`
- `examples/example_input/fasta_dir/`
并把所有中间产物与最终结果写到：
- `examples/example_output/run_full_pipeline_out/`

## 跑你自己的数据（全流程）

全流程的输入接口是**两个路径参数**：
- `--input_csv`：ColabFold 输入 CSV
- `--fasta_dir`：包含配对 `.fasta` 的目录（每个文件里包含 H/L 两条链）

```bash
bash scripts/run_full_pipeline.sh \
  --input_csv /path/to/input.csv \
  --fasta_dir /path/to/fasta_dir \
  --output_root ./output
```

可选参数：
- `--abfold_ckpt`：覆盖 AbFold checkpoint 路径（默认 `checkpoints/abfold/checkpoint_ema`）

## 输出目录结构（全部在 output_root 下）

`--output_root` 下会生成（不会散落到其他目录）：
- `AF2_repr_raw/`：colabfold 生成的原始 `.npy`
- `AF2_repr/`：合并后的 `.pkl`
- `igfold_embedding/`：IgFold `.pt`
- `abfold_embedding/`：AbFold encoder 输出 `*_pred.pt`
- `cdr_mask_H3/`：H3 mask `.pt`
- `gen_abdiff_embeddings/`：diffusion 采样输出 `.pt`
- `gen_structures/`：最终结构 `.pdb`

## 更详细的内部 pipeline 说明

请看：
- 中文：`docs/pipeline.md`
- English：`docs/pipeline.en.md`

