## AbDiff Pipeline（内部说明 / 中文）

本文档面向**仓库维护者/内部协作**，用于记录当前 AbDiff 仓库的流水线编排策略、输入输出约定、以及常见的失败点定位方式。

对外快速上手请看：
- `README.zh-CN.md`（中文）
- `README.md`（English）

### 总体策略

本仓库的核心约束是：
- **不重组项目结构**：`abdiff/`、`scripts/`、`environments/`、`checkpoints/` 等为同级目录
- **只在编排层解决问题**：尽量不修改 `abdiff/` 内部各模块逻辑（除非出现路径硬编码导致无法通过参数传递解决）
- **统一输出根目录**：用户指定 `--output_root` 后，所有中间产物与最终结果都必须落在该目录下

### 输入接口（对外）

全流程只暴露两个主要输入：
- `input_csv`：供 ColabFold 使用的输入 CSV（脚本参数 `--input_csv`）
- `fasta_dir`：包含配对 `.fasta` 文件的目录（脚本参数 `--fasta_dir`）

### 目录与产物（统一落在 output_root）

`scripts/run_full_pipeline.sh` 会在 `output_root` 下创建：
- `AF2_repr_raw/`：ColabFold 输出的 raw `.npy` representations
- `AF2_repr/`：合并后的 `.pkl`（每个样本一个）
- `igfold_embedding/`：IgFold 的 `.pt`
- `abfold_embedding/`：AbFold encoder 输出 `*_pred.pt`
- `cdr_mask_H3/`：CDR-H3 mask `.pt`
- `gen_abdiff_embeddings/`：diffusion sampling 输出 `.pt`
- `gen_structures/`：最终结构 `.pdb`

### Stage 编排顺序与命令

所有命令都从仓库根目录执行；实际由 `conda run -n <env>` 调用对应模块：

1. **AF2/ColabFold（env: `abdiff_colabfold`）**

```bash
conda run -n abdiff_colabfold python -m abdiff.af2.run_af2 \
  --fasta_csv "$INPUT_CSV" \
  --repr_dir "$OUTPUT_ROOT/AF2_repr_raw" \
  --af2_repr_dir "$OUTPUT_ROOT/AF2_repr"
```

2. **IgFold（env: `abdiff_igfold`）**

```bash
conda run -n abdiff_igfold python -m abdiff.igfold.run_igfold \
  --fasta_dir "$FASTA_DIR" \
  --output_dir "$OUTPUT_ROOT/igfold_embedding"
```

3. **AbFold encoder（env: `abdiff_abfold`）**

```bash
conda run -n abdiff_abfold python -m abdiff.abfold_encoder.run_abfold_encoder \
  --repr_dir "$OUTPUT_ROOT/AF2_repr" \
  --fasta_dir "$FASTA_DIR" \
  --point_feat_dir "$OUTPUT_ROOT/igfold_embedding" \
  --output_dir "$OUTPUT_ROOT/abfold_embedding" \
  --checkpoint_name "checkpoints/abfold/checkpoint_ema"
```

4. **H3 mask（env: `abdiff_abfold`）**

```bash
conda run -n abdiff_abfold python -m abdiff.h3_mask.run_h3_mask \
  --fasta_dir "$FASTA_DIR" \
  --output_dir "$OUTPUT_ROOT/cdr_mask_H3"
```

5. **AbDiff sampling（env: `abdiff_diffusion`）**

```bash
conda run -n abdiff_diffusion python -m abdiff.abdiff_sampling.sample_embedding \
  --cdr_mask_dir "$OUTPUT_ROOT/cdr_mask_H3" \
  --abfold_embedding_dir "$OUTPUT_ROOT/abfold_embedding" \
  --output_dir "$OUTPUT_ROOT/gen_abdiff_embeddings"
```

6. **Structure decode（env: `abdiff_abfold`）**

```bash
conda run -n abdiff_abfold python -m abdiff.structure_decode.run_structure_decode \
  --sample_emb_dir "$OUTPUT_ROOT/gen_abdiff_embeddings" \
  --output_dir "$OUTPUT_ROOT/gen_structures" \
  --checkpoint_name "checkpoints/abfold/checkpoint_ema"
```

### 权重与 checkpoint 约定

#### AbFold checkpoint（单文件）
- 期望路径：`checkpoints/abfold/checkpoint_ema`
- 注意：当前仓库快照中该文件**无 `.pt` 后缀**，脚本与编排层均以此为准。

#### AbDiff diffusion checkpoint（diffusers 目录）
- 期望目录：`checkpoints/abdiff/20250103_1_a_1/`
- 必需文件（最小集合）：
  - `model_index.json`
  - `unet/config.json`
  - `scheduler/scheduler_config.json`

`scripts/preparation.sh` 假设 diffusion checkpoint 以 zip 形式分发，并会自动解压到上述目录。

### 常见失败点（不跑模型也能提前发现/定位）

- **conda 不可用**：`scripts/check.py` 会直接报错
- **diffusers checkpoint 解压路径不对**：
  - `scripts/check.py --mode post` 会检查 `model_index.json`、`unet/config.json`、`scheduler/scheduler_config.json`
- **Step5 import 即加载 checkpoint**：
  - `abdiff/abdiff_sampling/diffusion/sample.py` 在 import 时加载 `checkpoints/abdiff/20250103_1_a_1`
  - 所以必须从仓库根目录运行，且 checkpoint 目录必须存在
- **采样默认 CUDA**：
  - 若机器无 GPU/显存不足，Step5 运行期会失败（此为模块逻辑限制，编排层当前不改变）

