# AbDiff

[![English](https://img.shields.io/badge/Language-English-blue)](README.md)
[![中文](https://img.shields.io/badge/语言-中文-green)](README.zh.md)

**AbDiff: Antibody Conformation Generation Using Denoising Diffusion Probabilistic Models**

AbDiff is an end-to-end pipeline for generating antibody 3D structures. It integrates AlphaFold2-based representation extraction, IgFold embedding, AbFold encoder fusion, CDR-H3 masked diffusion sampling, and structure decoding into a unified workflow, generating diverse antibody conformations from paired heavy/light chain sequences.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Output Description](#output-description)
- [Pipeline Details](#pipeline-details)
- [Citation](#citation)
- [License](#license)

---

## Overview

AbDiff generates antibody conformations through a six-stage pipeline:

1. **AF2/ColabFold** — Extract single/pair representations from antibody sequences
2. **IgFold** — Obtain structure-aware embeddings via IgFold
3. **AbFold Encoder** — Fuse AF2 representations, IgFold embeddings, and sequence features
4. **CDR-H3 Mask** — Identify the CDR-H3 loop region using ANARCI-based numbering
5. **AbDiff Sampling** — Apply denoising diffusion with CDR-H3 inpainting
6. **Structure Decode** — Decode sampled embeddings into 3D structures (PDB)

Each stage runs in a dedicated conda environment to isolate dependencies.

---

## Repository Structure

```
AbDiff/
├── README.md                  # This file (English)
├── README.zh.md               # 中文说明
├── scripts/
│   ├── preparation.sh         # Preparation: environment setup + weight download
│   ├── run_full_pipeline.sh   # Full pipeline orchestration
│   ├── example.sh             # Example script
│   └── check.py               # Validator
├── abdiff/                    # Core models
│   ├── af2/                   # Stage 1: ColabFold/AF2
│   ├── igfold/                # Stage 2: IgFold
│   ├── abfold_encoder/        # Stage 3: AbFold encoder
│   ├── h3_mask/               # Stage 4: CDR-H3 mask
│   ├── abdiff_sampling/       # Stage 5: Diffusion sampling
│   ├── structure_decode/      # Stage 6: Structure decoding
│   └── utils/                 
├── environments/              # Conda/Pip environment files
├── checkpoints/               # Model weights (auto-downloaded via preparation.sh)
├── examples/                  # Example input and output
│   ├── example_input/
│   └── example_output/
└── docs/                      # Detailed pipeline documentation
    ├── pipeline.zh.md         # Chinese technical reference
    └── pipeline.md            # English technical reference
```

> **Tip:** By default, please run script commands from the repository root `AbDiff/`.

---

## Requirements

- **OS:** Linux or other supported systems
- **GPU:** NVIDIA GPU with CUDA support (required for diffusion sampling)
- **Conda:** Anaconda or Miniconda
- **Disk:** ~5 GB for environments, ~1 GB for checkpoints

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/SII-TianleiYing/AbDiff.git
cd AbDiff
```

### Step 2 — Run the preparation script

The preparation script automatically creates conda environments, downloads model weights, and validates the setup.

```bash
bash scripts/preparation.sh
```

**Model weights** are hosted on Google Drive and will be downloaded automatically during preparation. Please ensure you have access to Google Drive.

```bash
bash scripts/preparation.sh
```

### Conda environments created

| Environment | Purpose | Key dependencies |
|---|---|---|
| `abdiff_colabfold` | AF2 representation extraction | ColabFold, JAX, TensorFlow |
| `abdiff_igfold` | IgFold embedding | IgFold, PyTorch |
| `abdiff_abfold` | AbFold encoder and structure decode | AbFold, PyTorch, ANARCI |
| `abdiff_diffusion` | Diffusion sampling | Diffusers, PyTorch |

---

## Quick Start

Run the bundled example:

```bash
bash scripts/example.sh
```

This uses `examples/example_input/` as input and writes all outputs to `examples/example_output/run_full_pipeline_out/`.

---

## Usage

### Input format

The pipeline requires two inputs:

- **`--input_csv`** — ColabFold input CSV file (see ColabFold documentation for format)
  ```
  id,sequence
  test,<H_chain_seq>:<L_chain_seq>
  ```
- **`--fasta_dir`** — A directory containing `.fasta` files, each with paired heavy and light chain sequences:
  ```
  >:H
  EVQLVESGGGLVQPGG...
  >:L
  DIQMTQSPSSLSASVG...
  ```

### Run the full pipeline

```bash
bash scripts/run_full_pipeline.sh \
  --input_csv /path/to/input.csv \
  --fasta_dir /path/to/fasta_dir \
  --output_root ./output
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input_csv` | Yes | — | ColabFold input CSV |
| `--fasta_dir` | Yes | — | Directory of paired `.fasta` files |
| `--output_root` | No | `./output` | Root directory for all outputs |
| `--abfold_ckpt` | No | `checkpoints/abfold/checkpoint_ema` | AbFold checkpoint path |

---

## Output Description

All intermediate and final outputs are organized under `output_root/`:

```
output_root/
├── AF2_repr_raw/              # Raw ColabFold .npy files
├── AF2_repr/                  # Merged single+pair representations (.pkl)
├── igfold_embedding/          # IgFold structure embeddings (.pt)
├── abfold_embedding/          # AbFold encoder outputs (*_pred.pt)
├── cdr_mask_H3/               # CDR-H3 binary masks (.pt)
├── gen_abdiff_embeddings/     # Diffusion-sampled embeddings (.pt)
└── gen_structures/            # Final predicted structures (.pdb)
```

The final PDB files in `gen_structures/` are the primary output.

---

## Pipeline Details

For detailed stage-by-stage commands, checkpoint conventions, and troubleshooting:

- Chinese: [`docs/pipeline.zh.md`](docs/pipeline.zh.md)
- English: [`docs/pipeline.md`](docs/pipeline.md)

---

## Citation

If you use AbDiff in your research, please cite:

```bibtex
@article{abdiff2026,
  title   = {AbDiff: Antibody Conformation Generation Using Denoising Diffusion Probabilistic Models},
  author  = {},
  journal = {},
  year    = {2026}
}
```

---

## License

This project is released under the [Apache License 2.0](LICENSE).
