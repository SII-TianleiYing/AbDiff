## AbDiff

AbDiff is an end-to-end antibody conformation generation pipeline based on diffusion models, orchestrating multiple external toolchains (ColabFold/AF2, IgFold, AbFold, AbDiff diffusion).

This repository is designed to be run **from the AbDiff repo root**, where `abdiff/`, `scripts/`, `environments/`, `checkpoints/`, etc. are sibling directories.

If you prefer Chinese documentation, see `README.zh-CN.md`.

## Repository layout (important)

Run everything from the repository root:

```bash
cd AbDiff
```

Key directories:
- `scripts/`: orchestration scripts (preparation + full pipeline)
- `environments/`: conda environment definitions (4 environments)
- `checkpoints/`: model weights and diffusion checkpoints
- `examples/`: example input and expected output structure
- `docs/`: internal pipeline notes

## Environments (4 conda envs)

Environment lock files live in `environments/` (`@EXPLICIT` format):
- `environments/abdiff_colabfold.txt` → env name `abdiff_colabfold`
- `environments/abdiff_igfold.txt` → env name `abdiff_igfold`
- `environments/abdiff_abfold.txt` → env name `abdiff_abfold`
- `environments/abdiff_diffusion.txt` → env name `abdiff_diffusion`

## Preparation (create envs + download weights)

You only need to fill in the URLs via environment variables:

```bash
export ABFOLD_CKPT_URL="__FILL_ME__"      # single checkpoint file
export ABDIFF_CKPT_URL="__FILL_ME__"      # diffusion checkpoint archive (.tar/.tar.gz/.tgz/.zip)
bash scripts/preparation.sh
```

What `scripts/preparation.sh` does:
- pre-check (validator only)
- create the 4 conda environments if missing
- download:
  - AbFold checkpoint → `checkpoints/abfold/checkpoint_ema`
  - AbDiff diffusion archive → extracted into `checkpoints/abdiff/20250103_1_a_1/`
- post-check (stricter validation)

## Run the example

```bash
bash scripts/example.sh
```

By default it uses:
- `examples/example_input/test.csv`
- `examples/example_input/fasta_dir/`
and writes everything under:
- `examples/example_output/run_full_pipeline_out/`

## Run your own data (full pipeline)

The pipeline interface is **path-based**:
- `--input_csv`: input CSV for ColabFold
- `--fasta_dir`: directory containing paired `.fasta` files

```bash
bash scripts/run_full_pipeline.sh \
  --input_csv /path/to/input.csv \
  --fasta_dir /path/to/fasta_dir \
  --output_root ./output
```

Optional:
- `--abfold_ckpt`: override AbFold checkpoint path (default `checkpoints/abfold/checkpoint_ema`)

## Output layout (under output_root)

All intermediate and final artifacts are written under `output_root/`:
- `AF2_repr_raw/` (raw colabfold `.npy`)
- `AF2_repr/` (merged `.pkl`)
- `igfold_embedding/` (`.pt`)
- `abfold_embedding/` (`*_pred.pt`)
- `cdr_mask_H3/` (`.pt`)
- `gen_abdiff_embeddings/` (sampled embedding `.pt`)
- `gen_structures/` (final `.pdb`)

## Internal pipeline notes

See:
- Chinese: `docs/pipeline.md`
- English: `docs/pipeline.en.md`

