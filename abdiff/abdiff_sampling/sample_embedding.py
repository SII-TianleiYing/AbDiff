import torch
import os
from tqdm import tqdm
import argparse

from abdiff.abdiff_sampling.dataset import AbDiffDataset
from abdiff.abdiff_sampling.diffusion.sample import sample, repaint_sample
from abdiff.abdiff_sampling.nn.mynet.transformer2D import model, Transformer2DModel

# model.save_pretrained("/home/pengchao/data/abdiff/test/model")
# te = Transformer2DModel.from_pretrained("/home/pengchao/data/abdiff/test/model")
def main(args):
    output_dir = args.output_dir
    cdr_mask_dir = args.cdr_mask_dir
    abfold_embedding_dir = args.abfold_embedding_dir
    
    # create output dir if missing
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = AbDiffDataset(abfold_embedding_dir)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    num_samples = 100 

    for batch in tqdm(test_dataloader):
        #'6xm2', '6xp6', '6xsw', '7a0x', '7ahu','7aql', '7pgb'
        # if batch['filename'][0][:4] not in ['6xp6']:
        #     continue
        z = batch['z']
        s = batch['s']
        cdr_mask_filepath = os.path.join(cdr_mask_dir, f'{batch["filename"][0][:-5]}.pt')
        cdr_mask = torch.load(cdr_mask_filepath)
        for i in range(num_samples):
            # sample_z = sample(s, z)
            sample_z = repaint_sample(s, z, cdr_mask)

            batch['z'] = sample_z

            torch.save(batch, os.path.join(output_dir, f'{batch["filename"][0]}_{i}.pt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AbDiff sampling to generate antibody embeddings."
    )

    # replaced server paths with repo-relative defaults
    parser.add_argument(
        "--cdr_mask_dir",
        type=str,
        default="examples/example_input/cdr_mask_H3",
        help="Directory containing CDR-H3 mask .pt files",
    )
    parser.add_argument(
        "--abfold_embedding_dir",
        type=str,
        default="examples/example_input/abfold_embedding",
        help="Directory containing AbFold embedding .pt files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="examples/example_output/gen_abdiff_embeddings",
        help="Directory to save generated sampled embeddings",
    )

    args = parser.parse_args()
    main(args)
