import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

""" This module will handle loading the antibody embedding features from AbFold and structuring them into a format compatible with the diffusion model.
"""

class AbDiffDataset(Dataset):
    def __init__(self, dir, std_path='checkpoints/std_mean.pt') -> None:
        super().__init__()

        self.dir = dir
        self.std = None
        self.mean = None
        if std_path is not None and os.path.exists(std_path):
            std_mean = torch.load(std_path, map_location='cpu')
            self.std = std_mean['std']
            self.mean = std_mean['mean']
        self.datas = []
        for file in os.listdir(self.dir):
            path = os.path.join(self.dir, file)
            if not path.endswith(".pt"):
                continue
            data = torch.load(path, map_location='cpu')
            self.transform(data)
            data['filename'] = os.path.splitext(file)[0]
            self.datas.append(data)

    def transform(self, data):
        z = data['z']
        # Normalize to the range of [-1, 1].
        # if self.std is not None and self.mean is not None:
        #     z = (z - self.mean) / self.std

        # z = z.permute(2, 0, 1)

        # Ensure consistent image size after downsampling and upsampling through padding.
        # Calculate padding size.
        # height, width = z.shape[1], z.shape[2]
        # base = 32
        # new_height = (height + (base - 1)) // base * base Round up to the nearest multiple of 8.
        # new_width = (width + (base - 1)) // base * base

        # Apply padding.
        # z = F.pad(z, (0, new_width - width, 0, new_height - height), mode="constant", value=0)
        data['z'] = z

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        return self.datas[index]
    

if __name__ == "__main__":
    # Obtain the mean and standard deviation of the training data.
    dataset_dir = 'examples/example_input/abfold_embedding'
    stats_out_path = 'checkpoints/std_mean.pt'

    tensors = []
    dataset = AbDiffDataset(dataset_dir, std_path=None)
    for i, data in enumerate(dataset):
        tensors.append(data['z'])

    all_data = torch.cat([t.view(-1, 128) for t in tensors], dim=0)
    # Calculate the mean and standard deviation.
    mean = all_data.mean(dim=0)
    std = all_data.std(dim=0)

    print("The Mean Value:", mean)
    print("The Standard Deviation:", std)

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({'std': std, 'mean': mean}, stats_out_path)
