from diffusers import UNet2DModel, UNet2DConditionModel

from dataset.dataset import AbDiffDataset

model = UNet2DConditionModel(
    # sample_size=config.image_size,  # the target image resolution
    in_channels=128,  # the number of input channels, 3 for RGB images
    out_channels=128,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128,),  # the number of output channels for each UNet block
    encoder_hid_dim=384,
    down_block_types=(
        # "DownBlock2D",  # a regular ResNet downsampling block
        # "DownBlock2D",
        # "DownBlock2D",
        "SimpleCrossAttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
    ),
    up_block_types=(
        "SimpleCrossAttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        # "UpBlock2D",
        # "UpBlock2D",
        # "UpBlock2D",
    ),
    cross_attention_dim=384,
)


if __name__ == "__main__":
    dataset = AbDiffDataset('/home/pengchao/data/abdiff/test/embedding')
    sample_image = dataset[0]['z'].unsqueeze(0)
    print("Input shape:", sample_image.shape)
    print("Output shape:", model(sample_image, timestep=0, encoder_hidden_states=dataset[0]['s'].unsqueeze(0)).sample.shape)