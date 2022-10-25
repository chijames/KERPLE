# KERPLE: Kernelized Relative Positional Embedding for Length Extrapolation

PyTorch implementation of the paper [KERPLE: Kernelized Relative Positional Embedding for Length Extrapolation](https://arxiv.org/abs/2205.09921) accepted at NeurIPS 2022. This repository is adapted from the awesome [gpt-neox](https://github.com/EleutherAI/gpt-neox) library.

## Important Changes and Information
1. This repository was developed based on commit 450b58c4ad7f36c319ca0b2f089c7349f34d8c3b of gpt-neox. We bump it to commit 738b87e73775e2cef4ea0a898b655f5d717cb8a0 to include some (irrelevant to this project) bug fixes. We only keep the main branch.
2. We remove the .github/ folder as it is not needed in our experiments.
3. The original gpt-neox readme is renamed as README_gpt_neox.md.
4. The config files used in our experiments are stored in kerple_configs/.
5. The two proposed positional embeddings are called **ParallelKerplePower** and **ParallelKerpleLog** in this repository. A simple grep will point you to our implementation.

## Installation
Please refer to the original readme README_gpt_neox.md for details. We use the Host Setup without fused kernels.

## Data Preparation
Warning: These datasets are huge! Please make sure you have at least **250 GB** of disk space before download them all.

We use the three preconfigured datasets in the orignal gpt-neox repository:
```
python prepare_data.py -d ./data openwebtext2
python prepare_data.py -d ./data arxiv
python prepare_data.py -d ./data github
```
Please refer to the original readme README_gpt_neox.md for details.

## Config Preparation
```
python generate_ymls.py
```

## Training
```
bash train.sh
```

## Testing
```
bash test.sh
```

## Pretrained Models
We release 6 pretrained checkpoints: **kerple_log** and **kerple_power** pretrained on the above three datasets.

1. Please navigate to [**Releases**](https://github.com/chijames/KERPLE/releases) to download the checkpoints.
2. You can right click on the filename, copy link address, and use wget to download it directly in a command line environment.
3. Once the files are downloaded, unzip them and leave them in the current directory.
4. Run *test.sh*, and the extrapolation performance should be very close to the numbers reported in Table 3 of the paper.
