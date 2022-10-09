# stylegan2-ada-PTI-ganspace

## Requirements
### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA(11.0 or later) CuDNN 
- Python 3.7
- PyTorch 1.7.1

### Installation
- Dependencies:  
    1. click 
    2. requests 
    3. tqdm 
    4. pyspng 
    5. ninja 
    6. imageio-ffmpeg==0.4.3
	7. lpips
	8. wandb
	9. pytorch
	10. torchvision
	11. matplotlib
	12. dlib
    13. pillow=6.2
    14. ffmpeg
    15. tqdm
    16. scipy
    17. scikit-learn
    18. scikit-image
    19. boto3
    20. requests
    21. nltk
    22. fbpca
    23. pyopengltk
    24. pycuda (enable OpenGL, Direct build recommended)
- All dependencies can be installed using *pip install* and the package name

## Getting Started
### Training new networks

In its most basic form, training new networks boils down to:

```.bash
python train.py --outdir=~/training-runs --data=~/mydataset.zip --gpus=1 --dry-run
python train.py --outdir=~/training-runs --data=~/mydataset.zip --gpus=1
```

The first command is optional; it validates the arguments, prints out the training configuration, and exits. The second command kicks off the actual training.

In this example, the results are saved to a newly created directory `~/training-runs/<ID>-mydataset-auto1`, controlled by `--outdir`. The training exports network pickles (`network-snapshot-<INT>.pkl`) and example images (`fakes<INT>.png`) at regular intervals (controlled by `--snap`). For each pickle, it also evaluates FID (controlled by `--metrics`) and logs the resulting scores in `metric-fid50k_full.jsonl` (as well as TFEvents if TensorBoard is installed).

The name of the output directory reflects the training configuration. For example, `00000-mydataset-auto1` indicates that the *base configuration* was `auto1`, meaning that the hyperparameters were selected automatically for training on one GPU. The base configuration is controlled by `--cfg`:

| Base config           | Description
| :-------------------- | :----------
| `auto`&nbsp;(default) | Automatically select reasonable defaults based on resolution and GPU count. Serves as a good starting point for new datasets but does not necessarily lead to optimal results.
| `stylegan2`           | Reproduce results for StyleGAN2 config F at 1024x1024 using 1, 2, 4, or 8 GPUs.
| `paper256`            | Reproduce results for FFHQ and LSUN Cat at 256x256 using 1, 2, 4, or 8 GPUs.
| `paper512`            | Reproduce results for BreCaHAD and AFHQ at 512x512 using 1, 2, 4, or 8 GPUs.
| `paper1024`           | Reproduce results for MetFaces at 1024x1024 using 1, 2, 4, or 8 GPUs.
| `cifar`               | Reproduce results for CIFAR-10 (tuned configuration) using 1 or 2 GPUs.

The training configuration can be further customized with additional command line options:

* `--aug=noaug` disables ADA.
* `--cond=1` enables class-conditional training (requires a dataset with labels).
* `--mirror=1` amplifies the dataset with x-flips. Often beneficial, even with ADA.
* `--resume=ffhq1024 --snap=10` performs transfer learning from FFHQ trained at 1024x1024.
* `--resume=~/training-runs/<NAME>/network-snapshot-<INT>.pkl` resumes a previous training run.
* `--gamma=10` overrides R1 gamma. We recommend trying a couple of different values for each new dataset.
* `--aug=ada --target=0.7` adjusts ADA target value (default: 0.6).
* `--augpipe=blit` enables pixel blitting but disables all other augmentations.
* `--augpipe=bgcfnc` enables all available augmentations (blit, geom, color, filter, noise, cutout).

Please refer to [`python train.py --help`](./docs/train-help.txt) for the full list.

### Running PTI
The main training script is `scripts/run_pti.py`. The script receives aligned and cropped images from paths configured in the "Input info" subscetion in
 `configs/paths_config.py`. 
Results are saved to directories found at "Dirs for output files" under `configs/paths_config.py`. This includes inversion latent codes and tuned generators. 
The hyperparametrs for the inversion task can be found at  `configs/hyperparameters.py`. They are intilized to the default values used in the paper. 