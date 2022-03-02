# Installation

Environment:
* Windows 10
* Python 3.7+
* PyTorch 1.6+
* mmf

### Create an environment 
```commandline
conda create -n trs python=3.7
```

### Activate the environment
```commandline
conda activate trs
```

### Install MMF from source
Run the command line at your own user folder. (e.g. C:/Users/YOUR USERNAME/)
 ```commandline
git clone https://github.com/facebookresearch/mmf.git
cd mmf
pip install --editable .
```
The installation of mmf may meet difficulties. For more installation details, please refer to [the official doc](https://mmf.sh/docs/).

MMF installs with related dependencies such as PyTorch, so no additional installation is required.

### Get TRS-MML

Run the command line at your own working folder.
 ```commandline
git clone https://github.com/Li-Qingyun/TRS_MML.git
```

# Preparing data

to be written


# Usage

Run the command line at TRS_MML folder.

### Training using a single GPU

 ```commandline
# base command:
python run.py \
    config=TRS_MML/configs/all_hsi_cls.yaml \
    datasets=indian,ksc,pavid \
    model=trs run_type=train training.batch_size=1 
    env.save_dir=./save/TRS_MML/all_hsi_cls
    env.data_dir=./

# resume training from the last checkpoint: add
    checkpoint.resume=True

# resume training from the best checkpoint: add
    checkpoint.resume=True
    checkpoint.resume_best=True

# resume training with a different lr: add
    checkpoint.resume=True
    checkpoint.reset.optimizer=True

# finetune from a checkpoint: add
    checkpoint.resume_file=/your/checkpoint/site  # Notice: Don't set checkpoint.resume=True
    checkpoint.reset.counts=True  # set iteration counter to 0

    ......
```

For the detailed usage, please refer to [the official website of the mmf](https://mmf.sh/), and [the homepage of UniT project](https://mmf.sh/docs/projects/unit)

For Chinese reader, blogs about mmf will be presented at [CSDN](qingyun.blog.csdn.net) and [my own blog](liqingyun.zone).