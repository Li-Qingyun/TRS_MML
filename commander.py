"""

"""

import os

# lr and lr_backbone
lrs_list = [('1e-5', '1e-5'), ('1e-5', '1e-4'), ('1e-5', '1e-3'), ('1e-5', '1e-6'), ('1e-4', '1e-5'),
            ('1e-4', '1e-4'), ('1e-4', '1e-3'), ('1e-6', '1e-7'), ('1e-6', '1e-6'), ('1e-6', '1e-5')]

for lrs in lrs_list:

    command = "python run.py config=TRS_MML/configs/single_task/indian_single.yaml " \
              "datasets=indian " \
              "model=trs " \
              "run_type=train " \
              "training.batch_size=100 " \
              "training.evaluation_interval=100 " \
              "training.max_updates=4000 " \
              "training.lr_steps=[3000] " \
              "env.save_dir=./save/TRS_MML/indian_single_ds=100_adjustlr={}_{}_lr_steps=[3000] " \
              "env.data_dir=C:/Users/LQY/Desktop " \
              "optimizer.params.lr={} " \
              "model_config.trs.base_args.lr_backbone={} " \
              "checkpoint.max_to_keep=1 ".format(
        lrs[0], lrs[1], lrs[0], lrs[1]
    )

    os.system(command)