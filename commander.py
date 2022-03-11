"""

"""

import os

# adjust lr (bs=100) smaller warmup: (lr, lr_backbone, warmup_steps)
settings_list = []
for i in ['', '', '']:  # lr for transformer
    for j in ['', '', '']:  # lr for cnn backbone
        warmup_steps = 20  # 2 steps equal to 1 epoch
        settings_list.append((i, j, warmup_steps))

for exp_inx, settings in enumerate(settings_list):

    command = "python run.py config=TRS_MML/configs/single_task/indian_single.yaml " \
              "datasets=indian " \
              "model=trs " \
              "run_type=train " \
              "training.batch_size=100 " \
              "training.evaluation_interval=100 " \
              "training.max_updates=2000 " \
              "optimizer.params.lr=1e-4 " \
              "model_config.trs.base_args.lr_backbone=1e-5 " \
              "env.save_dir=./save/TRS_MML/indian_single_ds=100_(frequentVAL)_adjust_lr/" \
              f"indian_single_lr={settings[0]}_{settings[1]} " \
              "env.data_dir=C:/Users/LQY/Desktop " \
              "checkpoint.max_to_keep=1 " \
              f"optimizer.params.lr={settings[0]} " \
              f"model_config.trs.base_args.lr_backbone={settings[1]} " \
              f"scheduler.params.num_warmup_steps={settings[2]} " \
              f"training.log_interval=2 " \    
              f"training.evaluation_interval=2 "    # evaluation frequency (2 steps equal to 1 epoch)

    for i in range(3):

        print(
            "\n\n\n"
            "################################################################################\n"
            "################################################################################\n\n"
            f"    EXPERIMENTS INDEX: {exp_inx+1}\n"
            f"    LR = {settings[0]}    LR_backbone = {settings[1]}    WARM_UP_STEPS={settings[2]}\n"
            f"    RUNNING FOR THE {i+1}th TIMES \n\n"
            "################################################################################\n"
            "################################################################################\n"
            "\n\n\n"
        )
        os.system(command)


# # adjust num_queries
# num_queries_list = [1, 5, 10, 20, 50, 80]
#
# for num_queries in num_queries_list:
#
#     command = "python run.py config=TRS_MML/configs/single_task/indian_single.yaml " \
#               "datasets=indian " \
#               "model=trs " \
#               "run_type=train " \
#               "training.batch_size=100 " \
#               "training.evaluation_interval=100 " \
#               "training.max_updates=2000 " \
#               "optimizer.params.lr=1e-4 " \
#               "model_config.trs.base_args.lr_backbone=1e-5 " \
#               f"env.save_dir=./save/TRS_MML/indian_single_ds=100_adjust_queries={num_queries} " \
#               "env.data_dir=C:/Users/LQY/Desktop " \
#               "checkpoint.max_to_keep=1 " \
#               f"model_config.trs.base_args.num_queries.hsi_cls.indian={num_queries}"
#
#     for i in range(3):
#
#         print(
#             "\n\n\n"
#             "################################################################################\n"
#             "################################################################################\n"
#             f"    NUM_QUERIES = {num_queries}  RUNNING FOR THE {i+1}th TIMES \n"
#             "################################################################################\n"
#             "################################################################################\n"
#             "\n\n\n"
#         )
#         os.system(command)