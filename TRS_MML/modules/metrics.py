from typing import Dict

import torch
from mmf.common.registry import registry
from mmf.modules.metrics import BaseMetric


@registry.register_metric("mAP@0.5")
class RSDetectionMeanAP(BaseMetric):
    """Metric for calculating the detection mean average precision (mAP) using the COCO
    evaluation toolkit, returning the default COCO-style mAP@IoU=0.50:0.95

    **Key:** ``detection_mean_ap``
    """

    def __init__(self, dataset_json_files, *args, **kwargs):
        """Initialization function detection mean AP (mAP)

        Args:
            dataset_json_files (Dict): paths to the dataset (instance) json files
                for each dataset type and dataset name in the following format:
                ``{'val/detection_coco': '/path/to/instances_val2017.json', ...}``

        """
        super().__init__("mAP@0.5")
        self.required_params = ["__prediction_report__"]
        self.dataset_json_files = dataset_json_files

    def calculate(
        self, sample_list, model_output, execute_on_master_only=True, *args, **kwargs
    ):
        """Calculate detection mean AP (mAP) from the prediction list and the dataset
        annotations. The function returns COCO-style mAP@IoU=0.50:0.95.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration.
            model_output (Dict): Dict returned by model. This should contain
                                "prediction_report" field, which is a list of
                                detection predictions from the model.
            execute_on_master_only (bool): Whether to only run mAP evaluation on the
                                master node over the gathered detection prediction
                                (to avoid wasting computation and CPU OOM).
                                Default: True (only run mAP evaluation on master).

        Returns:
            torch.FloatTensor: COCO-style mAP@IoU=0.50:0.95.

        """

        # as the detection mAP metric is run on the entire dataset-level predictions,
        # which are *already* gathered from all notes, the evaluation should only happen
        # in one node and broadcasted to other nodes (to avoid CPU OOM due to concurrent
        # mAP evaluation)
        from mmf.utils.distributed import broadcast_tensor, is_master
        from mmf.utils.general import get_current_device
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        device = get_current_device()
        if execute_on_master_only and not is_master():
            # dummy mAP to be override in boardcasting
            mAP = torch.tensor(-1, dtype=torch.float, device=device)
        else:
            predictions = model_output.prediction_report

            cocoGt = COCO(
                self.dataset_json_files[sample_list.dataset_name][
                    sample_list.dataset_type
                ]
            )
            cocoDt = cocoGt.loadRes(predictions)
            cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            mAP = torch.tensor(cocoEval.stats[1], dtype=torch.float, device=device)

        if execute_on_master_only:
            mAP = broadcast_tensor(mAP, src=0)
        return mAP
