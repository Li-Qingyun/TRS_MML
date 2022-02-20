from mmf.common.constants import IMAGE_COLOR_MEAN, IMAGE_COLOR_STD
from mmf.common.registry import registry
from mmf.datasets.processors import BaseProcessor
import torchvision.transforms as T

@registry.register_processor("optical_RSI")
class OpticalRSIProcessor(BaseProcessor):
    """Process a detection image and target in consistent with DETR. At training time,
    random crop is done. At test time, an image is deterministically resized with short
    side equal to `image_size` (while ensuring its long side no larger than `max_size`)
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.training_transform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                # T.Resize((224, 224)),
                T.Normalize(IMAGE_COLOR_MEAN, IMAGE_COLOR_STD),
            ]
        )
        self.inference_transform = T.Compose(
            [
                T.ToTensor(),
                # T.Resize((224, 224)),
                T.Normalize(IMAGE_COLOR_MEAN, IMAGE_COLOR_STD),
            ]
        )

    def __call__(self, item):
        dataset_type = item["dataset_type"]
        img = item["img"]
        target = item["target"]

        if dataset_type == "train":
            img = self.training_transform(img)
        elif dataset_type == "val" or dataset_type == "test":
            img = self.inference_transform(img)
        else:
            raise Exception(f"unknown dataset_type: {dataset_type}")

        return {"img": img, "target": target}
