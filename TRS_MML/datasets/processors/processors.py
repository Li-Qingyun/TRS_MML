from mmf.common.constants import IMAGE_COLOR_MEAN, IMAGE_COLOR_STD
from mmf.common.registry import registry
from mmf.datasets.processors import BaseProcessor
from mmf.datasets.processors import detection_transforms as T

@registry.register_processor("optical_RSI")
class OpticalRSIProcessor(BaseProcessor):
    """Process a detection image and target in consistent with DETR. At training time,
    random crop is done. At test time, an image is deterministically resized with short
    side equal to `image_size` (while ensuring its long side no larger than `max_size`)
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        train_image_sizes = list(config.train_image_sizes)
        self.training_transform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(train_image_sizes, max_size=config.max_size),
                    T.Compose(
                        [
                            T.RandomResize(list(config.train_resize_random_sizes)),
                            T.RandomSizeCrop(*config.train_crop_size),
                            T.RandomResize(train_image_sizes, max_size=config.max_size),
                        ]
                    ),
                ),
                T.ToTensor(),
                T.Normalize(IMAGE_COLOR_MEAN, IMAGE_COLOR_STD),
            ]
        )
        self.inference_transform = T.Compose(
            [
                T.RandomResize([config.test_image_size], max_size=config.max_size),
                T.ToTensor(),
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
