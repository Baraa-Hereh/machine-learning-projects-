import os
from pathlib import Path

from yolox.data import COCODataset, TrainTransform, ValTransform
from yolox.exp import Exp as BaseExp

ROOT = Path(__file__).resolve().parent.parent
COCO_DIR = ROOT / "data/processed/coco"
RAW_DIR = ROOT / "data/raw/DeepPCB/PCBData"
IMAGE_DIR = os.path.relpath(RAW_DIR, COCO_DIR)


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = "pcb_yolox_s"
        self.output_dir = str(ROOT / "models/yolox")
        self.seed = 0
        self.data_num_workers = 2

        self.num_classes = 6
        self.data_dir = str(COCO_DIR)
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.input_size = (640, 640)
        self.test_size = (640, 640)

        self.hsv_prob = 0.0
        self.enable_mixup = False
        self.mixup_prob = 0.0
        self.degrees = 0.0
        self.shear = 0.0
        self.mosaic_scale = (0.5, 1.5)

        self.max_epoch = 100
        self.warmup_epochs = 5
        self.no_aug_epochs = 15
        self.eval_interval = 5
        self.save_history_ckpt = False

    def get_dataset(self, cache=False, cache_type="ram"):
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name=IMAGE_DIR,
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name=IMAGE_DIR,
            img_size=self.test_size,
            preproc=ValTransform(legacy=kwargs.get("legacy", False)),
        )