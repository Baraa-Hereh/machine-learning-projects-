import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pcb_yolox_s_neg_e50 import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.val_ann = "test.json"