import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcb_yolox_s import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "pcb_yolox_s_e50"
        self.max_epoch = 50