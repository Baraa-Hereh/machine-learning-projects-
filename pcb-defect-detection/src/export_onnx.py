from pathlib import Path

import torch
from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module

from baseline import load_config

ROOT = Path(__file__).resolve().parent.parent


def main():
    cfg, _ = load_config(ROOT / "configs/config.yaml")
    final, v2 = cfg["final"], cfg["final"]["v2"]
    name = final["experiment"]

    exp = get_exp(str(ROOT / f"exps/{name}.py"), None)
    model = exp.get_model()
    ckpt = torch.load(ROOT / f"models/yolox/{name}_seed{v2['seed']}/best_ckpt.pth",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = replace_module(model, torch.nn.SiLU, SiLU)
    model.head.decode_in_inference = True
    model.eval()

    out = ROOT / f"models/onnx/{name}_seed{v2['seed']}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, *exp.test_size)
    torch.onnx.export(model, dummy, str(out), input_names=["images"], output_names=["output"],
                      opset_version=17, dynamo=False)
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()