import torch

from .MemFlow.configs.sintel_memflownet import get_cfg
from .MemFlow.core.Networks import build_network

# TODO: change to dynamic path if needed
CKPT = "/home/bobberman/programming/radar/radar-viz/simulation/modules/MemFlow/ckpts/MemFlowNet_sintel.pth"

class OpticalFlow:
    def __init__(self):
        self.cfg = get_cfg()
        self.model = build_network(self.cfg).cuda()
        print(f"Parameter count: {self.count_parameters()}")
        print(f"Loading ckpt from: {CKPT}")
        ckpt = torch.load(CKPT, map_location='cpu')
        ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt
        # some fix regarding mismatch between multi-gpu trained model and single gpu model
        if 'module' in list(ckpt_model.keys())[0]:
            for key in ckpt_model.keys():
                ckpt_model[key.replace('module.', '', 1)] = ckpt_model.pop(key)
            self.model.load_state_dict(ckpt_model, strict=True)
        else:
            self.model.load_state_dict(ckpt_model, strict=True)
        self.model.eval()

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    @torch.no_grad()
    def inference(self):
        print("hello from inference")

