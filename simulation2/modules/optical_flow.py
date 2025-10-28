import torch
import numpy as np
from scipy import interpolate
import torch.nn.functional as F
import cv2
from typing import Optional
from numpy.typing import NDArray

from .types import FlowField
from .MemFlow.configs.sintel_memflownet import get_cfg
from .MemFlow.core.Networks import build_network
from .MemFlow.inference import inference_core_skflow as inference_core

CKPT = "/home/bobberman/programming/radar/radar-viz/simulation/modules/MemFlow/ckpts/MemFlowNet_sintel.pth"

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', multiply=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // multiply) + 1) * multiply - self.ht) % multiply
        pad_wd = (((self.wd // multiply) + 1) * multiply - self.wd) % multiply
        self.mode = mode
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2, 0, 0]
        elif mode == "downzero":
            self._pad = [0, pad_wd, 0, pad_ht, 0, 0]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht, 0, 0]

    def pad(self, input):
        if self.mode == "downzero":
            return F.pad(input, self._pad)
        else:
            return F.pad(input, self._pad, mode='replicate')

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    if len(x1) == 0:
        return torch.zeros(flow.shape)

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (y1, x1), dy, (y0, x0), method='nearest', fill_value=0) # Fix: use (y1, x1) and (y0, x0) for y

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()

class OpticalFlow:
    """
    Manages frame history and calculates dense optical flow using OpenCV.
    """
    def __init__(self):
        """ Initializes the OpticalFlow calculator. """
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

        self.processor = inference_core.InferenceCore(self.model, config=self.cfg)
        self.flow_prev = None
        self.padder = None

        self.prev_gray: Optional[NDArray[np.uint8]] = None
        self.prev_memflow: Optional[np.ndarray] = None

        # --- Farneback Parameters (can be tuned) ---
        # These are common defaults, adjust as needed for your scene/speed
        self.fb_params = dict(
            pyr_scale=0.5,  # Pyramid scale (< 1)
            levels=3,       # Number of pyramid levels
            winsize=15,     # Averaging window size
            iterations=3,   # Iterations per pyramid level
            poly_n=5,       # Size of pixel neighborhood for polynomial expansion
            poly_sigma=1.2, # Std dev for Gaussian used for polynomial expansion
            flags=0         # Operation flags (0 for defaults)
        )
        print("OpticalFlow calculator initialized.")

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _preprocess_to_cpu_tensor(self, img: np.ndarray) -> torch.Tensor:
        """
        Converts a single (H, W, 3) NumPy image into a
        (3, H, W) PyTorch CPU tensor.
        """
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
            
        if len(img.shape) == 2:
            img = np.tile(img[..., None], (1, 1, 3))
        else:
            img = img[..., :3]
            
        # Convert to tensor, permute to (C, H, W), and convert to float
        return torch.from_numpy(img.copy()).permute(2, 0, 1).float()

    def calculate_flow(self, current_frame_bgr: NDArray[np.uint8]) -> Optional[FlowField]:
        """
        Calculates dense optical flow between the previous frame and the current frame.

        Parameters:
        ----------
        current_frame_bgr : NDArray[np.uint8]
            The current video frame (H x W x 3) in BGR format (common for OpenCV).
            Note: glReadPixels gives RGB, so conversion might be needed before passing.

        Returns:
        -------
        Optional[FlowField]
            The calculated flow field (H x W x 2) containing (dx, dy) flow vectors
            for each pixel, or None if this is the first frame processed.
        """
        # 1. Convert current frame to grayscale
        #    OpenCV typically works best with grayscale for flow
        current_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)

        flow: Optional[FlowField] = None

        # 2. Check if we have a previous frame
        if self.prev_gray is not None:
            # 3. Calculate Dense Optical Flow (Farneback)
            flow = cv2.calcOpticalFlowFarneback(
                prev=self.prev_gray,
                next=current_gray,
                flow=None, # Output flow image; computed by function
                **self.fb_params
            )
            # flow will have shape (height, width, 2)

        # 4. Update the previous frame store *after* calculation
        self.prev_gray = current_gray

        # 5. Return the calculated flow (or None if it was the first frame)
        return flow
    
    @torch.no_grad()
    def inference(self, img: np.ndarray) -> Optional[FlowField]:
        """
        Calculates optical flow from two consecutive (H, W, 3) np.ndarray images.
        This function is STATEFUL and updates self.flow_prev.
        """
        # 1. Initialize Padder on first run
        if self.padder is None:
            H, W, _ = img.shape
            # InputPadder expects (B, C, H, W) shape for dims
            self.padder = InputPadder((1, 3, H, W))

        if self.prev_memflow is None:
            self.prev_memflow = img.copy()
            return None

        # 2. Preprocess images to CPU tensors
        t1_cpu = self._preprocess_to_cpu_tensor(self.prev_memflow)
        t2_cpu = self._preprocess_to_cpu_tensor(img)
        
        # 3. Stack them (as the example code does)
        images_pair_cpu = torch.stack([t1_cpu, t2_cpu])
        
        # 4. Move to GPU and add Batch dimension (B=1, T=2, C, H, W)
        images_pair = images_pair_cpu.cuda().unsqueeze(0) 

        # 5. Apply padding and normalization
        images_pair = self.padder.pad(images_pair)
        images_pair = 2 * (images_pair / 255.0) - 1.0 # Normalize

        # 6. Run inference
        # We set end=True because we're only processing one pair
        flow_low, flow_pre = self.processor.step(
            images_pair, 
            end=True, 
            add_pe=('rope' in self.cfg and self.cfg.rope), 
            flow_init=self.flow_prev
        )
                                            
        # 7. Unpad the result
        flow_out = self.padder.unpad(flow_pre[0]).cpu()
        
        # 8. Update the state for the *next* call
        # if 'warm_start' in self.cfg and self.cfg.warm_start:
        self.flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
        # 9. Post-process to (H, W, 2) NumPy array and return
        flow_map_numpy = flow_out.numpy().transpose(1, 2, 0)

        self.prev_memflow = img.copy()

        return flow_map_numpy