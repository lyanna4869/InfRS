import numpy as np
import torch 
import cv2

def vis_single_tensor(im: "torch.Tensor"):
    im_np: "np.ndarray" = im.detach().cpu().numpy()
    im_np = im_np.astype(np.uint8)
    im_np = im_np.transpose(1, 2, 0)
    cv2.imwrite(im_np, "test.png")
