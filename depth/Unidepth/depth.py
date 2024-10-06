from unidepth.models import UniDepthV2

model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14") # or "lpiccinelli/unidepth-v1-cnvnxtl" for the ConvNext backbone

import numpy as np
import torch
from PIL import Image

image_path = "assets/demo/rgb.png"

# Move to CUDA, if any
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the RGB image and the normalization will be taken care of by the model
rgb = torch.from_numpy(np.array(Image.open(image_path))).permute(2, 0, 1) # C, H, W

predictions = model.infer(rgb)

# Metric Depth Estimation
depth = predictions["depth"]

# Point Cloud in Camera Coordinate
xyz = predictions["points"]

# Intrinsics Prediction
intrinsics = predictions["intrinsics"]

print(depth.shape)
