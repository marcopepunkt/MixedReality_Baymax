from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

# ZoeD_N
conf = get_config("zoedepth", "infer")
model_zoe_n = build_model(conf)

import torch

##### sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)


# Local file
from PIL import Image
image = Image.open("assets/rgb.png").convert("RGB")  # load

#depth_numpy = zoe.infer_pil(image)  # as numpy
#depth_tensor = zoe.infer(X)

depth = zoe.infer_pil(image)


# Save raw
#from zoedepth.utils.misc import save_raw_16bit
#fpath = "/assets/raw_output.png"
#save_raw_16bit(depth_numpy, fpath)

# Colorize output
from zoedepth.utils.misc import colorize

colored = colorize(depth)

# save colored output
fpath_colored = "assets/output_colored.png"
Image.fromarray(colored).save(fpath_colored)
