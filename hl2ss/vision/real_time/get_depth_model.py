import torch
from pathlib import Path

import openvino as ov

from real_time.depth_anything_v2.dpt import DepthAnythingV2

import numpy as np

import real_time.utils as utils

import requests
import os
import cv2

if __name__ == '__main__':

    ##########################
    # Load Pre-trained model #
    ##########################
    model_select = "vits"

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    encoder = 'vits' # or 'vits', 'vitb'
    dataset = 'vkitti' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 80 # 20 for indoor model, 80 for outdoor model

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})

    # Define the URL and destination folder (change url to load another model!!)
    url = 'https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth?download=true'
    destination_folder = 'checkpoints'
    filename = 'depth_anything_v2_metric_vkitti_vits.pth'

    # Create the checkpoints directory if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Full path to save the file
    weights_path = os.path.join(destination_folder, filename)

    if os.path.exists(weights_path):
        print(f"File already exists at {weights_path}, skipping download.")
    else:
        # Download the file
        print(f"Downloading file from {url} ...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the request was successful

        # Save the file to the destination
        with open(weights_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"Model weights downloaded and saved to {weights_path}")

    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))

    ########################
    # Get Sample RGB Image #
    ########################

    #image_url = "https://images.pexels.com/photos/5740792/pexels-photo-5740792.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
    #image = np.array(utils.download_image(image_url))
    image = cv2.imread("real_time\demo\pexels-photo.jpeg")
    ########################
    # Preprocess RGB Image #
    ########################
    input_tensor, image_size = utils.image_preprocess(image)

    #######################
    # Convert to OpenVINO #
    #######################

    ov_model_path = Path("models_ov") / Path(Path(weights_path).name.replace(".pth", ".xml"))
    if not ov_model_path.exists():
        ov_model = ov.convert_model(model, example_input=input_tensor, input=[1, 3, 518, 518])
        ov.save_model(ov_model, ov_model_path)
    
    