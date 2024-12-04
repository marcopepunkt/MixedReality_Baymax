import tarfile
from pathlib import Path

import openvino as ov
from openvino.tools.mo.front import tf as ov_tf_front
from openvino.tools import mo
import model_utils.utils as utils
import os

# A directory where the model will be downloaded.
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
base_model_dir = Path(os.path.join(current_dir, "model"))

# The name of the model from Open Model Zoo
model_name = "ssdlite_mobilenet_v2"

archive_name = Path(f"{model_name}_coco_2018_05_09.tar.gz")
model_url = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/{model_name}/{archive_name}"

# Download the archive
downloaded_model_path = base_model_dir / archive_name
if not downloaded_model_path.exists():
    utils.download_file(model_url, downloaded_model_path.name, downloaded_model_path.parent)

# Unpack the model
tf_model_path = base_model_dir / archive_name.with_suffix("").stem / "frozen_inference_graph.pb"
if not tf_model_path.exists():
    with tarfile.open(downloaded_model_path) as file:
        file.extractall(base_model_dir)

precision = "FP16"
# The output path for the conversion.
converted_model_path = base_model_dir / f"{model_name}_{precision.lower()}.xml"

# Convert it to IR if not previously converted
trans_config_path = Path(ov_tf_front.__file__).parent / "ssd_v2_support.json"
if not converted_model_path.exists():
    ov_model = mo.convert_model(
        tf_model_path,
        compress_to_fp16=(precision == "FP16"),
        transformations_config=trans_config_path,
        tensorflow_object_detection_api_pipeline_config=tf_model_path.parent / "pipeline.config",
        reverse_input_channels=True,
    )
    ov.save_model(ov_model, converted_model_path)
    del ov_model
