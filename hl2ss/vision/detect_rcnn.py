import torch
from detecto.core import Model
from detecto import utils, visualize
import ssl

ssl._create_default_https_context = ssl._create_stdlib_context

model = Model(device=torch.device('cpu'), pretrained=True)
image = utils.read_image('park.jpg')  # Helper function to read in images
labels, boxes, scores = model.predict(image)  # Get all predictions on an image
visualize.show_labeled_image(image, boxes, labels)

# visualize.detect_live(model, score_filter=0.7)
