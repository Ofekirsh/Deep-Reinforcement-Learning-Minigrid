import cv2
import numpy as np
import torch


def process_image(image, new_size=(84, 84)):
    # Convert to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Resize image.
    resized = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
    # Normalize pixel values.
    normalized = resized / 255.0
    # Convert to torch tensor and add a channel dimension.
    tensor_image = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return tensor_image
