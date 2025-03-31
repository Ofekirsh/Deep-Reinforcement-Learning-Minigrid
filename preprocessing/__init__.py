# preprocessing/__init__.py

from .custom_preprocess import process_image as custom_process_image
from .deepmind_preprocess import process_image as deepmind_process_image

__all__ = [
    "custom_process_image",
    "deepmind_process_image",
]
