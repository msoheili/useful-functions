import numpy as np
import cv2
from PIL import Image

def pil2cv(img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to an OpenCV (NumPy) array."""
    img = np.array(img)  # Convert to NumPy array
    
    # Convert RGB to BGR (OpenCV default format)
    if img.ndim == 3 and img.shape[2] == 3:  # RGB image
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:  # RGBA image
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    
    return img