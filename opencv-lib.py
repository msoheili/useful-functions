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

def cv2pil(img: np.ndarray) -> Image.Image:
    """Convert an OpenCV (NumPy) array to a PIL Image."""
    if img.ndim == 3:  # Color image
        if img.shape[2] == 3:  # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 4:  # BGRA to RGBA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    
    return Image.fromarray(img)



def crop_contour(img: np.ndarray, contour: np.ndarray, bg_color=(127, 127, 127)) -> np.ndarray:
    """Crop an image using a given contour and set a custom background color.

    Args:
        img (np.ndarray): Input image (grayscale or color).
        contour (np.ndarray): Contour points as a NumPy array.
        bg_color (tuple): Background color (B, G, R) for the cropped region.

    Returns:
        np.ndarray: Cropped image with the specified background color.
    """
    # Ensure contour is a NumPy array
    contour = np.asarray(contour, dtype=np.int32)

    # Create a blank mask with the same shape as the image (single channel)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Draw the filled contour on the mask
    cv2.drawContours(mask, [contour], 0, 255, thickness=-1)

    # Get bounding box coordinates of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Crop the mask and image to the bounding box
    mask_cropped = mask[y:y+h, x:x+w]
    img_cropped = img[y:y+h, x:x+w]

    # Create a new image filled with the background color
    if len(img.shape) == 3:  # Color image
        bg_image = np.full((h, w, 3), bg_color, dtype=np.uint8)
    else:  # Grayscale image
        bg_image = np.full((h, w), bg_color[0], dtype=np.uint8)

    # Combine the cropped image and the background
    result = np.where(mask_cropped[:, :, None] == 255, img_cropped, bg_image)

    return result


def get_vector_angle(p1, p2):
    """Calculate the angle of a vector in image coordinates.

    Args:
        p1 (tuple): (x1, y1) - Start point coordinates.
        p2 (tuple): (x2, y2) - End point coordinates.

    Returns:
        float: Angle in degrees (relative to the positive X-axis).
    """
    x1, y1 = p1
    x2, y2 = p2

    # Compute the horizontal and vertical changes
    dx = x2 - x1  # X increases to the right
    dy = y1 - y2  # Y decreases going up (correcting for image coordinates)

    # Compute angle in radians and convert to degrees
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)

    return angle_deg