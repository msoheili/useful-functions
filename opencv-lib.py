import numpy as np
import cv2
from PIL import Image
from PIL import ImageDraw, ImageFont

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




def major_axis_angle(contour):
    """
    Calculates the major axis angle of an object given its contour.
    
    Parameters:
    contour (array-like): A list or NumPy array representing the contour of an object.

    Returns:
    float: The angle (in degrees) of the major axis relative to the x-axis.
    """
    # Compute image moments from the given contour
    M = cv2.moments(np.array(contour))

    # Calculate the orientation (major axis angle) using central moments
    angle = 0.5 * np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"]) * 180 / np.pi        

    return angle




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

def order_points(pts):
    """
    Orders four points in the following order:
    [top-left, top-right, bottom-right, bottom-left]
    
    :param pts: List of four (x, y) tuples.
    :return: Ordered numpy array of shape (4,2).
    """
    pts = np.array(pts, dtype=np.float32)

    # Sort points based on sum (top-left: smallest sum, bottom-right: largest sum)
    sum_pts = pts.sum(axis=1)
    top_left = pts[np.argmin(sum_pts)]
    bottom_right = pts[np.argmax(sum_pts)]

    # Sort points based on difference (top-right: smallest difference, bottom-left: largest difference)
    diff_pts = np.diff(pts, axis=1)
    top_right = pts[np.argmin(diff_pts)]
    bottom_left = pts[np.argmax(diff_pts)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)



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



def get_text_size(text, font):
    """
    Calculate the width and height of a given text when rendered with a specific font.

    Args:
        text (str): The text to measure.
        font (ImageFont.FreeTypeFont): The font used to render the text.

    Returns:
        tuple: (width, height) of the rendered text in pixels.
    """
    # Create a temporary image (1x1) to use for text measurement
    im = Image.new(mode="L", size=(1, 1))  
    draw = ImageDraw.Draw(im)

    # Get bounding box of the text and extract width and height
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)

    return width, height
