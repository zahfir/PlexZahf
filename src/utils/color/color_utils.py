from typing import List
import matplotlib.pyplot as plt
import numpy as np
import cv2


def visualize_bgr_color(bgr_value):
    """
    Display a color patch for the given BGR value

    Args:
        bgr_value: A tuple or list of (Blue, Green, Red) values (0-255)
    """
    # Create a small image with the BGR color
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = bgr_value  # Fill with the BGR color

    # Convert BGR to RGB for matplotlib display
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the color
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_img)
    plt.title(f"BGR: {bgr_value}")
    plt.axis("off")
    plt.show()


def calculate_brightness(rgb_values):
    """
    Calculate brightness value from RGB values.
    Returns an integer from 0-100.

    Args:
        rgb_values: List or tuple of RGB values [R, G, B] (0-255)

    Returns:
        Brightness value from 0-100
    """
    # Extract R, G, B values
    r, g, b = rgb_values

    # Calculate V (value) from HSV color model
    # V is simply the maximum value among the RGB components
    v = max(r, g, b) / 255.0

    # Scale to 0-100 range and return as integer
    return int(v * 100)


def calculate_perceived_brightness(rgb_values):
    """Calculate perceived brightness using weighted RGB values"""
    r, g, b = rgb_values
    # Human eye perceives green as brighter than red, and red as brighter than blue
    return int((0.299 * r + 0.587 * g + 0.114 * b) / 2.55)  # Scale to 0-100


def bgr_to_rgb(bgr) -> List[int]:
    return [int(channel) for channel in reversed(bgr)]


def color_difference(rgb1, rgb2):
    """Compare colors in HSV space with special handling for hue"""
    # Convert RGB to HSV
    hsv1 = cv2.cvtColor(np.uint8([[rgb1]]), cv2.COLOR_RGB2HSV)[0][0]
    hsv2 = cv2.cvtColor(np.uint8([[rgb2]]), cv2.COLOR_RGB2HSV)[0][0]

    h1, s1, v1 = [float(x) for x in hsv1]
    h2, s2, v2 = [float(x) for x in hsv2]

    # Handle hue's circular nature (0 and 180 are far in value but could be similar)
    h_diff = min(abs(h1 - h2), 180 - abs(h1 - h2)) / 90.0
    s_diff = abs(s1 - s2) / 255.0
    v_diff = abs(v1 - v2) / 255.0

    # Weighted combination
    return h_diff * 0.6 + s_diff * 0.3 + v_diff * 0.1


def get_color_hsv_metrics(rgb_colors):
    """
    Convert RGB dominant colors to HSV color space and extract hue and saturation values.

    Args:
        dominant_colors: NumPy array of RGB colors with shape (n, 3)

    Returns:
        Hues and saturations arrays
    """
    # Initialize lists to store results
    hues = []
    saturations = []

    for color in rgb_colors:
        # Convert RGB to BGR (OpenCV format)
        bgr_color = np.uint8([[color[::-1]]])

        # Convert BGR to HSV
        hsv = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]

        # Extract components (H: 0-179 and S: 0-255 in OpenCV)
        hues.append(hsv[0])
        saturations.append(hsv[1])

    return np.array(hues), np.array(saturations)
