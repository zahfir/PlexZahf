import cv2
import numpy as np
from typing import List, Tuple


class ColorAnalysis:
    """
    Provides methods to analyze colors in video frames,
    particularly for extracting dominant colors.
    """

    @staticmethod
    def extract_dominant_color_kmeans(frame, k=5):
        """
        Extract dominant color using K-means clustering.

        Pros:
        - Very accurate in identifying true dominant colors
        - Can find multiple dominant colors by examining different clusters
        - Works well with complex images

        Cons:
        - Computationally expensive
        - Results can vary based on random initialization

        Args:
            frame: OpenCV image frame (numpy array)
            k: Number of clusters/colors to extract

        Returns:
            Dominant color as BGR array [B, G, R]
        """
        from sklearn.cluster import KMeans

        # Reshape the image data
        pixels = frame.reshape(-1, 3).astype(np.float32)

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(pixels)

        # Get the colors and counts for each cluster
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        counts = np.bincount(labels)

        # Return the color of the largest cluster
        dominant_color = colors[np.argmax(counts)]
        return dominant_color.astype(np.uint8)

    @staticmethod
    def extract_dominant_color_histogram(frame, bins=8):
        """
        Extract dominant color using color histograms.

        Pros:
        - Fast computation
        - Simple to implement
        - Works well for images with clear color dominance

        Cons:
        - Less accurate for complex images
        - Bin size affects results significantly

        Args:
            frame: OpenCV image frame (numpy array)
            bins: Number of histogram bins for each channel

        Returns:
            Dominant color as BGR array [B, G, R]
        """
        # Process each channel separately (BGR)
        histograms = []

        for channel in range(3):
            hist = np.histogram(frame[:, :, channel], bins=bins, range=(0, 256))[0]
            histograms.append(hist)

        # Find dominant bin for each channel
        dominant_vals = []
        for hist in histograms:
            bin_idx = np.argmax(hist)
            # Calculate middle value of the bin
            val = (bin_idx + 0.5) * (256 / bins)
            dominant_vals.append(int(val))

        return np.array(dominant_vals, dtype=np.uint8)

    @staticmethod
    def extract_dominant_color_average(frame):
        """
        Extract average color of the frame.

        Pros:
        - Extremely fast
        - Works well for solid-colored frames

        Cons:
        - Produces "muddy" colors for varied frames
        - Not truly a "dominant" color

        Args:
            frame: OpenCV image frame (numpy array)

        Returns:
            Average color as BGR array [B, G, R]
        """
        # Simply calculate the mean across all pixels
        avg_color = np.mean(frame, axis=(0, 1))
        return avg_color.astype(np.uint8)

    @staticmethod
    def extract_dominant_color_median_cut(frame, depth=3):
        """
        Extract dominant color using median cut quantization.

        Pros:
        - Good balance between speed and accuracy
        - Can identify multiple dominant colors
        - Works well on complex images

        Cons:
        - More complex implementation
        - Full implementation requires pixel assignment to quantized colors

        Args:
            frame: OpenCV image frame (numpy array)
            depth: Recursion depth for the median cut algorithm

        Returns:
            Dominant color as BGR array [B, G, R]
        """

        # Helper function to find the channel with the largest range
        def find_widest_channel(pixels):
            ranges = np.max(pixels, axis=0) - np.min(pixels, axis=0)
            return np.argmax(ranges)

        # Helper function to perform the median cut
        def median_cut(pixels, depth):
            if depth == 0:
                # Average the pixels in this box
                return [np.mean(pixels, axis=0).astype(np.uint8)]

            # Find the channel with the largest range
            channel = find_widest_channel(pixels)

            # Sort pixels by the selected channel
            pixels = pixels[pixels[:, channel].argsort()]

            # Split at median
            median = len(pixels) // 2

            # Recursively process both halves
            return median_cut(pixels[:median], depth - 1) + median_cut(
                pixels[median:], depth - 1
            )

        # Reshape and sample the image for faster computation
        # Sample every 10th pixel to speed up computation
        sampled = frame[::10, ::10].reshape(-1, 3)

        # Perform median cut
        colors = median_cut(sampled, depth)

        # For a full implementation, you would count pixels closest to each quantized color
        return colors[0]  # Return the first color as dominant

    @staticmethod
    def extract_dominant_color_hsv(frame, bins=16):
        """
        Extract dominant color using HSV color space analysis.

        Pros:
        - Perceptually more accurate
        - Works better for detecting meaningful colors
        - Better handles images with varying lighting

        Cons:
        - More complex
        - Conversion between color spaces adds computation

        Args:
            frame: OpenCV image frame (numpy array)
            bins: Number of histogram bins for the hue channel

        Returns:
            Dominant color as BGR array [B, G, R]
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create histogram for Hue channel only
        hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])

        # Find the dominant hue
        dominant_hue_bin = np.argmax(hist)
        dominant_hue = int((dominant_hue_bin + 0.5) * (180 / bins))

        # Create a mask to isolate pixels with the dominant hue
        hue_mask = cv2.inRange(
            hsv,
            np.array([dominant_hue - 180 // bins, 0, 0]),
            np.array([dominant_hue + 180 // bins, 255, 255]),
        )

        # Use the mask to find average saturation and value
        mask = hue_mask > 0
        if np.any(mask):
            s_vals = hsv[:, :, 1][mask]
            v_vals = hsv[:, :, 2][mask]

            avg_s = np.mean(s_vals)
            avg_v = np.mean(v_vals)

            # Convert back to BGR
            dominant_hsv = np.array([[[dominant_hue, avg_s, avg_v]]], dtype=np.uint8)
            dominant_bgr = cv2.cvtColor(dominant_hsv, cv2.COLOR_HSV2BGR)[0][0]
            return dominant_bgr

        # Fallback to average color if mask is empty
        return ColorAnalysis.extract_dominant_color_average(frame)

    @staticmethod
    def extract_color_multiple_frames(frames, extract_method, bins=16):
        """
        Returns the average dominant color across multiple frames.

        Args:
            frames: List of OpenCV image frames (numpy arrays)
            bins: Number of histogram bins for the hue channel

        Returns:
            Average dominant color as BGR array [B, G, R]
        """
        if not frames:
            return np.array([0, 0, 0], dtype=np.uint8)

        # Extract dominant color from each frame
        dominant_colors = []
        for frame in frames:
            color = extract_method(frame, bins)
            dominant_colors.append(color)

        # Convert to numpy array for easier averaging
        colors_array = np.array(dominant_colors)

        # Calculate average color
        avg_color = np.mean(colors_array, axis=0).astype(np.uint8)

        return avg_color
