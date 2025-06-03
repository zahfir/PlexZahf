import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

from constants import COLORS_PER_FRAME, PIXEL_SAMPLE_SIZE
from utils.color.color_utils import get_color_hsv_metrics


RANDOM_STATE = 42


class FrameAnalysis:
    """
    Class for analyzing video frames with various metrics.
    Provides color analysis and dominant color extraction.
    """

    @classmethod
    def get_top_colors(cls, image, show_plot=False):
        """
        Extract the dominant colors from an image using MiniBatchKMeans clustering.

        Args:
            image (numpy.ndarray): Input image in BGR or RGB format
            num_colors (int): Number of dominant colors to extract
            show_plot (bool): Whether to display a visualization of the colors

        Returns:
            dict: Dictionary containing the dominant colors, their proportions,
                  hues, and saturations
        """
        # Step 1: Flatten image to 2D array
        pixels = image.reshape(-1, 3)

        # Step 2: Sample subset of pixels (optional, speeds up clustering)
        if len(pixels) > PIXEL_SAMPLE_SIZE:
            idx = np.random.choice(len(pixels), PIXEL_SAMPLE_SIZE, replace=False)
            pixels = pixels[idx]

        # Step 3: MiniBatchKMeans clustering
        kmeans = MiniBatchKMeans(
            n_clusters=COLORS_PER_FRAME,
            random_state=RANDOM_STATE,
            batch_size=PIXEL_SAMPLE_SIZE,
        )
        kmeans.fit(pixels)

        # Step 4: Get RGB cluster centers and proportions
        centers = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_

        _, counts = np.unique(labels, return_counts=True)
        proportions = counts / counts.sum()

        # Step 5: Sort colors by dominance
        sorted_indices = np.argsort(-proportions)
        dominant_colors = centers[sorted_indices]
        dominant_proportions = proportions[sorted_indices]
        hues, saturations = get_color_hsv_metrics(dominant_colors)

        # Step 6: Optional – Plot the color bar
        if show_plot:
            cls._plot_color_bar(dominant_colors, dominant_proportions)

        return {
            "colors": dominant_colors,
            "proportions": dominant_proportions,
            "hues": hues,
            "saturations": saturations,
        }

    @staticmethod
    def _plot_color_bar(colors, proportions, title="Top Dominant Colors (RGB)"):
        """
        Plot a visualization of the dominant colors.

        Args:
            colors (numpy.ndarray): Array of dominant colors
            proportions (numpy.ndarray): Proportions for each color
            title (str): Title for the plot
        """
        bar = np.zeros((50, 300, 3), dtype="uint8")
        start_x = 0
        for percent, color in zip(proportions, colors):
            end_x = start_x + int(percent * 300)
            cv2.rectangle(bar, (start_x, 0), (end_x, 50), color.tolist(), -1)
            start_x = end_x
        plt.figure()
        plt.axis("off")
        plt.imshow(bar)
        plt.title(title)
        plt.show()
