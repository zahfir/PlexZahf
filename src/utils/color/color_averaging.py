from typing import List
import numpy as np
import cv2


class ColorAveraging:
    @staticmethod
    def mean(colors: List[List[int]]) -> List[int]:
        """Returns a single RGB average of multiple RGBs"""
        if not colors:
            return [0, 0, 0]

        return np.mean(colors, axis=0).astype(int).tolist()

    @staticmethod
    def median(colors: List[List[int]]) -> List[int]:
        """Returns median color"""
        return np.median(colors, axis=0).astype(int).tolist()

    @staticmethod
    def get_perceptual_average(colors: List[List[int]]) -> List[int]:
        """Average colors in LAB space for better perceptual results"""
        # Convert RGB to LAB
        lab_colors = [
            cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2LAB)[0][0]
            for color in colors
        ]

        # Average in LAB space
        avg_lab = np.mean(lab_colors, axis=0).astype(np.uint8)

        # Convert back to RGB
        avg_rgb = cv2.cvtColor(np.uint8([[avg_lab]]), cv2.COLOR_LAB2RGB)[0][0]
        return avg_rgb.tolist()
