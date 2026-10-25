"""
Utility functions for pork packaging defect severity assessment.

This module contains reusable functions developed for converting
YOLO bounding boxes, estimating defect severity and matching
predicted bounding boxes with ground-truth annotations.
"""

import numpy as np


def classify_severity(relative_area):
    """
    Classify defect severity using normalised bounding-box area.

    Parameters
    ----------
    relative_area : float
        Bounding-box area relative to the complete image.

    Returns
    -------
    str
        Severity category: "Low", "Medium" or "High".
    """

    if relative_area < 0.003254:
        return "Low"

    elif relative_area <= 0.089605:
        return "Medium"

    else:
        return "High"


def yolo_to_xyxy(x_center, y_center, width, height):
    """
    Convert a YOLO bounding box into corner coordinates.

    Parameters
    ----------
    x_center : float
        Normalised horizontal centre coordinate.
    y_center : float
        Normalised vertical centre coordinate.
    width : float
        Normalised bounding-box width.
    height : float
        Normalised bounding-box height.

    Returns
    -------
    numpy.ndarray
        Bounding box in [x_min, y_min, x_max, y_max] format.
    """

    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    return np.array([
        x_min,
        y_min,
        x_max,
        y_max
    ])


def calculate_iou(box_a, box_b):
    """
    Calculate Intersection over Union between two bounding boxes.

    Parameters
    ----------
    box_a : array-like
        First box in [x_min, y_min, x_max, y_max] format.
    box_b : array-like
        Second box in [x_min, y_min, x_max, y_max] format.

    Returns
    -------
    float
        IoU value between 0 and 1.
    """

    # Find coordinates of the overlapping region
    x_min = max(box_a[0], box_b[0])
    y_min = max(box_a[1], box_b[1])
    x_max = min(box_a[2], box_b[2])
    y_max = min(box_a[3], box_b[3])

    # Calculate overlapping width and height
    intersection_width = max(0.0, x_max - x_min)
    intersection_height = max(0.0, y_max - y_min)

    intersection_area = (
        intersection_width * intersection_height
    )

    # Calculate individual bounding-box areas
    area_a = (
        max(0.0, box_a[2] - box_a[0])
        * max(0.0, box_a[3] - box_a[1])
    )

    area_b = (
        max(0.0, box_b[2] - box_b[0])
        * max(0.0, box_b[3] - box_b[1])
    )

    # Calculate combined area of both boxes
    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area