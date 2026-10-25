import numpy as np

from SRC.severity import (
    classify_severity,
    yolo_to_xyxy,
    calculate_iou
)


# Test severity classification

def test_low_severity():
    """Check that a small defect area is classified as Low."""
    assert classify_severity(0.001) == "Low"


def test_medium_severity():
    """Check that a medium defect area is classified as Medium."""
    assert classify_severity(0.05) == "Medium"


def test_high_severity():
    """Check that a large defect area is classified as High."""
    assert classify_severity(0.15) == "High"


# Test bounding-box conversion

def test_yolo_to_xyxy():
    """Check conversion from YOLO coordinates to corner coordinates."""

    result = yolo_to_xyxy(
        x_center=0.5,
        y_center=0.5,
        width=0.2,
        height=0.4
    )

    expected = np.array([
        0.4,
        0.3,
        0.6,
        0.7
    ])

    assert np.allclose(result, expected)


# Test Intersection over Union

def test_identical_boxes_iou():
    """Check that identical bounding boxes have an IoU of 1."""

    box = np.array([
        0.2,
        0.2,
        0.8,
        0.8
    ])

    assert np.isclose(
        calculate_iou(box, box),
        1.0
    )


def test_non_overlapping_boxes_iou():
    """Check that non-overlapping bounding boxes have an IoU of 0."""

    box_a = np.array([
        0.0,
        0.0,
        0.2,
        0.2
    ])

    box_b = np.array([
        0.8,
        0.8,
        1.0,
        1.0
    ])

    assert np.isclose(
        calculate_iou(box_a, box_b),
        0.0
    )