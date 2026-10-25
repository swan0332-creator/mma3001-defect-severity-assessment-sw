#!/usr/bin/env python
# coding: utf-8

# # YOLOv8s Model Development

# Import the YOLOv8s model to this new file

# In[ ]:


"""
Check if the YOLOv8s is able to load similar to the n model.
"""
#Imports
import torch
from ultralytics import YOLO

# Load pretrained YOLOv8s
model_s = YOLO("yolov8s.pt")

print("YOLOv8s loaded successfully.")

# # Training test
# Using one epoch to make sure everything is working before doing the actual fine tuning. This uses the same values as the YOLOv8n model to ensure a fair comparison

# In[ ]:


"""
YOLOv8s one epoch training test
"""

# Dataset configuration used for BOTH models
dataset_yaml = r"C:\Users\sadul\OneDrive\Documents\1University\Year 3\MMA3001\Project\Pork Rasher Error-Packaging-.v4i.yolov8\data.yaml"

# Reload the original pretrained YOLOv8s model
model_s = YOLO("yolov8s.pt")

# Perform one test epoch
smoke_results_s = model_s.train(
    data=dataset_yaml,
    epochs=1,
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    name="yolov8s_smoke_test",
    seed=42,
    deterministic=True,
    verbose=True
)

print("YOLOv8s one-epoch test completed.")

# # Completing a full YOLOv8s Training run
# Fine-tuning the model for 30 epochs. The model performance is being monitored by using recall, mAP@0.5, and mAP@0.5:0.95. The same parameters set for the YOLOv8n model to ensure a fair and easy comparison. 

# In[ ]:


"""
Full YOLOv8s training using 30 epoch
"""

#Imports
from ultralytics import YOLO
import torch

# Path to the same dataset configuration used for YOLOv8n
dataset_yaml = r"C:\Users\sadul\OneDrive\Documents\1University\Year 3\MMA3001\Project\Pork Rasher Error-Packaging-.v4i.yolov8\data.yaml"

# Confirm GPU acceleration remains available.
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Training GPU:", torch.cuda.get_device_name(0))

# Load the ORIGINAL pretrained YOLOv8s model.
model_s = YOLO("yolov8s.pt")

# Train YOLOv8s using the same core experimental settings that were used for YOLOv8n.
yolov8s_results = model_s.train(

    # Dataset configuration
    data=dataset_yaml,

    # Maximum number of complete passes through the training set
    epochs=30,

    # Stop early if validation performance does not improve for eight consecutive epochs
    patience=8,

    # Input image resolution
    imgsz=640,

    # Number of images processed simultaneously
    batch=8,

    # Use the NVIDIA RTX 4060
    device=0,

    # Stable Windows/Jupyter data-loading configuration
    workers=0,

    # Separate output folder from YOLOv8n
    name="yolov8s_full_training",

    # Fixed seed supports experimental reproducibility
    seed=42,

    # Request more repeatable training behaviour
    deterministic=True,

    # Generate training curves and evaluation plots
    plots=True,

    # Display training progress
    verbose=True
)

print("Full YOLOv8s training completed successfully.")

# # Evaluation YOLOv8s
# Assessing the best performance of the model 

# In[ ]:


"""
Evaluate YOLOv8s on the held-out test set
"""

#Imports
from ultralytics import YOLO
import os

# Path to the same dataset configuration used throughout both model experiments.
dataset_yaml = r"C:\Users\sadul\OneDrive\Documents\1University\Year 3\MMA3001\Project\Pork Rasher Error-Packaging-.v4i.yolov8\data.yaml"

# Path to the checkpoint with the best validation performance during the full YOLOv8s training run.
best_model_s_path = os.path.join(
    "runs",
    "detect",
    "yolov8s_full_training",
    "weights",
    "best.pt"
)

# Load the best trained YOLOv8s checkpoint.
best_model_s = YOLO(best_model_s_path)

# Evaluate using exactly the same held-out test set used for YOLOv8n.
test_results_s = best_model_s.val(
    data=dataset_yaml,
    split="test",
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    plots=True
)

print("\nYOLOv8s TEST RESULTS")
print("--------------------")

print(f"Precision:      {test_results_s.box.mp:.4f}")
print(f"Recall:         {test_results_s.box.mr:.4f}")
print(f"mAP@0.5:        {test_results_s.box.map50:.4f}")
print(f"mAP@0.5:0.95:   {test_results_s.box.map:.4f}")

# These numbers are going to be used to compare the two models on their performance.

# Next, find the performance per class of packaging defect to comapre it to the findings for the per class defects for the previous model.

# In[ ]:


"""
YOLOv8s test performance per defect class
"""
#Imports
import pandas as pd

# Retrieve the test evaluation summary produced by YOLOv8s
class_results_s = test_results_s.summary()
class_results_s_df = pd.DataFrame(class_results_s)

# Display the per-class results.
class_results_s_df

# It was found during the YOLOv8n model evaluation that there is no wrinkle test data ground truth hence why it doesn't show up on this table either. Therefore this is still a limitation.

# # Defect Severity Analysis 
# Using the new fine-tuned YOLOv8s model, we are completing an analysis on if it can identify the severity of the defect using the previously established values that can be accessed in the data exploration section of this project.

# In[ ]:


"""
YOLOv8s defect-severity analysis
"""

#Imports
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import pandas as pd

# Root folder containing the complete dataset
dataset_path = Path(
    r"C:\Users\sadul\OneDrive\Documents\1University\Year 3\MMA3001\Project\Pork Rasher Error-Packaging-.v4i.yolov8"
)

# Held-out test images and ground-truth annotation folders
test_images_path = dataset_path / "test" / "images"
test_labels_path = dataset_path / "test" / "labels"

# Best YOLOv8s checkpoint from the completed training run
best_model_s_path = Path(
    "runs/detect/yolov8s_full_training/weights/best.pt"
)

# Confirm that Python can find the required files
print("Test images found:", test_images_path.exists())
print("Test labels found:", test_labels_path.exists())
print("Best YOLOv8s model found:", best_model_s_path.exists())

# Class names
class_names = {
    0: "loose-meat",
    1: "packaging-error",
    2: "twisted-meat",
    3: "unsealed",
    4: "wrinkle"
}

#Re-define the functions created in the second notebook
def classify_severity(relative_area):
    """
    Classify a defect according to its normalised bounding-box area.

    Parameters
    relative_area : float
        Bounding-box area expressed as a proportion of the complete image area.

    Returns
    str
        Low, Medium or High severity.
    """

    if relative_area < 0.003254:
        return "Low"

    elif relative_area <= 0.089605:
        return "Medium"

    else:
        return "High"


def yolo_to_xyxy(x_center, y_center, width, height):
    """
    Convert a normalised YOLO bounding box from centre format to corner format.

    Parameter
        x_center, y_center, width, height

    Returns
        x_min, y_min, x_max, y_max

    """

    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    return np.array(
        [x_min, y_min, x_max, y_max],
        dtype=float
    )


def calculate_iou(box_a, box_b):
    """
    Calculate Intersection over Union between two boxes.

    Parameters
    box_a : array-like
        First box in x_min, y_min, x_max, y_max format.

    box_b : array-like
        Second box in x_min, y_min, x_max, y_max format.

    Returns
    float
        IoU value between 0 and 1.
    """

    # Identify the corners of the overlapping region
    intersection_x_min = max(box_a[0], box_b[0])
    intersection_y_min = max(box_a[1], box_b[1])
    intersection_x_max = min(box_a[2], box_b[2])
    intersection_y_max = min(box_a[3], box_b[3])

    # Calculate the width and height of the overlap
    intersection_width = max(
        0.0,
        intersection_x_max - intersection_x_min
    )

    intersection_height = max(
        0.0,
        intersection_y_max - intersection_y_min
    )

    # Calculate the area shared by both boxes
    intersection_area = (
        intersection_width * intersection_height
    )

    # Calculate the area of each individual box
    area_a = (
        max(0.0, box_a[2] - box_a[0])
        * max(0.0, box_a[3] - box_a[1])
    )

    area_b = (
        max(0.0, box_b[2] - box_b[0])
        * max(0.0, box_b[3] - box_b[1])
    )

    # Calculate the combined area without double-counting the overlapping region
    union_area = area_a + area_b - intersection_area

    # Prevent division by zero
    if union_area <= 0:
        return 0.0

    return intersection_area / union_area

def load_ground_truth(label_path):
    """
    Read one YOLO label file.

    Returns
    - class ID
    - normalised bounding box
    - normalised bounding-box area
    - severity classification
    """

    annotations = []

    if not label_path.exists():
        return annotations

    with open(label_path, "r") as file:

        for line in file:

            # Ignore empty lines
            if not line.strip():
                continue

            values = line.strip().split()

            class_id = int(values[0])
            x_center = float(values[1])
            y_center = float(values[2])
            width = float(values[3])
            height = float(values[4])

            # Width and height are normalised, so their product gives the relative image area
            relative_area = width * height

            annotations.append(
                {
                    "class_id": class_id,

                    "box": yolo_to_xyxy(
                        x_center,
                        y_center,
                        width,
                        height
                    ),

                    "area": relative_area,

                    "severity": classify_severity(
                        relative_area
                    )
                }
            )

    return annotations


#Load model s
severity_model_s = YOLO(str(best_model_s_path))

# The same confidence threshold used for YOLOv8n
prediction_results_s = severity_model_s.predict(
    source=str(test_images_path),
    imgsz=640,
    conf=0.25,
    device=0,
    save=False,
    stream=True,
    verbose=False
)

#Threshold to ground truth
IOU_THRESHOLD = 0.50

matched_records_s = []

total_ground_truth_s = 0
total_predictions_s = 0
unmatched_ground_truth_s = 0
unmatched_predictions_s = 0


for result in prediction_results_s:

    # Identify the image and its corresponding label file
    image_path = Path(result.path)
    label_path = test_labels_path / f"{image_path.stem}.txt"

    # Load the ground-truth objects for this image
    ground_truth = load_ground_truth(label_path)

    # Store the predictions for this image
    predictions = []

    if result.boxes is not None:

        # Predicted boxes in normalised corner format
        predicted_boxes = (
            result.boxes.xyxyn.cpu().numpy()
        )

        # Predicted class IDs
        predicted_classes = (
            result.boxes.cls.cpu().numpy().astype(int)
        )

        # Prediction confidence scores
        predicted_confidences = (
            result.boxes.conf.cpu().numpy()
        )

        for box, class_id, confidence in zip(
            predicted_boxes,
            predicted_classes,
            predicted_confidences
        ):

            # Calculate normalised predicted box dimensions
            predicted_width = max(
                0.0,
                box[2] - box[0]
            )

            predicted_height = max(
                0.0,
                box[3] - box[1]
            )

            # Calculate normalised predicted box area
            predicted_area = (
                predicted_width * predicted_height
            )

            predictions.append(
                {
                    "class_id": class_id,
                    "box": box,
                    "area": predicted_area,

                    "severity": classify_severity(
                        predicted_area
                    ),

                    "confidence": confidence
                }
            )

    total_ground_truth_s += len(ground_truth)
    total_predictions_s += len(predictions)

    # Prevent one prediction from being matched to multiple ground-truth defects
    used_prediction_indices = set()

    for gt in ground_truth:

        best_prediction_index = None
        best_iou = 0.0

        # Find the unused same-class prediction with the highest IoU value
        for prediction_index, prediction in enumerate(
            predictions
        ):

            if prediction_index in used_prediction_indices:
                continue

            # Only match defects belonging to the same class
            if prediction["class_id"] != gt["class_id"]:
                continue

            current_iou = calculate_iou(
                gt["box"],
                prediction["box"]
            )

            if current_iou > best_iou:
                best_iou = current_iou
                best_prediction_index = prediction_index

        # Accept the match when IoU is at least 0.50
        if (
            best_prediction_index is not None
            and best_iou >= IOU_THRESHOLD
        ):

            used_prediction_indices.add(
                best_prediction_index
            )

            matched_prediction = predictions[
                best_prediction_index
            ]

            matched_records_s.append(
                {
                    "image": image_path.name,

                    "class_id": gt["class_id"],

                    "class_name": class_names[
                        gt["class_id"]
                    ],

                    "iou": best_iou,

                    "confidence": matched_prediction[
                        "confidence"
                    ],

                    "ground_truth_area": gt["area"],

                    "predicted_area": matched_prediction[
                        "area"
                    ],

                    "ground_truth_severity": gt[
                        "severity"
                    ],

                    "predicted_severity": matched_prediction[
                        "severity"
                    ],

                    "severity_correct": (
                        gt["severity"]
                        == matched_prediction["severity"]
                    )
                }
            )

        else:
            # Ground-truth defect was missed, misclassified or insufficiently localised
            unmatched_ground_truth_s += 1

    # Predictions that were not assigned to a ground-truth defect are counted as unmatched predictions
    unmatched_predictions_s += (
        len(predictions) - len(used_prediction_indices)
    )

#Create table of results
severity_results_s_df = pd.DataFrame(
    matched_records_s
)

print("YOLOv8s SEVERITY VALIDATION COMPLETE")
print("------------------------------------")

print(
    f"Ground-truth defects:       "
    f"{total_ground_truth_s}"
)

print(
    f"Model predictions:          "
    f"{total_predictions_s}"
)

print(
    f"Matched defects:            "
    f"{len(severity_results_s_df)}"
)

print(
    f"Unmatched ground truth:     "
    f"{unmatched_ground_truth_s}"
)

print(
    f"Unmatched predictions:      "
    f"{unmatched_predictions_s}"
)

if len(severity_results_s_df) > 0:

    # Severity accuracy among correctly matched defects
    severity_accuracy_s = (
        severity_results_s_df[
            "severity_correct"
        ].mean()
    )

    # Mean absolute error between predicted and ground-truth normalised box areas
    area_mae_s = np.mean(
        np.abs(
            severity_results_s_df["predicted_area"]
            - severity_results_s_df[
                "ground_truth_area"
            ]
        )
    )

    # Proportion of all ground-truth defects that were successfully detected and matched
    matched_detection_rate_s = (
        len(severity_results_s_df)
        / total_ground_truth_s
    )

    # Number of ground-truth defects that were detected and given the correct severity category
    correctly_assessed_s = int(
        severity_results_s_df[
            "severity_correct"
        ].sum()
    )

    # End-to-end performance treats missed detections as unsuccessful severity assessments
    end_to_end_severity_rate_s = (
        correctly_assessed_s
        / total_ground_truth_s
    )

    print("\nYOLOv8s SEVERITY RESULTS")
    print("------------------------")

    print(
        f"Matched severity accuracy: "
        f"{severity_accuracy_s:.4f}"
    )

    print(
        f"Matched detection rate:     "
        f"{matched_detection_rate_s:.4f}"
    )

    print(
        f"End-to-end severity rate:    "
        f"{end_to_end_severity_rate_s:.4f}"
    )

    print(
        f"Bounding-box area MAE:       "
        f"{area_mae_s:.6f}"
    )

else:
    print(
        "No defects satisfied the matching criteria."
    )


# Display the first ten successfully matched defects
severity_results_s_df.head(10)

# ### YOLOv8s Severity Validation Results
# 
# YOLOv8s successfully matched 57 of the 79 ground-truth defects, corresponding to a matched detection rate of 72.15%. This was slightly higher than the 70.89% matched detection rate achieved by YOLOv8n.
# 
# Among the matched defects, YOLOv8s correctly classified 55 of 57 severity categories, producing a matched severity accuracy of 96.49%. When missed defects were treated as unsuccessful severity assessments, the end-to-end severity-assessment rate was only 69.62%.
# 
# However, YOLOv8s produced 27 unmatched predictions compared with 20 for YOLOv8n. This supports the detection results showing that YOLOv8s achieved greater recall but lower precision. Therefore, YOLOv8s identified a greater proportion of defects but also produced more unmatched detections.

# # Confusion matrix

# In[ ]:


"""
Create a confusion matrix 
"""

#Imports
import pandas as pd

#Order
severity_order = ["Low", "Medium", "High"]

# Create a confusion matrix comparing the ground-truth severity category against the severity predicted by YOLOv8s.
severity_confusion_matrix_s = pd.crosstab(
    severity_results_s_df["ground_truth_severity"],
    severity_results_s_df["predicted_severity"],
    rownames=["Ground Truth"],
    colnames=["Predicted"],
    dropna=False
)

severity_confusion_matrix_s = severity_confusion_matrix_s.reindex(
    index=severity_order,
    columns=severity_order,
    fill_value=0
)

# Display the completed confusion matrix.
severity_confusion_matrix_s

# YOLOv8s correctly classified the severity of 55 of the 57 successfully matched defects, corresponding to a matched severity accuracy of 96.49%.
# 
# Of the 26 Medium-severity defects, 24 were correctly classified while two were overestimated as High severity. All 31 matched High-severity defects were correctly classified.
# 
# No Low-severity defects were present among the successfully matched test detections. Therefore, the performance of YOLOv8s for Low-severity defect assessment cannot be independently evaluated using the available test sample.
# 
# Compared with YOLOv8n, YOLOv8s demonstrated slightly improved severity classification performance, particularly for High-severity defects.
