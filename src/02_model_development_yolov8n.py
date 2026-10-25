#!/usr/bin/env python
# coding: utf-8

# # YOLOv8n Model Development

# Start by installing ultralytics, and then importing the first model

# In[ ]:


"""
Installs Ultralytics

"""
# INSTALL ULTRALYTICS IN THE NEW CUDA ENVIRONMENT
%pip install ultralytics

# # Training test
# Using one epoch to make sure everything is working before doing the actual fine tuning.
# 

# In[ ]:


"""
YOLOv8n One-epoch training test

"""

#Import YOLOv8
from ultralytics import YOLO
import torch

# Verify that GPU acceleration is available
print("CUDA available:", torch.cuda.is_available())
print("Training GPU:", torch.cuda.get_device_name(0))

# Path to the dataset configuration file.
# The YAML file contains the locations of the training,
# validation and test sets and the five defect classes.
dataset_yaml = r"C:\Users\sadul\OneDrive\Documents\1University\Year 3\MMA3001\Project\Pork Rasher Error-Packaging-.v4i.yolov8\data.yaml"

# Load the pretrained YOLOv8n model.
# The pretrained weights provide the starting point
model = YOLO("yolov8n.pt")

# Perform one epoch only.
test_results = model.train(
    data=dataset_yaml,         # Dataset configuration
    epochs=1,                  # Single test epoch
    imgsz=640,                 # Training image resolution
    batch=8,                   # Images processed together
    device=0,                  # NVIDIA GPU 0
    workers=0,                 # Safer setting for Windows/Jupyter
    name="yolov8n_smoke_test", # Name of this test experiment
    seed=42,                   # Reproducibility
    deterministic=True,        # Improve reproducibility
    verbose=True               # Display training information
)

print("One-epoch training test completed.")

# This confirms that the YOLOv8n model is working as intended and is ready for the actual run.

# # Completing a full YOLOv8n Training run
# Fine-tuning the model for 30 epochs. The model performance is being monitored by using recall, mAP@0.5, and mAP@0.5:0.95

# In[ ]:


"""
Full YOLOv8n training using 30 epoch
"""
# Full YOLOv8n training
from ultralytics import YOLO
import torch

# Dataset configuration file
dataset_yaml = r"C:\Users\sadul\OneDrive\Documents\1University\Year 3\MMA3001\Project\Pork Rasher Error-Packaging-.v4i.yolov8\data.yaml"

# Verify that GPU acceleration is available
print("CUDA available:", torch.cuda.is_available())
print("Training GPU:", torch.cuda.get_device_name(0))

# Reload the original pretrained model.
model = YOLO("yolov8n.pt")

# Train the first comparison model
yolov8n_results = model.train(
    data=dataset_yaml,             # Dataset paths and class names
    epochs=30,                     # Maximum training epochs
    patience=8,                    # Stop early if validation stops improving
    imgsz=640,                     # Input resolution
    batch=8,                       # Batch size suitable for initial testing
    device=0,                      # RTX 4060 GPU
    workers=0,                     # Stable setting for Windows notebooks
    name="yolov8n_full_training",  # Output directory name
    seed=42,                       # Reproducibility
    deterministic=True,            # More consistent repeated experiments
    plots=True,                    # Generate training and validation figures
    verbose=True                   # Display progress
)

print("Full YOLOv8n training completed.")

# # Evaluation YOLOv8n
# Assessing the best performance of the model 

# In[ ]:


"""
Evaluate YOLOv8n on the held-out test set
"""
#Imports
from ultralytics import YOLO
import os

# Path to the dataset configuration file
dataset_yaml = r"C:\Users\sadul\OneDrive\Documents\1University\Year 3\MMA3001\Project\Pork Rasher Error-Packaging-.v4i.yolov8\data.yaml"

# Path to the best model produced during the 30-epoch training run.
# best.pt represents the checkpoint with the best validation performance
best_model_path = os.path.join(
    "runs",
    "detect",
    "yolov8n_full_training",
    "weights",
    "best.pt"
)

# Load the best trained YOLOv8n checkpoint
best_model = YOLO(best_model_path)

# Evaluate the model using the held-out test dataset.
test_results = best_model.val(
    data=dataset_yaml,
    split="test",
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    plots=True
)

#Display all the best test results
print("\nYOLOv8n TEST RESULTS")
print("--------------------")

# Overall precision across defect classes
print(f"Precision:      {test_results.box.mp:.4f}")

# Overall recall across defect classes
print(f"Recall:         {test_results.box.mr:.4f}")

# Mean Average Precision using IoU = 0.50
print(f"mAP@0.5:        {test_results.box.map50:.4f}")

# Mean Average Precision averaged across IoU thresholds 0.50-0.95
print(f"mAP@0.5:0.95:   {test_results.box.map:.4f}")

# This forms the basis of the comparison between the n and s YOLO models

# Next find the performance per class of packaging defect.

# In[ ]:


"""
YOLOv8n test performance per defect class
"""
#Imports
import pandas as pd

# Retrieve the per-class results from the test evaluation
class_results = test_results.summary()

# Convert results into a readable table
class_results_df = pd.DataFrame(class_results)
class_results_df

# From this table it can be seen that there is no class for wrinkle. To check if this is an issue with the model developement or a limitation of the provided dataset, find the number of images per defect in the test set. 

# In[ ]:


"""
Count defect numbers in the test set
"""
#Import
import os
from collections import Counter

# Path to the complete dataset
dataset_path = r"C:\Users\sadul\OneDrive\Documents\1University\Year 3\MMA3001\Project\Pork Rasher Error-Packaging-.v4i.yolov8"

# Path containing the ground-truth YOLO annotations for the held-out test dataset
test_labels_path = os.path.join(dataset_path, "test", "labels")

# Defect classes defined in data.yaml
class_names = {
    0: "loose-meat",
    1: "packaging-error",
    2: "twisted-meat",
    3: "unsealed",
    4: "wrinkle"
}

# Create a counter for each class
class_counts = Counter()

# Loop through every label file in the test dataset
for label_file in os.listdir(test_labels_path):

    # Only process YOLO .txt annotation files
    if label_file.endswith(".txt"):

        label_path = os.path.join(test_labels_path, label_file)

        # Read each ground-truth object annotation
        with open(label_path, "r") as file:

            for line in file:

                # Ignore empty annotation lines
                if not line.strip():
                    continue

                # YOLO format:
                # class_id x_center y_center width height
                values = line.strip().split()

                # Retrieve the defect class ID
                class_id = int(values[0])

                # Add one instance to that class
                class_counts[class_id] += 1

print("TEST SET GROUND-TRUTH CLASS DISTRIBUTION")
print("----------------------------------------")

total_instances = 0

# Loop through ALL five classes.
for class_id, class_name in class_names.items():

    count = class_counts[class_id]

    print(f"{class_name:<20}: {count}")

    total_instances += count

print("----------------------------------------")
print(f"Total defect instances: {total_instances}")

# This reveals that there is a new limitation, as there is 0 ground-truth data instances for the wrinkle defect class type. 

# # Defect Severity Analysis 
# Using the new fine-tuned YOLOv8n model, we are completing an analysis on if it can identify the severity of the defect using the previously established values that can be accessed in the data exploration section of this project.

# In[ ]:


"""
YOLOv8n defect-severity analysis
"""
#Imports
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import pandas as pd

# Root folder of the downloaded Roboflow dataset
dataset_path = Path(
    r"C:\Users\sadul\OneDrive\Documents\1University\Year 3\MMA3001\Project\Pork Rasher Error-Packaging-.v4i.yolov8"
)

# Folder containing the 80 held-out test images
test_images_path = dataset_path / "test" / "images"

# Folder containing the corresponding ground-truth labels
test_labels_path = dataset_path / "test" / "labels"

# Path to the best YOLOv8n checkpoint
best_model_path = Path(
    "runs/detect/yolov8n_full_training/weights/best.pt"
)

# Confirm that all required paths are available
print("Test images found:", test_images_path.exists())
print("Test labels found:", test_labels_path.exists())
print("Best model found:", best_model_path.exists())


#Defect classes
class_names = {
    0: "loose-meat",
    1: "packaging-error",
    2: "twisted-meat",
    3: "unsealed",
    4: "wrinkle"
}


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
#Using values calculated in 01 notebook
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

    return np.array([x_min, y_min, x_max, y_max], dtype=float)


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

    # Find the overlapping rectangle
    intersection_x_min = max(box_a[0], box_b[0])
    intersection_y_min = max(box_a[1], box_b[1])
    intersection_x_max = min(box_a[2], box_b[2])
    intersection_y_max = min(box_a[3], box_b[3])

    # Calculate overlap width and height
    intersection_width = max(
        0.0,
        intersection_x_max - intersection_x_min
    )

    intersection_height = max(
        0.0,
        intersection_y_max - intersection_y_min
    )

    # Calculate the intersection area
    intersection_area = (
        intersection_width * intersection_height
    )

    # Calculate individual box areas
    area_a = max(0.0, box_a[2] - box_a[0]) * max(
        0.0,
        box_a[3] - box_a[1]
    )

    area_b = max(0.0, box_b[2] - box_b[0]) * max(
        0.0,
        box_b[3] - box_b[1]
    )

    # Calculate the union area
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

    # Some images may contain no labelled defects
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

            # Relative bounding-box area
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
                    "severity": classify_severity(relative_area)
                }
            )

    return annotations

#Load the model
severity_model = YOLO(str(best_model_path))

prediction_results = severity_model.predict(
    source=str(test_images_path),
    imgsz=640,
    #Only accepting predictions with a 25% or higher confidence
    conf=0.25,
    device=0,

    save=False,
    stream=True,
    verbose=False
)

#Match predictions to the threshold
IOU_THRESHOLD = 0.50

matched_records = []
total_ground_truth = 0
total_predictions = 0
unmatched_ground_truth = 0
unmatched_predictions = 0


for result in prediction_results:

    # Locate the test image and its matching label file
    image_path = Path(result.path)
    label_path = test_labels_path / f"{image_path.stem}.txt"

    # Read ground-truth annotations
    ground_truth = load_ground_truth(label_path)

    # Store model predictions for the current image
    predictions = []

    if result.boxes is not None:

        # Predicted boxes are converted to normalised xyxy format
        predicted_boxes = result.boxes.xyxyn.cpu().numpy()
        predicted_classes = (
            result.boxes.cls.cpu().numpy().astype(int)
        )
        predicted_confidences = (
            result.boxes.conf.cpu().numpy()
        )

        for box, class_id, confidence in zip(
            predicted_boxes,
            predicted_classes,
            predicted_confidences
        ):

            # Calculate normalised prediction area
            predicted_width = max(0.0, box[2] - box[0])
            predicted_height = max(0.0, box[3] - box[1])
            predicted_area = predicted_width * predicted_height

            predictions.append(
                {
                    "class_id": class_id,
                    "box": box,
                    "area": predicted_area,
                    "severity": classify_severity(predicted_area),
                    "confidence": confidence
                }
            )

    total_ground_truth += len(ground_truth)
    total_predictions += len(predictions)

    # Keep track of predictions already assigned to ground truth
    used_prediction_indices = set()

    # Match each ground-truth defect to the unused same-class prediction with the highest IoU
    for gt in ground_truth:

        best_prediction_index = None
        best_iou = 0.0

        for prediction_index, prediction in enumerate(predictions):

            # Do not reuse a prediction
            if prediction_index in used_prediction_indices:
                continue

            # Only compare detections of the same defect class
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

            used_prediction_indices.add(best_prediction_index)

            matched_prediction = predictions[
                best_prediction_index
            ]

            matched_records.append(
                {
                    "image": image_path.name,
                    "class_id": gt["class_id"],
                    "class_name": class_names[gt["class_id"]],
                    "iou": best_iou,
                    "confidence": matched_prediction["confidence"],
                    "ground_truth_area": gt["area"],
                    "predicted_area": matched_prediction["area"],
                    "ground_truth_severity": gt["severity"],
                    "predicted_severity": matched_prediction["severity"],
                    "severity_correct": (
                        gt["severity"]
                        == matched_prediction["severity"]
                    )
                }
            )

        else:
            # Ground truth was missed or incorrectly classified
            unmatched_ground_truth += 1

    # Predictions that were not paired are false positives
    unmatched_predictions += (
        len(predictions) - len(used_prediction_indices)
    )

#Generate a results table
severity_results_df = pd.DataFrame(matched_records)

print("SEVERITY VALIDATION COMPLETE")
print("----------------------------")
print(f"Ground-truth defects:       {total_ground_truth}")
print(f"Model predictions:          {total_predictions}")
print(f"Matched defects:            {len(severity_results_df)}")
print(f"Unmatched ground truth:     {unmatched_ground_truth}")
print(f"Unmatched predictions:      {unmatched_predictions}")

if len(severity_results_df) > 0:

    # Percentage of matched detections with the correct severity
    severity_accuracy = (
        severity_results_df["severity_correct"].mean()
    )

    # Mean absolute error between predicted and true
    # normalised bounding-box areas
    area_mae = np.mean(
        np.abs(
            severity_results_df["predicted_area"]
            - severity_results_df["ground_truth_area"]
        )
    )

    # Percentage of all ground-truth defects that were matched
    matched_detection_rate = (
        len(severity_results_df) / total_ground_truth
    )

    print("\nSEVERITY RESULTS")
    print("----------------")
    print(
        f"Matched severity accuracy: "
        f"{severity_accuracy:.4f}"
    )

    print(
        f"Matched detection rate:     "
        f"{matched_detection_rate:.4f}"
    )

    print(
        f"Bounding-box area MAE:       "
        f"{area_mae:.6f}"
    )

else:
    print("No defects satisfied the matching criteria.")


# Display the first matched results
severity_results_df.head(10)

# ### YOLOv8n Severity Validation Results
# 
# The YOLOv8n model matched 56 of the 79 ground-truth defects using criteria. This produced a detection rate of 70.89%.
# 
# From the matched defects, the predicted severity category agreed with the ground-truth severity category in 94.64% of cases. This indicates that bounding-box area provided a consistent severity estimate when the model successfully detected and localised a defect.
# 
# However, 23 ground-truth defects were not matched. Defect detection and localisation remain more significant limitations than severity classification itself.

# # Confusion matrix

# In[ ]:


"""
Create a confusion matrix 
"""

#Order
severity_order = ["Low", "Medium", "High"]

# Compare ground-truth severity with predicted severity
severity_confusion_matrix = pd.crosstab(
    severity_results_df["ground_truth_severity"],
    severity_results_df["predicted_severity"],
    rownames=["Ground Truth"],
    colnames=["Predicted"],
    dropna=False
)

severity_confusion_matrix = severity_confusion_matrix.reindex(
    index=severity_order,
    columns=severity_order,
    fill_value=0
)

severity_confusion_matrix

# Of the 56 correctly matched defects, 53 were assigned the correct severity category, corresponding to a matched severity accuracy of 94.64%.
# 
# For Medium-severity defects, 24 of 26 were correctly classified, while two were overestimated as High severity. For High-severity defects, 29 of 30 were correctly classified, while one was underestimated as Medium severity.
# 
# No low-severity defects were present among the successfully matched test detections.
# 
# These results indicate that bounding-box area provides a consistent severity estimate for successfully detected Medium and High-severity defects. However, overall system performance remains dependent on the object detector successfully identifying and localising the defect.
