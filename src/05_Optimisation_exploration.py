#!/usr/bin/env python
# coding: utf-8

# # YOLO model optimsation experiment

# Complete a targeted experiment on one of the YOLO models to see if reducing the learning rate can improve the models performance, without harming the positive aspects of the model. 

# In[2]:


"""
Load the fine-tuned YOLOv8s model from the 03 notebook

"""

#Imports
from ultralytics import YOLO

# The best checkpoint obtained during the original 30-epoch YOLOv8s training experiment.
baseline_model_path = (
    "runs/detect/yolov8s_full_training/weights/best.pt"
)

# Load the already fine-tuned model.
optimized_model = YOLO(baseline_model_path)

print("Baseline YOLOv8s model loaded successfully.")

# In[3]:


"""
Short YOLOv8s optimisation fine tuning run using 10 epoch

"""

# Dataset configuration
dataset_yaml = r"C:\Users\sadul\OneDrive\Documents\1University\Year 3\MMA3001\Project\Pork Rasher Error-Packaging-.v4i.yolov8\data.yaml"

# Continue fine-tuning the best YOLOv8s checkpoint.
optimization_results = optimized_model.train(

    # Same dataset used for the baseline model
    data=dataset_yaml,

    # Short additional optimisation stage
    epochs=10,

    # Stop if validation performance stops improving
    patience=4,

    # Same input resolution as baseline
    imgsz=640,

    # Same batch size as baseline
    batch=8,

    # Use RTX 4060 GPU
    device=0,

    # Stable Windows/Jupyter configuration
    workers=0,

    # Reduced learning rate for refinement
    lr0=0.001,

    # Save optimisation separately from baseline results
    name="yolov8s_optimized",

    # Same reproducibility settings used previously
    seed=42,
    deterministic=True,

    # Generate result figures
    plots=True,

    # Display training information
    verbose=True
)

print("YOLOv8s optimisation complete.")

# # Optimisation evaluation

# In[4]:



"""
Evaluating the optimisation fine tuning results

"""

#Imports
from ultralytics import YOLO

# Path to the best checkpoint created during optimisation
optimized_model_path = (
    "runs/detect/yolov8s_optimized/weights/best.pt"
)

# Load the best optimised model
optimized_model = YOLO(optimized_model_path)

# Use the same dataset configuration used for previous models
dataset_yaml = r"C:\Users\sadul\OneDrive\Documents\1University\Year 3\MMA3001\Project\Pork Rasher Error-Packaging-.v4i.yolov8\data.yaml"

# Evaluate on the same held-out test set
optimized_test_results = optimized_model.val(
    data=dataset_yaml,
    split="test",
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    plots=True
)

# Display the important test metrics
print("\nOPTIMISED YOLOv8s TEST RESULTS")
print("--------------------------------")

print(f"Precision:      {optimized_test_results.box.mp:.4f}")
print(f"Recall:         {optimized_test_results.box.mr:.4f}")
print(f"mAP@0.5:        {optimized_test_results.box.map50:.4f}")
print(f"mAP@0.5:0.95:   {optimized_test_results.box.map:.4f}")
