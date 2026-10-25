#!/usr/bin/env python
# coding: utf-8

# # YOLOv8n vs YOLOv8s Model Comparison
# 
# Compare the YOLOv8n and YOLOv8s models by comparing detection accuracy, defect-severity assessments and computational-efficiency metrics.
# 
# Both models were trained and evaluated using the same dataset splits, image resolution, batch size, random seed, GPU and severity-assessment framework. This allows differences in performance to be primarily attributed to model architecture and complexity.
# 
# The purpose of this comparison is to determine which model provides the most suitable balance between accuracy and computational efficiency for automated pork packaging quality assessment.
# 
# This will allow the models to be fairly and accurately compared to one another and combine the previous sections of this project into analysis.

# In[ ]:


"""
Model performance comparison table

"""
#Create table of all previously found values 
display_comparison = pd.DataFrame({
    "Metric": [
        "Precision",
        "Recall",
        "mAP@0.5",
        "mAP@0.5:0.95",
        "Matched Severity Accuracy",
        "Matched Detection Rate",
        "End-to-End Severity Rate",
        "Bounding-Box Area MAE",
        "Unmatched Predictions"
    ],

    "YOLOv8n": [
        "85.29%",
        "33.45%",
        "42.06%",
        "20.63%",
        "94.64%",
        "70.89%",
        "67.09%",
        "0.010867",
        "20"
    ],

    "YOLOv8s": [
        "77.11%",
        "43.83%",
        "52.09%",
        "26.68%",
        "96.49%",
        "72.15%",
        "69.62%",
        "0.010055",
        "27"
    ]
})

display_comparison

# # Performance comparisons
# By using the above table, it can be seen that YOLOv8n has around an 8% higher precision compared to the s model. This indicates that a larger amount of the predictions made by the n model actually corresponded to the valid ground truth data. 
# 
# On the other hand the s model managed to have over 10% higher recall and has considerable improvements to the mAP@0.5 and mAP@0.5:0.95. This demonstrates that it has a stronger localised ability to  detect the defects.
# 
# The severity accuracy, detection rates and end-to-end severity rates were extremely similar however the s model is slightly improved in comparison by around 2%. 
# 
# YOLOv8s generated more unmatched predictions than YOLOv8n. This indicates a trade-off between detecting a greater proportion of real defects and generating additional unmatched detections.

# # Computational cost comparison - Using Benchmark
# Both models were trained under the same system and circumstances (for example performance mode, plugged in, etc)

# In[ ]:


"""
Using a benchmark to test computational costs of both models

"""

#Imports
from ultralytics import YOLO
from pathlib import Path
import time
import pandas as pd

# Test images used to benchmark both models
test_images = r"C:\Users\sadul\OneDrive\Documents\1University\Year 3\MMA3001\Project\Pork Rasher Error-Packaging-.v4i.yolov8\test\images"

# Best trained model from each experiment
yolov8n_path = Path(
    "runs/detect/yolov8n_full_training/weights/best.pt"
)

yolov8s_path = Path(
    "runs/detect/yolov8s_full_training/weights/best.pt"
)

#Load models
model_n = YOLO(str(yolov8n_path))
model_s = YOLO(str(yolov8s_path))

# Count the number of parameters contained in each neural network
parameters_n = sum(
    parameter.numel()
    for parameter in model_n.model.parameters()
)

parameters_s = sum(
    parameter.numel()
    for parameter in model_s.model.parameters()
)

# Convert file size from bytes to megabytes
size_n = yolov8n_path.stat().st_size / (1024 ** 2)
size_s = yolov8s_path.stat().st_size / (1024 ** 2)

# Measure how long YOLOv8n takes to process the test dataset
start_time = time.perf_counter()

model_n.predict(
    source=test_images,
    imgsz=640,
    device=0,
    verbose=False
)

time_n = time.perf_counter() - start_time


# Measure how long YOLOv8s takes to process the same dataset
start_time = time.perf_counter()

model_s.predict(
    source=test_images,
    imgsz=640,
    device=0,
    verbose=False
)

time_s = time.perf_counter() - start_time


# The test dataset contains 80 images
number_of_images = 80

# Calculate average processing time per image
time_per_image_n = (time_n / number_of_images) * 1000
time_per_image_s = (time_s / number_of_images) * 1000

#Create a comparison table
efficiency_df = pd.DataFrame({

    "Metric": [
        "Parameters",
        "Checkpoint Size (MB)",
        "Test Set Inference Time (s)",
        "Average Time per Image (ms)"
    ],

    "YOLOv8n": [
        parameters_n,
        round(size_n, 2),
        round(time_n, 2),
        round(time_per_image_n, 2)
    ],

    "YOLOv8s": [
        parameters_s,
        round(size_s, 2),
        round(time_s, 2),
        round(time_per_image_s, 2)
    ]
})

efficiency_df

# # Computation cost comparison - using hardware

# In[6]:


# Install NVIDIA's Python GPU monitoring interface
%pip install nvidia-ml-py

# In[ ]:


"""
Compare the GPU usage and other hardware values for both models

"""

#Imports
from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import threading
import time
import pynvml

test_images = r"C:\Users\sadul\OneDrive\Documents\1University\Year 3\MMA3001\Project\Pork Rasher Error-Packaging-.v4i.yolov8\test\images"

yolov8n_path = Path(
    "runs/detect/yolov8n_full_training/weights/best.pt"
)

yolov8s_path = Path(
    "runs/detect/yolov8s_full_training/weights/best.pt"
)

#Load models
model_n = YOLO(str(yolov8n_path))
model_s = YOLO(str(yolov8s_path))

#Initialise
pynvml.nvmlInit()
gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

def monitor_gpu(stop_event, measurements):
    """
    Continuously record GPU measurements until the benchmark ends.

    Parameters
    stop_event : threading.Event
        Signal indicating when monitoring should stop.

    Returns
        List used to store GPU measurements.
    """

    while not stop_event.is_set():

        # GPU utilisation percentage
        utilisation = pynvml.nvmlDeviceGetUtilizationRates(gpu)

        # GPU memory information
        memory = pynvml.nvmlDeviceGetMemoryInfo(gpu)

        # GPU temperature
        temperature = pynvml.nvmlDeviceGetTemperature(
            gpu,
            pynvml.NVML_TEMPERATURE_GPU
        )

        # GPU power draw NVML reports power in milliwatts, so divide by 1000
        power = pynvml.nvmlDeviceGetPowerUsage(gpu) / 1000

        measurements.append({
            "gpu_utilisation": utilisation.gpu,
            "memory_used_mb": memory.used / (1024 ** 2),
            "temperature_c": temperature,
            "power_w": power
        })

        # Record approximately five measurements per second
        time.sleep(0.2)

def benchmark_gpu_usage(model, model_name, repetitions=10):
    """
    Run repeated inference while recording GPU resource usage.

    Parameters
    model : ultralytics.YOLO
        Trained YOLO model.

    model_name : str
        Name used to identify the model.

    repetitions : int
        Number of times the test dataset is processed.

    Returns
    dict
        Summary of GPU resource utilisation.
    """

    measurements = []

    stop_event = threading.Event()

    # Start GPU monitoring in the background
    monitor_thread = threading.Thread(
        target=monitor_gpu,
        args=(stop_event, measurements)
    )

    monitor_thread.start()

    # Start benchmark timer
    start_time = time.perf_counter()

    # Process the same test set repeatedly
    for repetition in range(repetitions):

        model.predict(
            source=test_images,
            imgsz=640,
            conf=0.25,
            device=0,
            save=False,
            verbose=False
        )

    # Stop benchmark timer
    total_time = time.perf_counter() - start_time

    # Stop GPU monitoring
    stop_event.set()
    monitor_thread.join()

    # Convert measurements into a DataFrame
    measurements_df = pd.DataFrame(measurements)

    # Produce the summary statistics
    results = {
        "Model": model_name,

        "Average GPU Usage (%)":
            measurements_df["gpu_utilisation"].mean(),

        "Peak GPU Usage (%)":
            measurements_df["gpu_utilisation"].max(),

        "Average VRAM (MB)":
            measurements_df["memory_used_mb"].mean(),

        "Peak VRAM (MB)":
            measurements_df["memory_used_mb"].max(),

        "Average Temperature (C)":
            measurements_df["temperature_c"].mean(),

        "Peak Temperature (C)":
            measurements_df["temperature_c"].max(),

        "Average Power (W)":
            measurements_df["power_w"].mean(),

        "Peak Power (W)":
            measurements_df["power_w"].max(),

        "Benchmark Time (s)":
            total_time
    }

    return results


print("Benchmarking YOLOv8n...")

gpu_results_n = benchmark_gpu_usage(
    model_n,
    "YOLOv8n",
    repetitions=10
)

print("YOLOv8n benchmark complete.")

print("Benchmarking YOLOv8s...")

gpu_results_s = benchmark_gpu_usage(
    model_s,
    "YOLOv8s",
    repetitions=10
)

print("YOLOv8s benchmark complete.")


#Create a comparison table for the two models
gpu_comparison_df = pd.DataFrame(
    [gpu_results_n, gpu_results_s]
)

# Round numerical results for readability
gpu_comparison_df = gpu_comparison_df.round(2)

gpu_comparison_df

# By comparing the benchmarks for both models, it can be seen that generally the s model is more gpu heavy. However interestingly, the n model had a slightly higher peak gpu usage and actually took slightly longer as well.

# # Overall comparisons of the completed model developments

# In[ ]:


"""


"""

#Imports
import pandas as pd

# Number of training images used for both models
training_images = 3349

# Both models were configured for 30 epochs
epochs = 30

# Training image resolution
image_size = 640

# Training batch size
batch_size = 8

# GPU used for both experiments
gpu_name = "NVIDIA GeForce RTX 4060 Laptop GPU"

#Observed training times from the runs in notebook 2 and 3
training_time_n_minutes = 58
training_time_s_minutes = 49

images_processed = training_images * epochs

# Convert training durations into seconds
training_time_n_seconds = training_time_n_minutes * 60
training_time_s_seconds = training_time_s_minutes * 60

# Estimate how many training images were processed per second during each complete experiment.
training_throughput_n = (
    images_processed / training_time_n_seconds
)

training_throughput_s = (
    images_processed / training_time_s_seconds
)

#Create comparison table
training_cost_df = pd.DataFrame({

    "Metric": [
        "Training Images",
        "Epochs",
        "Image Size",
        "Batch Size",
        "Observed Training Time (min)",
        "Approx. Images Processed",
        "Approx. Training Throughput (images/s)"
    ],

    "YOLOv8n": [
        training_images,
        epochs,
        image_size,
        batch_size,
        training_time_n_minutes,
        images_processed,
        round(training_throughput_n, 2)
    ],

    "YOLOv8s": [
        training_images,
        epochs,
        image_size,
        batch_size,
        training_time_s_minutes,
        images_processed,
        round(training_throughput_s, 2)
    ]
})


# Display experimental hardware separately
print("Training GPU:", gpu_name)

training_cost_df

# Interestingly, even though the s model is seemingly larger and should take longer in theory, the model development was actually completed around 9 minutes faster compared to the n model. This could be due to other factors, such as hardware. 
