# mma3001-defect-severity-assessment-sw
# Automated Pork Packaging Defect Severity Assessment

## Project overview

This project investigates the use of object detection models for automated pork rasher packaging quality assessment.

YOLOv8n and YOLOv8s models are compared using defect-detection accuracy, severity-assessment performance and computational efficiency.

## Research question

Which fine-tuned object detection model provides the most accurate and computationally efficient method for estimating the severity of defects in pork rasher packaging under varying image conditions?

## Dataset

The project uses the Pork Rasher Error (Packaging) object-detection dataset. Data set 3.

The dataset contains 3,549 images divided into:

- 3,349 training images
- 120 validation images
- 80 test images

Five defect classes are investigated:

- loose-meat
- packaging-error
- twisted-meat
- unsealed
- wrinkle

## Repository structure

- `doc/` - Project documentation
- `html_docs/` - Generated HTML source-code documentation
- `notebooks/` - Jupyter notebooks containing the computational workflow
- `src/` - Reusable Python source code
- `tests/` - Unit tests

## Computational workflow

The project was completed using five notebooks:

1. `01_dataset_exploration.ipynb` - Dataset exploration and severity framework
2. `02_model_development_yolov8n.ipynb` - YOLOv8n training and validation
3. `03_model_development_yolov8s.ipynb` - YOLOv8s training and validation
4. `04_model_comparison.ipynb` - Model accuracy and computational comparison
5. `05_Optimisation_exploration.ipynb` - Targeted YOLOv8s optimisation

## Environment

The project was developed using:

- Python
- PyTorch
- Ultralytics YOLOv8
- NumPy
- pandas
- Matplotlib
- Jupyter
- NVIDIA CUDA

## Running the project
Run the notebooks numerically from 01 to 05.

The notebooks contain the complete model-development, validation,
comparison and optimisation workflow.

## Testing

Reusable severity-assessment functions are located in `src/severity.py`.

Six unit tests were implemented using pytest to verify:

- Low, Medium and High severity classification
- YOLO bounding-box coordinate conversion
- IoU calculation for identical bounding boxes
- IoU calculation for non-overlapping bounding boxes

All six tests passed successfully.

The full pytest output is provided in `test_report.txt`.

## Documentation

HTML documentation for the reusable Python functions was generated using pdoc and is available in `html_docs/`.

## Main findings

YOLOv8s provided stronger overall defect-detection and severity-assessment performance than YOLOv8n, while YOLOv8n provided substantially lower model complexity and storage requirements.

## Limitations

Important limitations include:

- Significant class imbalance
- No wrinkle instances in the held-out test annotations
- Limited minority-class test examples
- No matched Low-severity test detections
- Severity is based on relative defect extent rather than expert-defined manufacturing severity
- Performance has only been validated using the provided dataset

## Licence

See the `LICENSE` file for licensing information.
