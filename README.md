# Adaptive Surveillance System

## Overview
This project implements an advanced car detection system using Faster R-CNN with PyTorch, featuring image augmentation, automatic annotation, model training, and comprehensive model evaluation.

## Project Structure
The project consists of three main Python scripts:

1. **Image Augmentation and Initial Annotation** (`paste.txt`)
   - Uses TensorFlow's ImageDataGenerator for data augmentation
   - Generates multiple variations of input car images
   - Uses Faster R-CNN for initial car detection and annotation

2. **Model Training and Video Processing** (`paste-2.txt`)
   - Implements a custom dataset for car detection
   - Trains Faster R-CNN model on annotated images
   - Provides video processing functionality to detect cars in video streams

3. **Model Evaluation** (`paste-3.txt`)
   - Comprehensive model performance evaluation
   - Calculates precision, recall, F1 score
   - Generates detailed visualization of model metrics
   - Creates performance plots and heatmaps

## Prerequisites
- Python 3.8+
- PyTorch
- TensorFlow
- OpenCV
- torchvision
- numpy
- matplotlib
- seaborn
- scikit-learn

## Installation
```bash
pip install torch torchvision opencv-python numpy matplotlib seaborn scikit-learn tensorflow
```

## Project Setup
1. Create project directories:
   ```
   ├── preview/             # Augmented training images
   ├── annotations/         # Annotation text files
   ├── output_annotated_images/  # Images with bounding boxes
   ├── detected_frames/     # Frames from video processing
   ```

2. Prepare your datasets:
   - Place original car images in `preview/`
   - Ensure corresponding annotation files in `annotations/`

## Workflow

### 1. Image Augmentation
- Generates multiple image variations
- Creates augmented training dataset
- Annotates images with Faster R-CNN

### 2. Model Training
- Uses augmented images for training
- Configurable training parameters
- Saves trained model

### 3. Video Processing
- Detect cars in video streams
- Save frames with car detections
- Configurable confidence threshold

### 4. Model Evaluation
- Comprehensive performance metrics
- Precision-Recall curves
- Visualization of model performance

## Usage Example
```python
# In main.py
from car_detection import train_model, process_video, evaluate_model

# Train the model
model = train_model(data_loader, optimizer, device)

# Process a video
process_video(model, 'car_video.mp4', 'detected_frames')

# Evaluate model performance
evaluation_results = evaluate_model(model, eval_data_loader)
```

## Customization
- Adjust `num_epochs`, `learning_rate`, `batch_size`
- Modify confidence thresholds
- Experiment with different augmentation techniques

## Output
- Trained model: `car_detection_model.pth`
- Annotated images in `output_annotated_images/`
- Detected video frames in `detected_frames/`
- Evaluation plots: `model_evaluation_results.png`

## Troubleshooting
- Ensure consistent image sizes
- Check annotation format
- Verify CUDA availability for GPU acceleration

## Contributing
Contributions are welcome! Please submit pull requests or open issues.

## Acknowledgements
The authors would like to express their sincere gratitude to Ms. T Archana, AP/CSE, Department of Computer Science and Engineering, SRM Institute of Science and Technology, for her invaluable guidance and support throughout this research project.
