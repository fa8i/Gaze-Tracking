# Research-Grade Gaze Tracking and Facial Gesture Interaction System

A high-precision, research-grade gaze tracking system enabling hands-free human-computer interaction using deep learning, geometric modeling, and personalized calibration.

This project implements a complete end-to-end gaze estimation pipeline, from camera calibration and neural gaze vector prediction to screen-space regression and gesture-based interaction. The system achieves highly competitive performance compared to state-of-the-art academic models and commercially available gaze tracking solutions.

Designed with modularity, extensibility, and real-world robustness in mind, the system supports personalized fine-tuning and real-time interaction.

---

## Key Features

• Appearance-based gaze estimation using a custom CNN architecture  
• Personalized gaze calibration via regression-based screen mapping  
• Real-time gaze tracking with temporal smoothing  
• Facial gesture recognition for hands-free interaction  
• On-screen keyboard controlled entirely by gaze and facial gestures  
• Dynamic gaze heatmap visualization  
• Fully modular and extensible architecture  

---

## System Overview

The system consists of four main components:

1. Camera Calibration  
   Computes intrinsic camera parameters required for geometric consistency.

2. Neural Gaze Vector Estimation  
   A deep convolutional neural network predicts the 3D gaze vector from face and eye appearance.

3. Personalized Screen Mapping  
   A regression model maps gaze vectors into precise screen coordinates, enabling high accuracy after calibration.

4. Interaction Layer  
   Enables real-time control via gaze position and facial gestures.

Pipeline summary:

Camera → Face/Eye Extraction → CNN → Gaze Vector → Regression Calibration → Screen Coordinates → Interaction Layer

---

## Architecture

The neural network is a custom-designed convolutional architecture optimized for gaze estimation accuracy and robustness under real-world conditions.

Architecture visualization:

![Model Architecture](docs/CNN_graph/cnn_layers.png)

Architecture diagrams were generated using PlotNeuralNet.

---

## Performance and Validation

The proposed model achieves highly competitive angular error performance compared to:

• MPIIGaze baseline models  
• MPIIFaceGaze baseline models  
• Modern CNN-based gaze estimation architectures  
• Recent state-of-the-art appearance-based methods  

Comparative evaluation:

![Comparison 2015](docs/results/MPI2015.png)

![Comparison 2017](docs/results/MPI2017.png)

![Comparison Modern Models](docs/results/actual_models.jpeg)

CNN validation results across multiple trained architectures:

![CNN Validation](docs/results/cnn_models_val_comparation.png)

---

## Personalized Calibration Performance

Screen-space regression significantly improves accuracy through user-specific calibration.

Pixel error decreases as calibration sample size increases:

![Regression 1](docs/results/fine-tuning_regression0.png)

![Regression 2](docs/results/fine-tuning_regression1.png)

![Regression 3](docs/results/fine-tuning_regression2.png)

Regression parity plots:

![Regression Parity](docs/results/fine-tuning_Catboost.png)

This demonstrates the effectiveness of combining appearance-based gaze estimation with personalized regression calibration.

---

## Real-Time Interaction Capabilities

The system supports real-time interaction through:

• Cursor control via gaze  
• Gesture-triggered actions  
• Gaze-controlled on-screen keyboard  
• Heatmap visualization  
• Temporal smoothing for stability  

Example:

![Demo](docs/demo_examples/Demo_Exponential_Moving_Average.gif)

---

## Installation

Install dependencies:

    pip install -r requirements.txt

---

## Usage

### Step 1 — Camera Calibration

Extract camera intrinsic parameters:

    python src/data_collection/camera_calibration.py

Requires a standard chessboard calibration pattern.

---

### Step 2 — Train or Load Gaze Estimation Model

Optional: Train from MPIIFaceGaze dataset:

    python src/train/preprocess_mpii_dataset.py
    python src/train/training.py

Or use the provided pretrained model.

---

### Step 3 — Personalized Calibration (Recommended)

Collect calibration samples:

    python src/data_collection/data_collection.py

Train regression mapping:

    python src/regressor/gaze_csv.py
    python src/regressor/regression.py

This step significantly improves prediction accuracy.

---

### Step 4 — Run Real-Time Demo

    python src/demo/main_demo.py

Optional arguments:

    --calibration_matrix_path    Path to camera calibration file
    --model_path                 Path to gaze estimation model
    --monitor_mm                Monitor size in millimeters
    --monitor_pixels            Monitor resolution in pixels
    --visualize_preprocessing   Show intermediate preprocessing results
    --smoothing                 Enable temporal smoothing
    --heatmap                   Enable gaze heatmap visualization
    --keyboard                  Enable gaze-controlled keyboard

---

## Technical Highlights

Key technical contributions:

• Custom CNN architecture optimized for gaze estimation  
• Hybrid geometric + learning-based gaze mapping pipeline  
• Personalized regression calibration  
• Real-time inference pipeline  
• Robust performance under unconstrained conditions  
• Modular and extensible system design  

---

## Project Structure

    src/
        data_collection/
        train/
        regressor/
        demo/
    
    docs/
        CNN_graph/
        results/
        demo_examples/

---

## Applications

This system is applicable to:

• Assistive technologies  
• Accessibility interfaces  
• Human-computer interaction research  
• Hands-free computer control  
• AR/VR interaction systems  
• Attention tracking  
• Behavioral analysis  

---

## Research Context

This project demonstrates that carefully designed appearance-based models combined with personalized calibration can achieve performance competitive with modern gaze tracking systems, while maintaining flexibility and deployability using standard RGB cameras.

---

## License

MIT License