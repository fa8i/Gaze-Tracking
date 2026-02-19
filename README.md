# Research-Grade Gaze Tracking and Facial Gesture Interaction System

A high-precision, appearance-based gaze tracking system enabling real-time, hands-free human-computer interaction using deep learning, geometric modeling, and personalized calibration.

This project implements a complete end-to-end gaze estimation pipeline, from camera calibration and gaze vector prediction to personalized screen mapping and gesture-based interaction. The system is evaluated across multiple generations of gaze estimation benchmarks, demonstrating strong performance compared to both historical and modern appearance-based gaze estimation models.

The system is designed to operate using standard RGB cameras under real-world conditions.

---

# Key Features

• Appearance-based gaze estimation using a custom CNN architecture  
• Full end-to-end gaze tracking pipeline  
• Personalized calibration via regression mapping  
• Real-time inference and interaction  
• Facial gesture recognition for hands-free control  
• Gaze-controlled virtual keyboard  
• Dynamic gaze heatmap visualization  
• Modular and extensible architecture  

---

# System Overview

The pipeline consists of four main stages:

1. Camera Calibration  
   Computes intrinsic camera parameters required for accurate geometric modeling.

2. Gaze Vector Estimation  
   A convolutional neural network predicts the 3D gaze vector from facial appearance.

3. Screen Mapping via Personalized Calibration  
   A regression model maps gaze vectors to screen-space coordinates.

4. Interaction Layer  
   Enables real-time hands-free interaction using gaze and facial gestures.

Pipeline summary:

Camera → Face Detection → Eye Extraction → CNN → Gaze Vector → Regression Mapping → Screen Coordinates → Interaction

---

# Model Architecture

The gaze estimation model is a custom convolutional neural network designed for appearance-based gaze prediction under unconstrained real-world conditions.

Architecture visualization:

![Model Architecture](docs/CNN_graph/cnn_layers.png)

The architecture was designed to:

• Extract robust gaze features from eye and facial appearance  
• Generalize across users and lighting conditions  
• Enable stable and accurate gaze vector prediction  

Architecture diagrams were generated using PlotNeuralNet.

---

# Training Convergence and Angular Error

The following figure shows the angular error evolution during training for multiple CNN architectures and configurations evaluated on the MPIIFaceGaze dataset.

Each curve represents a different model configuration. The minimum validation and test errors achieved by each model are highlighted.

![Training Convergence](docs/results/cnn_models_val_comparation.png)

Key observations:

• Best validation angular error: 1.119°  
• Best test angular error: 1.190°  
• Consistent convergence across multiple architectures  
• Stable training behavior and smooth error reduction  
• Minimal validation-to-test gap, indicating strong generalization  

These results demonstrate that the proposed architecture achieves stable convergence and low angular error under real-world appearance-based gaze estimation conditions.

---

# Benchmark Evaluation Across Generations of Gaze Estimation

This project evaluates performance across three major generations of appearance-based gaze estimation systems, reflecting the evolution of the field.

---

## First Generation: MPIIGaze (2015)

MPIIGaze was the first large-scale gaze dataset collected under unconstrained real-world conditions and established the first realistic benchmark for appearance-based gaze estimation.

The comparison below includes the original baseline model and regression-based methods evaluated within this benchmark framework.

![MPIIGaze Comparison](docs/results/MPI2015.png)

This generation established CNN-based appearance models as significantly more effective than traditional regression and geometric approaches.

---

## Second Generation: MPIIFaceGaze (2017)

MPIIFaceGaze introduced full-face gaze estimation and improved performance compared to earlier eye-only approaches.

The proposed architecture achieves competitive performance compared to established CNN-based models including:

• iTracker  
• Single-eye models  
• Two-eye models  
• Full-face CNN architectures  

![MPIIFaceGaze Comparison](docs/results/MPI2017.png)

This demonstrates the effectiveness of the architecture compared to standard full-face gaze estimation baselines.

---

## Third Generation: Modern Appearance-Based Gaze Estimation Models

This comparison includes modern gaze estimation architectures and recent appearance-based systems such as:

• RT-GENE  
• Gaze360  
• Attention-based CNN models  
• Geometry-based hybrid systems  

The proposed model is shown in yellow.

![State-of-the-Art Comparison](docs/results/actual_models.jpeg)

The convergence analysis presented earlier confirms that the reported angular error values are achieved through stable training and consistent validation performance.

These results demonstrate strong performance compared to modern gaze estimation systems.

---

# Personalized Calibration Performance

A personalized regression calibration stage significantly improves screen-space accuracy by adapting the system to individual users.

Pixel error decreases as calibration sample size increases:

![Regression 1](docs/results/fine-tuning_regression0.png)

![Regression 2](docs/results/fine-tuning_regression1.png)

![Regression 3](docs/results/fine-tuning_regression2.png)

Regression parity plots:

![Regression Parity](docs/results/fine-tuning_Catboost.png)

This calibration stage enables precise gaze-based interaction using standard RGB cameras.

---

# Real-Time Interaction

The system supports real-time interaction including:

• Cursor control via gaze  
• Facial gesture-based action triggering  
• Gaze-controlled virtual keyboard  
• Dynamic gaze heatmap visualization  
• Temporal smoothing for stable interaction  

Example:

![Demo](docs/demo_examples/Demo_Exponential_Moving_Average.gif)

---

# Installation

Install dependencies:

    pip install -r requirements.txt

---

# Usage

## Step 1 — Camera Calibration

Extract camera intrinsic parameters:

    python src/data_collection/camera_calibration.py

Requires a standard chessboard calibration pattern.

---

## Step 2 — Train or Load the Gaze Estimation Model

Optional: train using MPIIFaceGaze dataset:

    python src/train/preprocess_mpii_dataset.py
    python src/train/training.py

Alternatively, use the pretrained model.

---

## Step 3 — Personalized Calibration

Collect calibration samples:

    python src/data_collection/data_collection.py

Train regression mapping:

    python src/regressor/gaze_csv.py
    python src/regressor/regression.py

---

## Step 4 — Run Real-Time Demo

    python src/demo/main_demo.py

Optional arguments:

    --calibration_matrix_path  
    --model_path  
    --monitor_mm  
    --monitor_pixels  
    --visualize_preprocessing  
    --smoothing  
    --heatmap  
    --keyboard  

---

# Project Structure

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

# Applications

This system can be applied to:

• Assistive technologies  
• Accessibility interfaces  
• Human-computer interaction  
• Hands-free computer control  
• AR/VR systems  
• Attention tracking  
• Behavioral analysis  

---

# Technical Summary

This project demonstrates a complete appearance-based gaze estimation pipeline integrating:

• Deep learning-based gaze estimation  
• Geometric camera modeling  
• Personalized regression calibration  
• Real-time gaze interaction  

The system achieves stable convergence and strong performance across multiple generations of gaze estimation benchmarks.

---

# License

MIT License