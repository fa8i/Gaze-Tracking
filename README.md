# Research-Grade Gaze Tracking and Facial Gesture Interaction System

A high-precision, appearance-based gaze tracking system enabling real-time, hands-free human-computer interaction using deep learning, geometric modeling, and personalized calibration.

This project implements a complete end-to-end gaze estimation pipeline, from camera calibration and gaze vector prediction to personalized screen mapping and gesture-based interaction. The system is evaluated across multiple generations of gaze estimation benchmarks, demonstrating strong performance compared to both historical and modern appearance-based gaze estimation models.

The system operates using standard RGB cameras under real-world conditions.

---

# Key Features

• Appearance-based gaze estimation using a custom CNN architecture  
• End-to-end gaze tracking pipeline  
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
   Computes intrinsic camera parameters for accurate geometric modeling.

2. Gaze Vector Estimation  
   A convolutional neural network predicts the 3D gaze vector from facial appearance.

3. Personalized Screen Mapping  
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

These results confirm stable convergence and effective gaze representation learning.

---

# Benchmark Evaluation Across Generations of Gaze Estimation

This project evaluates performance across three major generations of appearance-based gaze estimation systems.

---

## First Generation: MPIIGaze (2015)

MPIIGaze established the first realistic benchmark for appearance-based gaze estimation under unconstrained real-world conditions.

![MPIIGaze Comparison](docs/results/MPI2015.png)

This benchmark marked the transition from geometric methods to deep learning-based appearance models.

---

## Second Generation: MPIIFaceGaze (2017)

MPIIFaceGaze introduced full-face gaze estimation and improved performance compared to earlier architectures.

![MPIIFaceGaze Comparison](docs/results/MPI2017.png)

The proposed architecture demonstrates competitive performance compared to established full-face CNN-based gaze estimation models.

---

## Third Generation: Modern Gaze Estimation Systems

This comparison includes modern appearance-based gaze estimation systems such as RT-GENE, Gaze360, and attention-based CNN architectures.

The proposed model is shown in yellow.

![State-of-the-Art Comparison](docs/results/actual_models.jpeg)

The convergence analysis presented earlier confirms that these results are achieved through stable and reproducible training.

---

# Personalized Calibration Performance

A regression-based calibration stage improves screen-space accuracy by adapting the model to individual users.

Pixel error decreases as calibration sample size increases:

![Regression 1](docs/results/fine-tuning_regression0.png)

![Regression 2](docs/results/fine-tuning_regression1.png)

![Regression 3](docs/results/fine-tuning_regression2.png)

Parity plots:

![Regression Parity](docs/results/fine-tuning_Catboost.png)

This calibration stage enables accurate gaze-based interaction using standard hardware.

---

# Real-Time Interaction

The system supports real-time interaction using gaze and facial gestures.

Example:

![Demo](docs/demo_examples/Demo_Exponential_Moving_Average.gif)

---

# Installation

Install dependencies:

    pip install -r requirements.txt

---

# Usage

Camera calibration:

    python src/data_collection/camera_calibration.py

Optional training:

    python src/train/preprocess_mpii_dataset.py
    python src/train/training.py

Personalized calibration:

    python src/data_collection/data_collection.py
    python src/regressor/gaze_csv.py
    python src/regressor/regression.py

Run demo:

    python src/demo/main_demo.py

---

# Applications

• Assistive technologies  
• Accessibility interfaces  
• Human-computer interaction  
• Hands-free computer control  
• AR/VR interaction  
• Attention tracking  
• Behavioral analysis  

---

# Technical Summary

This project demonstrates a complete appearance-based gaze estimation system combining deep learning, geometric modeling, personalized calibration, and real-time interaction.

The system achieves stable convergence and strong performance across multiple gaze estimation benchmarks.

---

# License

MIT License