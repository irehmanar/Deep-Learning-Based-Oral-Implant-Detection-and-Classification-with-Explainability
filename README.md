# Deep Learning-Based Oral Implant Detection and Classification

This repository contains the implementation of a comprehensive framework for oral implant detection and classification using state-of-the-art deep learning models. The project is enhanced by explainability through Grad-CAM visualizations and includes an interactive chatbot interface.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Explainability](#explainability)
- [Chatbot Integration](#chatbot-integration)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Introduction

Accurate identification and classification of dental implants in radiographic images are critical for clinical diagnosis, treatment planning, and post-surgical monitoring. This project presents a deep learning-based approach to automate these tasks, achieving high accuracy and providing visual explanations for model decisions.

## Dataset

We utilize the publicly available OII-DS dataset containing 3,834 oral CT images and 15,240 labeled implant crops across five classes:
- Single Implant
- Double Implant
- Compound Implant
- Steel Ball
- Crowns/Braces

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/irehmanar/Deep-Learning-Based-Oral-Implant-Detection-and-Classification-with-Explainability.git
   ```

2. Navigate to the project directory:
   ```bash
   cd oral-implant-detection
   ```

3. Run Classification.ipynb 


## Hugging face
https://huggingface.co/spaces/irehmanar/dental-implant-classifier

## Model Training

We employed Faster R-CNN for detection and ResNet50 for classification. To train the models:

1. Prepare the dataset as described in the Dataset section.

2. Run the training script:
   ```bash
   python train.py --config configs/train_config.yaml
   ```

## Explainability

Grad-CAM visualizations are used to provide intuitive explanations of model decisions. 

## Chatbot Integration

An interactive chatbot interface is included to assist clinicians in querying implant information. To start the chatbot:


## Results

Our Faster R-CNN detector achieves a mean average precision (mAP) of 92.53%, and the ResNet50 classification model attains a validation accuracy of 99.41%. Detailed results and evaluation metrics can be found in the results directory.

## Acknowledgements

We thank the original OII-DS authors for the publicly available dataset and foundational work.
