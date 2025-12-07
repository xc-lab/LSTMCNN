# LSTM-CNN
An efficient diagnostic network for Parkinson's disease utilizing dynamic handwriting analysis

<img src="https://github.com/xc-lab/LSTMCNN/blob/main/processing/chapt4_Fig1-1.jpg" width="800">

## Introduction
we propose a lightweight network architecture to analyse dynamic handwriting signal segments of patients and present visual diagnostic results, providing an efficient diagnostic method. 

To analyse subtle variations in handwriting, we investigate time-dependent patterns in local representation of handwriting signals. Specifically, we segment the handwriting signal into fixed-length sequential segments and design a compact one-dimensional (1D) hybrid network to extract discriminative temporal features for classifying each local segment. Finally, the category of the handwriting signal is fully diagnosed through a majority voting scheme.

## Getting Started
#### 2.2 training
Train a model to classify hand-drawn data:
```
python train.py
```

#### 2.3 testing
After the above training, we can test the model and output some visualizations of the metric curves. You can run the evaluation code with:
```
python test.py
```
## Model Architecture

<img src="https://github.com/xc-lab/LSTMCNN/blob/main/processing/chapt4_Fig4-1.jpg" width="800">
