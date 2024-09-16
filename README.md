# Real-Time Sound Event Localization and Detection: Deployment Challenges on Edge Devices

This repository implements the real-time SELD inference pipeline codes in the paper "Real-Time Sound Event Localization and Detection: Deployment Challenges on Edge Devices" submitted to ICASSP 2025. We do not provide training scripts for the models referred to in the paper. Instead, please refer to the [DCASE Challenge 2023 Task 3 baseline code](https://github.com/sharathadavanne/seld-dcase2023) for model training.

## Models
The model checkpoints in the Open Neural Network Exchange (ONNX) format in the `./models` folder contain randomly initialized weights, but use the same underlying model architecture as described in the paper. In other words, these models are not capable of inference, but can be used to benchmark speed/hardware performance on the edge device. 


## Scripts

Currently, this repository contains the scripts used to benchmark the feature extraction and inference speeds

* `audio_device_checker.py` helps you to determine which audio device index to use for the PyAudio integration.
* `comparing_feature_extraction.py` benchmarks the feature extraction speeds without any multi-threadining involved.
* `edge_inference.py` contains the real-time multi-thread recording and inference script. Use this to benchmark the feature extraction and inference speeds during live inference using a multi-threaded system.
* `model_utils.py` contains simple utility functions used by the inference pipeline.
* `single_thread_inference.py` implements a single-threaded inference pipeline using random data as input. This can be useful for debugging the inference and post-processing stage of the pipeline.

## Pre-requisties

To run `edge_inference.py`, you will need a 4-channel recording device similar to that of DCASE Challenge Task 3 recording setups. You may use the `audio_device_checker.py` to determine the appropriate recording device index to use. 

This code has been tested with `librosa==0.10.1`, `PyAudio==0.2.11` and `onnxruntime==1.19.0` on a `Raspberry Pi 3 Model B`. 

## Inference Speeds

Below results assumes $T_r = 1, n = 2, T_w = 2$

| Feature | Dimensions | System | Params (M) | MACs (G) | Feature Extraction (s) | Model Inference (s) | Excess (s) | 
| --- | --- | --- | --- | --- | --- | --- | --- |
| SALSA-Lite | 7 x 160 x 191 | SELDNet | 0.885 | 0.181 | 0.205 | 0.221 | 0.574 |
| SALSA-Lite | 7 x 160 x 191 | ResNet-18 | 3.867 | 1.104 | 0.205 | 0.666 | 0.129 |
| SALSA-Mel | 7 x 160 x 128 | SELDNet | 0.835 | 0.125 | 0.434 | 0.169 | 0.397 |
| SALSA-Mel | 7 x 160 x 128 | ResNet-18 | 3.867 | 0.774 | 0.434 | 0.463 | 0.103 |
| MelGCC | 10 x 160 x 128 | SELDNet | 0.837 | 0.161 | 0.433 | 0.198 | 0.369 |
| MelGCC | 10 x 160 x 128 | ResNet-18 | 3.868 | 0.791 | 0.433 | 0.488 | 0.079 |

Please note that these results may not be entirely replicable due to different hardware and device setups. However, you should observe similar trends with regards to these selected input features and models. 