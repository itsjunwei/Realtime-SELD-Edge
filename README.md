# Real-Time Sound Event Localization and Detection: Deployment Challenges on Edge Devices

This repository implements the real-time SELD inference pipeline codes in the paper "Real-Time Sound Event Localization and Detection: Deployment Challenges on Edge Devices" submitted to ICASSP 2025. We do not provide training scripts for the models referred to in the paper. Instead, please refer to the [DCASE Challenge 2023 Task 3 baseline code](https://github.com/sharathadavanne/seld-dcase2023) for model training.


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