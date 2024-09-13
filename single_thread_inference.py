"""Script to run inference using a single thread, using simulated audio data. This is just a testing grounds code sample before moving onto the 
actual multi-threaded edge inference in `edge_inference.py`."""

import numpy as np
import sys
from datetime import datetime
from collections import deque
import time
import warnings
from comparing_feature_extraction import extract_salsalite, get_gccphat
from model_utils import get_multi_accdoa_labels, reshape_3Dto2D
import onnxruntime
import os

#suppress warnings for numpy overflow encountered in exp function during sigmoid
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    if len(sys.argv) != 4:
        model_choice = "seldnet"
        feature_choice = "salsalite"
        record_length = 2
    else:
        # Get the feature choice from system arguments
        feature_choice = sys.argv[1]
        assert feature_choice in ['salsalite', 'gcc', 'salsamel'], "Choice of : {} is forbidden!".format(feature_choice)

        model_choice = sys.argv[2]
        assert model_choice in ['seldnet', 'resnet'], "Choice of : {} is forbidden!".format(model_choice)
        print("\n\nFeature selection : {} \t Model selection : {}".format(feature_choice, model_choice))

        record_length = int(sys.argv[3])

    print("Using {} with {} features at length of {} seconds!".format(model_choice, feature_choice, record_length))

    # Get the model running
    if model_choice == "seldnet":
        model = onnxruntime.InferenceSession("./models/{}_baseline.onnx".format(feature_choice))
    elif model_choice == "resnet":
        model = onnxruntime.InferenceSession("./models/{}_resnet.onnx".format(feature_choice))

    # Queue to store audio buffers 
    data_queue = deque()
    feat_extraction_times = []
    inference_times = []
    pp_times = []
    n_times = 0

    try:
        for ni in range(1010):
            audio_data = np.random.rand(4, int(record_length * 24000))
            record_time = time.time()

            # Feature Extraction Stage
            feat_start = time.time()
            if feature_choice == "salsalite":
                features = extract_salsalite(audio_data)
            elif feature_choice == "salsamel":
                features = extract_salsalite(audio_data, use_mel=True)
            elif feature_choice == "gcc":
                features = get_gccphat(audio_data)

            # Trim features to fit the model
            features = np.expand_dims(features, axis=0)
            feat_end = time.time()
            feat_time = feat_end - feat_start
            feat_extraction_times.append(feat_time)

            # Model prediction
            pred_start = time.time()
            ort_inputs = {model.get_inputs()[0].name : features.astype(float)}
            output = model.run(None, ort_inputs)
            output = np.array(output)

            pred_time = time.time() - pred_start
            inference_times.append(pred_time)

            # Postprocessing stage
            pp_start = time.time()

            sed_pred0, doa_pred0, sed_pred1, doa_pred1, sed_pred2, doa_pred2 = get_multi_accdoa_labels(output, 13)
            sed_pred0 = reshape_3Dto2D(sed_pred0)
            doa_pred0 = reshape_3Dto2D(doa_pred0)
            sed_pred1 = reshape_3Dto2D(sed_pred1)
            doa_pred1 = reshape_3Dto2D(doa_pred1)
            sed_pred2 = reshape_3Dto2D(sed_pred2)
            doa_pred2 = reshape_3Dto2D(doa_pred2)

            pp_time = time.time() - pp_start 
            pp_times.append(pp_time)

            # Just a verbose message
            n_times += 1
            print(
                '[In : {}, Out : {}]'
                '{} times | Feat: {:.3f}, Pred: {:.3f}, PP: {:.3f}'.format(record_time.strftime("%H:%M:%S.%f")[:-4],
                                                                           datetime.now().strftime("%H:%M:%S.%f")[:-4],
                                                                           n_times, feat_time, pred_time, pp_time)
                )

            # Small pause
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Recording stopped by user")


    # Subsequently, we are just assuming the feature extraction / inference times loosely follow a normal distribution
    # We also discard the first 10 iterations due to startup variance
    feat_extraction_times = np.array(feat_extraction_times[10:])
    inference_times = np.array(inference_times[10:])
    pp_times = np.array(pp_times[10:])

    os.makedirs("./logs", exist_ok=True)
    feat_extraction_times.tofile("./logs/{}_{}_{}_feature.csv".format(feature_choice, model_choice, record_length), sep="r")
    inference_times.tofile("./logs/{}_{}_{}_model.csv".format(feature_choice, model_choice, record_length), sep="r")

    print("Feature Extraction ~ [{:.4f} ± {:.4f}]".format(np.mean(feat_extraction_times), np.std(feat_extraction_times)))

    print("Model Inference ~ [{:.4f} ± {:.4f}]".format(np.mean(inference_times), np.std(inference_times)))

    print("Post-Processing ~ [{:.4f} ± {:.4f}]".format(np.mean(pp_times), np.std(pp_times)))