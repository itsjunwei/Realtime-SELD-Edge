import numpy as np
import pyaudio
import threading
import sys
from datetime import datetime
from collections import deque
import time
import warnings
from comparing_feature_extraction import extract_salsalite, get_gccphat
from model_utils import get_multi_accdoa_labels, reshape_3Dto2D
import onnxruntime

#suppress warnings just in case
warnings.filterwarnings('ignore')

# Queue to store audio buffers 
data_queue = deque()
feat_extraction_times = []
inference_times = []
pp_times = []
n_times = 0
lock = threading.Lock()

# Function to record audio
def record_audio(stream, stop_event):
    """Define the data buffer queue outside of the function and call it globally.
    This function is used to record audio data and push them to the 
    buffer queue for another function to use for inference. 

    Parameters
    ----------
    stream : pyaudio.stream
        Stream class defined by PyAudio
    stop_event : thread
        Thread to indicate whether this thread should continue running

    Returns
    -------
    None
    """

    global data_queue # Bufer queue for all the data

    while not stop_event.is_set():
        try:
            # Read the data from the buffer as np.float32
            buffer = np.frombuffer(stream.read(BLOCK_SIZE, exception_on_overflow=False), dtype=np.float32)
            time_now = datetime.now()

            # Append the audio data and recording time into the buffer queues
            data_queue.append((time_now, buffer))
        except Exception as e:
            print("Something went wrong!")
            print(e)
            break

# Function to infer audio
def infer_audio(model, feature_type):
    """Define the data buffer queue outside of the function. In fact, may not even need to 
    append the recording time into the buffer if do not wish to keep track of time. It was
    used to keep track of when the audio data was recorded and to make sure that the system
    is inferring on the correct audio data in the queue.

    Parameters
    ----------
    model
        Open Neural Network Exchange (ONNX) model used for inference
    feature_type : str
        One of `salsalite`, `salsamel` or `gcc` to determine the feature choice

    Returns
    -------
    None
    """

    global data_queue, n_times, full_block

    # Wait until there is something in the buffer queue
    while len(data_queue) == 0:
        pass

    # We take the latest (most recent) item and copy it in order to make modifications on the data
    all_data = data_queue.popleft()
    record_time = all_data[0] # No need copy for string data, apparently
    audio_data = all_data[1].copy() # Float data needs to be copied

    # Feature extraction
    feat_start = time.time()

    # Shape the audio data and convert to features
    audio_data = audio_data.reshape(-1,4).T # Reshape to our 4 channel audio array
    t_w = np.concatenate((full_block[:, BLOCK_SIZE:], audio_data), axis=-1) # Attach the latest block to the end of the full T_w
    full_block = t_w.copy() # Make a copy of the buffer in case of any in-place operations

    if feature_type == "salsalite":
        features = extract_salsalite(t_w)
    elif feature_type == "salsamel":
        features = extract_salsalite(t_w, use_mel=True)
    elif feature_type == "gcc":
        features = get_gccphat(t_w)
    else:
        print("Feature type of {} is not accepted. Please use one of `salsalite`, `salsamel` or `gcc`!".format(feature_type))
        exit()

    # Convert features
    features = np.expand_dims(features, axis=0)
    feat_time = time.time() - feat_start
    feat_extraction_times.append(feat_time)

    # Model prediction
    pred_start = time.time()

    # ONNX Runtime
    ort_inputs = {model.get_inputs()[0].name : features.astype(float)}
    output = model.run(None, ort_inputs)
    output = np.array(output[0])

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
    if n_times >= 1010: # Run this for only 1000 iterations for testing
        raise KeyboardInterrupt

def main(feature_type = "salsalite",
         model_type = "seldnet"):
    """Main function to do concurrent recording and inference"""

    # Determine feature type
    feature_type = feature_type.lower()

    # Determine the model we wish to test and set to eval mode
    model_type = model_type.lower()
    if model_type == "seldnet":
        model = onnxruntime.InferenceSession("./models/{}_baseline.onnx".format(feature_type))
    elif model_type == "resnet":
        model = onnxruntime.InferenceSession("./models/{}_resnet.onnx".format(feature_type))

    # Create an event to signal the threads to stop
    stop_event = threading.Event()
    record_thread = threading.Thread(target=record_audio,
                                    args=(stream,stop_event))
    record_thread.start()
    print("Threads started!")

    try:
        while True:
            infer_audio(model, feature_type)
    except KeyboardInterrupt:

        # Signal the threads to stop
        stop_event.set()

        # Wait for the threads to finish
        record_thread.join()
        print("Recording stopped by user")

        # End the stream gracefully
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":

    if len(sys.argv) != 3:
        

        print("\n\nScript will expect exactly two inputs for example, python3 edge_inference.py <feature> <model>")
        print("Please choose one of 'salsalite', 'salsamel', 'gccphat' for features")
        print("and choose one of 'seldnet' or 'resnet' for models")
        print("\nIn the future, more will be incrementally added in. For now, will default to <salsalite> and <seldnet>")

        feature_choice = "salsalite"
        model_choice = "seldnet"

    else:
        # Get the feature choice from system arguments
        feature_choice = sys.argv[1]
        assert feature_choice in ['salsalite', 'gcc', 'salsamel'], "Choice of : {} is forbidden!".format(feature_choice)

        model_choice = sys.argv[2]
        assert model_choice in ['seldnet', 'resnet'], "Choice of : {} is forbidden!".format(model_choice)
        print("\n\nFeature selection : {} \t Model selection : {}".format(feature_choice, model_choice))


    # Global variables
    FORMAT = pyaudio.paFloat32
    CHANNELS = 4
    RATE = 24000
    INPUT_DEVICE_INDEX = 1
    BLOCK_SIZE = 24000 # Example Tr = 1
    RECORD_SECONDS = 2 # Example n = 2
    full_block = np.zeros((CHANNELS, int(RECORD_SECONDS * RATE))) # Create the full Tw audio buffer

    print("Using Tr = 1, n = 2, Tw = 2") # For now this is hard fixed, adjust the variables above if needed

    # Stream
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=BLOCK_SIZE,
                        input_device_index=INPUT_DEVICE_INDEX)

    # Main recording and inference function
    main(feature_type=feature_choice, model_type=model_choice)

    # Subsequently, we are just assuming the feature extraction / inference times loosely follow a normal distribution
    # We also discard the first 10 iterations due to startup variance
    feat_extraction_times = np.array(feat_extraction_times[10:])
    inference_times = np.array(inference_times[10:])
    pp_times = np.array(pp_times[10:])

    print("Feature Extraction ~ [{:.4f} ± {:.4f}]".format(np.mean(feat_extraction_times), np.std(feat_extraction_times)))

    print("Model Inference ~ [{:.4f} ± {:.4f}]".format(np.mean(inference_times), np.std(inference_times)))

    print("Post-Processing ~ [{:.4f} ± {:.4f}]".format(np.mean(pp_times), np.std(pp_times)))
