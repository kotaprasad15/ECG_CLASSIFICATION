import os
import wfdb
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- CONFIG ---
DATASET_PATH = 'ptb_database'  # Root folder containing all patient folders
LEAD_INDEX = 0     # Use 0 for Lead I (or set 'all' for 12 leads)
WINDOW_SEC = 2     # window duration in seconds
STEP_SEC = 1       # overlap = WINDOW_SEC - STEP_SEC
SAMPLING_RATE = 1000  # PTB is usually at 1000Hz

WINDOW_SIZE = WINDOW_SEC * SAMPLING_RATE
STEP_SIZE = STEP_SEC * SAMPLING_RATE

def segment_ecg(signal, window_size, step_size):
    segments = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        end = start + window_size
        segments.append(signal[start:end])
    return np.array(segments)

def normalize_segment(seg):
    scaler = StandardScaler()
    return scaler.fit_transform(seg)
