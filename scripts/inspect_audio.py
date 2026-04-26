import sys
import numpy as np
from scipy.io import wavfile

file_path = "hc_test.wav"
print(f"--- Inspecting {file_path} ---")

try:
    sr, y = wavfile.read(file_path)
    print(f"Sample Rate: {sr}")
    print(f"Data Type: {y.dtype}")
    print(f"Shape: {y.shape}")
    
    # Check bounds
    min_val = np.min(y)
    max_val = np.max(y)
    mean_val = np.mean(y)
    
    print(f"Min Value: {min_val}")
    print(f"Max Value: {max_val}")
    print(f"Mean Value: {mean_val}")
    
    # Check Dynamic Range
    if max_val == min_val:
        print("DYNAMIC RANGE: 0 (Flat Line)")
    else:
        # DBFS approximation for integers
        # Assuming peak is type max
        if 'int' in str(y.dtype):
            type_max = np.iinfo(y.dtype).max
            peak_db = 20 * np.log10(max(abs(max_val), abs(min_val)) / type_max)
            print(f"Peak Level (approx): {peak_db:.2f} dBFS")
    
    # Check for silence
    if max_val == 0 and min_val == 0:
        print("RESULT: DIGITAL SILENCE (All Zeros)")
    else:
        print("RESULT: Contains Data")
        
except Exception as e:
    print(f"ERROR: {e}")
