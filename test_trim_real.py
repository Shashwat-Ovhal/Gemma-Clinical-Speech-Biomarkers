import numpy as np
from scipy.io import wavfile
from medgemma_pd.audio_pipeline.preprocessing import AudioPreprocessor

file = "hc_test.wav"
print(f"--- Analyzing {file} ---")

try:
    sr, y_raw = wavfile.read(file)
    print(f"Loaded. Raw Length: {len(y_raw)/sr}s")
    
    # Normalize manually just for test (simplified)
    if y_raw.dtype.kind == 'i':
         type_info = np.iinfo(y_raw.dtype)
         y = y_raw.astype(float) / max(abs(type_info.min), abs(type_info.max))
    else:
         y = y_raw
         
    if len(y.shape) > 1: y = np.mean(y, axis=1)

    y_trim, log = AudioPreprocessor._trim_silence_numpy(y, top_db=20)
    
    print(f"Trimmed Length: {len(y_trim)/sr:.4f}s")
    print(f"Trim Log: {log}")
    
    if len(y_trim) == 0:
        print("RESULT: ALL SILENCE")
    else:
        print("RESULT: CONTAINS AUDIO (Unvoiced/Noise)")

except Exception as e:
    print(f"ERROR: {e}")
