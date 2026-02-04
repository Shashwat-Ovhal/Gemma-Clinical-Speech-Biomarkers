import sys
print("Starting SciPy Test...")
try:
    import scipy.io.wavfile as wav
    import numpy as np
    
    sr, y = wav.read("hc_test.wav")
    print(f"Loaded audio: {len(y)} samples, sr={sr}")
    
    with open("scipy_ok.txt", "w") as f:
        f.write("OK")
    print("Test Complete.")
    
except Exception as e:
    print(f"ERROR: {e}")
    with open("scipy_error.txt", "w") as f:
        f.write(str(e))
