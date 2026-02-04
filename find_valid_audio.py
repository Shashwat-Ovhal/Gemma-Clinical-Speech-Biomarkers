import os
import glob
import numpy as np
from scipy.io import wavfile

def is_silent(file_path):
    try:
        sr, y = wavfile.read(file_path)
        if len(y) == 0: return True
        # Check max amplitude
        if y.dtype.kind == 'i':
            peak = np.max(np.abs(y))
            return peak == 0
        return np.max(np.abs(y)) < 1e-4
    except:
        return True

print("Scanning for VALID (Non-Silent) Audio...")
root = "dataset- MDVR-KCL Dataset"
candidates = glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)

found_pd = None
found_hc = None

for f in candidates:
    if "PD" in f and not found_pd:
        if not is_silent(f):
            print(f"Found Valid PD: {f}")
            found_pd = f
    elif "HC" in f and not found_hc:
        if not is_silent(f):
            print(f"Found Valid HC: {f}")
            found_hc = f
            
    if found_pd and found_hc:
        break

if not found_pd and not found_hc:
    print("CRITICAL: ALL SCANNED FILES APPEAR SILENT!")
