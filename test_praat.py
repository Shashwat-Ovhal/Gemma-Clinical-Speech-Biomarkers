import sys
print("Starting Praat Test...")
try:
    import parselmouth
    print(f"Parselmouth imported: {parselmouth.__version__}")
    
    sound = parselmouth.Sound("hc_test.wav")
    y = sound.values[0] # Channel 0
    sr = sound.sampling_frequency
    
    print(f"Loaded audio: {len(y)} samples, sr={sr}")
    
    with open("praat_ok.txt", "w") as f:
        f.write("OK")
    print("Test Complete.")
    
except Exception as e:
    print(f"ERROR: {e}")
    with open("praat_error.txt", "w") as f:
        f.write(str(e))
