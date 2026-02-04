import sys
print("Starting Librosa Test...")
try:
    import librosa
    print(f"Librosa imported: {librosa.__version__}")
    
    y, sr = librosa.load("hc_test.wav", sr=16000)
    print(f"Loaded audio: {len(y)} samples, sr={sr}")
    
    with open("librosa_ok.txt", "w") as f:
        f.write("OK")
    print("Test Complete.")
    
except Exception as e:
    print(f"ERROR: {e}")
    with open("librosa_error.txt", "w") as f:
        f.write(str(e))
