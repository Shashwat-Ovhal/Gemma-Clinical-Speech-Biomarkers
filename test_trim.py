import numpy as np
try:
    from medgemma_pd.audio_pipeline.preprocessing import AudioPreprocessor
    print("Imported AudioPreprocessor")
    
    # Create dummy silent signal with a pulse
    # 1 second silence, 1 second tone, 1 second silence
    sr = 16000
    t = np.linspace(0, 1, 16000)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    silence = np.zeros(16000)
    
    y = np.concatenate([silence, tone, silence])
    print(f"Original Duration: {len(y)/sr}s")
    
    y_trim, log = AudioPreprocessor._trim_silence_numpy(y, top_db=20)
    print(f"Trimmed Duration: {len(y_trim)/sr}s")
    print(f"Log: {log}")
    
    if len(y_trim) > 0 and len(y_trim) < len(y):
        print("PASS: Trimming worked")
    else:
        print("FAIL: Bad trim result")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"CRASH: {e}")
