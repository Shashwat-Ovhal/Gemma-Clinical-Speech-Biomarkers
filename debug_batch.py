import subprocess
import os
import glob

# Find a file
DATASET_ROOT = r"dataset- MDVR-KCL Dataset\26_29_09_2017_KCL\26-29_09_2017_KCL\ReadText"
pd_files = glob.glob(os.path.join(DATASET_ROOT, "PD", "*.wav"))
target = pd_files[0] if pd_files else "dataset- MDVR-KCL Dataset/hc_test.wav"

print(f"Testing target: {target}")

cmd = f'python main.py --file "{target}" --patient_id ID_DEBUG'

try:
    print("Running subprocess...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    with open("batch_error.log", "w") as f:
        f.write(f"Return Code: {result.returncode}\n")
        f.write("--- STDOUT ---\n")
        f.write(result.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(result.stderr)
        
    print(f"Done. RC={result.returncode}")
    
except Exception as e:
    print(f"Subprocess Crash: {e}")
