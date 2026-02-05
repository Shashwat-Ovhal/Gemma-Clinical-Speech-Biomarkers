print("1. Importing os, glob, csv, numpy...")
import os
import glob
import csv
import numpy as np
print("   OK")

print("2. Importing tqdm...")
from tqdm import tqdm
print("   OK")

print("3. Importing medgemma_pd.audio_pipeline.validation...")
from medgemma_pd.audio_pipeline.validation import InputValidator
print("   OK")

print("4. Importing medgemma_pd.audio_pipeline.preprocessing...")
from medgemma_pd.audio_pipeline.preprocessing import AudioPreprocessor
print("   OK")

print("5. Importing medgemma_pd.audio_pipeline.quality_control...")
from medgemma_pd.audio_pipeline.quality_control import QualityControl
print("   OK")

print("6. Importing medgemma_pd.audio_pipeline.features...")
from medgemma_pd.audio_pipeline.features import FeatureExtractor
print("   OK")

print("7. Importing medgemma_pd.reasoning.history_loader...")
from medgemma_pd.reasoning.history_loader import HistoryLoader
print("   OK")

print("8. Importing medgemma_pd.reasoning.engine...")
from medgemma_pd.reasoning.engine import MedGemmaEngine
print("   OK")

print("ALL IMPORTS SUCCESSFUL")
