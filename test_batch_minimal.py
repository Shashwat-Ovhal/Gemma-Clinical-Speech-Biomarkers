print("Importing core modules...")
import os
import glob
import csv
import numpy as np
# from tqdm import tqdm # Skipping tqdm
from medgemma_pd.audio_pipeline.validation import InputValidator
from medgemma_pd.audio_pipeline.preprocessing import AudioPreprocessor
from medgemma_pd.audio_pipeline.quality_control import QualityControl
from medgemma_pd.audio_pipeline.features import FeatureExtractor
from medgemma_pd.reasoning.history_loader import HistoryLoader
from medgemma_pd.reasoning.engine import MedGemmaEngine
print("Minimal Imports OK")
