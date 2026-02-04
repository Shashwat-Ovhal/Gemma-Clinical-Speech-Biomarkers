try:
    from medgemma_pd.reasoning.history_loader import HistoryLoader
    print("HistoryLoader Imported")
    
    # Test Mapping
    print(f"ID02 -> {HistoryLoader.get_patient_history('ID02').get('subject_id')}")
    
    with open("history_ok.txt", "w") as f: f.write("OK")
    
except Exception as e:
    print(f"FAIL: {e}")
