import argparse
import json
import os
import sys

# Ensure we can import modules from current dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from medgemma_pd.data.loader import DataLoader
# from medgemma_pd.reasoning.packet_builder import PacketBuilder
# from medgemma_pd.reasoning.engine import MedGemmaEngine

def main():
    parser = argparse.ArgumentParser(description="MedGemma-PD Pipeline CLI")
    parser.add_argument("--patient", type=str, default="P07", help="Patient ID (P07, P08)")
    parser.add_argument("--session", type=int, default=6, help="Session Month (0, 3, 6)")
    parser.add_argument("--root", type=str, default="./data", help="Data root directory")
    
    args = parser.parse_args()
    
    print(f"\n--- MedGemma-PD Pipeline ---")
    print(f"Patient: {args.patient} | Session: {args.session}")
    
    # Lazy imports to catch errors
    from medgemma_pd.data.loader import DataLoader
    from medgemma_pd.reasoning.packet_builder import PacketBuilder
    from medgemma_pd.reasoning.engine import MedGemmaEngine

    # 1. Init Data Layer
    loader = DataLoader(args.root)
    
    # 2. Build Evidence Packet
    print("Building Structured Evidence Packet...")
    builder = PacketBuilder(loader)
    packet = builder.build_packet(args.patient, args.session)
    
    print("\n[EVIDENCE PACKET SUMMARY]")
    print(f"Jitter: {packet['clinical_biomarkers']['voice_features'].get('jitter_local', 0):.4f}")
    print(f"Risk Score: {packet['model_signals']['risk_probability']}")
    print(f"Trend: {packet['longitudinal_context']['trend_analysis'].get('updrs_trend', 'N/A')}")
    
    # 3. MedGemma Reasoning
    print("\n[Running MedGemma AI]...")
    insight = MedGemmaEngine.generate_insight(packet)
    
    print("\n" + "="*40)
    print(insight)
    print("="*40 + "\n")
    
    # Generate Web Data
    output_path = os.path.join(os.path.dirname(__file__), "medgemma_pd/ui/data.js")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    web_data = {
        "packet": packet,
        "insight": insight
    }
    
    with open(output_path, "w") as f:
        f.write(f"window.medgemmaData = {json.dumps(web_data, indent=2)};")
    print(f"[UI] Generated dashboard data: {output_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL ERROR: {e}")
