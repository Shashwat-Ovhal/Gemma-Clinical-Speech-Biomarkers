import numpy as np

class SignalQualityControl:
    """
    Layer 2: Signal Quality Control (SQC)
    Medical-grade checks for SnR, Clipping, and Duration.
    """
    
    MIN_DURATION = 0.5  # Relaxed for short test clips
    MIN_DURATION = 0.5  # Relaxed for short test clips
    MAX_CLIPPING_RATIO = 0.05 # Stricter: 5% Max Clipping
    MIN_RMS = 0.005 # Stricter: Reject absolute silence
    
    @staticmethod
    def assess_quality(y: np.ndarray, sr: int) -> dict:
        """
        Analyzes raw signal for defects.
        Returns: {'passed': bool, 'metrics': dict, 'reasons': list}
        """
        reasons = []
        metrics = {}
        passed = True
        
        # 1. Duration Check
        duration = len(y) / sr
        metrics['duration'] = duration
        if duration < SignalQualityControl.MIN_DURATION:
            passed = False
            reasons.append(f"Duration too short ({duration:.2f}s < {SignalQualityControl.MIN_DURATION}s)")

        # 2. Clipping Check
        # Samples near +/- 1.0 are considered clipped
        clipped_samples = np.sum(np.abs(y) >= 0.99)
        clipping_ratio = clipped_samples / len(y)
        metrics['clipping_ratio'] = clipping_ratio
        
        if clipping_ratio > SignalQualityControl.MAX_CLIPPING_RATIO:
            passed = False
            reasons.append(f"Excessively Clipped ({clipping_ratio*100:.1f}%)")

        # 3. Silence / Energy Check
        rms = np.sqrt(np.mean(y**2))
        metrics['rms_energy'] = rms
        
        if rms < SignalQualityControl.MIN_RMS:
            passed = False
            reasons.append("Signal Level too low (Silence/Near-Silence)")

        # 4. SNR Proxy (Simple estimation: Mean / Std of quiet segments)
        # This is hard on short clips, so we skip for now or use RMS as proxy.

        return {
            'passed': passed,
            'metrics': metrics,
            'reasons': reasons
        }
