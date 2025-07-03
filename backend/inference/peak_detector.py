import numpy as np
from scipy.signal import find_peaks

def detect_peaks(trend, prominence=0.001, distance=2):
    """
    Detect peaks in a word's trend over time.

    Args:
        trend: List or np.array of floats (word importance over time)
        prominence: Required prominence of peaks (tune based on scale)
        distance: Minimum distance between peaks

    Returns:
        List of indices (timestamps) where peaks occur
    """
    trend = np.array(trend)
    peaks, _ = find_peaks(trend, prominence=prominence, distance=distance)
    return peaks.tolist()
