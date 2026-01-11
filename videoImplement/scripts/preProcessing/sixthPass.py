# sixth pass
# use savitzky-golay filter 

import numpy as np
import pandas as pd
from scripts.others.util import dprint
from scipy.signal import savgol_filter

"""
def rollingAverage(dataframe):
    dprint("Starting sixth pass preprocessing...")
    smoothedDiameters = []
    smoothedDiameters_mm = []
    diameters = dataframe['diameter'].tolist()
    diameters_mm = dataframe['diameter_mm'].tolist()
    totalPoints = len(diameters)

    for i in range(totalPoints):
        if i == 0:
            avg = (diameters[i] + diameters[i + 1]) / 2
            avg_mm = (diameters_mm[i] + diameters_mm[i + 1]) / 2
        elif i == totalPoints - 1:
            avg = (diameters[i - 1] + diameters[i]) / 2
            avg_mm = (diameters_mm[i - 1] + diameters_mm[i]) / 2
        else:
            avg = (diameters[i - 1] + diameters[i] + diameters[i + 1]) / 3
            avg_mm = (diameters_mm[i - 1] + diameters_mm[i] + diameters_mm[i + 1]) / 3
        smoothedDiameters.append(avg)
        smoothedDiameters_mm.append(avg_mm)
        dprint(f"Replacing diameter at frame {i} with smoothed value {avg} pixels ({avg_mm} mm)")

    dataframe['diameter'] = smoothedDiameters
    dataframe['diameter_mm'] = smoothedDiameters_mm
    dprint("Sixth pass preprocessing completed.")
    return dataframe
"""


def savgolSmoothing(dataframe, fps=None, target_window_ms=150):

    n_points = len(dataframe)

    # Infer fps if user did not provide it
    if fps is None:
        try:
            from videoImplement.settings import infer_fps_from_timestamps  # when run from project root
        except Exception:
            try:
                from settings import infer_fps_from_timestamps  # when run inside videoImplement
            except Exception:
                infer_fps_from_timestamps = lambda ts: None

        inferred = infer_fps_from_timestamps(dataframe.get('timestamp', []))
        if inferred is None:
            # Last-resort fallback (should rarely be needed)
            fps = 60.0
            dprint("Pass 6: fps not provided and could not be inferred; falling back to 60 fps")
        else:
            fps = float(round(inferred))
            dprint(f"Pass 6: inferred fps ≈ {fps}")

    fps = float(fps)

    diameters = dataframe['diameter'].astype(float).values.copy()
    diameters_mm = dataframe['diameter_mm'].astype(float).values.copy()

    # SavGol cannot handle NaNs -> temporarily interpolate for smoothing,
    # then restore original NaN locations afterwards.
    nan_mask = np.isnan(diameters) | np.isnan(diameters_mm)

    if np.any(np.isnan(diameters)):
        diameters = pd.Series(diameters).interpolate(method='linear', limit_direction='both').values
    if np.any(np.isnan(diameters_mm)):
        diameters_mm = pd.Series(diameters_mm).interpolate(method='linear', limit_direction='both').values

    # Signal properties
    signal_std = float(np.std(diameters_mm))

    # Window in frames (must be odd and >= 5)
    window_frames = int(target_window_ms / (1000.0 / fps))
    window_frames = max(5, window_frames)
    if window_frames % 2 == 0:
        window_frames += 1

    # Adaptive tweak based on noise
    if signal_std > 0.5:
        window_frames = min(window_frames + 2, 11)
        polyorder = 2
        dprint(f"Pass 6: noisy signal (std={signal_std:.3f}), window={window_frames}, polyorder={polyorder}")
    else:
        window_frames = max(5, window_frames)
        polyorder = 3
        dprint(f"Pass 6: clean signal (std={signal_std:.3f}), window={window_frames}, polyorder={polyorder}")

    if window_frames > n_points:
        window_frames = n_points if n_points % 2 == 1 else n_points - 1
        window_frames = max(5, window_frames)

    if polyorder >= window_frames:
        polyorder = window_frames - 1

    # Apply smoothing
    smoothed = savgol_filter(diameters, window_length=window_frames, polyorder=polyorder, mode='interp')
    smoothed_mm = savgol_filter(diameters_mm, window_length=window_frames, polyorder=polyorder, mode='interp')

    # Restore NaNs where original data was missing
    smoothed[nan_mask] = np.nan
    smoothed_mm[nan_mask] = np.nan

    dataframe['diameter'] = smoothed
    dataframe['diameter_mm'] = smoothed_mm
    return dataframe

