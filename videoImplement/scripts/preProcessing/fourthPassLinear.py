import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scripts.others.util import dprint

def linear_interpolation(df: pd.DataFrame, fps=60, max_gap_ms=400): #max 500ms
    df_interp = df.copy()
    frame_time_ms = 1000 / fps
    
    # maximum consecutive NaNs (in frames) allowed to be filled
    N_max = int(max_gap_ms / frame_time_ms)

    for col in ['diameter', 'diameter_mm']: # adjust to your column names if needed
        if col in df.columns:
            # Ensure numeric dtype so np.isnan works reliably
            values = pd.to_numeric(df[col], errors='coerce').values.copy()
            n = len(values)

            i = 0
            while i < n:
                if np.isnan(values[i]):
                    start = i

                    #find gap end
                    while i < n and np.isnan(values[i]):
                        i += 1
                    end = i - 1

                    gap_duration_ms = (end - start + 1) * frame_time_ms

                    #interpolate if gap < max_gap_ms
                    if gap_duration_ms <= max_gap_ms:
                        #find valid neighbours
                        before_idx = start - 1
                        after_idx = end + 1

                        if before_idx < 0:  # Gap at start (no before value)
                            if after_idx < n and not np.isnan(values[after_idx]):
                                after_val = values[after_idx]
                                for j in range(start, end + 1):
                                    values[j] = after_val  # Constant fill with after
                                i = end + 1
                                continue
                            else:    
                                i = end + 1
                            continue
                        elif after_idx >= n:  
                            if before_idx >= 0 and not np.isnan(values[before_idx]):
                                before_val = values[before_idx]
                                for j in range(start, end + 1):
                                    values[j] = before_val  
                                i = end + 1
                                continue
                            else:
                                i = end + 1
                                continue
                                
                        if before_idx >= 0 and after_idx < n:
                            if not np.isnan(values[before_idx]) and not np.isnan(values[after_idx]):
                                before_val = values[before_idx]
                                after_val = values[after_idx]

                                for j in range(start, end + 1):
                                    t = max(0, min(1, (j - before_idx) / (after_idx - before_idx)))
                                    values[j] = (1 - t) * before_val + t * after_val
                                    
                                dprint(f"Frames {start} - {end}: Linear interpolated {gap_duration_ms}")
                                # advance past this gap
                                i = end + 1
                                continue
                            else:
                                dprint(f"Frames {start} - {end}: skipped (missing numeric neighbor)")
                                i = end + 1
                                continue
                        else:
                            dprint(f"Frames {start} - {end}: skipped (gap touches boundary)")
                            i = end + 1
                            continue
                    else: 
                        dprint(f"Frames {start} - {end}: skipped (gap {gap_duration_ms} ms exceeds {max_gap_ms} ms cap)")
                        i = end + 1
                        continue
                else:
                    i += 1


        df_interp[col] = values

    return df_interp

def interpolateData(df: pd.DataFrame, fps: float, max_gap_ms: int = 400) -> pd.DataFrame:

    dprint(f"Pass 4: linear interpolation (max gap {max_gap_ms}ms, fps={fps})")
    df_interpolated = linear_interpolation(df, fps=float(fps), max_gap_ms=int(max_gap_ms))
    dprint("Pass 4: interpolation completed.")
    return df_interpolated

if __name__ == "__main__":
    # Simple manual test runner (optional).
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to a CSV containing diameter and diameter_mm")
    parser.add_argument("--fps", type=float, required=True, help="Video fps used to record this CSV (e.g., 30)")
    parser.add_argument("--max_gap_ms", type=int, default=400, help="Maximum gap duration to fill (ms)")
    parser.add_argument("--out", default="processed_interpolated.csv", help="Output CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df_interpolated = interpolateData(df, fps=args.fps, max_gap_ms=args.max_gap_ms)
    df_interpolated.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")
