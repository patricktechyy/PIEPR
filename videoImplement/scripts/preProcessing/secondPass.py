# second pass: check for impossible biology
# reject if difference between consecutive frames is too large
# i.e > 0.5 at 60fps 
#     > 1.0 at 30fps
# if difference between frame n and frame n+1 is too large, set frame n+1 to NaN

# also check and remove if pupil diameter is above 9mm and below 2mm
import pandas as pd
import numpy as np
from scripts.others.util import dprint

def removeSusBio(df, fps):
    dprint("Second pass preprocessing: removing biologically implausible changes")

    # Make sure arrays can hold NaN (must be float)
    diameters = df['diameter_mm'].astype(float).values
    pixelsDiameters = df['diameter'].astype(float).values

    # 1) Remove based on absolute diameter limits
    for i in range(len(diameters)):
        if not np.isnan(diameters[i]):
            if diameters[i] < 2.0 or diameters[i] > 9.0:
                dprint(f"Frame {i}: diameter {diameters[i]}mm outside 2-9mm. Setting to NaN.")
                diameters[i] = np.nan
                pixelsDiameters[i] = np.nan
                df.at[i, 'is_bad_data'] = True

    # 2) Remove based on diameter difference (spikes)
    # max change per frame scaled by fps
    maxChange = 0.5 * (60 / fps)

    n = len(diameters)
    i = 1

    while i < n:
        if np.isnan(diameters[i]):
            i += 1
            continue

        # Find the last valid (non-NaN) point before i
        prev = i - 1
        while prev >= 0 and np.isnan(diameters[prev]):
            prev -= 1

        if prev < 0:
            i += 1
            continue

        # Scale allowed change by how many frames we skipped
        frames_gap = i - prev
        change = abs(diameters[i] - diameters[prev])
        allowed_change = maxChange * frames_gap

        if change > allowed_change:
            # Rapid change -> try to see if this is a blink-like pattern
            blink_start = prev
            blink_end = i

            # Go backwards to extend blink_start
            j = blink_start - 1
            while j >= 0 and not np.isnan(diameters[j]):
                if j > 0 and not np.isnan(diameters[j - 1]):
                    if abs(diameters[j] - diameters[j - 1]) > maxChange:
                        blink_start = j
                        j -= 1
                    else:
                        break
                else:
                    break

            # Go forward to extend blink_end
            j = blink_end + 1
            while j < n and not np.isnan(diameters[j]):
                if j < n - 1 and not np.isnan(diameters[j + 1]):
                    if abs(diameters[j] - diameters[j + 1]) > maxChange:
                        blink_end = j
                        j += 1
                    else:
                        break
                else:
                    break

            blink_len = blink_end - blink_start + 1

            # Blink duration check (<= 0.5s)
            if blink_len >= 3 and blink_len <= int(fps * 0.5):
                segment = diameters[blink_start:blink_end + 1]

                if len(segment) >= 3:
                    min_idx = np.argmin(segment)

                    # V-shape: minimum not at edges
                    if 0 < min_idx < len(segment) - 1:
                        amplitude = max(segment[0], segment[-1]) - segment[min_idx]
                        if amplitude > 0.5 * (fps / 60):
                            dprint(f"Frames {blink_start}-{blink_end}: Detected blink")

                            for k in range(blink_start, blink_end + 1):
                                diameters[k] = np.nan
                                pixelsDiameters[k] = np.nan
                                df.at[k, 'is_bad_data'] = True

                            i = blink_end + 1
                            continue

            # Not a blink -> just mark this point as bad
            dprint(f"Frame {i}: change {change} exceeds allowed {allowed_change}. Marking as bad.")
            diameters[i] = np.nan
            pixelsDiameters[i] = np.nan
            df.at[i, 'is_bad_data'] = True

        i += 1

    df['diameter_mm'] = diameters
    df['diameter'] = pixelsDiameters
    return df
