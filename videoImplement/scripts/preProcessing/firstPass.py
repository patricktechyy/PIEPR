

import numpy as np
from scripts.others.util import dprint

DEFAULT_CONFIDENCE_THRESH = 0.75

def confidenceFilter(df, confidence_thresh: float = DEFAULT_CONFIDENCE_THRESH):
    dprint(f"Pass 1: setting diameters with confidence < {confidence_thresh} to NaN")

    if 'is_bad_data' not in df.columns:
        df['is_bad_data'] = False

    bad = df['confidence'].astype(float) < float(confidence_thresh)
    df.loc[bad, 'is_bad_data'] = True
    df.loc[bad, 'diameter'] = np.nan
    df.loc[bad, 'diameter_mm'] = np.nan

    return df
