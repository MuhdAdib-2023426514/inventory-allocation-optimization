"""Chronological whole-origin time-block split utilities (no temporal leakage)."""

from __future__ import annotations

import pandas as pd


def assign_chronological_splits(
    origins: list[str] | pd.Series,
    train_weeks: int,
    val_weeks: int,
    test_weeks: int,
) -> dict[str, str]:
    """Map sorted unique forecast_origin strings to 'train', 'val', or 'test'.

    Enforces whole-origin time blocks:
        max(origin_train) < min(origin_val) <= max(origin_val) < min(origin_test)
    """
    unique_origins = sorted(set(str(o) for o in origins))
    required = train_weeks + val_weeks + test_weeks
    if len(unique_origins) < required:
        raise ValueError(
            f"Not enough unique forecast origins ({len(unique_origins)}) for "
            f"train_weeks={train_weeks}, val_weeks={val_weeks}, test_weeks={test_weeks}"
        )
    # Use the trailing `required` origins so test is the last `test_weeks`,
    # val is the preceding `val_weeks`, and train is earlier history.
    selected = unique_origins[-required:]
    early_extra = unique_origins[:-required]

    mapping: dict[str, str] = {}
    for o in early_extra:
        mapping[o] = "train"
    for idx, o in enumerate(selected):
        if idx < train_weeks:
            mapping[o] = "train"
        elif idx < train_weeks + val_weeks:
            mapping[o] = "val"
        else:
            mapping[o] = "test"

    train_list = [o for o, s in mapping.items() if s == "train"]
    val_list = [o for o, s in mapping.items() if s == "val"]
    test_list = [o for o, s in mapping.items() if s == "test"]
    if not (max(train_list) < min(val_list) <= max(val_list) < min(test_list)):
        raise RuntimeError("Chronological split invariant violated")
    return mapping
