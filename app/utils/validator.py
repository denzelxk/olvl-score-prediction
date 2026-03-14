from typing import List, Tuple
import pandas as pd

REQUIRED_COLS = [
    "number_of_siblings","direct_admission","CCA","learning_style","gender",
    "tuition","n_male","n_female","age","hours_per_week","attendance_rate",
    "sleep_time","wake_time","mode_of_transport",
]
NUMERIC_COLS   = ["n_male","n_female","age","hours_per_week","attendance_rate","number_of_siblings"]
VALID_VALUES   = {
    "direct_admission": {"Yes","No"},
    "CCA":              {"Sports","Arts","Clubs","None","sports","arts","clubs","none", "ARTS", "CLUBS", "SPORTS"},
    "learning_style":   {"Visual","Auditory","visual","auditory"},
    "gender":           {"Male","Female","male","female"},
    "tuition":          {"Yes","No","Y","N","yes","no","0","1"},
    "mode_of_transport":{"private transport","walk","public transport","Private Transport","Walk","Public transport"},
}
_VALID_AGES = {15, 16, 5, 6, -5, -4}

def validate_csv(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        errors.append(f"Missing required column(s): **{', '.join(missing)}**. Download the CSV template.")
        return False, errors
    if len(df) == 0:
        return False, ["The file contains no data rows."]
    for col in NUMERIC_COLS:
        if col in df.columns:
            n_bad = pd.to_numeric(df[col], errors="coerce").isna().sum() - df[col].isna().sum()
            if n_bad > 0:
                errors.append(f"Column '{col}' has {n_bad} non-numeric value(s).")
    ages = pd.to_numeric(df["age"], errors="coerce")
    bad  = ages[~ages.isin(_VALID_AGES) & ages.notna()]
    if len(bad):
        errors.append(f"{len(bad)} row(s) have unexpected age values: {sorted(bad.unique().tolist())}.")
    for col, valid in VALID_VALUES.items():
        if col in df.columns:
            unexpected = set(df[col].dropna().astype(str)) - valid
            if unexpected:
                errors.append(f"Column '{col}' has unexpected value(s): {sorted(unexpected)}.")
    for col in ("sleep_time","wake_time"):
        if col in df.columns:
            for i, val in df[col].dropna().items():
                if len(str(val).strip().split(":")) != 2:
                    errors.append(f"Row {i+1}: '{col}'='{val}' is not HH:MM format (e.g. \'22:00\').")
                    break
    att = pd.to_numeric(df.get("attendance_rate"), errors="coerce")
    oob = att[(att < 0) | (att > 100)].dropna()
    if len(oob):
        errors.append(f"{len(oob)} row(s) have attendance_rate outside [0, 100].")
    return len(errors) == 0, errors
