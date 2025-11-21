import argparse
import re
import sys
import numpy as np
import pandas as pd

#!/usr/bin/env python3
"""
Clean healthcare patient records for machine learning training.
Usage:
    python task_3.py --input patient_records.csv --output cleaned_data.csv
"""

def try_cast_numeric_columns(df, threshold=0.5):
    # For object columns, coerce to numeric if a substantial fraction looks numeric
    for col in df.select_dtypes(include=["object"]).columns:
        coerced = pd.to_numeric(df[col].str.replace(r"[^\d\.\-\/]", "", regex=True), errors="coerce")
        num_coercible = coerced.notna().sum()
        if num_coercible / max(1, len(df)) >= threshold:
            df[col] = coerced
    return df


def parse_blood_pressure(df):
    if "blood_pressure" not in df.columns:
        return df
    s = df["blood_pressure"].astype(str).str.strip()
    # Handle '120/80' style
    mask_slash = s.str.contains(r"/")
    if mask_slash.any():
        parts = s[mask_slash].str.split("/", expand=True)
        df.loc[mask_slash, "bp_systolic"] = pd.to_numeric(parts[0], errors="coerce")
        df.loc[mask_slash, "bp_diastolic"] = pd.to_numeric(parts[1], errors="coerce")
    # Handle single numeric blood pressure entries as systolic
    mask_num = (~mask_slash) & s.str.replace(r"[^\d\.\-]", "", regex=True).str.len().gt(0)
    if mask_num.any():
        df.loc[mask_num, "bp_systolic"] = pd.to_numeric(s[mask_num].str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce")
    df.drop(columns=["blood_pressure"], inplace=True, errors="ignore")
    return df


def standardize_height(df):
    # Prefer explicit height_cm column
    if "height_cm" in df.columns:
        col = df["height_cm"].astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
        df["height_m"] = pd.to_numeric(col, errors="coerce") / 100.0
        df.drop(columns=["height_cm"], inplace=True, errors="ignore")
        return df

    # If generic 'height' exists, attempt to detect units and convert if needed
    if "height" in df.columns:
        s = df["height"].astype(str).str.strip()
        # remove trailing 'cm' or 'm'
        has_cm = s.str.lower().str.contains("cm")
        has_m = s.str.lower().str.contains("m")
        numeric = s.str.replace(r"[^\d\.\-]", "", regex=True)
        numeric_vals = pd.to_numeric(numeric, errors="coerce")
        # If many entries with 'cm' or numeric mean > 3 assume cm -> convert to meters
        if has_cm.sum() > 0 or (numeric_vals.mean(skipna=True) is not None and (numeric_vals.mean() > 3)):
            df["height_m"] = numeric_vals / 100.0
        else:
            df["height_m"] = numeric_vals  # likely already in meters
        df.drop(columns=["height"], inplace=True, errors="ignore")
    return df


def standardize_gender(df):
    # Find gender/sex column
    gender_col = None
    for c in df.columns:
        if re.search(r"\b(gender|sex)\b", c, flags=re.I):
            gender_col = c
            break
    if not gender_col:
        return df
    s = df[gender_col].astype(str).str.strip().str.lower()
    def map_gender(v):
        if pd.isna(v):
            return "Unknown"
        v = v.strip().lower()
        if v in ("m", "male", "man"):
            return "Male"
        if v in ("f", "female", "woman"):
            return "Female"
        if v in ("other", "o", "non-binary", "nonbinary", "nb"):
            return "Other"
        return "Unknown"
    df["gender"] = s.map(map_gender)
    if gender_col != "gender":
        df.drop(columns=[gender_col], inplace=True, errors="ignore")
    return df


def drop_irrelevant_ids(df):
    drop_cols = [c for c in df.columns if re.search(r"\b(patient_id|patientid|id|record_id|recordid)\b", c, flags=re.I)]
    # keep non-informative 'id' removal conservative: only drop if column looks like identifiers (mostly unique)
    to_drop = []
    for c in drop_cols:
        nunique = df[c].nunique(dropna=True)
        if nunique > max(1, 0.5 * len(df)):  # lots of unique values -> identifier
            to_drop.append(c)
        else:
            # also drop explicit names like patient_id
            if re.search(r"patient", c, flags=re.I) or re.search(r"record", c, flags=re.I):
                to_drop.append(c)
    df.drop(columns=to_drop, inplace=True, errors="ignore")
    return df


def fill_numeric_means(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        mean = df[c].mean(skipna=True)
        if pd.notna(mean):
            df[c].fillna(mean, inplace=True)
    return df


def clean_dataframe(df):
    df = df.copy()
    df = try_cast_numeric_columns(df)
    df = parse_blood_pressure(df)
    df = standardize_height(df)
    df = standardize_gender(df)
    df = drop_irrelevant_ids(df)
    df = fill_numeric_means(df)
    # Final: reorder to put target/important columns first (optional)
    return df


def main(args):
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)

    cleaned = clean_dataframe(df)

    try:
        cleaned.to_csv(args.output, index=False)
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Cleaned data saved to {args.output}")
    print(f"Rows: {len(cleaned)}, Columns: {len(cleaned.columns)}")
    print("Sample columns:", ", ".join(list(cleaned.columns)[:10]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Clean healthcare patient records for ML training.")
    p.add_argument("--input", "-i", required=True, help="Input CSV filepath")
    p.add_argument("--output", "-o", default="cleaned_data.csv", help="Output CSV filepath")
    args = p.parse_args()
    main(args)