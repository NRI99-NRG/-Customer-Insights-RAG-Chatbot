
import pandas as pd
import numpy as np


INPUT_PATH = "backend\data\enhanced_user_dataset_messy_v2.csv"
OUTPUT_PATH = "backend/data/enhanced_user_dataset_cleaned.csv"
REPORT_PATH = "backend/data/cleaning_report.csv"

def load_data(path=INPUT_PATH):
    print("Loading:", path)
    return pd.read_csv(path)

def basic_info(df, label="DATA"):
    print(f"--- {label} ---")
    print("Rows, Columns:", df.shape)
    print("Missing values per column:")
    print(df.isna().sum())
    print("Dtypes:")
    print(df.dtypes)
    print("Sample rows:")
    print(df.head(3).to_string())
    print("------------\n")

def drop_unused_columns(df):
    # drop unnamed-like columns and fully empty columns
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    all_nan = [c for c in df.columns if df[c].isna().all()]
    to_drop = list(set(unnamed + all_nan))
    if len(to_drop) > 0:
        print("Dropping columns:", to_drop)
        df = df.drop(columns=to_drop)
    return df

def strip_string_columns(df):
    str_cols = df.select_dtypes(include=["object"]).columns
    for c in str_cols:
        # preserve real NaN, strip whitespace, collapse repeated spaces, normalize common null tokens
        s = df[c]
        # keep NaN as is (avoid turning to "nan" string)
        mask_notna = s.notna()
        cleaned = s.astype(str).where(mask_notna, other=np.nan)
        cleaned = cleaned.str.strip()
        # normalize common tokens to actual NA
        cleaned = cleaned.replace({"nan": np.nan, "None": np.nan, "<NA>": np.nan, "none": np.nan})
        # collapse multiple spaces
        cleaned = cleaned.where(cleaned.isna(), cleaned.str.replace(r"\s+", " ", regex=True))
        df[c] = cleaned
    return df

def parse_dates(df):
    # heuristic: parse columns with date-like names
    date_cols = [c for c in df.columns if any(tok in c.lower() for tok in ("date", "joined", "created", "timestamp"))]
    for c in date_cols:
        try:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        except Exception:
            # silently skip if parse fails
            pass
    return df

def convert_types(df):
    # keep User_ID as string if present
    if "User_ID" in df.columns:
        df["User_ID"] = df["User_ID"].astype(str)

    # Newsletter subscription normalization
    if "Newsletter_Subscription" in df.columns:
        mapping = {True: True, False: False, "True": True, "False": False, "true": True, "false": False, "1": True, "0": False}
        df["Newsletter_Subscription"] = df["Newsletter_Subscription"].map(mapping).fillna(df["Newsletter_Subscription"])
        # keep as object/nullable boolean depending on pandas version
        try:
            df["Newsletter_Subscription"] = df["Newsletter_Subscription"].astype("boolean")
        except Exception:
            pass

    # numeric conversion (strip non-numeric chars first)
    numeric_candidates = [
        "Age", "Income", "Last_Login_Days_Ago", "Purchase_Frequency",
        "Average_Order_Value", "Total_Spending", "Time_Spent_on_Site_Minutes",
        "Pages_Viewed", "Churn_Risk_Score", "Customer_Lifetime_Value",
        "Satisfaction_Rating", "Last_Purchase_Days_Ago", "Return_Rate"
    ]
    for c in numeric_candidates:
        if c in df.columns:
            # remove non-digit characters (preserve - and .)
            cleaned = df[c].astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
            # convert empty strings to NaN before coercion
            cleaned = cleaned.replace("", np.nan)
            df[c] = pd.to_numeric(cleaned, errors="coerce")
    # attempt date parsing too
    df = parse_dates(df)
    return df

def impute_missing(df):
    # numeric -> median, categorical -> mode (or "Unknown")
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            med = df[c].median()
            if np.isnan(med):
                med = 0
            df[c] = df[c].fillna(med)
        else:
            try:
                mode_series = df[c].mode(dropna=True)
                fill_val = mode_series.iloc[0] if len(mode_series) > 0 else "Unknown"
            except Exception:
                fill_val = "Unknown"
            df[c] = df[c].fillna(fill_val)
    return df

def remove_duplicates(df):
    before = len(df)
    if "User_ID" in df.columns:
        df = df.drop_duplicates(subset=["User_ID"], keep="last")
    else:
        df = df.drop_duplicates(keep="last")
    after = len(df)
    print(f"Removed {before - after} duplicates.")
    return df

def cap_outliers_iqr(df, cols=None):
    # choose numeric columns if not specified
    if cols is None:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    for c in cols:
        series = pd.to_numeric(df[c], errors="coerce")
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        if pd.isna(IQR) or IQR == 0:
            continue
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[c] = series.clip(lower=lower, upper=upper)
    return df

def derive_features(df):
    if "Last_Login_Days_Ago" in df.columns:
        df["Recency_Bucket"] = pd.cut(
            df["Last_Login_Days_Ago"],
            bins=[-1, 1, 7, 30, 90, 1e9],
            labels=["Today", "Week", "30_days", "90_days", "Older"]
        )
    if "Customer_Lifetime_Value" in df.columns and "Income" in df.columns:
        # avoid division by zero
        income = df["Income"].replace(0, np.nan)
        df["Rel_CLV"] = df["Customer_Lifetime_Value"] / income
        # if still NaN, fallback to CLV
        df["Rel_CLV"] = df["Rel_CLV"].fillna(df["Customer_Lifetime_Value"])
    return df

def save_report(report_rows, path=REPORT_PATH):
    # report_rows: list of dicts -> create dataframe and save
    rpt_df = pd.DataFrame(report_rows)
    try:
        rpt_df.to_csv(path, index=False)
        print("Saved report to:", path)
    except Exception as e:
        print("Could not save report:", e)

def clean_pipeline(path=INPUT_PATH, out_path=OUTPUT_PATH):
    raw = load_data(path)
    basic_info(raw, "INITIAL")

    report = []
    report.append({
        "stage": "initial",
        "rows": raw.shape[0],
        "cols": raw.shape[1],
        "missing_total": int(raw.isna().sum().sum())
    })

    df = raw.copy()
    df = drop_unused_columns(df)
    df = strip_string_columns(df)
    df = convert_types(df)
    df = remove_duplicates(df)
    df = impute_missing(df)
    df = cap_outliers_iqr(df)
    df = derive_features(df)

    basic_info(df, "AFTER CLEANING")

    # final report entry
    report.append({
        "stage": "final",
        "rows": df.shape[0],
        "cols": df.shape[1],
        "missing_total": int(df.isna().sum().sum())
    })

    # save cleaned data and a tiny report CSV
    try:
        df.to_csv(out_path, index=False)
        print("Saved cleaned dataset to:", out_path)
    except Exception as e:
        print("Could not save cleaned dataset:", e)

    save_report(report, REPORT_PATH)
    return df

if __name__ == "__main__":
    clean_pipeline()
