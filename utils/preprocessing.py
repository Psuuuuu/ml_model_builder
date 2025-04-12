

# import pandas as pd
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# def preprocess_data(df, target_col, missing_strategy="drop", transformation_map=None):
#     df = df.copy()

#     # Global Missing Value Handling
#     if missing_strategy == "drop":
#         df = df.dropna()
#     elif missing_strategy in ["mean", "median"]:
#         numeric_cols = df.select_dtypes(include=["number"]).columns
#         non_numeric_cols = df.columns.difference(numeric_cols)
#         if missing_strategy == "mean":
#             df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
#         else:
#             df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
#         for col in non_numeric_cols:
#             if df[col].isna().sum() > 0:
#                 df[col] = df[col].fillna(df[col].mode()[0])
#     elif missing_strategy == "mode":
#         for col in df.columns:
#             if df[col].isna().sum() > 0:
#                 df[col] = df[col].fillna(df[col].mode()[0])

#     # Apply per-column transformations based on user selections
#     if transformation_map:
#         for col, transform in transformation_map.items():
#             if transform == "Label Encode":
#                 if df[col].dtype == "object" or str(df[col].dtype).startswith("category"):
#                     df[col] = LabelEncoder().fit_transform(df[col])
#                 else:
#                     df[col] = LabelEncoder().fit_transform(df[col].astype(str))
#             elif transform == "Normalize":
#                 scaler = StandardScaler()
#                 df[[col]] = scaler.fit_transform(df[[col]])
#             # "No Transformation" leaves the column unchanged

#     # ✅ Automatically encode target column if it's a string (so r2_score works)
#     if target_col and target_col in df.columns:
#         if df[target_col].dtype == "object":
#             df[target_col] = LabelEncoder().fit_transform(df[target_col])

#     return df

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df, target_col, missing_strategy="drop", transformation_map=None):
    df = df.copy()

    # 1. Handle missing values
    if missing_strategy == "drop":
        df = df.dropna()
    elif missing_strategy in ["mean", "median"]:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        non_numeric_cols = df.columns.difference(numeric_cols)
        if missing_strategy == "mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        else:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        for col in non_numeric_cols:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
    elif missing_strategy == "mode":
        for col in df.columns:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])

    # 2. Apply feature transformations
    if transformation_map:
        for col, transform in transformation_map.items():
            if transform == "Label Encode":
                if df[col].dtype == "object" or str(df[col].dtype).startswith("category"):
                    df[col] = LabelEncoder().fit_transform(df[col])
                else:
                    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            elif transform == "Normalize":
                scaler = StandardScaler()
                df[[col]] = scaler.fit_transform(df[[col]])
            # "No Transformation" = leave column as is

    # 3. Label encode target column if it's a string
    if target_col and target_col in df.columns:
        if df[target_col].dtype == "object" or str(df[target_col].dtype).startswith("category"):
            df[target_col] = LabelEncoder().fit_transform(df[target_col])

    return df

