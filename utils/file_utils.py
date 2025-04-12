import pandas as pd

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df, None
    except Exception as e:
        return None, str(e)

def preview_dataframe(df, num_rows=5):
    return df.head(num_rows)

def get_column_names(df):
    return list(df.columns)
