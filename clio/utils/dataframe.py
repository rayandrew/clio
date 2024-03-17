import pandas as pd


def append_to_df(df: pd.DataFrame | None, data: dict) -> pd.DataFrame:
    if df is None or df.empty:
        df = pd.DataFrame([data])
        return df

    df.loc[len(df)] = data
    return df


__all__ = ["append_to_df"]
