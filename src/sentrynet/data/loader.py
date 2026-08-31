from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "isFraud",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "D1",
}


def validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def load_transactions(
    transaction_path: str | Path, identity_path: str | Path | None = None
) -> pd.DataFrame:
    df = pd.read_csv(transaction_path)
    if identity_path is not None:
        identity_df = pd.read_csv(identity_path)
        df = df.merge(identity_df, on="TransactionID", how="left")
    validate_schema(df)
    return df
