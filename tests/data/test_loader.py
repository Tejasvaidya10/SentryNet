import pandas as pd
import pytest

from sentrynet.data.loader import validate_schema, REQUIRED_COLUMNS


def test_validate_schema_raises_on_missing_columns():
    df = pd.DataFrame({"TransactionID": [1, 2]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df)


def test_validate_schema_passes_with_all_columns():
    df = pd.DataFrame({col: [0] for col in REQUIRED_COLUMNS})
    validate_schema(df)  # should not raise
