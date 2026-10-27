from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

NumericValue = Annotated[
    float,
    Field(allow_inf_nan=False)
]


class TransactionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Time: NumericValue
    V1: NumericValue
    V2: NumericValue
    V3: NumericValue
    V4: NumericValue
    V5: NumericValue
    V6: NumericValue
    V7: NumericValue
    V8: NumericValue
    V9: NumericValue
    V10: NumericValue
    V11: NumericValue
    V12: NumericValue
    V13: NumericValue
    V14: NumericValue
    V15: NumericValue
    V16: NumericValue
    V17: NumericValue
    V18: NumericValue
    V19: NumericValue
    V20: NumericValue
    V21: NumericValue
    V22: NumericValue
    V23: NumericValue
    V24: NumericValue
    V25: NumericValue
    V26: NumericValue
    V27: NumericValue
    V28: NumericValue
    Amount: NumericValue

