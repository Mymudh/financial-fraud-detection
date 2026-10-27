from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.data_format import ParquetFormat
from feast.types import Float64, Int64, ValueType

transaction = Entity(
    name="transaction",
    join_keys=["transaction_id"],
    value_type=ValueType.INT64,
    description="Unique identifier for each transaction",
)

transaction_source = FileSource(
    name="transaction_features_source",
    path="../artifacts/features/engineered_features.parquet",
    timestamp_field="event_timestamp",
    file_format=ParquetFormat(),
)

transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="Amount", dtype=Float64),
        Field(name="Amount_Log", dtype=Float64),
        Field(name="Transaction_Time_Seconds", dtype=Float64),
        Field(name="Time_Sin", dtype=Float64),
        Field(name="Time_Cos", dtype=Float64),
        Field(name="Amount_Rank_Percentile", dtype=Float64),
        Field(name="Amount_ZScore", dtype=Float64),
        Field(name="Class", dtype=Int64),
    ],
    source=transaction_source,
    online=True,
)