from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from feast.data_format import ParquetFormat
from datetime import timedelta

transaction = Entity(
name="transaction_id",
join_keys=["transaction_id"],
)

transaction_source = FileSource(
name="transaction_features_source",
path="data/transaction_features.parquet",
timestamp_field="event_timestamp",
file_format=ParquetFormat(),
)

transaction_features = FeatureView(
name="transaction_features",
entities=[transaction],
ttl=timedelta(days=365),
schema=[
Field(name="Amount", dtype=Float32),
Field(name="Amount_Log", dtype=Float32),
Field(name="Amount_ZScore", dtype=Float32),
Field(name="Amount_Rank_Percentile", dtype=Float32),
Field(name="Time_Sin", dtype=Float32),
Field(name="Time_Cos", dtype=Float32),
],
source=transaction_source,
)
