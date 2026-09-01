from datetime import timedelta
from feast import Entity, FeatureView, Field, PushSource
from feast.types import Float32, Int64, String
from feast.infra.offline_stores.file_source import FileSource

# Define an entity for the account
account_entity = Entity(
    name="account_id",
    description="The ID of the bank account",
    value_type=String
)

# Define a batch source (required as a fallback for the PushSource)
offline_source = FileSource(
    name="account_features_offline",
    path="data/offline_store.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define the PushSource to receive stream updates
push_source = PushSource(
    name="graph_mutations_push_source",
    batch_source=offline_source,
)

# Define the Feature View that materializes the pushed data
account_graph_features = FeatureView(
    name="account_graph_features",
    entities=[account_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="transaction_volume_30d", dtype=Float32),
        Field(name="node_degree", dtype=Int64),
        Field(name="scc_community_id", dtype=Int64),
        Field(name="louvain_community_id", dtype=Int64),
        Field(name="tda_cycle_count", dtype=Int64),
        Field(name="tda_h1_persistence", dtype=Float32),
    ],
    online=True,
    source=push_source,
    description="Real-time graph features for AML GNN models",
)
